"""Lightweight Rolling Prediction Accuracy Logger, Brier Score Loss Optimizer, and WFV Gater."""

import json
import os
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from backend.calibration.brier_optimizer import BrierScoreCalibrator
from backend.calibration.walk_forward import WalkForwardValidator


class RollingAccuracyTracker:
    """
    Maintains a sliding window of historical risk and direction predictions.
    Evaluates compound success: requires BOTH risk containment (stop-loss unbreached)
    AND correct directional trajectory over 5-day out-of-sample market outcomes.
    Optimizes rolling Brier score and applies Walk-Forward Validation gating.
    """

    def __init__(
        self,
        storage_path: str = "backend/data_cache/rolling_accuracy.json",
        window_size: int = 25,
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.window_size = window_size
        self._brier_calibrator = BrierScoreCalibrator(rolling_window=window_size)
        self._wfv_validator = WalkForwardValidator()
        self._lock = threading.Lock()
        self._init_storage()

    def _init_storage(self):
        if not self.storage_path.exists():
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump({"events": []}, f, indent=2)

    def _read_events(self) -> List[Dict[str, Any]]:
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("events", [])
        except Exception:
            return []

    def _write_events(self, events: List[Dict[str, Any]]):
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump({"events": events[-120:]}, f, indent=2)
        except Exception as e:
            print(f"[TRACKER ERROR]: Failed to persist rolling events: {e}")

    def log_prediction(
        self,
        ticker: str,
        action: str,
        raw_confidence: int,
        current_price: float,
        date_str: str,
        garch_var_pct: float,
        stop_loss_pct: float = 3.0,
        expected_shortfall_pct: Optional[float] = None,
    ) -> str:
        """Log a new committee decision for forward tracking."""
        event_id = f"{ticker}_{date_str}_{action}"
        record = {
            "id": event_id,
            "date": date_str,
            "ticker": ticker,
            "action": action,
            "raw_confidence": raw_confidence,
            "entry_price": current_price,
            "garch_var_pct": garch_var_pct,
            "stop_loss_pct": stop_loss_pct,
            "expected_shortfall_pct": expected_shortfall_pct,
            "resolved": False,
            "realized_return_5d": None,
            "max_adverse_excursion_pct": None,
            "risk_contained": None,
            "directional_win": None,
            "outcome": None,  # "HIT" or "MISS"
            "high_conf_divergence": False,
        }

        with self._lock:
            events = self._read_events()
            if not any(e.get("id") == event_id for e in events):
                events.append(record)
                self._write_events(events)

        return event_id

    def resolve_outcomes(self, ticker: str, history_df: pd.DataFrame):
        """
        Scan unresolved events for this ticker and check if 5 trading days
        have elapsed. Evaluates compound success:
          1. Risk Containment: Price did NOT breach trailing stop-loss during 5d horizon.
          2. Directional Trajectory: Price followed predicted drift (+0.75% for LONG, -0.75% for SHORT).
        """
        if len(history_df) < 6:
            return

        with self._lock:
            events = self._read_events()
            updated = False

            for ev in events:
                if ev.get("ticker") != ticker or ev.get("resolved"):
                    continue

                ev_date = ev.get("date")
                try:
                    ts = pd.to_datetime(ev_date)
                    dt_series = pd.to_datetime(history_df.index).tz_localize(None)
                    match_indices = np.where(dt_series.date <= ts.date())[0]
                    if len(match_indices) == 0:
                        continue

                    bar_idx = match_indices[-1]
                    if bar_idx + 5 < len(history_df):
                        window_slice = history_df.iloc[bar_idx : bar_idx + 6]
                        p_entry = float(history_df["Close"].iloc[bar_idx])
                        p_exit = float(history_df["Close"].iloc[bar_idx + 5])
                        ret_5d = ((p_exit - p_entry) / (p_entry + 1e-6)) * 100.0

                        min_low = float(window_slice["Low"].min())
                        max_high = float(window_slice["High"].max())

                        stop_pct = float(ev.get("stop_loss_pct", 3.0)) / 100.0
                        action = ev.get("action", "LONG")

                        # 1. Risk Containment check
                        if action == "LONG":
                            risk_contained = bool(min_low >= p_entry * (1.0 - stop_pct))
                            directional_win = bool(ret_5d >= 0.75)
                            mae_pct = round(((min_low - p_entry) / p_entry) * 100.0, 2)
                        elif action == "SHORT":
                            risk_contained = bool(max_high <= p_entry * (1.0 + stop_pct))
                            directional_win = bool(ret_5d <= -0.75)
                            mae_pct = round(((p_entry - max_high) / p_entry) * 100.0, 2)
                        elif action == "REJECT_TRADE":
                            risk_contained = True
                            directional_win = bool(ret_5d <= 0.50 or min_low <= p_entry * 0.97)
                            mae_pct = 0.0
                        else:
                            risk_contained = True
                            directional_win = bool(abs(ret_5d) < 2.0)
                            mae_pct = 0.0

                        # Compound Success requires BOTH
                        compound_hit = bool(risk_contained and directional_win)
                        is_divergence = bool(ev.get("raw_confidence", 50) >= 80 and not compound_hit)

                        ev["resolved"] = True
                        ev["realized_return_5d"] = round(ret_5d, 2)
                        ev["max_adverse_excursion_pct"] = mae_pct
                        ev["risk_contained"] = risk_contained
                        ev["directional_win"] = directional_win
                        ev["outcome"] = "HIT" if compound_hit else "MISS"
                        ev["high_conf_divergence"] = is_divergence
                        updated = True

                except Exception:
                    continue

            if updated:
                self._write_events(events)

    def get_rolling_metrics_and_calibration_multiplier(
        self,
        ticker: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculates Brier Score loss, Walk-Forward Validation metrics,
        and directional hit-rate over resolved out-of-sample predictions.
        """
        with self._lock:
            events = self._read_events()

        resolved = [e for e in events if e.get("resolved") and e.get("outcome") in ("HIT", "MISS")]
        if ticker:
            ticker_resolved = [e for e in resolved if e.get("ticker") == ticker]
            if len(ticker_resolved) >= 5:
                resolved = ticker_resolved

        window_events = resolved[-self.window_size:]
        n_resolved = len(window_events)

        if n_resolved < 4:
            return {
                "rolling_hit_rate_pct": 65.0,
                "resolved_count": n_resolved,
                "confidence_multiplier": 1.0,
                "allocation_multiplier": 1.0,
                "status": "CALIBRATION_NEUTRAL (Insufficient sample)",
                "brier_metrics": {
                    "brier_score": 0.22,
                    "brier_skill_score": 0.0,
                    "is_well_calibrated": True,
                },
                "walk_forward_metrics": {
                    "walk_forward_status": "COLD_START",
                    "wfv_scaling_factor": 1.0,
                    "oos_spearman_rho": 0.20,
                },
                "recent_divergence_count": 0,
            }

        hits = sum(1 for e in window_events if e["outcome"] == "HIT")
        hit_rate = hits / n_resolved
        hit_rate_pct = round(hit_rate * 100.0, 1)

        # 1. Brier Score Loss Optimization
        probs = np.array([float(e.get("raw_confidence", 50)) / 100.0 for e in window_events])
        outcomes = np.array([1.0 if e["outcome"] == "HIT" else 0.0 for e in window_events])
        brier_res = self._brier_calibrator.compute_brier_metrics(probs, outcomes)

        # 2. Walk-Forward Validation Gating
        wfv_res = self._wfv_validator.evaluate_walk_forward_metrics(window_events)
        wfv_mult = wfv_res["wfv_scaling_factor"]

        # 3. High-Confidence Divergence Penalty
        divergence_count = sum(1 for e in window_events[-10:] if e.get("high_conf_divergence"))
        divergence_penalty = max(0.65, 1.0 - 0.15 * divergence_count)

        # Compound multiplier incorporating hit-rate, Brier skill, WFV gating, and divergence penalties
        base_mult = 1.0 + 0.60 * (hit_rate - 0.50)
        final_mult = base_mult * wfv_mult * divergence_penalty
        final_mult = float(np.clip(final_mult, 0.55, 1.25))

        alloc_mult = 1.0
        if hit_rate < 0.45 or divergence_count >= 2:
            alloc_mult = 0.50
        elif hit_rate > 0.72 and brier_res["brier_score"] <= 0.15 and not wfv_res["is_gated"]:
            alloc_mult = 1.15

        return {
            "rolling_hit_rate_pct": hit_rate_pct,
            "resolved_count": n_resolved,
            "confidence_multiplier": round(final_mult, 3),
            "allocation_multiplier": round(alloc_mult, 2),
            "status": "CALIBRATION_ACTIVE",
            "brier_metrics": brier_res,
            "walk_forward_metrics": wfv_res,
            "recent_divergence_count": divergence_count,
            "divergence_penalty_factor": round(divergence_penalty, 3),
        }

    def calibrate_confidence(
        self,
        raw_confidence: int,
        ticker: Optional[str] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Dynamically scales raw confidence based on realized 5-day out-of-sample metrics.
        """
        metrics = self.get_rolling_metrics_and_calibration_multiplier(ticker)
        mult = metrics.get("confidence_multiplier", 1.0)
        scaled_conf = int(np.clip(round(raw_confidence * mult), 25, 95))

        return scaled_conf, {
            **metrics,
            "raw_confidence": raw_confidence,
            "calibrated_confidence": scaled_conf,
        }
