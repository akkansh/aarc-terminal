"""Lightweight Rolling Prediction Accuracy Logger and Dynamic AI Confidence Scaler."""

import json
import os
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class RollingAccuracyTracker:
    """
    Maintains a sliding window of historical risk and direction predictions,
    evaluating them against realized 5-day price action.
    Dynamically scales the final AI confidence score and allocation based on recent hit rate.
    """

    def __init__(
        self,
        storage_path: str = "backend/data_cache/rolling_accuracy.json",
        window_size: int = 20,
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.window_size = window_size
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
                json.dump({"events": events[-100:]}, f, indent=2)
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
            "resolved": False,
            "realized_return_5d": None,
            "outcome": None,  # "HIT" or "MISS"
        }

        with self._lock:
            events = self._read_events()
            # Avoid duplicate logs for same ticker and date
            if not any(e.get("id") == event_id for e in events):
                events.append(record)
                self._write_events(events)

        return event_id

    def resolve_outcomes(self, ticker: str, history_df: pd.DataFrame):
        """
        Scan unresolved events for this ticker and check if 5 trading days
        have elapsed to evaluate realized return and assign HIT/MISS.
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
                    # Find bar position of event
                    ts = pd.to_datetime(ev_date)
                    dt_series = pd.to_datetime(history_df.index).tz_localize(None)
                    match_indices = np.where(dt_series.date <= ts.date())[0]
                    if len(match_indices) == 0:
                        continue

                    bar_idx = match_indices[-1]
                    # Check if at least 5 subsequent bars are available
                    if bar_idx + 5 < len(history_df):
                        p_entry = float(history_df["Close"].iloc[bar_idx])
                        p_exit = float(history_df["Close"].iloc[bar_idx + 5])
                        ret_5d = ((p_exit - p_entry) / (p_entry + 1e-6)) * 100.0

                        action = ev.get("action", "LONG")
                        # Evaluate outcome
                        if action == "LONG":
                            hit = (ret_5d > 0.3)
                        elif action == "SHORT":
                            hit = (ret_5d < -0.3)
                        elif action == "REJECT_TRADE":
                            # Successful rejection if asset dropped or stayed flat
                            hit = (ret_5d <= 0.5)
                        else:
                            hit = (abs(ret_5d) < 2.0)

                        ev["resolved"] = True
                        ev["realized_return_5d"] = round(ret_5d, 2)
                        ev["outcome"] = "HIT" if hit else "MISS"
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
        Calculates rolling hit rate over the sliding window.
        Computes dynamic confidence multiplier:
        Multiplier = clip(1.0 + 0.8 * (HitRate - 0.50), 0.70, 1.25)
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
            # Cold start: neutral baseline
            return {
                "rolling_hit_rate_pct": 65.0,
                "resolved_count": n_resolved,
                "confidence_multiplier": 1.0,
                "allocation_multiplier": 1.0,
                "status": "CALIBRATION_NEUTRAL (Insufficient sample)",
                "recent_history": [],
            }

        hits = sum(1 for e in window_events if e["outcome"] == "HIT")
        hit_rate = hits / n_resolved
        hit_rate_pct = round(hit_rate * 100.0, 1)

        # Scale multiplier: 50% baseline -> 1.0
        # 75% accuracy -> 1.20 multiplier
        # 30% accuracy -> 0.84 multiplier
        mult = 1.0 + 0.80 * (hit_rate - 0.50)
        mult = float(np.clip(mult, 0.70, 1.25))

        # Dynamic allocation safety throttle if hit rate falls below 45%
        if hit_rate < 0.45:
            alloc_mult = 0.60
            status_desc = f"CAUTION: Rolling win rate ({hit_rate_pct}%) below 45% threshold. Confidence dampened, sizing throttled."
        elif hit_rate >= 0.65:
            alloc_mult = 1.15
            status_desc = f"MOMENTUM: High rolling accuracy ({hit_rate_pct}%). Sizing and confidence boosted."
        else:
            alloc_mult = 1.0
            status_desc = f"NOMINAL: Balanced predictive accuracy ({hit_rate_pct}%)."

        recent_summary = [
            {"date": e["date"], "action": e["action"], "ret_5d": e["realized_return_5d"], "outcome": e["outcome"]}
            for e in window_events[-5:]
        ]

        return {
            "rolling_hit_rate_pct": hit_rate_pct,
            "resolved_count": n_resolved,
            "confidence_multiplier": round(mult, 2),
            "allocation_multiplier": round(alloc_mult, 2),
            "status": status_desc,
            "recent_history": recent_summary,
        }

    def calibrate_confidence(
        self,
        raw_confidence: int,
        ticker: Optional[str] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Dynamically adjusts raw AI confidence score based on recent rolling hit rate.
        Returns: (calibrated_confidence: int [10, 99], telemetry: Dict)
        """
        telemetry = self.get_rolling_metrics_and_calibration_multiplier(ticker=ticker)
        mult = telemetry["confidence_multiplier"]

        # Scale and clamp to [25, 95]
        adjusted_conf = int(np.clip(round(raw_confidence * mult), 25, 95))
        telemetry["raw_confidence"] = raw_confidence
        telemetry["calibrated_confidence"] = adjusted_conf

        return adjusted_conf, telemetry
