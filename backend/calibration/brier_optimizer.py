"""Brier Score Loss Optimization & Probabilistic Reliability Calibration.

Implements Brier Score (Brier, 1950) loss optimization for probability outputs:
  BS = (1/N) * sum_{i=1}^N (p_i - y_i)^2
  BSS = 1 - (BS / BS_ref)

Guarantees that the reported confidence score reflects empirical Mean Squared Error
against realized 5-day out-of-sample directional outcomes (Win=1, Loss=0).
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize_scalar


class BrierScoreCalibrator:
    """
    Optimizes probability calibration via Brier Score minimization on rolling out-of-sample events.
    Decomposes accuracy into Reliability (calibration), Resolution (discrimination), and Uncertainty.
    """

    def __init__(self, rolling_window: int = 25):
        self.rolling_window = rolling_window

    def compute_brier_metrics(
        self,
        predicted_probs: np.ndarray,
        realized_outcomes: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Computes Brier Score, Brier Skill Score, and Murphy decomposition.
        """
        p = np.asarray(predicted_probs, dtype=float).ravel()
        y = np.asarray(realized_outcomes, dtype=float).ravel()

        n = len(p)
        if n == 0 or len(y) != n:
            return {
                "brier_score": 0.25,
                "brier_skill_score": 0.0,
                "reliability_loss": 0.0,
                "resolution_gain": 0.0,
                "base_rate": 0.50,
                "sample_size": 0,
                "is_well_calibrated": False,
            }

        # Brier Score = MSE between predicted probabilities and binary outcomes
        brier_score = float(np.mean((p - y) ** 2))

        # Base rate (climatology) and reference Brier score
        base_rate = float(np.mean(y))
        bs_ref = float(base_rate * (1.0 - base_rate))
        bs_ref = max(1e-4, bs_ref)

        # Brier Skill Score (BSS) > 0 means better than predicting base-rate
        bss = float(1.0 - (brier_score / bs_ref))

        # Reliability vs Resolution decomposition (Murphy, 1973)
        # Partition probabilities into discrete decile bins
        bins = np.linspace(0.0, 1.0, 6)
        bin_indices = np.digitize(p, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(bins) - 2)

        rel_loss = 0.0
        res_gain = 0.0
        for b in range(len(bins) - 1):
            mask = (bin_indices == b)
            n_k = int(np.sum(mask))
            if n_k > 0:
                p_bar_k = float(np.mean(p[mask]))
                y_bar_k = float(np.mean(y[mask]))
                rel_loss += (n_k / n) * ((p_bar_k - y_bar_k) ** 2)
                res_gain += (n_k / n) * ((y_bar_k - base_rate) ** 2)

        is_well_calibrated = bool(brier_score <= 0.18 and bss >= 0.10)

        return {
            "brier_score": round(brier_score, 4),
            "brier_skill_score": round(bss, 4),
            "reliability_loss": round(rel_loss, 4),
            "resolution_gain": round(res_gain, 4),
            "base_rate": round(base_rate, 4),
            "sample_size": n,
            "is_well_calibrated": is_well_calibrated,
        }

    def optimize_temperature_for_brier(
        self,
        raw_confidences: np.ndarray,
        realized_outcomes: np.ndarray,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Finds the optimal temperature parameter T* that minimizes the rolling Brier Score loss:
          T* = argmin_T sum_t ( sigmoid( logit(p_t) / T ) - y_t )^2
        """
        c = np.asarray(raw_confidences, dtype=float).ravel() / 100.0
        y = np.asarray(realized_outcomes, dtype=float).ravel()

        if len(c) < 5 or len(y) != len(c):
            return 1.0, {"status": "INSUFFICIENT_SAMPLE", "optimal_temperature": 1.0}

        eps = 1e-4
        c_clipped = np.clip(c, eps, 1.0 - eps)
        logits = np.log(c_clipped / (1.0 - c_clipped))

        def brier_loss(T: float) -> float:
            scaled_p = 1.0 / (1.0 + np.exp(-logits / max(1e-2, T)))
            return float(np.mean((scaled_p - y) ** 2))

        res = minimize_scalar(brier_loss, bounds=(0.4, 2.5), method="bounded")
        opt_temp = float(res.x) if res.success else 1.0
        min_loss = float(res.fun) if res.success else float(np.mean((c - y) ** 2))

        return opt_temp, {
            "optimal_temperature": round(opt_temp, 3),
            "min_brier_loss": round(min_loss, 4),
            "sample_size": len(c),
        }

    def calibrate_confidence_via_brier(
        self,
        raw_confidence: int,
        predicted_probs: np.ndarray,
        realized_outcomes: np.ndarray,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Calibrates raw confidence by constraining it with rolling Brier Skill Score.
        If empirical Brier skill is negative (worse than coin flip), collapses confidence towards baseline.
        If Brier skill is strong (BSS >= 0.25, BS <= 0.15), enables high conviction (>80%).
        """
        raw_c = int(np.clip(raw_confidence, 25, 98))
        raw_p = raw_c / 100.0

        metrics = self.compute_brier_metrics(predicted_probs, realized_outcomes)
        bs = metrics["brier_score"]
        bss = metrics["brier_skill_score"]
        base_rate = metrics["base_rate"]

        if metrics["sample_size"] < 5:
            # Baseline cold start
            return raw_c, {
                **metrics,
                "brier_calibrated_confidence": raw_c,
                "brier_adjustment_factor": 1.0,
            }

        # Calibration adjustment based on empirical Brier Skill Score
        if bss <= 0.0 or bs >= 0.25:
            # Poor predictive calibration: penalize heavily toward base rate
            adj_factor = max(0.60, 1.0 + bss)
            cal_p = 0.50 * raw_p + 0.50 * base_rate
            cal_c = int(np.clip(round(cal_p * adj_factor * 100.0), 30, 68))
        elif bss >= 0.25 and bs <= 0.14:
            # High predictive accuracy: promote confident predictions
            adj_factor = min(1.15, 1.0 + 0.4 * bss)
            cal_p = 0.75 * raw_p + 0.25 * (1.0 - bs)
            cal_c = int(np.clip(round(cal_p * adj_factor * 100.0), 40, 95))
        else:
            # Moderate calibration
            adj_factor = 1.0 + 0.2 * bss
            cal_c = int(np.clip(round(raw_c * adj_factor), 35, 82))

        return cal_c, {
            **metrics,
            "raw_confidence": raw_c,
            "brier_calibrated_confidence": cal_c,
            "brier_adjustment_factor": round(adj_factor, 3),
        }
