"""Bayesian Confidence Calibration via Beta-Binomial Conjugate Updating."""

import json
import threading
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
from scipy.stats import beta as beta_dist


class BetaBinomialConfidenceCalibrator:
    """
    Bayesian confidence calibration engine replacing heuristic confidence scores
    with conjugate Beta-Binomial updating:
      - Prior: theta ~ Beta(alpha_0, beta_0) where theta is model predictive reliability.
      - Online Update: alpha += 1 (on true positive / hit), beta += 1 (on false positive / miss).
      - Exponential decay (gamma = 0.98) accounts for non-stationary market regimes.
      - Generates posterior expected accuracy, variance, and 90% credible intervals.
    """

    def __init__(
        self,
        alpha_0: float = 7.0,
        beta_0: float = 3.0,
        gamma: float = 0.95,
        storage_path: str = "backend/data_cache/bayesian_calibration.json",
    ):
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.gamma = gamma
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_storage()

    def _init_storage(self):
        if not self.storage_path.exists():
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump({
                    "global": {"alpha": self.alpha_0, "beta": self.beta_0, "n_updates": 0},
                    "tickers": {},
                }, f, indent=2)

    def _load_state(self) -> Dict[str, Any]:
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {
                "global": {"alpha": self.alpha_0, "beta": self.beta_0, "n_updates": 0},
                "tickers": {},
            }

    def _save_state(self, state: Dict[str, Any]):
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"[BAYESIAN ERROR]: Failed to persist state: {e}")

    def update_evidence(self, is_hit: bool, ticker: Optional[str] = None) -> Dict[str, Any]:
        """
        Bayesian online update:
        alpha_t = gamma * alpha_{t-1} + I(HIT)
        beta_t  = gamma * beta_{t-1}  + I(MISS)
        """
        with self._lock:
            state = self._load_state()

            # Update global parameters
            g = state.get("global", {"alpha": self.alpha_0, "beta": self.beta_0, "n_updates": 0})
            old_a, old_b = g["alpha"], g["beta"]

            new_a = (self.gamma * (old_a - self.alpha_0)) + self.alpha_0 + (1.0 if is_hit else 0.0)
            new_b = (self.gamma * (old_b - self.beta_0)) + self.beta_0 + (0.0 if is_hit else 1.0)
            g["alpha"] = round(new_a, 4)
            g["beta"] = round(new_b, 4)
            g["n_updates"] = g.get("n_updates", 0) + 1
            state["global"] = g

            # Update ticker-specific parameters if provided
            if ticker:
                tickers = state.setdefault("tickers", {})
                t_state = tickers.get(ticker, {"alpha": self.alpha_0, "beta": self.beta_0, "n_updates": 0})
                t_a = (self.gamma * (t_state["alpha"] - self.alpha_0)) + self.alpha_0 + (1.0 if is_hit else 0.0)
                t_b = (self.gamma * (t_state["beta"] - self.beta_0)) + self.beta_0 + (0.0 if is_hit else 1.0)
                t_state["alpha"] = round(t_a, 4)
                t_state["beta"] = round(t_b, 4)
                t_state["n_updates"] = t_state.get("n_updates", 0) + 1
                tickers[ticker] = t_state

            self._save_state(state)
            return self.get_posterior_summary(ticker)

    def get_posterior_summary(self, ticker: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculates posterior Beta distribution moments:
          - Expected accuracy: E[theta] = alpha / (alpha + beta)
          - Variance: Var[theta] = alpha * beta / ((alpha + beta)^2 * (alpha + beta + 1))
          - 90% Credible Interval: [theta_0.05, theta_0.95]
        """
        with self._lock:
            state = self._load_state()

        params = state.get("global", {"alpha": self.alpha_0, "beta": self.beta_0, "n_updates": 0})
        if ticker and ticker in state.get("tickers", {}):
            t_params = state["tickers"][ticker]
            if t_params.get("n_updates", 0) >= 3:
                params = t_params

        a = max(1.0, float(params["alpha"]))
        b = max(1.0, float(params["beta"]))

        post_mean = a / (a + b)
        post_var = (a * b) / (((a + b) ** 2) * (a + b + 1.0))
        post_std = np.sqrt(post_var)

        ci_lower = float(beta_dist.ppf(0.05, a, b))
        ci_upper = float(beta_dist.ppf(0.95, a, b))

        return {
            "alpha": round(a, 2),
            "beta": round(b, 2),
            "n_updates": int(params.get("n_updates", 0)),
            "posterior_mean_accuracy": round(post_mean, 4),
            "posterior_std": round(post_std, 4),
            "credible_interval_90": [round(ci_lower, 3), round(ci_upper, 3)],
        }

    def calibrate_score(
        self,
        raw_ai_confidence: int,
        ticker: Optional[str] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Calibrates raw AI confidence score against the Bayesian posterior:
        Combines model output with posterior mean and penalizes high posterior uncertainty.
        Returns: (calibrated_confidence: int [25, 95], telemetry: Dict)
        """
        summary = self.get_posterior_summary(ticker)
        post_mean = summary["posterior_mean_accuracy"]
        post_std = summary["posterior_std"]
        ci_lower = summary["credible_interval_90"][0]

        # Weight: 60% empirical Bayesian posterior, 40% current model conviction
        model_frac = float(raw_ai_confidence) / 100.0
        blended_prob = (0.40 * model_frac) + (0.60 * post_mean)

        # Uncertainty penalty based on posterior standard deviation
        uncertainty_penalty = float(np.clip(1.5 * post_std, 0.0, 0.15))
        calibrated_prob = max(ci_lower, blended_prob - uncertainty_penalty)

        final_conf = int(np.clip(round(calibrated_prob * 100.0), 25, 95))

        # Recommended allocation multiplier
        if ci_lower < 0.40:
            alloc_mult = 0.50  # Heavy defensive contraction
        elif post_mean > 0.70 and post_std < 0.10:
            alloc_mult = 1.15  # Statistically verified edge
        else:
            alloc_mult = 1.00

        telemetry = {
            "raw_ai_confidence": raw_ai_confidence,
            "calibrated_confidence": final_conf,
            "posterior_mean_accuracy_pct": round(post_mean * 100.0, 1),
            "credible_lower_bound_pct": round(ci_lower * 100.0, 1),
            "credible_upper_bound_pct": round(summary["credible_interval_90"][1] * 100.0, 1),
            "bayesian_allocation_multiplier": alloc_mult,
            "alpha": summary["alpha"],
            "beta": summary["beta"],
            "sample_updates": summary["n_updates"],
        }

        return final_conf, telemetry
