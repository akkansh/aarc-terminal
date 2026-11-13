"""Bayesian Confidence Calibration via Beta-Binomial Conjugate Updating.

Upgrades:
1. Exponential decay forgetting factor (lambda = 0.85) directly scaling alpha and beta
   before evidence updates, enabling rapid recovery from historical regime errors.
2. Temperature scaling integration for logit probabilities.
3. Direct Gaussian HMM Regime-Conditional Confidence Gating.
"""

import json
import threading
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
from scipy.stats import beta as beta_dist

from backend.models.temperature_scaling import (
    scale_probability_temperature,
    DEFAULT_RISK_TEMPERATURE,
)
from backend.models.regime_gater import RegimeConditionalConfidenceGater


class BetaBinomialConfidenceCalibrator:
    """
    Bayesian confidence calibration engine replacing heuristic confidence scores
    with conjugate Beta-Binomial updating:
      - Prior: theta ~ Beta(alpha_0, beta_0) where theta is model predictive reliability.
      - Online Update: alpha_t = lambda * alpha_{t-1} + I(HIT), beta_t = lambda * beta_{t-1} + I(MISS)
      - Exponential decay forgetting factor (lambda = 0.85) ensures historical errors
        decay rapidly, allowing confidence to track recent predictive accuracy.
      - Temperature scaling & HMM regime gating ensure high conviction (>80) in clean regimes
        while strictly suppressing false positives in panic/choppy markets.
    """

    def __init__(
        self,
        alpha_0: float = 7.0,
        beta_0: float = 3.0,
        gamma: float = 0.85,  # lambda forgetting factor
        storage_path: str = "backend/data_cache/bayesian_calibration.json",
        temperature: float = DEFAULT_RISK_TEMPERATURE,
    ):
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.gamma = gamma  # Exponential decay forgetting factor lambda
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.temperature = temperature
        self._gater = RegimeConditionalConfidenceGater()
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
        Bayesian online update with exponential decay forgetting factor (lambda = 0.85):
          alpha_t = lambda * alpha_{t-1} + I(HIT)
          beta_t  = lambda * beta_{t-1}  + I(MISS)

        Directly multiplies existing alpha and beta values by lambda before adding evidence,
        allowing the AI's confidence score to quickly recover from historical regime errors.
        """
        with self._lock:
            state = self._load_state()

            # Update global parameters
            g = state.get("global", {"alpha": self.alpha_0, "beta": self.beta_0, "n_updates": 0})
            old_a = float(g.get("alpha", self.alpha_0))
            old_b = float(g.get("beta", self.beta_0))

            # Multiply existing alpha and beta by decay factor lambda
            decayed_a = max(1.0, self.gamma * old_a)
            decayed_b = max(1.0, self.gamma * old_b)
            new_a = decayed_a + (1.0 if is_hit else 0.0)
            new_b = decayed_b + (0.0 if is_hit else 1.0)

            g["alpha"] = round(new_a, 4)
            g["beta"] = round(new_b, 4)
            g["n_updates"] = g.get("n_updates", 0) + 1
            state["global"] = g

            # Update ticker-specific parameters if provided
            if ticker:
                tickers = state.setdefault("tickers", {})
                t_state = tickers.get(ticker, {"alpha": self.alpha_0, "beta": self.beta_0, "n_updates": 0})
                t_old_a = float(t_state.get("alpha", self.alpha_0))
                t_old_b = float(t_state.get("beta", self.beta_0))

                t_a = max(1.0, self.gamma * t_old_a) + (1.0 if is_hit else 0.0)
                t_b = max(1.0, self.gamma * t_old_b) + (0.0 if is_hit else 1.0)

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
            if t_params.get("n_updates", 0) >= 2:
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
            "forgetting_factor_lambda": self.gamma,
        }

    def calibrate_score(
        self,
        raw_ai_confidence: int,
        ticker: Optional[str] = None,
        hmm_regime: Optional[Dict[str, Any]] = None,
        action: str = "LONG",
        volatility_shock_ratio: float = 1.0,
        temperature: Optional[float] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Calibrates raw AI confidence score against the Bayesian posterior, temperature scaling,
        and Gaussian HMM regime gating:

        1. Temperature Scaling: Scales logits so strong signals cleanly cross 0.80+.
        2. Bayesian Posterior Blending: Weights recent empirical hit-rate without rigid historical penalties.
        3. Regime-Conditional Gating: Allows high confidence (>80) in clean/trending regimes,
           while suppressing low-tier alerts and capping confidence in panic/choppy regimes.

        Returns: (calibrated_confidence: int [25, 95], telemetry: Dict)
        """
        use_temp = self.temperature if temperature is None else float(temperature)
        raw_conf = int(np.clip(raw_ai_confidence, 25, 98))

        # 1. Temperature Scaling of Raw Model Output
        raw_prob = float(raw_conf) / 100.0
        scaled_prob, temp_telemetry = scale_probability_temperature(raw_prob, temperature=use_temp)

        # 2. Bayesian Posterior Blending
        summary = self.get_posterior_summary(ticker)
        post_mean = summary["posterior_mean_accuracy"]
        post_std = summary["posterior_std"]
        ci_lower = summary["credible_interval_90"][0]

        # Weight allocation: higher weight to model conviction when post_mean is healthy
        w_model = 0.65 if post_mean >= 0.65 else 0.45
        blended_prob = (w_model * scaled_prob) + ((1.0 - w_model) * post_mean)

        # Dynamic uncertainty penalty proportional to posterior spread and historical inaccuracy
        # (Prevents rigid -15% deductions that previously trapped scores in the 60s)
        uncertainty_penalty = float(np.clip(1.2 * post_std * max(0.0, 0.80 - post_mean), 0.0, 0.08))
        calibrated_prob = max(ci_lower, blended_prob - uncertainty_penalty)

        pre_gated_conf = int(np.clip(round(calibrated_prob * 100.0), 25, 95))

        # 3. Regime-Conditional Confidence Gating (HMM Integration)
        gated_conf, regime_telemetry = self._gater.gate_confidence(
            confidence=pre_gated_conf,
            hmm_regime=hmm_regime,
            action=action,
            volatility_shock_ratio=volatility_shock_ratio,
        )

        final_conf = gated_conf

        # Allocation multiplier
        if regime_telemetry.get("is_suppressed", False):
            alloc_mult = 0.0
        elif ci_lower < 0.40 or "PANIC" in regime_telemetry.get("regime_name", ""):
            alloc_mult = 0.50
        elif post_mean >= 0.70 and post_std < 0.12 and final_conf >= 80:
            alloc_mult = 1.20
        else:
            alloc_mult = 1.00

        telemetry = {
            "raw_ai_confidence": raw_conf,
            "calibrated_confidence": final_conf,
            "posterior_mean_accuracy_pct": round(post_mean * 100.0, 1),
            "credible_lower_bound_pct": round(ci_lower * 100.0, 1),
            "credible_upper_bound_pct": round(summary["credible_interval_90"][1] * 100.0, 1),
            "bayesian_allocation_multiplier": alloc_mult,
            "alpha": summary["alpha"],
            "beta": summary["beta"],
            "sample_updates": summary["n_updates"],
            "forgetting_factor_lambda": self.gamma,
            "temperature_scaling": temp_telemetry,
            "regime_gating": regime_telemetry,
        }

        return final_conf, telemetry
