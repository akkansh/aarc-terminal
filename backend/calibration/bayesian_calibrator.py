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

    def update_evidence(
        self,
        is_hit: bool,
        ticker: Optional[str] = None,
        is_high_conf_divergence: bool = False,
    ) -> Dict[str, Any]:
        """
        Bayesian online update with exponential decay forgetting factor (lambda = 0.85):
          alpha_t = lambda * alpha_{t-1} + I(HIT)
          beta_t  = lambda * beta_{t-1}  + I(MISS)

        If is_high_conf_divergence is True (model asserted >=80% confidence but suffered
        a realized stop-loss breach or directional failure), applies an immediate asymmetric penalty:
          alpha_t = 0.70 * lambda * alpha_{t-1}
          beta_t  = lambda * beta_{t-1} + 2.5
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

            if is_high_conf_divergence:
                new_a = max(1.0, decayed_a * 0.70)
                new_b = decayed_b + 2.5
            else:
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

                t_decayed_a = max(1.0, self.gamma * t_old_a)
                t_decayed_b = max(1.0, self.gamma * t_old_b)

                if is_high_conf_divergence:
                    t_a = max(1.0, t_decayed_a * 0.70)
                    t_b = t_decayed_b + 2.5
                else:
                    t_a = t_decayed_a + (1.0 if is_hit else 0.0)
                    t_b = t_decayed_b + (0.0 if is_hit else 1.0)

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
        expected_shortfall_pct: Optional[float] = None,
        empirical_coverage_pct: Optional[float] = None,
        brier_metrics: Optional[Dict[str, Any]] = None,
        walk_forward_metrics: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Calibrates raw AI confidence score against the Bayesian posterior, temperature scaling,
        Gaussian HMM regime gating, rolling Brier Score loss, Walk-Forward Validation (WFV),
        and quantitative Expected Shortfall (ES) tail risk grounding.

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

        # 4. Brier Score Loss & Probabilistic Skill Gating
        brier_telemetry: Dict[str, Any] = {}
        if brier_metrics:
            bss = float(brier_metrics.get("brier_skill_score", 0.0))
            bs = float(brier_metrics.get("brier_score", 0.22))
            n_samples = int(brier_metrics.get("sample_size", 0))
            if n_samples >= 4:
                if bss <= 0.0 or bs >= 0.25:
                    # Inverted or zero skill: shrink toward base rate and impose strict 65 cap
                    brier_factor = max(0.65, 1.0 + bss)
                    gated_conf = min(65, int(round(gated_conf * brier_factor)))
                elif bss >= 0.25 and bs <= 0.14:
                    brier_factor = min(1.12, 1.0 + 0.3 * bss)
                    gated_conf = int(round(gated_conf * brier_factor))
                else:
                    brier_factor = 1.0 + 0.15 * bss
                    gated_conf = int(round(gated_conf * brier_factor))
                brier_telemetry = {
                    "brier_score": bs,
                    "brier_skill_score": bss,
                    "brier_adjustment_factor": round(brier_factor, 3),
                    "is_well_calibrated": brier_metrics.get("is_well_calibrated", False),
                }
            else:
                brier_telemetry = {
                    "status": "COLD_START_NEUTRAL",
                    "brier_score": bs,
                    "brier_skill_score": bss,
                }

        # 5. Walk-Forward Validation (WFV) Gating
        wfv_telemetry: Dict[str, Any] = {}
        is_wfv_gated = False
        if walk_forward_metrics:
            wfv_mult = float(walk_forward_metrics.get("wfv_scaling_factor", 1.0))
            is_wfv_gated = bool(walk_forward_metrics.get("is_gated", False))
            gated_conf = int(round(gated_conf * wfv_mult))
            if is_wfv_gated:
                # Strictly prevent AI confidence >75% when out-of-sample validation degrades
                gated_conf = min(gated_conf, 75)
            wfv_telemetry = {
                "walk_forward_status": walk_forward_metrics.get("walk_forward_status", "ALIGNED"),
                "oos_spearman_rho": walk_forward_metrics.get("oos_spearman_rho", 0.0),
                "high_conf_hit_rate": walk_forward_metrics.get("high_conf_hit_rate", 0.75),
                "wfv_scaling_factor": wfv_mult,
                "is_gated": is_wfv_gated,
                "gating_reason": walk_forward_metrics.get("gating_reason"),
            }

        # 6. Expected Shortfall (ES) & VaR Tail-Risk Grounding
        tail_grounding_telemetry: Dict[str, Any] = {}
        if empirical_coverage_pct is not None or expected_shortfall_pct is not None:
            es_cap = 95
            if empirical_coverage_pct is not None and empirical_coverage_pct > 0:
                # Reported confidence is bounded by empirical coverage (e.g. 91% if breach freq is 9%)
                es_cap = int(np.floor(empirical_coverage_pct))
                if gated_conf > es_cap:
                    gated_conf = es_cap

            es_factor = 1.0
            if expected_shortfall_pct is not None:
                abs_es = abs(float(expected_shortfall_pct))
                if abs_es > 3.5:
                    # Scale down conviction if fat-tail loss exceeds 3.5% daily
                    es_factor = max(0.70, 3.5 / abs_es)
                    gated_conf = int(round(gated_conf * es_factor))

            tail_grounding_telemetry = {
                "empirical_coverage_pct": empirical_coverage_pct,
                "expected_shortfall_95_pct": expected_shortfall_pct,
                "es_grounding_cap_pct": es_cap,
                "es_penalty_factor": round(es_factor, 3),
            }

        final_conf = int(np.clip(gated_conf, 25, 95))

        # Dynamic allocation multiplier
        if regime_telemetry.get("is_suppressed", False):
            alloc_mult = 0.0
        elif ci_lower < 0.40 or "PANIC" in regime_telemetry.get("regime_name", ""):
            alloc_mult = 0.50
        elif is_wfv_gated or (brier_telemetry.get("brier_skill_score", 0.0) < 0):
            alloc_mult = 0.65
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
            "brier_score_gating": brier_telemetry,
            "walk_forward_validation": wfv_telemetry,
            "tail_risk_grounding": tail_grounding_telemetry,
        }

        return final_conf, telemetry
