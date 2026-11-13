"""Regime-Conditional Confidence Gating linked to Gaussian HMM Market States."""

from typing import Dict, Any, Optional, Tuple
import numpy as np


class RegimeConditionalConfidenceGater:
    """
    Regime-conditional confidence gate linking scoring directly to 3-state Gaussian HMM:
      - LOW_VOL_TRENDING: Clean directional environment. Permits high-confidence scores (>80).
      - HIGH_VOL_PANIC: Severe macro uncertainty. Suppresses low-tier directional alerts
        and imposes defensive ceilings to prevent catastrophic whipsaws.
      - MEAN_REVERTING_CHOPPY: Sideways/choppy regime. Suppresses marginal alerts (<75)
        and discounts confidence to reflect regime uncertainty.
    """

    def __init__(
        self,
        high_confidence_threshold: int = 80,
        panic_confidence_cap: int = 68,
        choppy_confidence_cap: int = 74,
        choppy_suppression_cutoff: int = 75,
    ):
        self.high_confidence_threshold = high_confidence_threshold
        self.panic_confidence_cap = panic_confidence_cap
        self.choppy_confidence_cap = choppy_confidence_cap
        self.choppy_suppression_cutoff = choppy_suppression_cutoff

    def gate_confidence(
        self,
        confidence: int,
        hmm_regime: Optional[Dict[str, Any]] = None,
        action: str = "LONG",
        volatility_shock_ratio: float = 1.0,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Gates and scales confidence based on the active HMM market regime.

        Returns:
          (gated_confidence: int, gating_telemetry: Dict[str, Any])
        """
        raw_conf = int(np.clip(confidence, 20, 98))
        regime = hmm_regime or {}
        regime_name = str(regime.get("regime_name", "LOW_VOL_TRENDING")).upper()
        probs = regime.get("probabilities", {})
        p_trending = float(probs.get("P_Trending", 0.60))
        p_panic = float(probs.get("P_Panic", 0.15))
        p_choppy = float(probs.get("P_Choppy", 0.25))

        is_suppressed = False
        suppression_reason: Optional[str] = None
        regime_scale_factor = 1.0
        gated_conf = raw_conf

        # 1. HIGH VOLATILITY / PANIC REGIME
        if "PANIC" in regime_name or p_panic >= 0.40 or volatility_shock_ratio > 1.8:
            # Capital preservation mandates (REJECT_TRADE) are allowed high confidence
            if action == "REJECT_TRADE":
                regime_scale_factor = 1.05
                gated_conf = int(min(95, round(raw_conf * regime_scale_factor)))
                gating_status = "PANIC_DEFENSE_APPROVED"
            else:
                # Directional bets in panic are high-risk; suppress low-tier alerts
                if raw_conf < 78:
                    is_suppressed = True
                    suppression_reason = (
                        f"Low-tier {action} alert suppressed: Active market regime is HIGH_VOL_PANIC "
                        f"(P_Panic: {p_panic:.1%}, Vol Shock: {volatility_shock_ratio:.2f}x)."
                    )
                # Scale down confidence and enforce panic ceiling
                regime_scale_factor = 0.78
                gated_conf = int(np.clip(round(raw_conf * regime_scale_factor), 25, self.panic_confidence_cap))
                gating_status = "SUPPRESSED_PANIC_UNCERTAINTY" if is_suppressed else "PANIC_SCALED"

        # 2. MEAN-REVERTING / CHOPPY SIDEWAYS REGIME
        elif "CHOPPY" in regime_name or p_choppy >= 0.45:
            # Suppress low-tier alerts below cutoff
            if raw_conf < self.choppy_suppression_cutoff and action != "REJECT_TRADE":
                is_suppressed = True
                suppression_reason = (
                    f"Low-tier {action} alert suppressed: Active market regime is MEAN_REVERTING_CHOPPY "
                    f"(P_Choppy: {p_choppy:.1%}). Noise threshold requires >= {self.choppy_suppression_cutoff}% conviction."
                )
            # Scale confidence down to reflect choppy whipsaw risk
            regime_scale_factor = 0.85
            gated_conf = int(np.clip(round(raw_conf * regime_scale_factor), 25, self.choppy_confidence_cap))
            gating_status = "SUPPRESSED_CHOPPY_NOISE" if is_suppressed else "CHOPPY_SCALED"

        # 3. LOW VOLATILITY / TRENDING REGIME (Clean Market State)
        else:
            # Clean trending regime promotes high conviction
            if p_trending >= 0.65 and raw_conf >= 75:
                regime_scale_factor = 1.08  # Boost strong signals in trending regimes
                gated_conf = int(np.clip(round(raw_conf * regime_scale_factor), 25, 96))
            else:
                regime_scale_factor = 1.02
                gated_conf = int(np.clip(round(raw_conf * regime_scale_factor), 25, 95))

            gating_status = "CLEAN_TRENDING_PROMOTED"

        telemetry = {
            "regime_name": regime_name,
            "regime_status": gating_status,
            "is_suppressed": is_suppressed,
            "suppression_reason": suppression_reason,
            "regime_scale_factor": round(regime_scale_factor, 3),
            "raw_input_confidence": raw_conf,
            "gated_confidence": gated_conf,
            "cleared_high_confidence": bool(gated_conf >= self.high_confidence_threshold),
            "p_trending": round(p_trending, 3),
            "p_panic": round(p_panic, 3),
            "p_choppy": round(p_choppy, 3),
        }

        return gated_conf, telemetry
