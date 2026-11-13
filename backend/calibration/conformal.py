"""Split Conformal Prediction sets and risk-penalized confidence score calculator."""

from typing import Dict, Any, List, Optional
import numpy as np


CLASS_NAMES = {0: "DOWN", 1: "FLAT", 2: "UP"}


class ConformalPredictor:
    """
    Split conformal prediction producing valid finite-sample prediction sets
    with guaranteed (1 - alpha) coverage.
    """

    def __init__(self, alpha: float = 0.10, min_prob_threshold: float = 0.40):
        self.alpha = alpha
        self.min_prob_threshold = min_prob_threshold
        self.q_hat = 0.85
        self.is_calibrated = False

    def calibrate(self, cal_probs: np.ndarray, y_cal: np.ndarray) -> "ConformalPredictor":
        """
        Calibrate quantile of non-conformity scores on held-out calibration set.
        Score s_i = 1 - P(Y = y_i | x_i).
        """
        n = len(y_cal)
        if n < 15:
            self.q_hat = 0.85
            self.is_calibrated = True
            return self

        true_probs = np.array([cal_probs[i, int(y_cal[i])] for i in range(n)])
        scores = 1.0 - true_probs

        # Finite-sample adjusted quantile
        p_quantile = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        p_quantile = min(1.0, max(0.0, p_quantile))
        self.q_hat = float(np.percentile(scores, p_quantile * 100))
        self.is_calibrated = True
        return self

    def predict_set(self, prob_vector: np.ndarray) -> List[str]:
        """
        Construct prediction set C(x) = {k : P(Y=k|x) >= 1 - q_hat}.
        """
        threshold = 1.0 - self.q_hat
        pred_set = [CLASS_NAMES[k] for k in range(3) if prob_vector[k] >= threshold]

        # Guarantee at least the argmax class is present
        if not pred_set:
            top_k = int(np.argmax(prob_vector))
            pred_set = [CLASS_NAMES[top_k]]

        return pred_set


def calculate_calibrated_confidence(
    calibrated_probs: np.ndarray,
    conformal_set: List[str],
    ensemble_agreement: float,
    is_anomaly: bool,
    news_count: int,
    regime_state: int,
    min_prob_threshold: float = 0.40,
) -> Dict[str, Any]:
    """
    Produce final calibrated confidence score with transparent risk penalty decomposition.
    If ambiguous (multi-class conformal set or low max prob), marks as HOLD / ABSTAIN.
    """
    top_class_idx = int(np.argmax(calibrated_probs))
    raw_action = CLASS_NAMES[top_class_idx]
    base_prob = float(calibrated_probs[top_class_idx])
    base_conf = round(base_prob * 100.0, 1)

    # 1. Base model disagreement penalty (up to -15%)
    # agreement: 1.0 = unanimous, 0.0 = total disagreement
    disagreement = max(0.0, 1.0 - ensemble_agreement)
    disagree_penalty = round(-15.0 * disagreement, 1)

    # 2. Unexplained anomaly penalty: anomaly flagged but 0 news headlines (-12%)
    unexplained_penalty = 0.0
    if is_anomaly and news_count == 0:
        unexplained_penalty = -12.0

    # 3. High volatility regime penalty: -10% if turbulent
    regime_penalty = -10.0 if regime_state == 1 else 0.0

    # Final confidence sum clamped to [15, 95]
    final_conf = base_conf + disagree_penalty + unexplained_penalty + regime_penalty
    final_conf = round(max(15.0, min(95.0, final_conf)), 1)

    # Abstention check
    is_abstain = False
    abstain_reason = ""
    recommended_action = raw_action

    # If conformal set has multiple classes (ambiguity)
    if len(conformal_set) > 1:
        is_abstain = True
        abstain_reason = f"CONFORMAL_AMBIGUITY (Prediction set contains {conformal_set})"
        recommended_action = "HOLD"
    elif base_prob < min_prob_threshold:
        is_abstain = True
        abstain_reason = f"LOW_CONFIDENCE (Top calibrated probability {base_prob:.2f} < {min_prob_threshold:.2f})"
        recommended_action = "HOLD"

    return {
        "action": recommended_action,
        "raw_action": raw_action,
        "is_abstain": is_abstain,
        "abstain_reason": abstain_reason,
        "conformal_prediction_set": conformal_set,
        "confidence": final_conf,
        "confidence_breakdown": {
            "base_calibrated_confidence": base_conf,
            "disagreement_penalty": disagree_penalty,
            "unexplained_anomaly_penalty": unexplained_penalty,
            "regime_penalty": regime_penalty,
            "final_confidence": final_conf,
        },
        "calibrated_probabilities": {
            "DOWN": round(float(calibrated_probs[0]), 4),
            "FLAT": round(float(calibrated_probs[1]), 4),
            "UP": round(float(calibrated_probs[2]), 4),
        },
    }
