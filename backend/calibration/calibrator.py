"""Probability calibration (Isotonic / Platt Scaling), Brier score, and ECE."""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression


def compute_brier_score(y_true_one_hot: np.ndarray, y_prob: np.ndarray) -> float:
    """Multiclass Brier score: mean squared difference between one-hot truth and predicted probs."""
    return float(np.mean(np.sum((y_prob - y_true_one_hot) ** 2, axis=1)))


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[float, Dict[str, Any]]:
    """
    Expected Calibration Error (ECE) and reliability curve bins.
    ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_data = []

    n = len(y_true)
    if n == 0:
        return 0.0, {"bins": []}

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_size = int(np.sum(in_bin))

        if bin_size > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            ece += (bin_size / n) * abs(bin_acc - bin_conf)
            bin_data.append({
                "bin_idx": i,
                "range": [round(float(bin_boundaries[i]), 2), round(float(bin_boundaries[i + 1]), 2)],
                "count": bin_size,
                "accuracy": round(bin_acc, 4),
                "confidence": round(bin_conf, 4),
                "gap": round(abs(bin_acc - bin_conf), 4),
            })

    return round(float(ece), 4), {"bins": bin_data}


class TemperatureCalibrator:
    """
    Multiclass Temperature Scaling (Platt scaling extension) for probability calibration.
    Preserves class ranking while optimally softening overconfident probability outputs.
    """

    def __init__(self):
        self.temperature: float = 1.0
        self.is_fitted = False

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> "TemperatureCalibrator":
        from scipy.optimize import minimize
        from sklearn.metrics import log_loss

        n_samples = len(y_true)
        if n_samples < 10:
            return self

        logits = np.log(np.clip(raw_probs, 1e-7, 1.0 - 1e-7))

        def obj(t):
            temp = max(float(t[0]), 0.1)
            scaled = logits / temp
            exp_s = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
            p = exp_s / (np.sum(exp_s, axis=1, keepdims=True) + 1e-12)
            return log_loss(y_true, p, labels=[0, 1, 2])

        res = minimize(obj, [1.0], bounds=[(0.2, 5.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0]) if res.success else 1.0
        self.is_fitted = True
        return self

    def predict_proba(self, raw_probs: np.ndarray) -> np.ndarray:
        if not self.is_fitted or abs(self.temperature - 1.0) < 1e-4:
            return raw_probs

        logits = np.log(np.clip(raw_probs, 1e-7, 1.0 - 1e-7))
        scaled = logits / self.temperature
        exp_s = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        probs = exp_s / (np.sum(exp_s, axis=1, keepdims=True) + 1e-12)
        return probs


class MultiClassIsotonicCalibrator:
    """Robust probability calibrator using Temperature Scaling / Platt Scaling."""

    def __init__(self, method: str = "temperature"):
        self.method = method
        self.temp_calibrator = TemperatureCalibrator()
        self.is_fitted = False

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> "MultiClassIsotonicCalibrator":
        n_samples = len(y_true)
        if n_samples < 10:
            return self

        self.temp_calibrator.fit(raw_probs, y_true)
        self.is_fitted = True
        return self

    def predict_proba(self, raw_probs: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return raw_probs

        return self.temp_calibrator.predict_proba(raw_probs)
