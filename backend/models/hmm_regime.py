"""3-State Gaussian Hidden Markov Model for Dynamic Market Regime Detection."""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.special import logsumexp


class GaussianHMM3State:
    """
    Vectorized, robust 3-State Gaussian Hidden Markov Model for market regimes:
      - State 0: Low-Vol / Bullish Trending  (mu > 0, sigma is low)
      - State 1: High-Vol / Crisis / Panic   (mu < 0, sigma is high)
      - State 2: Mean-Reverting / Choppy     (mu ~ 0, sigma is moderate)
    Fitted via Baum-Welch Expectation-Maximization in log-space to prevent underflow.
    """

    def __init__(self, n_iter: int = 40, tol: float = 1e-4, random_state: int = 42):
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.n_states = 3

        # Priors: initial state probabilities
        self.pi = np.array([0.50, 0.20, 0.30])
        # Transition matrix (sticky regimes)
        self.A = np.array([
            [0.92, 0.03, 0.05],
            [0.08, 0.85, 0.07],
            [0.10, 0.08, 0.82],
        ])
        # Emission parameters: Means and Standard Deviations
        self.means = np.array([0.0010, -0.0025, 0.0000])
        self.sigmas = np.array([0.009, 0.028, 0.016])
        self.is_fitted = False

    def _log_gaussian_pdf(self, x: np.ndarray, mean: float, std: float) -> np.ndarray:
        std = max(1e-5, std)
        return -0.5 * np.log(2 * np.pi) - np.log(std) - 0.5 * ((x - mean) / std) ** 2

    def fit(self, returns: np.ndarray) -> "GaussianHMM3State":
        x = np.asarray(returns, dtype=float).ravel()
        n = len(x)
        if n < 45:
            return self

        # Heuristic quantile partition for EM parameter initialization
        vol_window = pd.Series(x).rolling(window=15, min_periods=5).std().fillna(np.std(x)).values
        high_vol_mask = vol_window > np.percentile(vol_window, 75)
        low_vol_mask = (vol_window <= np.percentile(vol_window, 40)) & (x > 0)
        choppy_mask = ~(high_vol_mask | low_vol_mask)

        if np.sum(low_vol_mask) > 5:
            self.means[0] = float(np.mean(x[low_vol_mask]))
            self.sigmas[0] = max(0.004, float(np.std(x[low_vol_mask])))
        if np.sum(high_vol_mask) > 5:
            self.means[1] = min(-0.0005, float(np.mean(x[high_vol_mask])))
            self.sigmas[1] = max(0.018, float(np.std(x[high_vol_mask])))
        if np.sum(choppy_mask) > 5:
            self.means[2] = float(np.mean(x[choppy_mask]))
            self.sigmas[2] = max(0.008, float(np.std(x[choppy_mask])))

        # Enforce canonical ordering: sigma[0] (low) < sigma[2] (mid) < sigma[1] (high)
        sort_order = np.argsort(self.sigmas)
        # We want: lowest sigma -> 0, middle sigma -> 2, highest sigma -> 1
        desired_idx = [sort_order[0], sort_order[2], sort_order[1]]
        self.means = self.means[desired_idx]
        self.sigmas = self.sigmas[desired_idx]

        log_A = np.log(np.maximum(self.A, 1e-12))
        log_pi = np.log(np.maximum(self.pi, 1e-12))

        # Baum-Welch EM in log space
        prev_log_lik = -np.inf

        for _ in range(self.n_iter):
            # Log Emission probabilities: (N, 3)
            log_B = np.zeros((n, 3))
            for k in range(3):
                log_B[:, k] = self._log_gaussian_pdf(x, self.means[k], self.sigmas[k])

            # Forward pass (alpha in log space)
            log_alpha = np.zeros((n, 3))
            log_alpha[0] = log_pi + log_B[0]

            for t in range(1, n):
                for k in range(3):
                    log_alpha[t, k] = logsumexp(log_alpha[t - 1] + log_A[:, k]) + log_B[t, k]

            total_log_lik = logsumexp(log_alpha[-1])
            if abs(total_log_lik - prev_log_lik) < self.tol:
                break
            prev_log_lik = total_log_lik

            # Backward pass (beta in log space)
            log_beta = np.zeros((n, 3))
            log_beta[-1] = 0.0

            for t in range(n - 2, -1, -1):
                for i in range(3):
                    log_beta[t, i] = logsumexp(log_A[i, :] + log_B[t + 1, :] + log_beta[t + 1, :])

            # Posterior state probabilities gamma = P(S_t = k | X)
            log_gamma = log_alpha + log_beta - total_log_lik
            gamma = np.exp(np.clip(log_gamma, -700, 0))
            gamma = gamma / (np.sum(gamma, axis=1, keepdims=True) + 1e-12)

            # Update parameters
            self.pi = np.clip(gamma[0], 0.05, 0.90)
            self.pi /= np.sum(self.pi)

            gamma_sums = np.sum(gamma, axis=0) + 1e-12
            for k in range(3):
                self.means[k] = float(np.sum(gamma[:, k] * x) / gamma_sums[k])
                sq_diff = (x - self.means[k]) ** 2
                self.sigmas[k] = float(np.sqrt(np.sum(gamma[:, k] * sq_diff) / gamma_sums[k]))
                self.sigmas[k] = max(0.003, self.sigmas[k])

        # Final sort to guarantee canonical regime identities
        sort_order = np.argsort(self.sigmas)
        desired_idx = [sort_order[0], sort_order[2], sort_order[1]]
        self.means = self.means[desired_idx]
        self.sigmas = self.sigmas[desired_idx]

        self.is_fitted = True
        return self

    def predict_state_probabilities(self, returns: np.ndarray) -> np.ndarray:
        """Returns posterior state probabilities [P(LowVol), P(Panic), P(Choppy)] across all bars."""
        x = np.asarray(returns, dtype=float).ravel()
        n = len(x)
        if not self.is_fitted or n == 0:
            return np.tile([0.60, 0.15, 0.25], (max(1, n), 1))

        log_B = np.zeros((n, 3))
        for k in range(3):
            log_B[:, k] = self._log_gaussian_pdf(x, self.means[k], self.sigmas[k])

        log_A = np.log(np.maximum(self.A, 1e-12))
        log_alpha = np.zeros((n, 3))
        log_alpha[0] = np.log(np.maximum(self.pi, 1e-12)) + log_B[0]

        for t in range(1, n):
            for k in range(3):
                log_alpha[t, k] = logsumexp(log_alpha[t - 1] + log_A[:, k]) + log_B[t, k]

        # Normalized posterior probability distributions
        probs = np.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))
        return np.clip(probs, 0.001, 0.999) / np.sum(probs, axis=1, keepdims=True)


class MarketRegimePipeline:
    """Institutional wrapper extracting 3-state HMM regime signals and conditioning factors."""

    def __init__(self, random_state: int = 42):
        self.hmm = GaussianHMM3State(random_state=random_state)
        self.state_labels = {
            0: "LOW_VOL_TRENDING",
            1: "HIGH_VOL_PANIC",
            2: "MEAN_REVERTING_CHOPPY",
        }

    def analyze_market_regime(self, returns_series: pd.Series) -> Dict[str, Any]:
        ret = returns_series.dropna().values
        if len(ret) < 30:
            return {
                "active_state": 0,
                "regime_name": "LOW_VOL_TRENDING",
                "probabilities": {"P_Trending": 0.65, "P_Panic": 0.10, "P_Choppy": 0.25},
                "regime_vol_annual_pct": 18.0,
                "volatility_multiplier": 1.0,
                "kelly_scale_factor": 1.0,
                "stop_loss_multiplier": 1.0,
            }

        self.hmm.fit(ret)
        probs_matrix = self.hmm.predict_state_probabilities(ret)
        latest_probs = probs_matrix[-1]

        active_state = int(np.argmax(latest_probs))
        regime_label = self.state_labels[active_state]

        # Expected conditional volatility across state distribution
        annual_vols = self.hmm.sigmas * np.sqrt(252) * 100.0
        expected_ann_vol = float(np.sum(latest_probs * annual_vols))

        # Dynamic risk conditioning factors
        if active_state == 1:  # Panic
            kelly_scale = 0.50
            stop_mult = 1.60
            vol_mult = 1.80
        elif active_state == 2:  # Choppy
            kelly_scale = 0.75
            stop_mult = 1.25
            vol_mult = 1.20
        else:  # Low-Vol Trending
            kelly_scale = 1.15
            stop_mult = 1.00
            vol_mult = 0.90

        return {
            "active_state": active_state,
            "regime_name": regime_label,
            "probabilities": {
                "P_Trending": round(float(latest_probs[0]), 3),
                "P_Panic": round(float(latest_probs[1]), 3),
                "P_Choppy": round(float(latest_probs[2]), 3),
            },
            "regime_vol_annual_pct": round(expected_ann_vol, 2),
            "volatility_multiplier": round(vol_mult, 2),
            "kelly_scale_factor": round(kelly_scale, 2),
            "stop_loss_multiplier": round(stop_mult, 2),
            "state_parameters": {
                "trending_annual_vol": round(float(annual_vols[0]), 2),
                "panic_annual_vol": round(float(annual_vols[1]), 2),
                "choppy_annual_vol": round(float(annual_vols[2]), 2),
            },
        }
