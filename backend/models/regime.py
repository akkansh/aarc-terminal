"""Regime detection (2-state Gaussian HMM) and GARCH(1,1) volatility forecasting."""

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from arch import arch_model


class GaussianHMM2State:
    """
    Self-contained, robust 2-State Gaussian Hidden Markov Model for market regimes.
    State 0: Low Volatility / Normal Regime
    State 1: High Volatility / Turbulent Regime
    Trained via Expectation-Maximization (Baum-Welch algorithm).
    """

    def __init__(self, n_iter: int = 35, tol: float = 1e-4, random_state: int = 42):
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.mu = np.array([0.0005, -0.001])
        self.sigma = np.array([0.010, 0.025])
        self.trans_mat = np.array([[0.95, 0.05], [0.10, 0.90]])
        self.pi = np.array([0.8, 0.2])
        self.is_fitted = False

    def _gaussian_pdf(self, x: np.ndarray, mean: float, std: float) -> np.ndarray:
        std = max(1e-6, std)
        denom = np.sqrt(2 * np.pi) * std
        num = np.exp(-0.5 * ((x - mean) / std) ** 2)
        return np.maximum(num / denom, 1e-12)

    def fit(self, returns: np.ndarray) -> "GaussianHMM2State":
        x = np.asarray(returns, dtype=float).ravel()
        n = len(x)
        if n < 30:
            return self

        # Initialize: partition by rolling absolute deviation
        abs_dev = np.abs(x - np.median(x))
        thresh = np.percentile(abs_dev, 70)
        low_mask = abs_dev <= thresh
        high_mask = ~low_mask

        self.mu[0] = float(np.mean(x[low_mask])) if np.sum(low_mask) > 5 else 0.0005
        self.mu[1] = float(np.mean(x[high_mask])) if np.sum(high_mask) > 5 else -0.001
        self.sigma[0] = float(np.std(x[low_mask])) if np.sum(low_mask) > 5 else 0.010
        self.sigma[1] = float(np.std(x[high_mask])) if np.sum(high_mask) > 5 else 0.025

        # Ensure state 0 is lower volatility than state 1
        if self.sigma[0] > self.sigma[1]:
            self.mu = self.mu[::-1]
            self.sigma = self.sigma[::-1]

        # EM iterations
        for _ in range(self.n_iter):
            # E-step: compute emission densities
            B = np.zeros((n, 2))
            B[:, 0] = self._gaussian_pdf(x, self.mu[0], self.sigma[0])
            B[:, 1] = self._gaussian_pdf(x, self.mu[1], self.sigma[1])

            # Forward pass with scaling
            alpha = np.zeros((n, 2))
            c = np.zeros(n)

            alpha[0] = self.pi * B[0]
            c[0] = np.sum(alpha[0]) + 1e-12
            alpha[0] /= c[0]

            for t in range(1, n):
                alpha[t] = (alpha[t - 1] @ self.trans_mat) * B[t]
                c[t] = np.sum(alpha[t]) + 1e-12
                alpha[t] /= c[t]

            # Backward pass
            beta = np.zeros((n, 2))
            beta[-1] = 1.0 / c[-1]

            for t in range(n - 2, -1, -1):
                beta[t] = (self.trans_mat @ (B[t + 1] * beta[t + 1])) / c[t]

            # Smoothed state probabilities (gamma)
            gamma = alpha * beta
            gamma_sum = np.sum(gamma, axis=1, keepdims=True) + 1e-12
            gamma /= gamma_sum

            # Transition probabilities (xi)
            xi = np.zeros((n - 1, 2, 2))
            for t in range(n - 1):
                numer = alpha[t, :, None] * self.trans_mat * B[t + 1, None, :] * beta[t + 1, None, :]
                denom = np.sum(numer) + 1e-12
                xi[t] = numer / denom

            # M-step: parameter updates
            self.pi = gamma[0]
            sum_gamma = np.sum(gamma, axis=0) + 1e-8
            sum_gamma_t1 = np.sum(gamma[:-1], axis=0)[:, None] + 1e-8
            self.trans_mat = np.sum(xi, axis=0) / sum_gamma_t1
            self.trans_mat /= np.sum(self.trans_mat, axis=1, keepdims=True)

            for k in range(2):
                self.mu[k] = np.sum(gamma[:, k] * x) / sum_gamma[k]
                self.sigma[k] = np.sqrt(np.sum(gamma[:, k] * (x - self.mu[k]) ** 2) / sum_gamma[k])
                self.sigma[k] = max(1e-4, self.sigma[k])

            # Maintain state 0 as low vol
            if self.sigma[0] > self.sigma[1]:
                self.mu = self.mu[::-1]
                self.sigma = self.sigma[::-1]
                self.trans_mat = self.trans_mat[::-1, ::-1]
                self.pi = self.pi[::-1]

        self.is_fitted = True
        return self

    def predict_proba(self, returns: np.ndarray) -> np.ndarray:
        """Return posterior probability for each state: shape (N, 2)."""
        x = np.asarray(returns, dtype=float).ravel()
        n = len(x)
        if not self.is_fitted or n == 0:
            return np.full((n, 2), 0.5)

        B = np.zeros((n, 2))
        B[:, 0] = self._gaussian_pdf(x, self.mu[0], self.sigma[0])
        B[:, 1] = self._gaussian_pdf(x, self.mu[1], self.sigma[1])

        # Forward filter
        alpha = np.zeros((n, 2))
        alpha[0] = self.pi * B[0]
        s0 = np.sum(alpha[0]) + 1e-12
        alpha[0] /= s0

        for t in range(1, n):
            alpha[t] = (alpha[t - 1] @ self.trans_mat) * B[t]
            st = np.sum(alpha[t]) + 1e-12
            alpha[t] /= st

        return alpha


class MarketRegimeDetector:
    """Estimates HMM regime and fits GARCH(1,1) volatility model."""

    def __init__(self, random_state: int = 42):
        self.hmm = GaussianHMM2State(random_state=random_state)
        self.random_state = random_state

    def fit_predict(self, returns_series: pd.Series) -> Dict[str, Any]:
        """
        Fit regime HMM and GARCH(1,1) on returns.
        Returns:
          - regime_state: 0 (NORMAL/LOW_VOL) or 1 (TURBULENT/HIGH_VOL)
          - regime_name: 'NORMAL' or 'TURBULENT'
          - p_high_vol: probability of high volatility regime
          - garch_vol_5d_forecast: forecasted annualized 5d volatility
          - regime_risk_multiplier: scaling factor (0.5 to 1.0) for sizing
        """
        ret = returns_series.dropna().values
        if len(ret) < 40:
            return {
                "regime_state": 0,
                "regime_name": "NORMAL",
                "p_high_vol": 0.20,
                "garch_vol_5d_forecast": 20.0,
                "regime_risk_multiplier": 1.0,
            }

        # 1. HMM Regime
        self.hmm.fit(ret)
        proba = self.hmm.predict_proba(ret)
        current_p_high = float(proba[-1, 1])
        current_regime = 1 if current_p_high >= 0.50 else 0
        regime_label = "TURBULENT" if current_regime == 1 else "NORMAL"

        # 2. GARCH(1,1) forward volatility forecast
        # Scale returns to percent for GARCH stability
        ret_pct = ret * 100.0
        garch_vol_annual = float(np.std(ret) * np.sqrt(252) * 100.0)

        try:
            am = arch_model(ret_pct, vol="Garch", p=1, q=1, rescale=False, mean="Constant")
            res = am.fit(disp="off", show_warning=False)
            f = res.forecast(horizon=5)
            # 5-day variance average converted to annualized std
            var_5d_daily = float(f.variance.values[-1, -1])
            garch_vol_annual = float(np.sqrt(max(0.01, var_5d_daily) * 252))
        except Exception:
            pass

        # Risk multiplier: scale down allocation in turbulent regime
        if current_regime == 1:
            risk_mult = round(max(0.40, 1.0 - (current_p_high * 0.60)), 2)
        else:
            risk_mult = 1.0

        return {
            "regime_state": current_regime,
            "regime_name": regime_label,
            "p_high_vol": round(current_p_high, 4),
            "garch_vol_5d_forecast": round(garch_vol_annual, 2),
            "regime_risk_multiplier": risk_mult,
        }
