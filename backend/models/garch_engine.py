"""Dynamic GARCH(1,1) Conditional Volatility and Regime-Aware Risk Engine."""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from arch import arch_model


class GARCHVolatilityEngine:
    """
    Fits a dynamic GARCH(1,1) conditional heteroskedasticity model using `arch`.
    Replaces static unconditional historical quantile VaR with a dynamic,
    time-varying conditional 95% Value-at-Risk (VaR) and forward volatility forecast.
    """

    def __init__(self, p: int = 1, q: int = 1, dist: str = "StudentsT"):
        self.p = p
        self.q = q
        self.dist = dist
        self.fitted_model = None

    def compute_dynamic_volatility_and_var(
        self,
        returns_series: pd.Series,
        confidence_level: float = 0.95,
        horizon_days: int = 5,
        lookback_window: int = 252,
    ) -> Dict[str, Any]:
        """
        Fits GARCH(1,1) with Student's t-distribution on rolling lookback window (default 252 trading days).
        Computes:
        - current_conditional_vol_daily: % daily volatility sigma_t
        - current_conditional_vol_annual: % annualized volatility sigma_t * sqrt(252)
        - dynamic_garch_var_95: dynamic 95% 1-day conditional VaR % (negative number)
        - forward_vol_forecast_annual: annualized horizon-day volatility forecast
        - volatility_shock_ratio: sigma_t / rolling_historical_std (regime stress metric)
        """
        raw_clean = returns_series.dropna().replace([np.inf, -np.inf], np.nan).dropna()
        # Enforce exact 252-day lookback window if history is longer
        clean_ret = raw_clean.tail(lookback_window) if len(raw_clean) > lookback_window else raw_clean
        n_obs = len(clean_ret)

        # Baseline fallback metrics if series is too short
        std_sample = float(clean_ret.std()) if n_obs > 1 else 0.015
        hist_ann_vol = float(std_sample * np.sqrt(252) * 100.0)
        hist_var_95 = float(np.percentile(clean_ret, (1.0 - confidence_level) * 100.0) * 100.0) if n_obs >= 20 else -2.5

        if n_obs < 40:
            return {
                "conditional_vol_daily_pct": round(std_sample * 100.0, 2),
                "conditional_vol_annual_pct": round(hist_ann_vol, 2),
                "dynamic_garch_var_95_pct": round(hist_var_95, 2),
                "garch_vol_forecast_annual_pct": round(hist_ann_vol, 2),
                "volatility_shock_ratio": 1.0,
                "is_garch_converged": False,
                "model_summary": "Sample Standard Deviation (Insufficient observations for GARCH)",
            }

        # Rescale returns to percentage points for numerical stability in GARCH optimization
        ret_pct = clean_ret * 100.0

        try:
            am = arch_model(
                ret_pct,
                mean="Constant",
                vol="GARCH",
                p=self.p,
                q=self.q,
                dist=self.dist,
                rescale=False,
            )
            res = am.fit(disp="off", show_warning=False)
            self.fitted_model = res

            # 1. Current Conditional Volatility (sigma_t on the latest bar)
            cond_vol_daily = float(res.conditional_volatility.iloc[-1])
            cond_vol_annual = float(cond_vol_daily * np.sqrt(252))

            # 2. Dynamic Conditional VaR (95%)
            # Formula: VaR_alpha = - ( mu + quantile_alpha * sigma_t )
            mu_hat = float(res.params.get("mu", 0.0))
            if self.dist.lower() in ("studentst", "t") and "nu" in res.params:
                from scipy.stats import t as student_t
                nu = max(2.1, float(res.params["nu"]))
                # Standardized Student-t quantile
                q_alpha = float(student_t.ppf(1.0 - confidence_level, df=nu) * np.sqrt((nu - 2.0) / nu))
            else:
                q_alpha = -1.644853  # Standard Normal 95% quantile

            # Expected 1-day conditional loss percentage at 95% confidence
            # Expressed as a negative percentage return (e.g., -2.85%)
            dynamic_var_daily_pct = -(abs(mu_hat) + abs(q_alpha) * cond_vol_daily)

            # 3. Forward Volatility Forecast over horizon
            forecasts = res.forecast(horizon=horizon_days)
            var_horizon_daily = float(forecasts.variance.values[-1, -1])
            forward_vol_annual = float(np.sqrt(max(0.01, var_horizon_daily) * 252))

            # 4. Volatility Shock Ratio (current conditional vol vs historical 60d realized vol)
            hist_60d_vol = float(clean_ret.tail(60).std() * 100.0) if n_obs >= 60 else (std_sample * 100.0)
            shock_ratio = round(cond_vol_daily / max(0.01, hist_60d_vol), 2)

            return {
                "conditional_vol_daily_pct": round(cond_vol_daily, 2),
                "conditional_vol_annual_pct": round(cond_vol_annual, 2),
                "dynamic_garch_var_95_pct": round(dynamic_var_daily_pct, 2),
                "garch_vol_forecast_annual_pct": round(forward_vol_annual, 2),
                "volatility_shock_ratio": shock_ratio,
                "is_garch_converged": bool(res.convergence_flag == 0),
                "model_summary": f"GARCH({self.p},{self.q})-{self.dist} (Persistence: {round(float(res.params.get('alpha[1]', 0.1) + res.params.get('beta[1]', 0.8)), 3)})",
            }

        except Exception as e:
            # Robust analytical EWMA fallback if optimization encounters singular covariance
            decay = 0.94
            squared_ret = (clean_ret ** 2).values
            ewma_var = squared_ret[0]
            for r2 in squared_ret[1:]:
                ewma_var = decay * ewma_var + (1 - decay) * r2
            ewma_vol_daily = np.sqrt(ewma_var) * 100.0
            ewma_vol_annual = ewma_vol_daily * np.sqrt(252)
            ewma_var_95 = -(1.645 * ewma_vol_daily)

            return {
                "conditional_vol_daily_pct": round(ewma_vol_daily, 2),
                "conditional_vol_annual_pct": round(ewma_vol_annual, 2),
                "dynamic_garch_var_95_pct": round(ewma_var_95, 2),
                "garch_vol_forecast_annual_pct": round(ewma_vol_annual, 2),
                "volatility_shock_ratio": 1.0,
                "is_garch_converged": False,
                "model_summary": f"RiskMetrics EWMA Fallback (GARCH error: {str(e)[:40]})",
            }
