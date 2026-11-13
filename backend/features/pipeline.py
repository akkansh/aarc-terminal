"""Master feature engineering and labeling pipeline."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .technical import compute_technical_features
from .sentiment import compute_sentiment_features
from .anomaly import compute_rolling_anomalies
from .context import compute_context_features


def create_forward_labels(
    df: pd.DataFrame,
    horizon_days: int = 5,
    volatility_factor: float = 0.5,
) -> pd.DataFrame:
    """
    Compute forward return and volatility-adjusted triple-barrier class labels.
    Label mapping:
      0: DOWN  (forward return < -threshold)
      1: FLAT  (-threshold <= forward return <= +threshold)
      2: UP    (forward return > +threshold)
    """
    close = df["Close"].astype(float)
    forward_close = close.shift(-horizon_days)
    forward_return = (forward_close - close) / close

    # 20-day rolling daily volatility
    daily_ret = close.pct_change(1)
    daily_vol = daily_ret.rolling(window=20, min_periods=10).std()
    horizon_vol = daily_vol * np.sqrt(horizon_days)

    # Dynamic threshold: max(1.0%, factor * horizon_vol)
    threshold = (volatility_factor * horizon_vol).clip(lower=0.010, upper=0.15)

    labels = pd.DataFrame(index=df.index)
    labels["forward_return"] = forward_return
    labels["threshold"] = threshold

    # Classification
    target_class = pd.Series(index=df.index, dtype=float)
    target_name = pd.Series(index=df.index, dtype=object)

    valid_mask = ~forward_return.isna() & ~threshold.isna()

    is_up = valid_mask & (forward_return > threshold)
    is_down = valid_mask & (forward_return < -threshold)
    is_flat = valid_mask & (~is_up & ~is_down)

    target_class[is_down] = 0
    target_class[is_flat] = 1
    target_class[is_up] = 2

    target_name[is_down] = "DOWN"
    target_name[is_flat] = "FLAT"
    target_name[is_up] = "UP"

    labels["target_class"] = target_class
    labels["target_name"] = target_name
    return labels


class FeaturePipeline:
    """End-to-end pipeline generating the unified feature matrix X and targets y."""

    def __init__(
        self,
        horizon_days: int = 5,
        volatility_factor: float = 0.5,
        rolling_anomaly_window: int = 60,
    ):
        self.horizon_days = horizon_days
        self.volatility_factor = volatility_factor
        self.rolling_anomaly_window = rolling_anomaly_window

    def build_feature_table(
        self,
        stock_df: pd.DataFrame,
        benchmark_df: pd.DataFrame,
        vix_df: pd.DataFrame,
        scored_headlines: List[Dict[str, Any]],
        fundamentals: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Produce:
        - X: Feature matrix (all stationary, causal features)
        - y: Target labels (forward return, class 0/1/2)
        """
        # 1. Technical features
        df_tech = compute_technical_features(stock_df)

        # 2. Sentiment features
        df_sent = compute_sentiment_features(stock_df.index, scored_headlines)

        # 3. Rolling Anomaly features
        df_anom = compute_rolling_anomalies(
            df_tech,
            rolling_window=self.rolling_anomaly_window,
            contamination=0.05,
        )

        # 4. Context & Fundamentals
        df_context = compute_context_features(stock_df, benchmark_df, vix_df, fundamentals)

        # Merge all stationary features
        X = pd.concat([df_tech, df_sent, df_anom, df_context], axis=1)

        # Drop initial warmup NaN rows (up to 50 rows for SMA50)
        valid_indices = X.dropna(subset=["return_20d", "ma_ratio_close_sma50"]).index
        X = X.loc[valid_indices].copy()

        # Fill any residual NaNs with column medians or 0.0
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # 5. Compute forward labels
        y = create_forward_labels(
            stock_df.loc[valid_indices],
            horizon_days=self.horizon_days,
            volatility_factor=self.volatility_factor,
        )

        return X, y
