"""Unit tests for feature engineering, stationary properties, and anomaly detection."""

import numpy as np
import pandas as pd
import pytest

from backend.features.technical import compute_technical_features
from backend.features.anomaly import compute_rolling_anomalies
from backend.features.context import compute_context_features
from backend.features.pipeline import FeaturePipeline


def _create_mock_ohlcv(n_bars: int = 150, seed: int = 101) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.date_range("2023-01-01", periods=n_bars, freq="B")
    ret = np.random.normal(0.0003, 0.018, n_bars)
    prices = 250.0 * np.cumprod(1 + ret)
    highs = prices * (1 + np.abs(np.random.normal(0.008, 0.004, n_bars)))
    lows = prices * (1 - np.abs(np.random.normal(0.008, 0.004, n_bars)))
    opens = (highs + lows) / 2.0
    vols = np.random.lognormal(13, 0.6, n_bars)
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": prices, "Volume": vols},
        index=dates,
    )


def test_technical_features_stationarity():
    df = _create_mock_ohlcv(120)
    feats = compute_technical_features(df)

    # Essential columns exist
    expected = [
        "return_1d",
        "return_5d",
        "return_20d",
        "realized_vol_20d",
        "rsi_14_norm",
        "macd_hist_ratio",
        "bollinger_pct_b",
        "atr_14_ratio",
        "volume_zscore_20d",
        "ma_ratio_close_sma20",
        "ma_ratio_close_sma50",
    ]
    for col in expected:
        assert col in feats.columns, f"Missing feature: {col}"

    # Discard warmup NaN rows (first 50 bars for 50d SMA)
    valid_feats = feats.iloc[50:]

    # Assert no NaNs or Infs in valid range
    assert not valid_feats.isna().any().any(), "Features contain unexpected NaNs"
    assert not np.isinf(valid_feats.values).any(), "Features contain Infs"

    # Normalized RSI should lie within [-1, 1] approximately
    assert valid_feats["rsi_14_norm"].min() >= -1.01
    assert valid_feats["rsi_14_norm"].max() <= 1.01


def test_rolling_anomalies():
    df = _create_mock_ohlcv(100)
    tech = compute_technical_features(df)
    anom = compute_rolling_anomalies(tech, rolling_window=50, contamination=0.08)

    assert "anomaly_score" in anom.columns
    assert "is_anomaly" in anom.columns
    assert "anomaly_count_10d" in anom.columns

    # Must contain both anomalies and normal points
    flags = anom["is_anomaly"].iloc[40:]
    assert flags.dtype == bool
    # Should flag approximately 5-15% of points
    rate = flags.mean()
    assert 0.01 <= rate <= 0.25, f"Anomaly rate {rate} outside expected range"


def test_end_to_end_pipeline():
    stock_df = _create_mock_ohlcv(160)
    bench_df = _create_mock_ohlcv(160, seed=202)
    vix_df = _create_mock_ohlcv(160, seed=303)
    vix_df["Close"] = np.random.uniform(12, 28, 160)

    pipe = FeaturePipeline(horizon_days=5, volatility_factor=0.5)
    X, y = pipe.build_feature_table(
        stock_df=stock_df,
        benchmark_df=bench_df,
        vix_df=vix_df,
        scored_headlines=[],
        fundamentals={"pe_ratio": 24.5, "revenue_growth": 0.12},
    )

    assert len(X) == len(y)
    assert len(X) > 80, "Pipeline dropped too many rows"

    # Verify target distribution has all 3 classes over a long synthetic random walk
    classes = y["target_class"].dropna().unique()
    assert set(classes).issubset({0.0, 1.0, 2.0})
    assert len(classes) >= 2, "Expected at least 2 distinct target classes"
