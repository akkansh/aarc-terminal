"""Strict verification tests preventing look-ahead bias and data leakage."""

import numpy as np
import pandas as pd
import pytest

from backend.features.technical import compute_technical_features
from backend.features.sentiment import compute_sentiment_features
from backend.features.pipeline import FeaturePipeline, create_forward_labels


def _create_mock_ohlcv(n_bars: int = 120, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV data."""
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="B")
    
    returns = np.random.normal(0.0005, 0.015, n_bars)
    prices = 100.0 * np.cumprod(1 + returns)
    
    highs = prices * (1 + np.abs(np.random.normal(0.005, 0.003, n_bars)))
    lows = prices * (1 - np.abs(np.random.normal(0.005, 0.003, n_bars)))
    opens = (highs + lows) / 2.0
    volumes = np.random.lognormal(14, 0.5, n_bars)
    
    df = pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": prices,
            "Volume": volumes,
        },
        index=dates,
    )
    return df


def test_feature_lookahead_leakage():
    """
    CRITICAL LEAKAGE TEST:
    Features computed at bar t must be strictly invariant to any modifications
    applied to future bars (t+1, t+2, ...).
    If altering future data alters features at bar t, look-ahead bias exists.
    """
    df_original = _create_mock_ohlcv(100)
    t_idx = 70  # Test point

    # 1. Compute features on unmodified series
    feats_orig = compute_technical_features(df_original)
    row_t_orig = feats_orig.iloc[t_idx].copy()

    # 2. Severely mutate future data from t+1 onwards (simulate a 50% crash or spike)
    df_mutated = df_original.copy()
    df_mutated.iloc[t_idx + 1 :, df_mutated.columns.get_loc("Close")] *= 1.50
    df_mutated.iloc[t_idx + 1 :, df_mutated.columns.get_loc("High")] *= 1.50
    df_mutated.iloc[t_idx + 1 :, df_mutated.columns.get_loc("Low")] *= 1.50
    df_mutated.iloc[t_idx + 1 :, df_mutated.columns.get_loc("Volume")] *= 10.0

    feats_mutated = compute_technical_features(df_mutated)
    row_t_mutated = feats_mutated.iloc[t_idx].copy()

    # 3. Assert zero leakage: features at t must be identical
    for col in feats_orig.columns:
        val_orig = row_t_orig[col]
        val_mutated = row_t_mutated[col]
        assert np.isclose(val_orig, val_mutated, atol=1e-10, equal_nan=True), (
            f"LOOK-AHEAD LEAKAGE DETECTED in feature '{col}' at bar {t_idx}! "
            f"Original: {val_orig}, Mutated: {val_mutated}"
        )


def test_news_timestamp_alignment():
    """
    Assert that news items published strictly after date t do not leak into date t features.
    """
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    
    # News published on bar 5 (index 5)
    future_news = [
        {
            "title": "Massive positive earnings surprise",
            "published_date": dates[5].strftime("%Y-%m-%d"),
            "net_score": 0.95,
            "prob_positive": 0.98,
            "prob_negative": 0.01,
            "prob_neutral": 0.01,
        }
    ]

    sent_df = compute_sentiment_features(dates, future_news)

    # For dates 0..4 (prior to news publish date), raw sentiment must be 0.0
    for i in range(5):
        assert sent_df.iloc[i]["sentiment_score_raw"] == 0.0, (
            f"Future news leaked into past bar {i} ({dates[i]})!"
        )
        assert sent_df.iloc[i]["news_count_daily"] == 0

    # On bar 5, news is registered
    assert sent_df.iloc[5]["sentiment_score_raw"] == 0.95
    assert sent_df.iloc[5]["news_count_daily"] == 1


def test_label_creation_future_dependency():
    """
    Verify that target labels depend strictly on future return (t -> t+H),
    while leaving the current bar unpolluted.
    """
    df = _create_mock_ohlcv(80)
    labels = create_forward_labels(df, horizon_days=5, volatility_factor=0.5)

    # The last 5 rows must have NaN for forward return (because future is unknown)
    last_5_returns = labels["forward_return"].iloc[-5:]
    assert last_5_returns.isna().all(), "Forward returns must be NaN for the final H bars!"

    # Valid labels must be only {0, 1, 2}
    valid_classes = labels["target_class"].dropna()
    assert set(valid_classes.unique()).issubset({0.0, 1.0, 2.0})
