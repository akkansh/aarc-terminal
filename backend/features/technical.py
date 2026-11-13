"""Stationary technical indicator calculation with strict past-only lookback."""

import numpy as np
import pandas as pd
from typing import Dict


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a stationary technical feature matrix from OHLCV data.
    All features are stationary (returns, ratios, z-scores) and computed
    strictly causally using rolling windows (no future bars).
    """
    if len(df) < 55:
        raise ValueError(f"Insufficient history ({len(df)} rows). Need at least 55 rows for 50d SMA/indicators.")

    feats = pd.DataFrame(index=df.index)
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    # 1. Returns (1-day, 5-day, 20-day)
    feats["return_1d"] = close.pct_change(1)
    feats["return_5d"] = close.pct_change(5)
    feats["return_20d"] = close.pct_change(20)

    # 2. Realized Volatility (20-day annualized)
    log_ret = np.log(close / close.shift(1))
    feats["realized_vol_20d"] = log_ret.rolling(window=20).std() * np.sqrt(252)

    # Volatility ratio: short-term vs longer-term volatility
    vol_60d = log_ret.rolling(window=60, min_periods=20).std() * np.sqrt(252)
    feats["vol_ratio_20_60"] = feats["realized_vol_20d"] / (vol_60d + 1e-8)

    # 3. RSI (14-day) - Normalized to [-1, 1]
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi_raw = 100.0 - (100.0 / (1.0 + rs))
    feats["rsi_14_raw"] = rsi_raw
    feats["rsi_14_norm"] = (rsi_raw - 50.0) / 50.0  # Center around 0, [-1, 1]

    # 4. MACD (12, 26, 9) - Price-normalized
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    feats["macd_hist_ratio"] = macd_hist / (close + 1e-8)
    feats["macd_signal_ratio"] = signal_line / (close + 1e-8)

    # 5. Bollinger Bands (20, 2.0) - Stationary %B and Bandwidth
    sma20 = close.rolling(window=20).mean()
    std20 = close.rolling(window=20).std()
    bb_upper = sma20 + (2.0 * std20)
    bb_lower = sma20 - (2.0 * std20)
    bb_range = bb_upper - bb_lower
    feats["bollinger_pct_b"] = (close - bb_lower) / (bb_range + 1e-8)
    feats["bollinger_bandwidth"] = bb_range / (sma20 + 1e-8)
    feats["sma20_z_score"] = (close - sma20) / (std20 + 1e-8)

    # 6. ATR (14-day) - Normalized by Close
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14).mean()
    feats["atr_14_ratio"] = atr14 / (close + 1e-8)

    # 7. Volume Z-Score & Volume Trend
    vol_mean20 = volume.rolling(window=20).mean()
    vol_std20 = volume.rolling(window=20).std()
    feats["volume_zscore_20d"] = (volume - vol_mean20) / (vol_std20 + 1e-8)
    feats["volume_ratio_5_20"] = volume.rolling(window=5).mean() / (vol_mean20 + 1e-8)

    # 8. Moving Average Ratios (Price to SMA20, Price to SMA50, SMA20 to SMA50)
    sma50 = close.rolling(window=50).mean()
    feats["ma_ratio_close_sma20"] = (close / (sma20 + 1e-8)) - 1.0
    feats["ma_ratio_close_sma50"] = (close / (sma50 + 1e-8)) - 1.0
    feats["ma_ratio_sma20_sma50"] = (sma20 / (sma50 + 1e-8)) - 1.0

    # 9. Intraday Range / High-Low Spread
    feats["hl_spread"] = (high - low) / (close + 1e-8)

    # Clean any inf / nan from initial rolling windows
    feats = feats.replace([np.inf, -np.inf], np.nan)
    return feats
