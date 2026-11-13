"""Macro context, benchmark relative strength, volatility index (VIX), and fundamental features."""

import numpy as np
import pandas as pd
from typing import Any, Dict


def compute_context_features(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    fundamentals: Dict[str, Any],
) -> pd.DataFrame:
    """
    Compute market index relative strength, VIX dynamics, and fundamental metrics.
    All aligned strictly to stock trading calendar dates.
    """
    out = pd.DataFrame(index=stock_df.index)

    # Stock 20d return for relative strength
    stock_ret_20d = stock_df["Close"].pct_change(20).fillna(0.0)

    # 1. Benchmark index features
    if not benchmark_df.empty and "Close" in benchmark_df.columns:
        bench_close = benchmark_df["Close"].reindex(stock_df.index).ffill().bfill()
        out["market_return_1d"] = bench_close.pct_change(1).fillna(0.0)
        out["market_return_5d"] = bench_close.pct_change(5).fillna(0.0)
        bench_ret_20d = bench_close.pct_change(20).fillna(0.0)
        out["relative_strength_20d"] = stock_ret_20d - bench_ret_20d
    else:
        out["market_return_1d"] = 0.0
        out["market_return_5d"] = 0.0
        out["relative_strength_20d"] = 0.0

    # 2. Volatility index (VIX) features
    if not vix_df.empty and "Close" in vix_df.columns:
        vix_close = vix_df["Close"].reindex(stock_df.index).ffill().bfill()
        out["vix_level"] = vix_close
        out["vix_change_5d"] = vix_close.pct_change(5).fillna(0.0)
        vix_mean20 = vix_close.rolling(window=20, min_periods=5).mean()
        vix_std20 = vix_close.rolling(window=20, min_periods=5).std()
        out["vix_zscore_20d"] = ((vix_close - vix_mean20) / (vix_std20 + 1e-8)).fillna(0.0)
    else:
        out["vix_level"] = 15.0
        out["vix_change_5d"] = 0.0
        out["vix_zscore_20d"] = 0.0

    # 3. Fundamental Context
    pe = float(fundamentals.get("pe_ratio", 0.0) or 0.0)
    rev_growth = float(fundamentals.get("revenue_growth", 0.0) or 0.0)
    margin = float(fundamentals.get("profit_margins", 0.0) or 0.0)

    # Stationary transformations for fundamentals
    out["pe_ratio_log"] = np.log(max(1.0, pe)) if pe > 0 else 0.0
    out["revenue_growth"] = rev_growth
    out["profit_margin"] = margin
    out["has_fundamentals"] = 1.0 if fundamentals.get("has_fundamentals") else 0.0

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out
