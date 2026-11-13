"""Vectorized backtesting engine factoring in transaction costs, slippage, and turnover."""

from typing import Dict, Any
import numpy as np
import pandas as pd


def run_realistic_backtest(
    forward_returns: pd.Series,
    predicted_actions: pd.Series,
    confidences: pd.Series,
    transaction_cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    risk_free_rate: float = 0.05,
    allow_short: bool = True,
) -> Dict[str, Any]:
    """
    Backtest directional prediction signals over 5-day horizons.
    Positions:
      UP: +1.0 * (conf / 100)
      DOWN: -1.0 * (conf / 100) if allow_short else 0.0
      HOLD/FLAT: 0.0 (cash)
    """
    valid = (~forward_returns.isna()) & (~predicted_actions.isna())
    returns = forward_returns[valid].values
    actions = predicted_actions[valid].values
    confs = confidences[valid].values / 100.0

    n = len(returns)
    if n == 0:
        return {}

    # Target positions
    target_pos = np.zeros(n)
    for i in range(n):
        act = str(actions[i]).upper()
        if act == "UP" or act == "LONG":
            target_pos[i] = confs[i]
        elif act == "DOWN" or act == "SHORT":
            target_pos[i] = -confs[i] if allow_short else 0.0
        else:
            target_pos[i] = 0.0

    # Turnover and execution friction (fees + slippage)
    fee_pct = (transaction_cost_bps + slippage_bps) / 10000.0
    pos_change = np.zeros(n)
    pos_change[0] = abs(target_pos[0])
    pos_change[1:] = np.abs(target_pos[1:] - target_pos[:-1])
    frictions = pos_change * fee_pct

    # Gross & Net returns
    gross_returns = target_pos * returns
    net_returns = gross_returns - frictions

    # Cumulative growth
    strat_equity = np.cumprod(1.0 + net_returns)
    bnh_equity = np.cumprod(1.0 + returns)

    strat_total_ret = float((strat_equity[-1] - 1.0) * 100.0)
    bnh_total_ret = float((bnh_equity[-1] - 1.0) * 100.0)

    # Max Drawdown
    peaks = np.maximum.accumulate(strat_equity)
    drawdowns = (strat_equity - peaks) / peaks
    max_dd = float(np.min(drawdowns) * 100.0)

    bnh_peaks = np.maximum.accumulate(bnh_equity)
    bnh_dd = (bnh_equity - bnh_peaks) / bnh_peaks
    bnh_max_dd = float(np.min(bnh_dd) * 100.0)

    # Sharpe ratio (annualized, assuming ~50 5-day periods per year)
    periods_per_year = 52
    mean_ret = np.mean(net_returns)
    std_ret = np.std(net_returns) + 1e-8
    rf_per_period = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    sharpe = float((mean_ret - rf_per_period) / std_ret * np.sqrt(periods_per_year))

    # Win rate on active trades
    active_mask = target_pos != 0.0
    if np.sum(active_mask) > 0:
        win_rate = float(np.mean(net_returns[active_mask] > 0) * 100.0)
    else:
        win_rate = 0.0

    annual_turnover = float(np.sum(pos_change) * (periods_per_year / n))

    return {
        "strategy_total_return_pct": round(strat_total_ret, 2),
        "benchmark_total_return_pct": round(bnh_total_ret, 2),
        "strategy_sharpe_ratio": round(sharpe, 2),
        "strategy_max_drawdown_pct": round(abs(max_dd), 2),
        "benchmark_max_drawdown_pct": round(abs(bnh_max_dd), 2),
        "active_trade_win_rate_pct": round(win_rate, 2),
        "annualized_turnover": round(annual_turnover, 2),
        "n_periods": n,
    }
