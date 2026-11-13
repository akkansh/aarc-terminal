"""Regression tests for unadjusted, ticker-accurate price ingestion."""

import numpy as np
import pandas as pd
import pytest

from backend.data import fetcher as fetcher_module
from backend.data.fetcher import OHLCV_CACHE_VERSION, MarketDataFetcher
from backend.tools import quant_math


def _ohlcv_frame(n_bars: int = 40, start: str = "2024-01-02") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n_bars)
    close = np.linspace(100.0, 140.0, n_bars)
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(n_bars, 1_000_000.0),
        },
        index=dates,
    )


def _install_empty_history(monkeypatch):
    class EmptyTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            return pd.DataFrame()

    monkeypatch.setattr(quant_math.yf, "Ticker", EmptyTicker)


def test_quant_download_requests_unadjusted_close(monkeypatch):
    frame = _ohlcv_frame()
    calls = []

    def fake_download(symbol, **kwargs):
        calls.append((symbol, kwargs))
        return frame

    monkeypatch.setattr(quant_math.yf, "download", fake_download)

    result = quant_math.analyze_ticker_quant("AAPL")

    assert calls == [("AAPL", {
        "period": "1y",
        "progress": False,
        "auto_adjust": False,
    })]
    assert result["ticker"] == "AAPL"
    assert result["current_price"] == pytest.approx(140.0, abs=0.01)


def test_quant_history_fallback_uses_same_ticker_unadjusted(monkeypatch):
    frame = _ohlcv_frame()
    history_calls = []

    monkeypatch.setattr(quant_math.yf, "download", lambda *a, **k: pd.DataFrame())

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            history_calls.append((self.symbol, kwargs))
            return frame

    monkeypatch.setattr(quant_math.yf, "Ticker", FakeTicker)

    result = quant_math.analyze_ticker_quant("MSFT")

    assert history_calls[0][0] == "MSFT"
    assert history_calls[0][1]["auto_adjust"] is False
    assert result["ticker"] == "MSFT"


def test_quant_raises_instead_of_synthesizing_prices(monkeypatch):
    monkeypatch.setattr(quant_math.yf, "download", lambda *a, **k: pd.DataFrame())
    _install_empty_history(monkeypatch)

    with pytest.raises(ValueError, match="No price data found"):
        quant_math.analyze_ticker_quant("NOT_A_REAL_TICKER")


class _RecordingCache:
    def __init__(self):
        self.get_calls = []
        self.set_calls = []
        self.data = {}

    def get(self, key):
        self.get_calls.append(key)
        return self.data.get(key)

    def set(self, key, value, ttl_seconds=None):
        self.set_calls.append((key, ttl_seconds))
        self.data[key] = value


def test_fetcher_uses_versioned_unadjusted_cache_key(monkeypatch):
    frame = _ohlcv_frame()
    calls = []

    def fake_download(symbol, **kwargs):
        calls.append((symbol, kwargs))
        return frame

    monkeypatch.setattr(fetcher_module.yf, "download", fake_download)

    fetcher = MarketDataFetcher()
    cache = _RecordingCache()
    fetcher.cache = cache

    # Simulate an old auto-adjusted cache entry under the previous key format.
    cache.data["ohlcv:AAPL:None:None:6mo"] = frame.assign(Close=frame["Close"] * 0.5)[["Close"]]

    result = fetcher.fetch_ohlcv("AAPL", period="6mo")

    assert calls == [("AAPL", {
        "period": "6mo",
        "progress": False,
        "auto_adjust": False,
    })]
    assert "ohlcv:AAPL:None:None:6mo" not in cache.get_calls
    assert cache.get_calls[0].startswith(f"ohlcv:{OHLCV_CACHE_VERSION}:AAPL:")
    assert cache.set_calls[0][0].startswith(f"ohlcv:{OHLCV_CACHE_VERSION}:AAPL:")
    assert result["Close"].iloc[-1] == pytest.approx(frame["Close"].iloc[-1])


def test_fetcher_history_fallback_uses_same_ticker_unadjusted(monkeypatch):
    frame = _ohlcv_frame()
    history_calls = []

    monkeypatch.setattr(fetcher_module.yf, "download", lambda *a, **k: pd.DataFrame())

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            history_calls.append((self.symbol, kwargs))
            return frame

    monkeypatch.setattr(fetcher_module.yf, "Ticker", FakeTicker)

    fetcher = MarketDataFetcher()
    fetcher.cache = _RecordingCache()

    result = fetcher.fetch_ohlcv("TSLA", start_date="2024-01-01")

    assert history_calls[0][0] == "TSLA"
    assert history_calls[0][1]["auto_adjust"] is False
    assert result["Close"].iloc[-1] == pytest.approx(frame["Close"].iloc[-1])


def test_fetcher_raises_when_provider_returns_no_data(monkeypatch):
    monkeypatch.setattr(fetcher_module.yf, "download", lambda *a, **k: pd.DataFrame())

    class EmptyTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            return pd.DataFrame()

    monkeypatch.setattr(fetcher_module.yf, "Ticker", EmptyTicker)

    fetcher = MarketDataFetcher()
    fetcher.cache = _RecordingCache()

    with pytest.raises(ValueError, match="No price data available"):
        fetcher.fetch_ohlcv("NOT_A_REAL_TICKER")
