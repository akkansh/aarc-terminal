"""Regression tests for the news node's failure and cutoff behaviour."""

import asyncio

import backend.graph as graph


def _quant_payload(ticker: str = "AAPL"):
    return {
        "ticker": ticker,
        "current_price": 190.5,
        "annualized_volatility": 22.0,
        "daily_VaR_95_pct": -1.8,
        "suggested_stop_loss_atr_pct": 3.1,
        "half_kelly_allocation_pct": 12.0,
        "recent_10d_anomalies": 0,
        "rsi_14": 58.0,
        "recent_5d_return_pct": 1.2,
        "sma20_trend": "BULLISH",
        "close_gt_sma20": True,
        "sma20_val": 185.0,
        "sma50_val": 180.0,
        "sma20_z_score": 0.8,
        "bb_upper": 195.0,
        "bb_middle": 185.0,
        "bb_lower": 175.0,
        "bb_width_pct": 10.8,
        "chart_data": [],
        "time_machine_actual_return_pct": None,
        "max_drawdown_5d_pct": None,
    }


def _collect(ticker="AAPL", cutoff_date=None):
    async def _run():
        return [
            event
            async for event in graph.stream_committee(ticker, cutoff_date)
        ]

    return asyncio.run(_run())


def test_news_failure_reports_unavailable_not_neutral(monkeypatch):
    monkeypatch.setattr(
        graph,
        "analyze_ticker_quant",
        lambda ticker, cutoff_date=None: _quant_payload(ticker),
    )

    def broken_sentiment(ticker, cutoff_date=None):
        raise RuntimeError("rss unavailable")

    monkeypatch.setattr(graph, "analyze_ticker_sentiment", broken_sentiment)
    monkeypatch.setattr(graph, "_get_gemini_client", lambda: None)

    events = _collect("AAPL")
    nodes = [event["node"] for event in events]

    assert nodes == ["quant", "news", "bull", "bear", "cro"]
    assert "error" not in nodes

    news = events[1]["data"]
    assert news["ticker"] == "AAPL"
    assert news["news_available"] is False
    assert news["aggregate_sentiment_score"] is None
    assert news["headlines"] == []
    assert "rss unavailable" in news["news_error"]


def test_news_node_forwards_requested_ticker_and_cutoff(monkeypatch):
    quant = _quant_payload("MSFT")
    calls = []

    monkeypatch.setattr(
        graph,
        "analyze_ticker_quant",
        lambda ticker, cutoff_date=None: quant,
    )

    def fake_sentiment(ticker, cutoff_date=None):
        calls.append((ticker, cutoff_date))
        return {
            "ticker": ticker,
            "aggregate_sentiment_score": 0.2,
            "headlines": [],
            "news_available": True,
        }

    monkeypatch.setattr(graph, "analyze_ticker_sentiment", fake_sentiment)
    monkeypatch.setattr(graph, "_get_gemini_client", lambda: None)

    events = _collect("MSFT", "2024-06-03")

    assert calls == [("MSFT", "2024-06-03")]
    assert events[1]["data"]["ticker"] == "MSFT"


def test_index_does_not_generate_trade_recommendations(monkeypatch):
    monkeypatch.setattr(graph, "analyze_ticker_quant", lambda ticker, cutoff_date=None: _quant_payload(ticker))
    monkeypatch.setattr(graph, "analyze_ticker_sentiment", lambda ticker, cutoff_date=None: {
        "ticker": ticker, "aggregate_sentiment_score": 0.0, "headlines": [],
    })

    def forbidden(*args, **kwargs):
        raise AssertionError("index must not invoke a trading agent")

    monkeypatch.setattr(graph, "run_bull_node", forbidden)
    monkeypatch.setattr(graph, "run_bear_node", forbidden)
    monkeypatch.setattr(graph, "run_cro_node", forbidden)
    events = _collect("^NSEI")
    assert [event["node"] for event in events] == ["quant", "news", "index_complete"]
