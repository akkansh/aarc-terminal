"""Integration and unit tests for FastAPI REST endpoints:
- GET /api/regime/status
- GET /api/anomaly/status
- GET /api/sentiment/aspects
- GET /api/risk/evaluate
- POST /api/confidence/update
- GET /api/confidence/status
"""

import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from backend.main import app


@pytest.fixture
def client():
    return TestClient(app)


def _mock_ohlcv_frame(n_bars: int = 40) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=n_bars)
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


def test_regime_status_endpoint(client, monkeypatch):
    from backend.tools import quant_math
    frame = _mock_ohlcv_frame()
    monkeypatch.setattr(quant_math.yf, "download", lambda *a, **k: frame)

    response = client.get("/api/regime/status?ticker=RELIANCE.NS")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "RELIANCE.NS"
    assert "active_state" in data
    assert data["active_state"] in (0, 1, 2)
    assert "regime_name" in data
    assert "probabilities" in data
    assert len(data["probabilities"]) == 3
    assert "kelly_scale_factor" in data
    assert "stop_loss_multiplier" in data
    assert data["kelly_scale_factor"] > 0


def test_anomaly_status_endpoint(client, monkeypatch):
    from backend.tools import quant_math
    frame = _mock_ohlcv_frame()
    monkeypatch.setattr(quant_math.yf, "download", lambda *a, **k: frame)

    response = client.get("/api/anomaly/status?ticker=RELIANCE.NS")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "RELIANCE.NS"
    assert "is_temporal_anomaly" in data
    assert isinstance(data["is_temporal_anomaly"], bool)
    assert "reconstruction_error" in data
    assert "reconstruction_threshold" in data
    assert "feature_breakdown" in data
    assert "ensemble_isolation_forest" in data


class _FakeAnalyzer:
    def score_headlines(self, items):
        return {
            "aggregate_sentiment_score": -0.45,
            "headlines": [
                {
                    "title": item["title"],
                    "label": "negative",
                    "score": 0.88,
                    "net_polarity": -0.85,
                    "published_at": item.get("published_at", ""),
                }
                for item in items
            ],
            "aspect_sentiment": {
                "risk_solvency": -0.65,
                "liquidity_cashflow": -0.70,
                "margin_health": 0.0,
            },
            "composite_aspect_score": -0.505,
            "tone_shift_index": -1.35,
            "tone_shift_regime": "SHARP_DETERIORATION",
            "aspect_relevance": {"risk_solvency": 1, "liquidity_cashflow": 1, "margin_health": 0},
        }


def test_sentiment_aspects_endpoint(client, monkeypatch):
    from backend.tools import sentiment_nlp

    class FakeFeed:
        entries = [
            {"title": "Company faces debt and insolvency probe amid credit risk", "published": "Thu, 12 Nov 2026 18:57:34 GMT"},
            {"title": "Free cash flow drops significantly squeezing working capital", "published": "Thu, 12 Nov 2026 18:00:00 GMT"},
        ]

    monkeypatch.setattr(sentiment_nlp.feedparser, "parse", lambda *a, **k: FakeFeed())
    monkeypatch.setattr(sentiment_nlp.FinBertSentimentAnalyzer, "get_instance", lambda: _FakeAnalyzer())

    response = client.get("/api/sentiment/aspects?ticker=RELIANCE.NS")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "RELIANCE.NS"
    assert "aspect_scores" in data
    assert "risk_solvency" in data["aspect_scores"]
    assert "liquidity_cashflow" in data["aspect_scores"]
    assert "margin_health" in data["aspect_scores"]
    assert "composite_aspect_score" in data
    assert "tone_shift_regime" in data


def test_confidence_update_and_status_endpoints(client):
    # Test POST /api/confidence/update
    res = client.post("/api/confidence/update", json={"is_hit": True, "ticker": "TCS.NS"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["is_hit"] is True
    assert "posterior_summary" in data
    assert data["posterior_summary"]["posterior_mean_accuracy"] > 0.0

    # Test GET /api/confidence/status
    res2 = client.get("/api/confidence/status?ticker=TCS.NS")
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["ticker"] == "TCS.NS"
    assert "posterior_summary" in data2
    assert "credible_interval_90" in data2["posterior_summary"]


def test_risk_evaluate_endpoint(client, monkeypatch):
    from backend.tools import quant_math, sentiment_nlp
    frame = _mock_ohlcv_frame()
    monkeypatch.setattr(quant_math.yf, "download", lambda *a, **k: frame)

    class FakeFeed:
        entries = [
            {"title": "Company reports strong quarterly results", "published": "Thu, 12 Nov 2026 18:57:34 GMT"},
        ]
    monkeypatch.setattr(sentiment_nlp.feedparser, "parse", lambda *a, **k: FakeFeed())
    monkeypatch.setattr(sentiment_nlp.FinBertSentimentAnalyzer, "get_instance", lambda: _FakeAnalyzer())

    response = client.get("/api/risk/evaluate?ticker=NVDA")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "NVDA"
    assert data["action"] in ("LONG", "SHORT", "REJECT_TRADE")
    assert "calibrated_confidence" in data
    assert "bayesian_calibration" in data
    assert "regime" in data
    assert "garch_volatility" in data
    assert "sequential_anomaly" in data
    assert "sentiment_and_tone_shift" in data
    assert "risk_boundaries" in data
    assert "theses" in data
