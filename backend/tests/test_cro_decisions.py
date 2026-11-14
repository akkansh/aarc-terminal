"""CRO model responses must not override the configured risk rules."""

import json
from types import SimpleNamespace

import pytest

import backend.graph as graph


@pytest.fixture(autouse=True)
def isolate_calibration(monkeypatch):
    class Tracker:
        def calibrate_confidence(self, confidence, ticker):
            return confidence, {"allocation_multiplier": 1.0}

        def log_prediction(self, *args):
            pass

    class Calibrator:
        def calibrate_score(self, raw_ai_confidence, **kwargs):
            return raw_ai_confidence, {"bayesian_allocation_multiplier": 1.0}

    monkeypatch.setattr(graph, "RollingAccuracyTracker", Tracker)
    monkeypatch.setattr(graph, "BetaBinomialConfidenceCalibrator", Calibrator)


@pytest.mark.parametrize(
    "quant,model_action,expected",
    [
        ({"rsi_14": 80}, "REJECT_TRADE", "SHORT"),
        ({"close_gt_sma20": False}, "REJECT_TRADE", "SHORT"),
        ({"close_gt_sma20": True}, "REJECT_TRADE", "LONG"),
        ({"recent_10d_anomalies": 3, "daily_VaR_95_pct": -3.1}, "LONG", "REJECT_TRADE"),
        ({"rsi_14": 80}, "SHORT", "SHORT"),
    ],
)
def test_model_verdict_obeys_risk_rules(monkeypatch, quant, model_action, expected):
    payload = {
        "action": model_action,
        "confidence": 80,
        "recommended_allocation_pct": 8,
        "stop_loss_pct": 3,
        "kill_switch_reason": "Model risk",
        "verdict_summary": "Model narrative",
    }
    monkeypatch.setattr(graph, "_get_gemini_client", lambda: object())
    monkeypatch.setattr(graph, "_call_gemini", lambda *args, **kwargs: SimpleNamespace(text=json.dumps(payload)))
    result = graph.run_cro_node(
        {"ticker": "TEST", "half_kelly_allocation_pct": 10, **quant},
        {"aggregate_sentiment_score": -0.2}, "bull", "bear",
    )
    assert result["action"] == result["verdict"] == expected
    assert (result["recommended_allocation_pct"] == 0) == (expected == "REJECT_TRADE")
    if model_action != expected:
        assert result["verdict_summary"] != "Model narrative"
    else:
        assert result["verdict_summary"] == "Model narrative"
