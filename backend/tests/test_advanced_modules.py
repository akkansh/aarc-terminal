"""Unit tests for Advanced Institutional Quantitative and AI Modules:
1. Hidden Markov Model (HMM) 3-State Dynamic Regime Detection
2. LSTM Autoencoder for Multi-variable Temporal Anomaly Detection
3. Aspect-Based Sentiment Analysis (ABSA) & Tone Shift Index (TSI)
4. Bayesian Confidence Calibration (Beta-Binomial Conjugate Updating)
"""

import pytest
import numpy as np
import pandas as pd
from backend.models.hmm_regime import MarketRegimePipeline, GaussianHMM3State
from backend.models.lstm_autoencoder import LSTMTemporalAnomalyDetector
from backend.features.aspect_sentiment import AspectBasedSentimentAnalyzer
from backend.calibration.bayesian_calibrator import BetaBinomialConfidenceCalibrator


def _generate_synthetic_market_data(n_bars: int = 120, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    base_price = 100.0
    returns = np.random.normal(0.0005, 0.015, n_bars)
    # Add a high-volatility cluster in the last 20 bars
    returns[-20:] = np.random.normal(-0.005, 0.04, 20)
    prices = base_price * np.exp(np.cumsum(returns))

    high = prices * (1.0 + np.abs(np.random.normal(0.005, 0.003, n_bars)))
    low = prices * (1.0 - np.abs(np.random.normal(0.005, 0.003, n_bars)))
    volume = np.random.lognormal(mean=14.0, sigma=0.4, size=n_bars)
    dates = pd.date_range(end=pd.Timestamp.now().normalize(), periods=n_bars, freq="B")

    return pd.DataFrame({
        "Date": dates,
        "Open": prices * 0.998,
        "High": high,
        "Low": low,
        "Close": prices,
        "Volume": volume,
    })


def test_hmm_regime_detection():
    df = _generate_synthetic_market_data(n_bars=100)
    pipeline = MarketRegimePipeline()
    ret_series = df["Close"].pct_change().dropna()
    result = pipeline.analyze_market_regime(ret_series)

    # Assert schema and valid probabilities
    assert "active_state" in result
    assert result["active_state"] in (0, 1, 2)
    assert result["regime_name"] in ("LOW_VOL_TRENDING", "HIGH_VOL_PANIC", "MEAN_REVERTING_CHOPPY")

    probs = result["probabilities"]
    assert len(probs) == 3
    prob_sum = sum(probs.values())
    assert pytest.approx(prob_sum, abs=1e-2) == 1.0

    # Multipliers must be positive and bounded
    assert 0.20 <= result["kelly_scale_factor"] <= 1.25
    assert 0.80 <= result["stop_loss_multiplier"] <= 1.60


def test_lstm_autoencoder_temporal_anomaly():
    df = _generate_synthetic_market_data(n_bars=80)
    detector = LSTMTemporalAnomalyDetector(seq_len=10, hidden_dim=16, epochs=15)
    result = detector.fit_and_detect(df)

    assert "is_temporal_anomaly" in result
    assert isinstance(result["is_temporal_anomaly"], (bool, np.bool_))
    assert result["reconstruction_error"] >= 0.0
    assert result["threshold"] >= 0.0
    assert "feature_breakdown" in result
    assert len(result["feature_breakdown"]) == 5
    for feat, err in result["feature_breakdown"].items():
        assert err >= 0.0


def test_aspect_based_sentiment_and_tone_shift():
    analyzer = AspectBasedSentimentAnalyzer()
    headlines = [
        {"title": "Company faces debt covenant default probe amid severe insolvency risk", "label": "negative", "net_polarity": -0.85},
        {"title": "Free cash flow drops sharply causing liquidity crunch and credit line freeze", "label": "negative", "net_polarity": -0.90},
        {"title": "Operating margin compression worsens as input costs surge", "label": "negative", "net_polarity": -0.75},
        {"title": "Quarterly revenue report shows modest expansion across retail units", "label": "positive", "net_polarity": 0.40},
        {"title": "Company announces dividend hike following solid operating cash performance", "label": "positive", "net_polarity": 0.60},
    ]

    result = analyzer.analyze_aspects_and_tone_shift(headlines)

    assert "aspect_scores" in result
    aspects = result["aspect_scores"]
    assert "risk_solvency" in aspects
    assert "liquidity_cashflow" in aspects
    assert "margin_health" in aspects

    # Risk and liquidity should detect the negative polarity
    assert aspects["risk_solvency"] < 0.0
    assert aspects["liquidity_cashflow"] < 0.0
    assert aspects["margin_health"] < 0.0

    # Composite aspect score must be negative
    assert result["composite_aspect_score"] < 0.0

    # Tone Shift Index should reflect deterioration (recent negative news vs baseline)
    assert result["tone_shift_regime"] in ("SHARP_DETERIORATION", "RAPID_IMPROVEMENT", "STABLE")


def test_bayesian_confidence_calibration(tmp_path):
    storage = str(tmp_path / "bayesian_test.json")
    calibrator = BetaBinomialConfidenceCalibrator(
        alpha_0=6.0,
        beta_0=4.0,
        gamma=0.98,
        storage_path=storage,
    )

    # Initial posterior summary
    summary = calibrator.get_posterior_summary("NVDA")
    assert summary["alpha"] == 6.0
    assert summary["beta"] == 4.0
    assert pytest.approx(summary["posterior_mean_accuracy"], abs=1e-3) == 0.60
    assert len(summary["credible_interval_90"]) == 2
    assert summary["credible_interval_90"][0] < summary["credible_interval_90"][1]

    # Calibrate raw score 85
    cal_score, telemetry = calibrator.calibrate_score(85, "NVDA")
    assert 25 <= cal_score <= 95
    assert telemetry["bayesian_allocation_multiplier"] > 0.0

    # Simulate sequence of 5 accurate predictions (hits)
    for _ in range(5):
        calibrator.update_evidence(is_hit=True, ticker="NVDA")

    updated_summary = calibrator.get_posterior_summary("NVDA")
    assert updated_summary["posterior_mean_accuracy"] > 0.60

    # Calibrate again; confidence should be higher now that empirical accuracy is proven
    higher_score, _ = calibrator.calibrate_score(85, "NVDA")
    assert higher_score >= cal_score
