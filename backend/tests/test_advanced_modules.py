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


def test_temperature_scaling_probability_and_reconstruction():
    from backend.models.temperature_scaling import (
        scale_probability_temperature,
        scale_reconstruction_error_temperature,
    )

    # 1. Strong conviction signal (0.78) scaled with T=0.72 clears 0.80+ threshold
    scaled_p, tele_p = scale_probability_temperature(0.78, temperature=0.72)
    assert scaled_p >= 0.82
    assert tele_p["temperature_scaled_probability"] >= 0.82

    # 2. Neutral signal (0.50) stays near 0.50
    neutral_p, _ = scale_probability_temperature(0.50, temperature=0.72)
    assert 0.48 <= neutral_p <= 0.52

    # 3. Weak signal (0.35) is suppressed further (down to ~0.30)
    weak_p, _ = scale_probability_temperature(0.35, temperature=0.72)
    assert weak_p < 0.35

    # 4. LSTM reconstruction error above threshold clears 0.80+
    # Error = 2.5, Threshold = 1.4, IQR = 0.4
    prob_recon, tele_recon = scale_reconstruction_error_temperature(
        reconstruction_error=2.5,
        threshold=1.4,
        iqr=0.4,
        temperature=0.70,
    )
    assert prob_recon >= 0.80
    assert tele_recon["cleared_high_confidence"] is True

    # 5. Reconstruction error below threshold stays <= 0.50
    prob_clean, tele_clean = scale_reconstruction_error_temperature(
        reconstruction_error=1.0,
        threshold=1.4,
        iqr=0.4,
        temperature=0.70,
    )
    assert prob_clean < 0.50
    assert tele_clean["cleared_high_confidence"] is False


def test_regime_conditional_confidence_gating():
    from backend.models.regime_gater import RegimeConditionalConfidenceGater

    gater = RegimeConditionalConfidenceGater()

    # Case A: Clean Trending Regime -> Promotes high confidence (>80)
    trending_regime = {
        "regime_name": "LOW_VOL_TRENDING",
        "probabilities": {"P_Trending": 0.80, "P_Panic": 0.05, "P_Choppy": 0.15},
    }
    gated_conf, tele_trending = gater.gate_confidence(
        confidence=82,
        hmm_regime=trending_regime,
        action="LONG",
    )
    assert gated_conf >= 82
    assert tele_trending["is_suppressed"] is False
    assert tele_trending["cleared_high_confidence"] is True
    assert tele_trending["regime_status"] == "CLEAN_TRENDING_PROMOTED"

    # Case B: High Vol Panic Regime -> Suppresses directional alert and caps confidence
    panic_regime = {
        "regime_name": "HIGH_VOL_PANIC",
        "probabilities": {"P_Trending": 0.10, "P_Panic": 0.70, "P_Choppy": 0.20},
    }
    gated_panic, tele_panic = gater.gate_confidence(
        confidence=75,
        hmm_regime=panic_regime,
        action="LONG",
    )
    assert tele_panic["is_suppressed"] is True
    assert gated_panic <= 68
    assert "PANIC" in tele_panic["regime_status"]

    # Case C: Choppy Sideways Regime -> Suppresses low-tier alert (<75)
    choppy_regime = {
        "regime_name": "MEAN_REVERTING_CHOPPY",
        "probabilities": {"P_Trending": 0.20, "P_Panic": 0.10, "P_Choppy": 0.70},
    }
    gated_choppy, tele_choppy = gater.gate_confidence(
        confidence=70,
        hmm_regime=choppy_regime,
        action="LONG",
    )
    assert tele_choppy["is_suppressed"] is True
    assert gated_choppy <= 74
    assert "CHOPPY" in tele_choppy["regime_status"]


def test_exponential_decay_forgetting_factor_lambda(tmp_path):
    storage = str(tmp_path / "bayesian_lambda_decay.json")
    calibrator = BetaBinomialConfidenceCalibrator(
        alpha_0=7.0,
        beta_0=5.0,  # Simulate rigid historical errors accumulated in beta
        gamma=0.85,  # lambda = 0.85
        storage_path=storage,
    )

    # Initial posterior mean with beta=5.0 is 7 / (7+5) = 58.3%
    init_summary = calibrator.get_posterior_summary("RELIANCE.NS")
    assert pytest.approx(init_summary["posterior_mean_accuracy"], abs=1e-2) == 0.58

    # Process 3 consecutive prediction hits
    for _ in range(3):
        calibrator.update_evidence(is_hit=True, ticker="RELIANCE.NS")

    # Due to lambda=0.85 decay:
    # beta decays by 0.85^3: 5.0 * 0.85 * 0.85 * 0.85 ~ 3.07
    # alpha updates: 7.0*0.85 + 1 -> 6.95; 6.95*0.85 + 1 -> 6.91; 6.91*0.85 + 1 -> 6.87
    # posterior accuracy rapidly recovers
    recovered_summary = calibrator.get_posterior_summary("RELIANCE.NS")
    assert recovered_summary["beta"] < 4.0
    assert recovered_summary["posterior_mean_accuracy"] > 0.65

    # Calibrating with clean trending regime enables high-confidence score (>80)
    trending_regime = {
        "regime_name": "LOW_VOL_TRENDING",
        "probabilities": {"P_Trending": 0.85, "P_Panic": 0.03, "P_Choppy": 0.12},
    }
    final_conf, cal_meta = calibrator.calibrate_score(
        raw_ai_confidence=84,
        ticker="RELIANCE.NS",
        hmm_regime=trending_regime,
        action="LONG",
    )
    assert final_conf >= 80
    assert cal_meta["forgetting_factor_lambda"] == 0.85
