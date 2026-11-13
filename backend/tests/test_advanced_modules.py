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


def test_brier_score_loss_optimization():
    from backend.calibration.brier_optimizer import BrierScoreCalibrator

    brier = BrierScoreCalibrator()

    # Synthetic well-calibrated predictions
    probs_good = np.array([0.9, 0.85, 0.8, 0.2, 0.15, 0.1, 0.88, 0.75])
    outcomes_good = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    metrics_good = brier.compute_brier_metrics(probs_good, outcomes_good)

    assert metrics_good["brier_score"] < 0.12
    assert metrics_good["brier_skill_score"] > 0.30
    assert metrics_good["is_well_calibrated"] is True

    # Synthetic poorly calibrated predictions (inverted conviction)
    probs_bad = np.array([0.95, 0.90, 0.85, 0.80, 0.88, 0.92])
    outcomes_bad = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    metrics_bad = brier.compute_brier_metrics(probs_bad, outcomes_bad)

    assert metrics_bad["brier_score"] > 0.40
    assert metrics_bad["brier_skill_score"] < 0.0
    assert metrics_bad["is_well_calibrated"] is False

    # Temperature optimization for Brier loss
    opt_t, opt_info = brier.optimize_temperature_for_brier(
        raw_confidences=np.array([90, 85, 80, 75, 70, 65]),
        realized_outcomes=np.array([1, 1, 1, 0, 0, 0]),
    )
    assert 0.4 <= opt_t <= 2.5
    assert "min_brier_loss" in opt_info

    # Calibrating with poor Brier skill collapses confidence toward base rate
    cal_c, tele = brier.calibrate_confidence_via_brier(90, probs_bad, outcomes_bad)
    assert cal_c <= 68
    assert tele["brier_adjustment_factor"] < 1.0


def test_walk_forward_validation_gating():
    from backend.calibration.walk_forward import WalkForwardValidator

    wfv = WalkForwardValidator(target_high_conf_hit_rate=0.75)

    # Historical predictions where high confidence predictions suffered severe drawdowns (hit rate 33%)
    degraded_predictions = [
        {"raw_confidence": 88, "outcome": "MISS", "resolved": True},
        {"raw_confidence": 85, "outcome": "MISS", "resolved": True},
        {"raw_confidence": 82, "outcome": "HIT", "resolved": True},
        {"raw_confidence": 70, "outcome": "HIT", "resolved": True},
        {"raw_confidence": 65, "outcome": "MISS", "resolved": True},
        {"raw_confidence": 60, "outcome": "HIT", "resolved": True},
    ]

    metrics = wfv.evaluate_walk_forward_metrics(degraded_predictions)
    assert metrics["is_gated"] is True
    assert metrics["high_conf_hit_rate"] <= 0.40
    assert metrics["wfv_scaling_factor"] < 0.85
    assert "WFV Gating Applied" in metrics["gating_reason"]

    # Ingest into Bayesian Calibrator: should strictly cap confidence at <= 75%
    calibrator = BetaBinomialConfidenceCalibrator()
    gated_score, tele = calibrator.calibrate_score(
        raw_ai_confidence=92,
        walk_forward_metrics=metrics,
    )
    assert gated_score <= 75
    assert tele["walk_forward_validation"]["is_gated"] is True


def test_compound_directional_and_risk_containment_outcomes(tmp_path):
    from backend.calibration.rolling_tracker import RollingAccuracyTracker

    tracker_storage = str(tmp_path / "rolling_test.json")
    tracker = RollingAccuracyTracker(storage_path=tracker_storage)

    # Create 10 bars of OHLCV history
    dates = pd.date_range(end=pd.Timestamp.now().normalize(), periods=10, freq="B")
    
    # Scenario A: LONG trade where low price breached stop-loss (-3.5%) before recovering (+2.0%)
    # Compound evaluation MUST mark this as MISS due to stop-loss breach
    p0 = 100.0
    prices_a = [100.0, 98.0, 96.0, 96.2, 99.0, 102.0, 103.0, 103.0, 104.0, 105.0]
    lows_a =   [ 99.0, 97.0, 95.8, 95.5, 98.0, 101.0, 102.0, 102.0, 103.0, 104.0] # 95.5 breaches 3% stop on 100
    highs_a =  [101.0, 99.0, 97.0, 98.0, 100.0, 103.0, 104.0, 104.0, 105.0, 106.0]

    df_a = pd.DataFrame({
        "Open": prices_a,
        "High": highs_a,
        "Low": lows_a,
        "Close": prices_a,
        "Volume": [1000] * 10,
    }, index=dates)

    ev_id = tracker.log_prediction(
        ticker="RELIANCE.NS",
        action="LONG",
        raw_confidence=88,
        current_price=100.0,
        date_str=dates[0].strftime("%Y-%m-%d"),
        garch_var_pct=-2.5,
        stop_loss_pct=3.0,
    )

    tracker.resolve_outcomes("RELIANCE.NS", df_a)
    events = tracker._read_events()
    assert len(events) == 1
    ev = events[0]
    assert ev["resolved"] is True
    assert ev["risk_contained"] is False  # Min low was 95.5 (< 97.0)
    assert ev["directional_win"] is True  # Final 5d close was 102.0 (+2.0%)
    assert ev["outcome"] == "MISS"        # Compound requires BOTH
    assert ev["high_conf_divergence"] is True  # Conf 88 >= 80 and outcome == MISS


def test_expected_shortfall_and_var_grounding():
    from backend.models.garch_engine import GARCHVolatilityEngine

    engine = GARCHVolatilityEngine(p=1, q=1, dist="StudentsT")
    np.random.seed(42)
    # Generate 300 returns with heavy negative outliers
    returns = np.random.standard_t(df=4.0, size=300) * 0.015
    ret_series = pd.Series(returns)

    garch_res = engine.compute_dynamic_volatility_and_var(ret_series, confidence_level=0.95)

    assert "dynamic_garch_es_95_pct" in garch_res
    assert "empirical_coverage_pct" in garch_res
    assert "kupiec_pof_pvalue" in garch_res

    # Expected Shortfall must be strictly more conservative than VaR: |ES_95| >= |VaR_95|
    es = abs(garch_res["dynamic_garch_es_95_pct"])
    var = abs(garch_res["dynamic_garch_var_95_pct"])
    assert es >= var

    # Confidence Grounding test in Bayesian Calibrator:
    # If empirical coverage is 91.2%, calibrated score cannot exceed 91%
    calibrator = BetaBinomialConfidenceCalibrator()
    gated_conf, tele = calibrator.calibrate_score(
        raw_ai_confidence=96,
        empirical_coverage_pct=91.2,
        expected_shortfall_pct=-4.2,  # Shock ES expands beyond 3.5%
    )
    assert gated_conf <= 91
    assert tele["tail_risk_grounding"]["es_grounding_cap_pct"] == 91
    assert tele["tail_risk_grounding"]["es_penalty_factor"] < 1.0


def test_high_confidence_divergence_asymmetric_penalty(tmp_path):
    storage = str(tmp_path / "asymmetric_test.json")
    calibrator = BetaBinomialConfidenceCalibrator(
        alpha_0=10.0,
        beta_0=2.0,
        gamma=0.90,
        storage_path=storage,
    )

    # Initial high posterior accuracy: 10 / 12 = 83.3%
    init_summary = calibrator.get_posterior_summary("TCS.NS")
    assert pytest.approx(init_summary["posterior_mean_accuracy"], abs=1e-2) == 0.83

    # High confidence divergence occurs: model asserted high conviction but trade failed
    calibrator.update_evidence(is_hit=False, ticker="TCS.NS", is_high_conf_divergence=True)

    diverged_summary = calibrator.get_posterior_summary("TCS.NS")
    # Beta should receive +2.5 penalty and alpha should be decayed by 0.70 * 0.90
    assert diverged_summary["beta"] >= 4.0
    assert diverged_summary["posterior_mean_accuracy"] < 0.65
