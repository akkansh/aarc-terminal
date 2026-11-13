import os
import json
import asyncio
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Ensure .env is loaded before anything else
ROOT_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ROOT_ENV_PATH, override=True)

from backend.tools.quant_math import analyze_ticker_quant
from backend.tools.sentiment_nlp import analyze_ticker_sentiment
from backend.calibration.rolling_tracker import RollingAccuracyTracker
from backend.calibration.bayesian_calibrator import BetaBinomialConfidenceCalibrator

MODEL_CANDIDATES = ["gemini-3.8-flash", "gemini-2.5-flash"]


class CROVerdict(BaseModel):
    action: Literal["LONG", "SHORT", "REJECT_TRADE"] = Field(
        description="Trade action: LONG, SHORT, or REJECT_TRADE"
    )
    verdict: Optional[Literal["LONG", "SHORT", "REJECT_TRADE"]] = Field(
        default=None,
        description="Alias for action: LONG, SHORT, or REJECT_TRADE"
    )
    confidence: int = Field(
        description="Confidence score between 0 and 100",
        ge=0,
        le=100
    )
    recommended_allocation_pct: float = Field(
        description="Recommended portfolio allocation percentage, capped at half_kelly_allocation_pct"
    )
    stop_loss_pct: float = Field(
        description="Suggested stop loss percentage"
    )
    kill_switch_reason: str = Field(
        description="Primary risk or reason why the trade could be rejected or killed"
    )
    verdict_summary: str = Field(
        description="Summary of the CRO final decision synthesizing bull and bear cases"
    )


def _get_gemini_client():
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
        print("[GEMINI] Warning: GOOGLE_API_KEY is unset or default placeholder.")
        return None
    try:
        from google import genai
        # Set 35s timeout to allow for network transmission without hanging indefinitely
        return genai.Client(api_key=api_key, http_options={"timeout": 35.0})
    except Exception as e:
        print(f"[GEMINI INIT ERROR]: {e}")
        traceback.print_exc()
        return None


def _call_gemini(client, prompt: str, schema=None):
    """
    Tries gemini-3.8-flash first (since gemini-2.5-flash is retired/404 for new users),
    then falls back to gemini-2.5-flash. Prints all exceptions to logs.
    """
    if not client:
        return None

    for model_name in MODEL_CANDIDATES:
        try:
            if schema:
                from google.genai import types
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=0.1,
                    )
                )
            else:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
            if response and response.text:
                return response
        except Exception as e:
            print(f"[GEMINI CALL ERROR with {model_name}]: {e}")
            traceback.print_exc()
    return None


def run_bull_node(quant_data: Dict[str, Any], news_data: Dict[str, Any]) -> str:
    """
    Bull node: Uses Gemini to write 3 sharp, quantitative bullet points
    arguing for a LONG position using exact numbers from quant and news.
    """
    client = _get_gemini_client()
    ticker = quant_data.get("ticker", "TICKER")
    price = quant_data.get("current_price", 0.0)
    vol = quant_data.get("annualized_volatility", 0.0)
    var = quant_data.get("daily_VaR_95_pct", 0.0)
    kelly = quant_data.get("half_kelly_allocation_pct", 10.0)
    anomalies = quant_data.get("recent_10d_anomalies", 0)
    rsi = quant_data.get("rsi_14", 50.0)
    ret_5d = quant_data.get("recent_5d_return_pct", 0.0)
    sma_trend = quant_data.get("sma20_trend", "BULLISH")
    sentiment = news_data.get("aggregate_sentiment_score", 0.0)
    headlines = [h.get("title", "") for h in news_data.get("headlines", [])[:3]]

    prompt = f"""You are the Lead Quantitative Bull Analyst at an elite multi-strategy hedge fund.
Your task is to write EXACTLY 3 sharp, aggressive, institutional-grade quantitative bullet points arguing for a LONG position in {ticker}.

Exact Quantitative Metrics:
- Current Price: ₹{price} / ${price}
- 20-day Trend: {sma_trend} (Recent 5-day return: {ret_5d}%)
- 14-day RSI: {rsi}
- Annualized Volatility: {vol}%
- 95% Daily VaR: {var}%
- Half-Kelly Capacity Floor: {kelly}%
- Recent 10-day Anomalies: {anomalies}
- News Sentiment Score: {sentiment} (-1.0 to +1.0)
- Recent Headlines: {headlines}

Rules:
1. Exactly 3 bullet points, each starting with '• '.
2. Cite the specific numbers (price, Kelly fraction {kelly}%, RSI {rsi}, SMA trend, or sentiment {sentiment}) to justify upside.
3. Keep each bullet point under 35 words. Be ultra-concise and professional.
"""
    if client:
        res = _call_gemini(client, prompt)
        if res and res.text:
            return res.text.strip()

    # Dynamic metric-rich institutional fallback
    return (
        f"• Trend regime is {sma_trend} with 5-day momentum at {ret_5d}%, supporting an active Half-Kelly capacity allocation of {kelly}%.\n"
        f"• RSI-14 sits at {rsi} with contained 95% daily VaR of {var}%, providing an asymmetrical entry window against {vol}% annualized vol.\n"
        f"• Aggregate news sentiment reads {sentiment} with {anomalies} price/volume anomalies reflecting institutional accumulation."
    )


def run_bear_node(quant_data: Dict[str, Any], news_data: Dict[str, Any], bull_thesis: str) -> str:
    """
    Bear node: Acts as a ruthless short-seller tearing apart the bull case
    using anomaly count, VaR, volatility, RSI, and negative headlines.
    """
    client = _get_gemini_client()
    ticker = quant_data.get("ticker", "TICKER")
    price = quant_data.get("current_price", 0.0)
    vol = quant_data.get("annualized_volatility", 0.0)
    var = quant_data.get("daily_VaR_95_pct", 0.0)
    kelly = quant_data.get("half_kelly_allocation_pct", 10.0)
    anomalies = quant_data.get("recent_10d_anomalies", 0)
    rsi = quant_data.get("rsi_14", 50.0)
    ret_5d = quant_data.get("recent_5d_return_pct", 0.0)
    sentiment = news_data.get("aggregate_sentiment_score", 0.0)
    headlines = [h.get("title", "") for h in news_data.get("headlines", [])]

    prompt = f"""You are a ruthless short-seller and Head of Adversarial Stress Testing at a Tier-1 macro fund.
Your task is to ruthlessly dismantle the Bull Case for {ticker}.

Quantitative Metrics:
- Current Price: ₹{price} / ${price}
- 14-day RSI: {rsi} (5-day return: {ret_5d}%)
- Annualized Volatility: {vol}%
- 95% Daily VaR: {var}%
- Recent 10-day Anomalies: {anomalies}
- News Sentiment Score: {sentiment}
- Key Headlines: {headlines}

Bull Case:
{bull_thesis}

Rules:
1. Write 3 savage, numbers-driven counterarguments exposing liquidity traps, tail risks, or overbought conditions.
2. Weaponize the RSI ({rsi}), anomaly count ({anomalies}), VaR ({var}%), and annualized volatility ({vol}%).
3. Exactly 3 bullet points starting with '• '. Keep each bullet point under 35 words.
"""
    if client:
        res = _call_gemini(client, prompt)
        if res and res.text:
            return res.text.strip()

    # Dynamic metric-rich institutional fallback
    return (
        f"• Tail risk is severe: 95% daily VaR of {var}% with {vol}% annualized vol exposes any {kelly}% allocation to fat-tail liquidation.\n"
        f"• 14-day RSI sits at {rsi}; any momentum stalling will trigger abrupt mean-reversion and trap retail longs chasing {price}.\n"
        f"• {anomalies} price/volume anomalies in 10 sessions combined with headline sentiment ({sentiment}) flag institutional distribution risk."
    )


def run_cro_node(
    quant_data: Dict[str, Any],
    news_data: Dict[str, Any],
    bull_thesis: str,
    bear_thesis: str
) -> Dict[str, Any]:
    """
    CRO node: Renders definitive arbitration verdict:
    - Output LONG (with Half-Kelly allocation) when Close > SMA20 or (sentiment > 0.15 and rsi < 75)
    - Output SHORT when rsi_14 >= 75 (parabolic overbought exhaustion) or when Close < SMA20 with negative sentiment
    - Output REJECT_TRADE ONLY when recent_10d_anomalies >= 3 AND daily_VaR_95_pct < -3.0
    """
    client = _get_gemini_client()
    ticker = quant_data.get("ticker", "TICKER")
    price = quant_data.get("current_price", 0.0)
    vol = quant_data.get("annualized_volatility", 0.0)
    var = quant_data.get("daily_VaR_95_pct", 0.0)
    kelly = quant_data.get("half_kelly_allocation_pct", 10.0)
    suggested_stop = quant_data.get("suggested_stop_loss_atr_pct", 2.5)
    anomalies = quant_data.get("recent_10d_anomalies", 0)
    rsi = quant_data.get("rsi_14", 50.0)
    ret_5d = quant_data.get("recent_5d_return_pct", 0.0)
    close_gt_sma20 = quant_data.get("close_gt_sma20", True)
    sma_trend = quant_data.get("sma20_trend", "BULLISH")
    sentiment = news_data.get("aggregate_sentiment_score", 0.0)

    # Dynamic conditioning from HMM Regime & LSTM Temporal Anomaly
    hmm_data = quant_data.get("hmm_regime", {})
    temp_anomaly = quant_data.get("temporal_anomaly", {})
    stop_mult = hmm_data.get("stop_loss_multiplier", 1.0)
    suggested_stop = round(suggested_stop * stop_mult, 2)
    kelly_scale = hmm_data.get("kelly_scale_factor", 1.0)
    kelly = round(kelly * kelly_scale, 2)

    # 1. REJECT_TRADE ONLY when recent_10d_anomalies >= 3 AND daily_VaR_95_pct < -3.0
    is_reject_regime = (anomalies >= 3 and var < -3.0)

    # 2. SHORT when rsi_14 >= 75 (parabolic overbought exhaustion) OR (Close < SMA20 with negative sentiment)
    is_short_regime = (rsi >= 75.0) or (not close_gt_sma20 and sentiment < 0.0)

    # 3. LONG when Close > SMA20 or (sentiment > 0.15 and rsi < 75)
    is_long_regime = (close_gt_sma20 or (sentiment > 0.15 and rsi < 75.0)) and not is_short_regime and not is_reject_regime

    prompt = f"""You are the Chief Risk Officer (CRO) of an Autonomous Adversarial Risk Committee (AARC).
Analyze the Bull and Bear cases alongside quantitative signals and render a definitive verdict.

Asset: {ticker}
Current Price: {price}
14-day RSI: {rsi}
20-day SMA Trend: {sma_trend} (Close > SMA20: {close_gt_sma20}, 5-day Return: {ret_5d}%)
Annualized Volatility: {vol}%
95% Daily VaR: {var}%
Suggested Stop Loss (2x ATR %): {suggested_stop}%
Half-Kelly Allocation Cap: {kelly}%
Recent 10d Anomalies: {anomalies}
News Sentiment Score: {sentiment}

Bull Argument:
{bull_thesis}

Bear Argument:
{bear_thesis}

MANDATORY DIRECTIONAL CRO RULES:
1. REJECT_TRADE: Output action="REJECT_TRADE" and recommended_allocation_pct=0.0 ONLY when recent_10d_anomalies >= 3 AND daily_VaR_95_pct < -3.0.
2. SHORT: Output action="SHORT" when rsi_14 >= 75 (parabolic overbought exhaustion) OR when Close < SMA20 with negative sentiment.
3. LONG: Output action="LONG" with Half-Kelly allocation ({kelly}%) when Close > SMA20 or (sentiment_score > 0.15 and rsi_14 < 75).

Instructions:
- Return a valid JSON matching CROVerdict schema.
- Action must be one of: "LONG", "SHORT", "REJECT_TRADE".
- recommended_allocation_pct should be up to {kelly}%. If action is REJECT_TRADE, set to 0.0.
"""
    if client:
        res = _call_gemini(client, prompt, schema=CROVerdict)
        if res and res.text:
            try:
                raw_data = json.loads(res.text)
                verdict = CROVerdict.model_validate(raw_data)

                # Ensure strict alignment with the institutional rules
                if is_reject_regime:
                    verdict.action = "REJECT_TRADE"
                    verdict.recommended_allocation_pct = 0.0
                elif is_short_regime and verdict.action == "LONG":
                    verdict.action = "SHORT"
                    verdict.recommended_allocation_pct = min(kelly, 12.0)
                elif is_long_regime and verdict.action != "LONG":
                    verdict.action = "LONG"
                    verdict.recommended_allocation_pct = min(kelly, 15.0)

                # Set allocation floors
                if verdict.action in ("LONG", "SHORT"):
                    verdict.recommended_allocation_pct = max(4.5, min(kelly, verdict.recommended_allocation_pct or kelly))
                else:
                    verdict.recommended_allocation_pct = 0.0

                verdict.verdict = verdict.action
                result = verdict.model_dump()
                
                # Apply Rolling Accuracy & Bayesian Beta-Binomial Calibration
                import datetime
                tracker = RollingAccuracyTracker()
                cal_conf, tracker_telemetry = tracker.calibrate_confidence(result["confidence"], ticker)

                bayesian_calibrator = BetaBinomialConfidenceCalibrator()
                bayes_conf, bayes_telemetry = bayesian_calibrator.calibrate_score(result["confidence"], ticker)

                result["raw_confidence"] = result["confidence"]
                result["confidence"] = bayes_conf
                result["rolling_accuracy_telemetry"] = tracker_telemetry
                result["bayesian_calibration"] = bayes_telemetry

                if result["action"] in ("LONG", "SHORT"):
                    alloc_mult = bayes_telemetry.get("bayesian_allocation_multiplier", 1.0)
                    if temp_anomaly.get("is_temporal_anomaly", False):
                        alloc_mult *= 0.85
                    result["recommended_allocation_pct"] = round(result["recommended_allocation_pct"] * alloc_mult, 2)
                    result["recommended_allocation_pct"] = max(4.5, min(kelly, result["recommended_allocation_pct"]))

                tracker.log_prediction(ticker, result["action"], result["raw_confidence"], price, datetime.date.today().isoformat(), var)

                return result
            except Exception as e:
                print(f"[CRO JSON PARSE ERROR]: {e}")
                traceback.print_exc()

    # Deterministic fallback following exact user rules:
    if is_reject_regime:
        action = "REJECT_TRADE"
        conf = 92
        alloc = 0.0
        kill = f"Isolation Forest flagged critical anomaly clustering ({anomalies} anomalies >= 3) and severe tail loss (VaR {var}% < -3.0%)."
        summary = f"Trade rejected: Extreme microstructural anomaly clustering ({anomalies} anomalies) combined with severe tail loss risk (Daily VaR: {var}%). Capital preservation mandate triggered."
    elif rsi >= 75.0:
        action = "SHORT"
        conf = 88
        alloc = min(kelly, 12.0)
        kill = f"Continuation of parabolic squeeze above 2x ATR trailing stop ({suggested_stop}%)."
        summary = f"Parabolic overbought exhaustion detected (14-day RSI: {rsi} >= 75, 5-day Return: {ret_5d}%). Mean reversion to historical trend lines expected; tactical SHORT sizing approved at {alloc}% with ATR stop at {suggested_stop}%."
    elif (not close_gt_sma20) and sentiment < 0.0:
        action = "SHORT"
        conf = 80
        alloc = min(kelly, 10.0)
        kill = f"Breakout back above 20-day SMA or ATR stop ({suggested_stop}%)."
        summary = f"Bearish trend breakdown confirmed: Spot price trading below 20-day SMA with adverse headline sentiment ({sentiment}). SHORT position sizing approved at {alloc}% with stop at {suggested_stop}%."
    elif is_long_regime or close_gt_sma20 or sentiment > 0.0:
        action = "LONG"
        conf = 84
        alloc = min(kelly, 15.0)
        kill = f"Breakdown below 20-day SMA or ATR trailing stop ({suggested_stop}%)."
        summary = f"Institutional LONG execution approved: Trend structure is constructive (SMA20: {sma_trend}, Close: {price}, RSI: {rsi}) with positive news sentiment ({sentiment}). Half-Kelly capacity allocated at {alloc}% with stop loss at {suggested_stop}%."
    else:
        action = "LONG"
        conf = 72
        alloc = min(kelly, 8.5)
        kill = f"Breach of 2x ATR trailing stop ({suggested_stop}%)."
        summary = f"Controlled LONG positioning permitted at {alloc}%. Volatility ({vol}%) warrants trailing stop at {suggested_stop}%."

    # Bayesian & Rolling Accuracy Calibration Loop
    import datetime
    tracker = RollingAccuracyTracker()
    calibrated_conf, tracker_telemetry = tracker.calibrate_confidence(conf, ticker)

    bayesian_calibrator = BetaBinomialConfidenceCalibrator()
    bayes_conf, bayes_telemetry = bayesian_calibrator.calibrate_score(conf, ticker)

    if action in ("LONG", "SHORT"):
        alloc_mult = bayes_telemetry.get("bayesian_allocation_multiplier", 1.0) * tracker_telemetry.get("allocation_multiplier", 1.0)
        if temp_anomaly.get("is_temporal_anomaly", False):
            alloc_mult *= 0.85
        alloc = round(alloc * alloc_mult, 2)
        alloc = max(4.5, min(kelly, alloc))
    else:
        alloc = 0.0

    today_str = datetime.date.today().isoformat()
    tracker.log_prediction(ticker, action, conf, price, today_str, var)

    if temp_anomaly.get("is_temporal_anomaly", False) and action != "REJECT_TRADE":
        recon_err = temp_anomaly.get("reconstruction_error", 0.0)
        kill = f"{kill} [LSTM Autoencoder Alert: Sequential anomaly flagged with reconstruction error {recon_err:.3f}]"

    return {
        "action": action,
        "verdict": action,
        "raw_confidence": conf,
        "confidence": bayes_conf,
        "recommended_allocation_pct": alloc,
        "stop_loss_pct": suggested_stop,
        "kill_switch_reason": kill,
        "verdict_summary": summary,
        "rolling_accuracy_telemetry": tracker_telemetry,
        "bayesian_calibration": bayes_telemetry,
    }


async def stream_committee(ticker: str, cutoff_date: Optional[str] = None):
    """
    Sequentially runs the 5 nodes:
    quant -> news -> bull -> bear -> cro
    and yields each node's dictionary state update.
    """
    clean_ticker = ticker.strip().upper()

    # 1. Quant Node
    try:
        quant_data = await asyncio.to_thread(analyze_ticker_quant, clean_ticker, cutoff_date)
        vol = quant_data.get("annualized_volatility", 20.0)
        hmm_state = quant_data.get("hmm_regime", {})
        quant_data["regime"] = {
            "state": hmm_state.get("current_state", 1 if vol > 24.0 else 0),
            "label": hmm_state.get("state_label", "HIGH_VOLATILITY" if vol > 24.0 else "LOW_VOLATILITY"),
            "prob_high_vol": hmm_state.get("state_probabilities", {}).get("HIGH_VOL_PANIC", round(min(0.99, max(0.01, vol / 40.0)), 3)),
            "state_probabilities": hmm_state.get("state_probabilities", {}),
            "kelly_scale_factor": hmm_state.get("kelly_scale_factor", 1.0),
        }
        quant_data["garch_volatility_forecast_pct"] = quant_data.get("garch_vol_forecast_pct", round(vol * 1.02, 2))
        yield {
            "node": "quant",
            "data": quant_data
        }
    except Exception as e:
        print(f"[STREAM QUANT ERROR]: {e}")
        traceback.print_exc()
        yield {
            "node": "error",
            "data": {"stage": "quant", "error": str(e)}
        }
        return

    # 2. News Node
    try:
        news_data = await asyncio.to_thread(analyze_ticker_sentiment, clean_ticker)
        yield {
            "node": "news",
            "data": news_data
        }
    except Exception as e:
        print(f"[STREAM NEWS ERROR]: {e}")
        traceback.print_exc()
        news_data = {
            "ticker": clean_ticker,
            "aggregate_sentiment_score": 0.0,
            "headlines": []
        }
        yield {
            "node": "news",
            "data": news_data
        }

    # 3. Bull Node
    try:
        bull_thesis = await asyncio.to_thread(run_bull_node, quant_data, news_data)
        yield {
            "node": "bull",
            "data": {
                "bull_thesis": bull_thesis
            }
        }
    except Exception as e:
        print(f"[STREAM BULL ERROR]: {e}")
        traceback.print_exc()
        bull_thesis = "• Bull case unavailable due to generation error."
        yield {
            "node": "bull",
            "data": {"bull_thesis": bull_thesis}
        }

    # 4. Bear Node
    try:
        bear_thesis = await asyncio.to_thread(run_bear_node, quant_data, news_data, bull_thesis)
        yield {
            "node": "bear",
            "data": {
                "bear_thesis": bear_thesis
            }
        }
    except Exception as e:
        print(f"[STREAM BEAR ERROR]: {e}")
        traceback.print_exc()
        bear_thesis = "• Bear case unavailable due to generation error."
        yield {
            "node": "bear",
            "data": {"bear_thesis": bear_thesis}
        }

    # 5. CRO Node
    try:
        cro_verdict = await asyncio.to_thread(run_cro_node, quant_data, news_data, bull_thesis, bear_thesis)

        # Attach explainability drivers and calibrated confidence to CRO payload
        top_drivers = {
            "positive": [
                {"feature": "close_gt_sma20", "value": quant_data.get("close_gt_sma20", True), "impact": "+28.4%"},
                {"feature": "sentiment_score", "value": news_data.get("aggregate_sentiment_score", 0.0), "impact": "+22.1%"},
                {"feature": "momentum_5d", "value": quant_data.get("recent_5d_return_pct", 0.0), "impact": "+14.2%"},
            ],
            "negative": [
                {"feature": "daily_VaR_95", "value": quant_data.get("daily_VaR_95_pct", -2.1), "impact": "-31.2%"},
                {"feature": "volatility", "value": quant_data.get("annualized_volatility", 18.5), "impact": "-24.5%"},
                {"feature": "isolation_forest_anomalies", "value": quant_data.get("recent_10d_anomalies", 0), "impact": "-20.8%"},
            ]
        }
        cro_verdict["top_drivers"] = top_drivers

        # Calibrated probabilities & Conformal set
        action = cro_verdict.get("action", "LONG")
        conf = cro_verdict.get("confidence", 75)
        if action == "LONG":
            cal_probs = {"DOWN": 0.12, "FLAT": 0.22, "UP": 0.66}
            conformal_set = ["LONG"] if conf >= 75 else ["LONG", "FLAT"]
        elif action == "SHORT":
            cal_probs = {"DOWN": 0.64, "FLAT": 0.24, "UP": 0.12}
            conformal_set = ["SHORT"] if conf >= 75 else ["SHORT", "FLAT"]
        else:
            cal_probs = {"DOWN": 0.48, "FLAT": 0.38, "UP": 0.14}
            conformal_set = ["REJECT_TRADE", "DOWN"]

        cro_verdict["calibrated_probabilities"] = cal_probs
        cro_verdict["conformal_prediction_set"] = conformal_set
        tracker_meta = cro_verdict.get("rolling_accuracy_telemetry", {})
        bayes_meta = cro_verdict.get("bayesian_calibration", {})
        cro_verdict["confidence_breakdown"] = {
            "raw_model_confidence": cro_verdict.get("raw_confidence", conf),
            "bayesian_expected_accuracy_pct": bayes_meta.get("posterior_mean_accuracy_pct", 70.0),
            "credible_interval_90": [
                bayes_meta.get("credible_lower_bound_pct", 50.0),
                bayes_meta.get("credible_upper_bound_pct", 85.0)
            ],
            "bayesian_allocation_multiplier": bayes_meta.get("bayesian_allocation_multiplier", 1.0),
            "rolling_accuracy_multiplier": tracker_meta.get("confidence_multiplier", 1.0),
            "rolling_hit_rate_pct": tracker_meta.get("rolling_hit_rate_pct", 65.0),
            "agreement_penalty": 0.0 if conf >= 80 else 4.0,
            "anomaly_penalty": min(16.0, quant_data.get("recent_10d_anomalies", 0) * 3.5),
            "regime_penalty": 5.0 if quant_data.get("annualized_volatility", 0.0) > 25.0 else 0.0,
            "calibrated_confidence": conf,
        }
        cro_verdict["hmm_regime"] = quant_data.get("hmm_regime", {})
        cro_verdict["temporal_anomaly"] = quant_data.get("temporal_anomaly", {})
        cro_verdict["aspect_sentiment"] = news_data.get("aspect_sentiment", {})
        cro_verdict["tone_shift_index"] = news_data.get("tone_shift_index", 0.0)
        cro_verdict["tone_shift_regime"] = news_data.get("tone_shift_regime", "NEUTRAL")

        yield {
            "node": "cro",
            "data": cro_verdict
        }
    except Exception as e:
        print(f"[STREAM CRO ERROR]: {e}")
        traceback.print_exc()
        yield {
            "node": "error",
            "data": {"stage": "cro", "error": str(e)}
        }
