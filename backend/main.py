import os
import asyncio
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Ensure .env is loaded before anything else
ROOT_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ROOT_ENV_PATH, override=True)
from fastapi import FastAPI, Query, HTTPException, Body, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from backend.graph import stream_committee, run_cro_node, run_bull_node, run_bear_node
from backend.tools.quant_math import analyze_ticker_quant
from backend.tools.sentiment_nlp import analyze_ticker_sentiment
from backend.calibration.bayesian_calibrator import BetaBinomialConfidenceCalibrator
from backend.data.news_stream import NewsStreamBroker
import yfinance as yf

app = FastAPI(
    title="Autonomous Adversarial Risk Committee (AARC) API",
    description="Institutional Dark-Mode Multi-Agent Risk Committee Backend",
    version="1.0.0"
)

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CACHE_FILE = Path(__file__).resolve().parent / "demo_cache.json"


def load_demo_cache() -> Dict[str, Any]:
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[CACHE] Error reading demo cache: {e}")
    return {}


def save_to_demo_cache(key: str, events: List[Dict[str, Any]]):
    try:
        cache = load_demo_cache()
        cache[key] = events
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
        print(f"[CACHE] Saved successful run for key '{key}' to {CACHE_FILE}")
    except Exception as e:
        print(f"[CACHE] Error saving to demo cache: {e}")


@app.get("/health")
@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "service": "Autonomous Adversarial Risk Committee (AARC)"
    }


TICKER_ALIASES = {
    "ZOMATO": "ETERNAL.NS",
    "ZOMATO.NS": "ETERNAL.NS",
    "ZOMATO.BO": "ETERNAL.BO",
}


@app.get("/api/instrument")
async def instrument_name(ticker: str = Query(min_length=1, max_length=32)):
    """Resolve only an exact symbol match; never assign a different company's name."""
    symbol = ticker.strip().upper()
    if symbol in ("ZOMATO", "ZOMATO.NS"):
        return {"ticker": "ETERNAL.NS", "name": "Eternal Limited (formerly Zomato)"}
    elif symbol == "ETERNAL.NS":
        return {"ticker": "ETERNAL.NS", "name": "Eternal Limited (formerly Zomato)"}
    try:
        search = await asyncio.to_thread(yf.Search, symbol, max_results=10, news_count=0)
        for quote in search.quotes:
            if str(quote.get("symbol", "")).upper() == symbol:
                name = quote.get("longname") or quote.get("shortname")
                if name:
                    return {"ticker": symbol, "name": name}
    except Exception as exc:
        print(f"[INSTRUMENT] Lookup failed for {symbol}: {exc}")
    return {"ticker": symbol, "name": None}


@app.get("/api/instruments/search")
async def search_instruments(q: str = Query(min_length=1, max_length=80)):
    """Search company names and symbols, returning only equity listings."""
    query = q.strip()
    if not query:
        return {"results": []}
    try:
        search = await asyncio.to_thread(yf.Search, query, max_results=20, news_count=0)
        results = []
        seen = set()

        if "zomato" in query.lower() or "eternal" in query.lower():
            results.append({
                "ticker": "ETERNAL.NS",
                "name": "Eternal Limited (formerly Zomato)",
                "exchange": "NSE"
            })
            seen.add("ETERNAL.NS")
        for quote in search.quotes:
            if quote.get("quoteType", "").upper() != "EQUITY":
                continue
            symbol = quote.get("symbol")
            name = quote.get("longname") or quote.get("shortname")
            if not isinstance(symbol, str) or not isinstance(name, str) or not symbol or not name:
                continue
            symbol = symbol.upper()
            if symbol in seen:
                continue
            seen.add(symbol)
            results.append({"ticker": symbol, "name": name, "exchange": quote.get("exchDisp") or quote.get("exchange") or ""})
            if len(results) == 8:
                break
        return {"results": results}
    except Exception as exc:
        print(f"[INSTRUMENT] Search failed: {exc}")
        raise HTTPException(status_code=503, detail="Stock search is temporarily unavailable") from exc


@app.get("/api/stream")
async def stream_committee_endpoint(
    ticker: str = Query(default="RELIANCE.NS", description="Asset ticker symbol"),
    cutoff_date: Optional[str] = Query(default=None, description="Optional historical cutoff date (YYYY-MM-DD) for backtesting")
):
    """
    SSE endpoint that streams node completion events:
    quant -> news -> bull -> bear -> cro
    as `data: <json>\\n\\n`.
    Failures are reported explicitly as `node: error` events; demo-cache replay is
    intentionally disabled so a request for one ticker can never be served another.
    """
    clean_t = ticker.strip().upper()
    clean_t = TICKER_ALIASES.get(clean_t, clean_t)
    cleaned_cutoff = cutoff_date.strip() if cutoff_date and cutoff_date.strip() else None
    cache_key = f"{clean_t}_{cleaned_cutoff or ''}"

    async def event_generator():
        recorded_events: List[Dict[str, Any]] = []

        try:
            async for node_event in stream_committee(ticker=clean_t, cutoff_date=cleaned_cutoff):
                recorded_events.append(node_event)
                yield {
                    "data": json.dumps(node_event, allow_nan=False)
                }

                # Do not replay cache after a failure: surface the error event and stop.
                if node_event.get("node") == "error":
                    return

            # Only persist a fully successful run; incomplete streams are reported as errors.
            if recorded_events and recorded_events[-1].get("node") == "index_complete":
                return
            if len(recorded_events) >= 5:
                save_to_demo_cache(cache_key, recorded_events)
            else:
                yield {
                    "data": json.dumps({
                        "node": "error",
                        "data": {
                            "stage": "stream",
                            "error": "Committee stream ended before all nodes completed",
                        },
                    })
                }

        except Exception as e:
            print(f"[STREAM] Execution failed or encountered error: {e}")
            yield {
                "data": json.dumps({"node": "error", "data": {"stage": "stream", "error": str(e)}})
            }

    return EventSourceResponse(event_generator())


class ConfidenceUpdateRequest(BaseModel):
    is_hit: bool = Field(description="Whether the previous risk or direction prediction was accurate")
    ticker: Optional[str] = Field(default=None, description="Optional asset ticker symbol")


@app.get("/api/regime/status")
async def get_regime_status(
    ticker: str = Query(default="RELIANCE.NS", description="Asset ticker symbol"),
    cutoff_date: Optional[str] = Query(default=None, description="Optional historical cutoff date (YYYY-MM-DD)"),
):
    """
    Exposes Gaussian HMM 3-state market regime classification and GARCH(1,1) volatility dynamics.
    Outputs active state probabilities, conditional volatility, and regime risk boundaries.
    """
    clean_t = ticker.strip().upper()
    clean_t = TICKER_ALIASES.get(clean_t, clean_t)
    try:
        quant_data = await asyncio.to_thread(analyze_ticker_quant, clean_t, cutoff_date)
        regime = quant_data.get("hmm_regime", {})
        return {
            "ticker": clean_t,
            "active_state": regime.get("active_state", 0),
            "regime_name": regime.get("regime_name", "LOW_VOL_TRENDING"),
            "probabilities": regime.get("probabilities", {"P_Trending": 0.65, "P_Panic": 0.10, "P_Choppy": 0.25}),
            "garch_vol_forecast_annual_pct": quant_data.get("garch_vol_forecast_pct"),
            "volatility_shock_ratio": quant_data.get("volatility_shock_ratio"),
            "regime_vol_annual_pct": regime.get("regime_vol_annual_pct"),
            "kelly_scale_factor": regime.get("kelly_scale_factor", 1.0),
            "stop_loss_multiplier": regime.get("stop_loss_multiplier", 1.0),
            "state_parameters": regime.get("state_parameters", {}),
            "current_price": quant_data.get("current_price"),
            "annualized_volatility": quant_data.get("annualized_volatility"),
            "daily_VaR_95_pct": quant_data.get("daily_VaR_95_pct"),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/anomaly/status")
async def get_anomaly_status(
    ticker: str = Query(default="RELIANCE.NS", description="Asset ticker symbol"),
    cutoff_date: Optional[str] = Query(default=None, description="Optional historical cutoff date (YYYY-MM-DD)"),
):
    """
    Exposes PyTorch LSTM Autoencoder sequence reconstruction anomaly detection
    alongside multi-timeframe ensemble Isolation Forest anomaly votes.
    """
    clean_t = ticker.strip().upper()
    clean_t = TICKER_ALIASES.get(clean_t, clean_t)
    try:
        quant_data = await asyncio.to_thread(analyze_ticker_quant, clean_t, cutoff_date)
        temporal = quant_data.get("temporal_anomaly", {})
        return {
            "ticker": clean_t,
            "is_temporal_anomaly": temporal.get("is_temporal_anomaly", False),
            "temporal_anomaly_score": temporal.get("temporal_anomaly_score", 0.0),
            "reconstruction_error": temporal.get("reconstruction_error", 0.0),
            "reconstruction_threshold": temporal.get("threshold", 0.0),
            "feature_breakdown": temporal.get("feature_breakdown", {}),
            "recent_10d_temporal_count": temporal.get("recent_10d_temporal_count", 0),
            "ensemble_isolation_forest": quant_data.get("anomaly_telemetry", {}),
            "point_in_time_anomalies_10d": quant_data.get("recent_10d_anomalies", 0),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/anomaly/stream")
async def stream_anomaly_status(
    ticker: str = Query(default="RELIANCE.NS", description="Asset ticker symbol"),
    cutoff_date: Optional[str] = Query(default=None, description="Optional historical cutoff date (YYYY-MM-DD)"),
):
    """
    Real-time SSE telemetry stream for temporal anomaly monitoring.
    """
    clean_t = ticker.strip().upper()
    clean_t = TICKER_ALIASES.get(clean_t, clean_t)

    async def anomaly_generator():
        try:
            quant_data = await asyncio.to_thread(analyze_ticker_quant, clean_t, cutoff_date)
            temporal = quant_data.get("temporal_anomaly", {})
            yield {
                "data": json.dumps({
                    "event": "anomaly_telemetry",
                    "ticker": clean_t,
                    "is_temporal_anomaly": temporal.get("is_temporal_anomaly", False),
                    "reconstruction_error": temporal.get("reconstruction_error", 0.0),
                    "threshold": temporal.get("threshold", 0.0),
                    "feature_breakdown": temporal.get("feature_breakdown", {}),
                    "recent_10d_temporal_count": temporal.get("recent_10d_temporal_count", 0),
                    "ensemble_isolation_forest": quant_data.get("anomaly_telemetry", {}),
                })
            }
        except Exception as exc:
            yield {
                "data": json.dumps({"event": "error", "error": str(exc)})
            }

    return EventSourceResponse(anomaly_generator())


@app.get("/api/sentiment/aspects")
async def get_sentiment_aspects(
    ticker: str = Query(default="RELIANCE.NS", description="Asset ticker symbol"),
    cutoff_date: Optional[str] = Query(default=None, description="Optional historical cutoff date (YYYY-MM-DD)"),
):
    """
    Exposes FinBERT sentiment analysis with Loughran-McDonald confidence thresholding,
    Aspect-Based Sentiment Analysis (Risk/Solvency, Liquidity/Cash Flow, Margin Health),
    and Tone Shift Index (TSI) narrative momentum.
    """
    clean_t = ticker.strip().upper()
    clean_t = TICKER_ALIASES.get(clean_t, clean_t)
    try:
        news_data = await asyncio.to_thread(analyze_ticker_sentiment, clean_t, cutoff_date)
        return {
            "ticker": clean_t,
            "aggregate_sentiment_score": news_data.get("aggregate_sentiment_score"),
            "ema_smoothed_sentiment": news_data.get("ema_smoothed_sentiment", news_data.get("aggregate_sentiment_score")),
            "aspect_scores": news_data.get("aspect_sentiment", {}),
            "composite_aspect_score": news_data.get("composite_aspect_score", 0.0),
            "tone_shift_index": news_data.get("tone_shift_index", 0.0),
            "tone_shift_regime": news_data.get("tone_shift_regime", "NEUTRAL"),
            "aspect_relevance": news_data.get("aspect_relevance", {}),
            "news_available": news_data.get("news_available", True),
            "headlines": news_data.get("headlines", []),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


class NewsPublishRequest(BaseModel):
    ticker: str = Field(description="Asset ticker symbol")
    title: str = Field(description="News or regulatory filing headline")
    source: Optional[str] = Field(default="FINANCIAL_NEWS", description="'FINANCIAL_NEWS' or 'REGULATORY_FILING'")
    published_at: Optional[str] = Field(default=None, description="Optional publication timestamp string")
    url: Optional[str] = Field(default=None, description="Optional source article link")


@app.get("/api/news/stream")
async def stream_news_feed(
    ticker: str = Query(default="RELIANCE.NS", description="Asset ticker symbol"),
    cutoff_date: Optional[str] = Query(default=None, description="Optional historical cutoff date (YYYY-MM-DD)"),
):
    """
    Real-time Server-Sent Events (SSE) endpoint streaming live financial news,
    regulatory disclosures, and sub-second FinBERT sentiment updates.
    Uses an asynchronous event-driven queue listener with automatic heartbeat keepalives.
    """
    clean_t = ticker.strip().upper()
    clean_t = TICKER_ALIASES.get(clean_t, clean_t)
    broker = NewsStreamBroker.get_instance()

    return EventSourceResponse(
        broker.news_event_generator(ticker=clean_t, cutoff_date=cutoff_date)
    )


@app.post("/api/news/publish")
async def publish_live_news_item(body: NewsPublishRequest = Body(...)):
    """
    Asynchronously ingests and publishes a live news item or regulatory filing,
    scoring it sub-second with FinBERT and broadcasting it to all connected SSE clients.
    """
    clean_t = body.ticker.strip().upper()
    clean_t = TICKER_ALIASES.get(clean_t, clean_t)
    broker = NewsStreamBroker.get_instance()
    event = await broker.publish_news_item(
        ticker=clean_t,
        title=body.title,
        source=body.source or "FINANCIAL_NEWS",
        published_at=body.published_at,
        url=body.url,
    )
    return {
        "status": "published",
        "event": event,
        "active_subscribers": broker.get_subscriber_count(clean_t),
    }


@app.websocket("/ws/news/{ticker}")
async def websocket_news_stream(websocket: WebSocket, ticker: str):
    """
    Real-time WebSocket endpoint streaming live financial news,
    regulatory disclosures, and sub-second FinBERT sentiment updates.
    """
    clean_t = ticker.strip().upper()
    clean_t = TICKER_ALIASES.get(clean_t, clean_t)
    await websocket.accept()
    broker = NewsStreamBroker.get_instance()
    queue = await broker.subscribe(clean_t)

    try:
        # Send initial snapshot immediately
        snapshot = await broker.get_initial_snapshot(clean_t)
        await websocket.send_text(json.dumps(snapshot))

        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=12.0)
                await websocket.send_text(json.dumps(item))
            except asyncio.TimeoutError:
                from datetime import datetime, timezone
                heartbeat = {
                    "type": "heartbeat",
                    "ticker": clean_t,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": "HEALTHY_STREAMING",
                }
                await websocket.send_text(json.dumps(heartbeat))
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        await broker.unsubscribe(clean_t, queue)


@app.get("/api/risk/evaluate")
async def evaluate_risk_consolidated(
    ticker: str = Query(default="RELIANCE.NS", description="Asset ticker symbol"),
    cutoff_date: Optional[str] = Query(default=None, description="Optional historical cutoff date (YYYY-MM-DD)"),
):
    """
    Consolidated institutional risk arbitration endpoint:
    Synthesizes HMM dynamic regimes, GARCH(1,1) volatility forecasts, LSTM Autoencoder sequential anomalies,
    Aspect-Based Sentiment Analysis (ABSA), Tone Shift Index (TSI), and Chief Risk Officer (CRO) arbitration
    calibrated with Bayesian Beta-Binomial conjugate updating.
    """
    clean_t = ticker.strip().upper()
    clean_t = TICKER_ALIASES.get(clean_t, clean_t)
    try:
        quant_data = await asyncio.to_thread(analyze_ticker_quant, clean_t, cutoff_date)
        news_data = await asyncio.to_thread(analyze_ticker_sentiment, clean_t, cutoff_date)
        bull_thesis = await asyncio.to_thread(run_bull_node, quant_data, news_data)
        bear_thesis = await asyncio.to_thread(run_bear_node, quant_data, news_data, bull_thesis)
        cro_verdict = await asyncio.to_thread(run_cro_node, quant_data, news_data, bull_thesis, bear_thesis)

        return {
            "ticker": clean_t,
            "current_price": quant_data.get("current_price"),
            "action": cro_verdict.get("action"),
            "verdict": cro_verdict.get("verdict"),
            "calibrated_confidence": cro_verdict.get("confidence"),
            "raw_confidence": cro_verdict.get("raw_confidence"),
            "bayesian_calibration": cro_verdict.get("bayesian_calibration"),
            "regime_gating": cro_verdict.get("regime_gating"),
            "temperature_scaling": cro_verdict.get("temperature_scaling"),
            "confidence_breakdown": cro_verdict.get("confidence_breakdown"),
            "regime": quant_data.get("hmm_regime"),
            "garch_volatility": {
                "5d_forecast_annual_pct": quant_data.get("garch_vol_forecast_pct"),
                "historical_annual_vol_pct": quant_data.get("annualized_volatility"),
                "volatility_shock_ratio": quant_data.get("volatility_shock_ratio"),
                "daily_VaR_95_pct": quant_data.get("daily_VaR_95_pct"),
            },
            "sequential_anomaly": quant_data.get("temporal_anomaly"),
            "ensemble_isolation_forest": quant_data.get("anomaly_telemetry"),
            "sentiment_and_tone_shift": {
                "aggregate_sentiment": news_data.get("aggregate_sentiment_score"),
                "aspects": news_data.get("aspect_sentiment"),
                "tone_shift_index": news_data.get("tone_shift_index"),
                "tone_shift_regime": news_data.get("tone_shift_regime"),
            },
            "risk_boundaries": {
                "recommended_allocation_pct": cro_verdict.get("recommended_allocation_pct"),
                "half_kelly_cap_pct": quant_data.get("half_kelly_allocation_pct"),
                "suggested_stop_loss_pct": cro_verdict.get("stop_loss_pct"),
                "kill_switch_reason": cro_verdict.get("kill_switch_reason"),
            },
            "theses": {
                "bull_thesis": bull_thesis,
                "bear_thesis": bear_thesis,
            },
            "verdict_summary": cro_verdict.get("verdict_summary"),
            "top_explainability_drivers": cro_verdict.get("top_drivers"),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/confidence/update")
async def update_bayesian_confidence(body: ConfidenceUpdateRequest = Body(...)):
    """
    Bayesian confidence updating endpoint:
    Increments alpha on successful risk/anomaly prediction and beta on false positives.
    Updates the Beta-Binomial conjugate posterior moments and 90% credible intervals.
    """
    calibrator = BetaBinomialConfidenceCalibrator()
    summary = calibrator.update_evidence(is_hit=body.is_hit, ticker=body.ticker)
    return {
        "status": "success",
        "ticker": body.ticker,
        "is_hit": body.is_hit,
        "posterior_summary": summary,
    }


@app.get("/api/confidence/status")
async def get_confidence_status(ticker: Optional[str] = Query(default=None)):
    """
    Returns current Bayesian Beta-Binomial posterior accuracy, variance, and 90% credible intervals.
    """
    calibrator = BetaBinomialConfidenceCalibrator()
    summary = calibrator.get_posterior_summary(ticker=ticker)
    return {
        "ticker": ticker or "GLOBAL",
        "posterior_summary": summary,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
