import os
import asyncio
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Ensure .env is loaded before anything else
ROOT_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ROOT_ENV_PATH, override=True)
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from backend.graph import stream_committee
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


@app.get("/api/instrument")
async def instrument_name(ticker: str = Query(min_length=1, max_length=32)):
    """Resolve only an exact symbol match; never assign a different company's name."""
    symbol = ticker.strip().upper()
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
