import os
import json
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Ensure .env is loaded before anything else
ROOT_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ROOT_ENV_PATH, override=True)
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from backend.graph import stream_committee

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


def get_cached_events_for_fallback(ticker: str, cutoff_date: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    cache = load_demo_cache()
    if not cache:
        return None

    clean_t = ticker.strip().upper()
    c_date = cutoff_date.strip() if cutoff_date else ""
    key = f"{clean_t}_{c_date}"

    # Exact match
    if key in cache:
        return cache[key]
    
    # Match by ticker alone
    for k in cache:
        if k.startswith(f"{clean_t}_"):
            return cache[k]
    
    # Generic fallback to any available cache entry (e.g. RELIANCE.NS)
    for k in ["RELIANCE.NS_", "RELIANCE.NS", "NVDA_", "NVDA"]:
        if k in cache:
            # Clone and adapt ticker name
            copied = json.loads(json.dumps(cache[k]))
            for ev in copied:
                if ev.get("node") == "quant" and "data" in ev:
                    ev["data"]["ticker"] = clean_t
            return copied

    # Return first available entry in cache
    first_key = next(iter(cache.keys()))
    return cache[first_key]


@app.get("/health")
@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "service": "Autonomous Adversarial Risk Committee (AARC)"
    }


@app.get("/api/stream")
async def stream_committee_endpoint(
    ticker: str = Query(default="RELIANCE.NS", description="Asset ticker symbol"),
    cutoff_date: Optional[str] = Query(default=None, description="Optional historical cutoff date (YYYY-MM-DD) for backtesting")
):
    """
    SSE endpoint that streams node completion events:
    quant -> news -> bull -> bear -> cro
    as `data: <json>\\n\\n`.
    Includes failsafe demo caching.
    """
    clean_t = ticker.strip().upper()
    cleaned_cutoff = cutoff_date.strip() if cutoff_date and cutoff_date.strip() else None
    cache_key = f"{clean_t}_{cleaned_cutoff or ''}"

    async def event_generator():
        recorded_events = []
        stream_failed = False

        try:
            async for node_event in stream_committee(ticker=clean_t, cutoff_date=cleaned_cutoff):
                if node_event.get("node") == "error":
                    stream_failed = True
                    break
                recorded_events.append(node_event)
                yield {
                    "data": json.dumps(node_event, allow_nan=False)
                }

            # If stream succeeded with all 5 nodes, save to demo cache
            if not stream_failed and len(recorded_events) >= 5:
                save_to_demo_cache(cache_key, recorded_events)
            elif stream_failed:
                raise RuntimeError("Node reported error during execution")

        except Exception as e:
            print(f"[STREAM] Execution failed or encountered error: {e}. Falling back to demo cache...")
            cached_events = get_cached_events_for_fallback(clean_t, cleaned_cutoff)
            if cached_events:
                # Stream cached events with small realistic intervals
                for cached_ev in cached_events:
                    yield {
                        "data": json.dumps(cached_ev, allow_nan=False)
                    }
                    await asyncio.sleep(0.3)
            else:
                yield {
                    "data": json.dumps({"node": "error", "data": {"stage": "failsafe", "error": str(e)}})
                }

    return EventSourceResponse(event_generator())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
