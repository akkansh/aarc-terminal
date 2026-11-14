"""Unit and integration tests for Real-Time Asynchronous News Streaming:
- NewsStreamBroker pub/sub queue subscription & deregistration
- Sub-second FinBERT scoring & payload generation
- GET /api/news/stream SSE endpoint (initial snapshot + live event streaming)
- POST /api/news/publish real-time ingestion endpoint
- Resilient heartbeat keepalive emission
"""

import asyncio
import json
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.data.news_stream import NewsStreamBroker


@pytest.fixture
def client():
    return TestClient(app)


def test_news_stream_broker_pub_sub():
    async def _run():
        broker = NewsStreamBroker.get_instance()
        ticker = "RELIANCE.NS"

        # Subscribe client
        queue = await broker.subscribe(ticker)
        assert broker.get_subscriber_count(ticker) >= 1

        # Publish news item
        published_event = await broker.publish_news_item(
            ticker=ticker,
            title="Reliance Industries expands green hydrogen capacity with new electrolyzer facility",
            source="FINANCIAL_NEWS",
        )

        assert published_event["ticker"] == ticker
        assert "label" in published_event
        assert "score" in published_event
        assert published_event["source"] == "FINANCIAL_NEWS"

        # Queue should receive the event
        received = await asyncio.wait_for(queue.get(), timeout=2.0)
        assert received["id"] == published_event["id"]
        assert received["title"] == published_event["title"]

        # Unsubscribe client
        await broker.unsubscribe(ticker, queue)

    asyncio.run(_run())


def test_regulatory_filing_detection():
    async def _run():
        broker = NewsStreamBroker.get_instance()
        ticker = "TCS.NS"

        event = await broker.publish_news_item(
            ticker=ticker,
            title="Regulatory Filing: TCS announces Q3 board meeting for interim dividend declaration",
            source="AUTO_DETECT",
        )

        assert event["source"] == "REGULATORY_FILING"
        assert event["ticker"] == ticker
        assert event["score"] > 0.0

    asyncio.run(_run())


def test_news_publish_endpoint(client):
    payload = {
        "ticker": "ETERNAL.NS",
        "title": "Exchange Disclosure: Eternal Ltd completes acquisition of regional supply partner",
        "source": "REGULATORY_FILING",
    }
    response = client.post("/api/news/publish", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "published"
    assert data["event"]["ticker"] == "ETERNAL.NS"
    assert data["event"]["source"] == "REGULATORY_FILING"
    assert "score" in data["event"]
    assert "label" in data["event"]


def test_news_event_generator_snapshot_and_heartbeat():
    async def _run():
        broker = NewsStreamBroker.get_instance()
        ticker = "NVDA"

        # Create generator with short heartbeat timeout
        gen = broker.news_event_generator(ticker=ticker, heartbeat_interval=0.5)

        # First message should be initial snapshot
        first_msg = await anext(gen)
        assert first_msg["event"] == "snapshot"
        snapshot = json.loads(first_msg["data"])
        assert snapshot["type"] == "snapshot"
        assert snapshot["ticker"] == ticker

        # Wait for heartbeat
        second_msg = await anext(gen)
        assert second_msg["event"] == "heartbeat"
        heartbeat = json.loads(second_msg["data"])
        assert heartbeat["type"] == "heartbeat"
        assert heartbeat["status"] == "HEALTHY_STREAMING"

    asyncio.run(_run())
