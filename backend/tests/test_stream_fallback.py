"""Regression tests ensuring failed streams are explicit and never replay cache."""

import json

from fastapi.testclient import TestClient

import backend.main as main


def _sse_events(body: str):
    events = []
    for line in body.splitlines():
        if line.startswith("data:"):
            events.append(json.loads(line[len("data:"):].strip()))
    return events


def _client():
    return TestClient(main.app)


def test_error_event_is_forwarded_without_cache_replay(monkeypatch):
    async def fake_stream(ticker, cutoff_date=None):
        yield {"node": "quant", "data": {"ticker": ticker}}
        yield {"node": "error", "data": {"stage": "news", "error": "provider down"}}

    monkeypatch.setattr(main, "stream_committee", fake_stream)
    monkeypatch.setattr(main, "save_to_demo_cache", lambda key, events: None)

    response = _client().get("/api/stream", params={"ticker": "AAPL"})
    events = _sse_events(response.text)

    assert response.status_code == 200
    assert [event["node"] for event in events] == ["quant", "error"]
    assert events[-1]["data"]["stage"] == "news"
    assert "RELIANCE" not in response.text


def test_partial_stream_emits_error_instead_of_replaying(monkeypatch):
    async def partial_stream(ticker, cutoff_date=None):
        yield {"node": "quant", "data": {"ticker": ticker}}
        yield {"node": "news", "data": {"ticker": ticker}}

    saved = []
    monkeypatch.setattr(main, "stream_committee", partial_stream)
    monkeypatch.setattr(
        main, "save_to_demo_cache", lambda key, events: saved.append((key, events))
    )

    response = _client().get("/api/stream", params={"ticker": "AAPL"})
    events = _sse_events(response.text)

    assert [event["node"] for event in events] == ["quant", "news", "error"]
    assert events[-1]["data"]["stage"] == "stream"
    assert "RELIANCE" not in response.text
    assert saved == []


def test_stream_exception_emits_error_without_cache_replay(monkeypatch):
    async def exploding_stream(ticker, cutoff_date=None):
        yield {"node": "quant", "data": {"ticker": ticker}}
        raise RuntimeError("upstream failed")

    monkeypatch.setattr(main, "stream_committee", exploding_stream)
    monkeypatch.setattr(main, "save_to_demo_cache", lambda key, events: None)

    response = _client().get("/api/stream", params={"ticker": "AAPL"})
    events = _sse_events(response.text)

    assert [event["node"] for event in events] == ["quant", "error"]
    assert events[-1]["data"]["error"] == "upstream failed"
    assert "RELIANCE" not in response.text


def test_complete_stream_is_cached(monkeypatch):
    async def complete_stream(ticker, cutoff_date=None):
        for node in ["quant", "news", "bull", "bear", "cro"]:
            yield {"node": node, "data": {"ticker": ticker}}

    saved = []
    monkeypatch.setattr(main, "stream_committee", complete_stream)
    monkeypatch.setattr(
        main, "save_to_demo_cache", lambda key, events: saved.append((key, events))
    )

    response = _client().get(
        "/api/stream", params={"ticker": "AAPL", "cutoff_date": "2024-06-03"}
    )
    events = _sse_events(response.text)

    assert [event["node"] for event in events] == ["quant", "news", "bull", "bear", "cro"]
    assert len(saved) == 1
    assert saved[0][0] == "AAPL_2024-06-03"
