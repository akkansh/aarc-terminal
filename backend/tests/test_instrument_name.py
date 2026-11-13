import asyncio
from types import SimpleNamespace

from backend import main


def test_instrument_name_requires_exact_symbol(monkeypatch):
    monkeypatch.setattr(main.yf, "Search", lambda *args, **kwargs: SimpleNamespace(quotes=[
        {"symbol": "OTHER", "longname": "Wrong company"},
        {"symbol": "AVGO", "longname": "Broadcom Inc."},
    ]))
    assert asyncio.run(main.instrument_name("avgo")) == {"ticker": "AVGO", "name": "Broadcom Inc."}
    assert asyncio.run(main.instrument_name("UNKNOWN")) == {"ticker": "UNKNOWN", "name": None}


def test_instrument_name_handles_provider_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("provider offline")
    monkeypatch.setattr(main.yf, "Search", fail)
    assert asyncio.run(main.instrument_name("ABC")) == {"ticker": "ABC", "name": None}
