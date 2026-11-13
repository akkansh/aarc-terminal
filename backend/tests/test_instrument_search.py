import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from backend import main


def test_search_by_name_filters_and_deduplicates_equities(monkeypatch):
    calls = []

    def search(query, **kwargs):
        calls.append((query, kwargs))
        return SimpleNamespace(quotes=[
            {"symbol": "^NSEI", "quoteType": "INDEX", "shortname": "Nifty 50"},
            {"symbol": "AVGO", "quoteType": "EQUITY", "longname": "Broadcom Inc.", "exchDisp": "NASDAQ"},
            {"symbol": "AVGO", "quoteType": "EQUITY", "shortname": "Broadcom"},
            {"symbol": "BRCM.L", "quoteType": "EQUITY", "shortname": "Broadcom London", "exchange": "LSE"},
            {"symbol": "MISSING", "quoteType": "EQUITY"},
        ])

    monkeypatch.setattr(main.yf, "Search", search)
    result = asyncio.run(main.search_instruments("Broadcom"))
    assert calls[0][0] == "Broadcom"
    assert result == {"results": [
        {"ticker": "AVGO", "name": "Broadcom Inc.", "exchange": "NASDAQ"},
        {"ticker": "BRCM.L", "name": "Broadcom London", "exchange": "LSE"},
    ]}


def test_search_reports_provider_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("offline")
    monkeypatch.setattr(main.yf, "Search", fail)
    with pytest.raises(HTTPException) as error:
        asyncio.run(main.search_instruments("Apple"))
    assert error.value.status_code == 503
