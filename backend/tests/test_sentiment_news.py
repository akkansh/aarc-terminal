"""Regression tests for per-ticker news selection, cutoff handling, and unavailability."""

from types import SimpleNamespace

import pytest

from backend.tools import sentiment_nlp


def _entry(title: str, published: str = ""):
    return SimpleNamespace(title=title, published=published, link="", source=None)


def _feed(*entries):
    return SimpleNamespace(entries=list(entries))


class _FakeAnalyzer:
    def __init__(self):
        self.received = None

    def score_headlines(self, items):
        self.received = items
        return {
            "aggregate_sentiment_score": 0.42,
            "headlines": [
                {
                    "title": item["title"],
                    "label": "positive",
                    "score": 0.9,
                    "published_at": item["published_at"],
                }
                for item in items
            ],
        }


@pytest.fixture
def fake_analyzer(monkeypatch):
    analyzer = _FakeAnalyzer()
    monkeypatch.setattr(
        sentiment_nlp.FinBertSentimentAnalyzer,
        "get_instance",
        lambda: analyzer,
    )
    return analyzer


def test_sentiment_uses_requested_ticker_and_locale(monkeypatch, fake_analyzer):
    urls = []

    def fake_parse(url):
        urls.append(url)
        return _feed(_entry("Headline", "Mon, 06 May 2024 07:00:00 GMT"))

    monkeypatch.setattr(sentiment_nlp.feedparser, "parse", fake_parse)

    sentiment_nlp.analyze_ticker_sentiment("AAPL")
    sentiment_nlp.analyze_ticker_sentiment("RELIANCE.NS")
    sentiment_nlp.analyze_ticker_sentiment("^NSEI")

    assert "AAPL" in urls[0]
    assert "hl=en-US" in urls[0]
    assert "RELIANCE" in urls[1]
    assert "hl=en-IN" in urls[1]
    assert "NSEI" in urls[2]
    assert "hl=en-IN" in urls[2]


def test_sentiment_cutoff_discards_future_and_undated_news(monkeypatch, fake_analyzer):
    def fake_parse(url):
        return _feed(
            _entry("Before cutoff", "Mon, 06 May 2024 07:00:00 GMT"),
            _entry("After cutoff", "Mon, 10 Jun 2024 07:00:00 GMT"),
            _entry("Undated"),
        )

    monkeypatch.setattr(sentiment_nlp.feedparser, "parse", fake_parse)

    result = sentiment_nlp.analyze_ticker_sentiment("RELIANCE.NS", cutoff_date="2024-06-03")

    assert [item["title"] for item in fake_analyzer.received] == ["Before cutoff"]
    assert result["news_available"] is True
    assert result["aggregate_sentiment_score"] == 0.42
    assert [headline["title"] for headline in result["headlines"]] == ["Before cutoff"]


def test_sentiment_unavailable_when_all_news_is_after_cutoff(monkeypatch, fake_analyzer):
    def fake_parse(url):
        return _feed(
            _entry("After cutoff", "Mon, 10 Jun 2024 07:00:00 GMT"),
            _entry("Undated"),
        )

    monkeypatch.setattr(sentiment_nlp.feedparser, "parse", fake_parse)

    result = sentiment_nlp.analyze_ticker_sentiment("RELIANCE.NS", cutoff_date="2024-06-03")

    assert fake_analyzer.received is None
    assert result["news_available"] is False
    assert result["aggregate_sentiment_score"] is None
    assert result["headlines"] == []
    assert "cutoff" in result["news_error"].lower()


def test_sentiment_unavailable_when_provider_returns_no_headlines(monkeypatch):
    monkeypatch.setattr(sentiment_nlp.feedparser, "parse", lambda url: _feed())
    monkeypatch.setattr(
        sentiment_nlp.FinBertSentimentAnalyzer,
        "get_instance",
        lambda: pytest.fail("analyzer should not be called without headlines"),
    )

    result = sentiment_nlp.analyze_ticker_sentiment("NVDA")

    assert result["news_available"] is False
    assert result["aggregate_sentiment_score"] is None
    assert result["headlines"] == []


def test_sentiment_unavailable_when_analyzer_fails(monkeypatch):
    def fake_parse(url):
        return _feed(_entry("Headline", "Mon, 06 May 2024 07:00:00 GMT"))

    monkeypatch.setattr(sentiment_nlp.feedparser, "parse", fake_parse)

    class BrokenAnalyzer:
        def score_headlines(self, items):
            raise RuntimeError("model unavailable")

    monkeypatch.setattr(
        sentiment_nlp.FinBertSentimentAnalyzer,
        "get_instance",
        lambda: BrokenAnalyzer(),
    )

    result = sentiment_nlp.analyze_ticker_sentiment("NVDA")

    assert result["news_available"] is False
    assert result["aggregate_sentiment_score"] is None
    assert "model unavailable" in result["news_error"]


def test_sentiment_unavailable_when_rss_fetch_fails(monkeypatch):
    def fake_parse(url):
        raise RuntimeError("network down")

    monkeypatch.setattr(sentiment_nlp.feedparser, "parse", fake_parse)

    result = sentiment_nlp.analyze_ticker_sentiment("NVDA")

    assert result["news_available"] is False
    assert result["aggregate_sentiment_score"] is None
    assert "network down" in result["news_error"]
