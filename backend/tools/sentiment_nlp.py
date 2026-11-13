import urllib.parse
import feedparser
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any, List, Optional, Union
from backend.features.financial_lexicon import LoughranMcDonaldVerifier
from backend.features.aspect_sentiment import AspectBasedSentimentAnalyzer

_GLOBAL_FINBERT_INSTANCE = None


class FinBertSentimentAnalyzer:
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @classmethod
    def get_instance(cls) -> "FinBertSentimentAnalyzer":
        global _GLOBAL_FINBERT_INSTANCE
        if _GLOBAL_FINBERT_INSTANCE is None:
            _GLOBAL_FINBERT_INSTANCE = cls()
        return _GLOBAL_FINBERT_INSTANCE

    def score_headlines(self, headlines: Union[List[str], List[Dict[str, Any]]]) -> Dict[str, Any]:
        if not headlines:
            return {
                "aggregate_sentiment_score": 0.0,
                "headlines": []
            }

        titles: List[str] = []
        metadata_list: List[Dict[str, Any]] = []

        for item in headlines:
            if isinstance(item, dict):
                t = str(item.get("title", "")).strip()
                titles.append(t if t else "Market news")
                metadata_list.append(item)
            else:
                t = str(item).strip()
                titles.append(t if t else "Market news")
                metadata_list.append({"title": t, "published_at": ""})

        inputs = self.tokenizer(
            titles,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()

        id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        
        # Determine index for positive and negative
        pos_idx = None
        neg_idx = None
        for idx, lbl in id2label.items():
            if lbl == "positive":
                pos_idx = idx
            elif lbl == "negative":
                neg_idx = idx

        results: List[Dict[str, Any]] = []
        net_scores: List[float] = []
        verifier = LoughranMcDonaldVerifier(confidence_threshold=0.75)

        for i, meta in enumerate(metadata_list):
            title = meta.get("title", "")
            published_at = meta.get("published_at", "")
            row_probs = probs[i]
            top_idx = int(np.argmax(row_probs))
            top_label = id2label.get(top_idx, "neutral")
            top_score = float(row_probs[top_idx])

            if pos_idx is not None and neg_idx is not None:
                net_score = float(row_probs[pos_idx] - row_probs[neg_idx])
            else:
                net_score = top_score if top_label == "positive" else (-top_score if top_label == "negative" else 0.0)

            # FinBERT Confidence Thresholding & Heuristic Cross-Verification Gate (< 0.75)
            calibrated = verifier.verify_and_calibrate(
                headline=title,
                finbert_label=top_label,
                finbert_confidence=top_score,
                finbert_net_score=net_score,
            )
            final_net = calibrated["final_score"]
            final_label = calibrated["final_label"]
            cal_conf = calibrated["calibrated_confidence"]

            net_scores.append(final_net)
            results.append({
                "title": title,
                "label": final_label,
                "score": round(cal_conf, 3),
                "published_at": str(published_at).strip() if published_at else "",
                "net_polarity": round(final_net, 3),
                "heuristic_routing": calibrated["routing"],
                "lm_matches": calibrated["lm_matches"],
            })

        aggregate_score = float(np.mean(net_scores)) if net_scores else 0.0
        # Clamp to [-1.0, 1.0] and round
        aggregate_score = round(max(-1.0, min(1.0, aggregate_score)), 2)

        absa = AspectBasedSentimentAnalyzer()
        absa_result = absa.analyze_aspects_and_tone_shift(results)

        return {
            "aggregate_sentiment_score": aggregate_score,
            "ema_smoothed_sentiment": absa_result.get("ema_smoothed_sentiment", aggregate_score),
            "headlines": results,
            "aspect_sentiment": absa_result["aspect_scores"],
            "composite_aspect_score": absa_result["composite_aspect_score"],
            "tone_shift_index": absa_result["tone_shift_index"],
            "tone_shift_regime": absa_result["tone_shift_regime"],
            "aspect_relevance": absa_result["aspect_relevance"],
        }


_INDIAN_INDEX_SYMBOLS = {"^NSEI", "^NSEBANK", "^BSESN", "^INDIAVIX"}


def _is_indian_ticker(ticker: str) -> bool:
    """True for NSE/BSE listings and Indian benchmark indices."""
    clean = ticker.strip().upper()
    return (
        clean.endswith(".NS")
        or clean.endswith(".BO")
        or clean in _INDIAN_INDEX_SYMBOLS
    )


def _news_search_term(ticker: str) -> str:
    """Build a search term for an arbitrary ticker (strips exchange suffix/caret)."""
    clean = ticker.strip().upper()
    if clean in ("ETERNAL.NS", "ETERNAL.BO", "ETERNAL", "ZOMATO.NS", "ZOMATO.BO", "ZOMATO"):
        return "Eternal Zomato"
    if "." in clean:
        clean = clean.split(".")[0]
    return clean.lstrip("^")


def _unavailable(ticker: str, reason: str) -> Dict[str, Any]:
    """Explicitly mark news sentiment as unavailable instead of faking neutral."""
    return {
        "ticker": ticker,
        "aggregate_sentiment_score": None,
        "ema_smoothed_sentiment": None,
        "headlines": [],
        "news_available": False,
        "news_error": reason,
        "aspect_sentiment": {
            "risk_solvency": 0.0,
            "liquidity_cashflow": 0.0,
            "margin_health": 0.0,
        },
        "composite_aspect_score": 0.0,
        "tone_shift_index": 0.0,
        "tone_shift_regime": "NEUTRAL",
        "aspect_relevance": {
            "risk_solvency": 0,
            "liquidity_cashflow": 0,
            "margin_health": 0,
        },
    }


def _parse_published_at(value: str) -> Optional[pd.Timestamp]:
    if not value:
        return None
    try:
        parsed = pd.to_datetime(value)
    except Exception:
        return None
    if parsed is None or pd.isna(parsed):
        return None
    if getattr(parsed, "tzinfo", None) is not None:
        parsed = parsed.tz_localize(None)
    return parsed


def analyze_ticker_sentiment(
    ticker: str,
    cutoff_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Scrapes top 8 headlines from Google News RSS for the requested ticker
    and scores each using ProsusAI/finbert.

    Uses the locale appropriate to the ticker. When cutoff_date is supplied,
    news published after the cutoff and undated headlines are discarded so a
    historical request cannot use unverifiable or future news. If no usable
    headlines or the analyzer fails, sentiment is reported as unavailable
    rather than as a fabricated neutral score.
    """
    clean_ticker = _news_search_term(ticker)
    encoded_ticker = urllib.parse.quote_plus(clean_ticker)
    if _is_indian_ticker(ticker):
        locale = "hl=en-IN&gl=IN&ceid=IN:en"
    else:
        locale = "hl=en-US&gl=US&ceid=US:en"
    rss_url = f"https://news.google.com/rss/search?q={encoded_ticker}+stock+finance&{locale}"

    try:
        feed = feedparser.parse(rss_url)
    except Exception as e:
        return _unavailable(ticker, f"RSS fetch failed: {e}")

    cutoff_dt = None
    if cutoff_date:
        try:
            cutoff_dt = pd.to_datetime(cutoff_date).tz_localize(None)
        except Exception:
            cutoff_dt = None

    items: List[Dict[str, str]] = []

    if hasattr(feed, "entries") and feed.entries:
        for entry in feed.entries[:8]:
            title = getattr(entry, "title", "")
            if not title and isinstance(entry, dict):
                title = entry.get("title", "")
            title = str(title).strip() if title else ""

            pub_date = (
                getattr(entry, "published", "")
                or getattr(entry, "pubDate", "")
                or getattr(entry, "updated", "")
            )
            if not pub_date and isinstance(entry, dict):
                pub_date = (
                    entry.get("published", "")
                    or entry.get("pubDate", "")
                    or entry.get("updated", "")
                )
            pub_dt = _parse_published_at(str(pub_date).strip() if pub_date else "")

            if cutoff_dt is not None:
                # Historical mode: discard future news and undated news, since
                # neither can be verified to predate the cutoff.
                if pub_dt is None or pub_dt.date() > cutoff_dt.date():
                    continue

            if title:
                items.append({
                    "title": title,
                    "published_at": str(pub_date).strip() if pub_date else ""
                })

    if not items:
        reason = (
            "No dated headlines available on or before the cutoff"
            if cutoff_dt is not None
            else "No headlines returned by the news provider"
        )
        return _unavailable(ticker, reason)

    try:
        analyzer = FinBertSentimentAnalyzer.get_instance()
        scored = analyzer.score_headlines(items)
    except Exception as e:
        return _unavailable(ticker, f"Sentiment analyzer failed: {e}")

    return {
        "ticker": ticker,
        "aggregate_sentiment_score": scored["aggregate_sentiment_score"],
        "ema_smoothed_sentiment": scored.get("ema_smoothed_sentiment", scored["aggregate_sentiment_score"]),
        "headlines": scored["headlines"],
        "news_available": True,
        "aspect_sentiment": scored.get("aspect_sentiment", {}),
        "composite_aspect_score": scored.get("composite_aspect_score", 0.0),
        "tone_shift_index": scored.get("tone_shift_index", 0.0),
        "tone_shift_regime": scored.get("tone_shift_regime", "NEUTRAL"),
        "aspect_relevance": scored.get("aspect_relevance", {}),
    }
