import urllib.parse
import feedparser
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
            "headlines": results,
            "aspect_sentiment": absa_result["aspect_scores"],
            "composite_aspect_score": absa_result["composite_aspect_score"],
            "tone_shift_index": absa_result["tone_shift_index"],
            "tone_shift_regime": absa_result["tone_shift_regime"],
            "aspect_relevance": absa_result["aspect_relevance"],
        }


def analyze_ticker_sentiment(ticker: str) -> Dict[str, Any]:
    """
    Scrapes top 8 headlines from Google News RSS for the ticker
    and scores each using ProsusAI/finbert.
    """
    clean_ticker = ticker.split(".")[0].strip() if "." in ticker else ticker.strip()
    encoded_ticker = urllib.parse.quote_plus(clean_ticker)
    rss_url = f"https://news.google.com/rss/search?q={encoded_ticker}+stock+finance&hl=en-IN&gl=IN&ceid=IN:en"

    feed = feedparser.parse(rss_url)
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

            if title:
                items.append({
                    "title": title,
                    "published_at": str(pub_date).strip() if pub_date else ""
                })

    # If no headlines found, return graceful fallback
    if not items:
        return {
            "ticker": ticker,
            "aggregate_sentiment_score": 0.0,
            "headlines": [],
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

    analyzer = FinBertSentimentAnalyzer.get_instance()
    scored = analyzer.score_headlines(items)

    return {
        "ticker": ticker,
        "aggregate_sentiment_score": scored["aggregate_sentiment_score"],
        "headlines": scored["headlines"],
        "aspect_sentiment": scored.get("aspect_sentiment", {}),
        "composite_aspect_score": scored.get("composite_aspect_score", 0.0),
        "tone_shift_index": scored.get("tone_shift_index", 0.0),
        "tone_shift_regime": scored.get("tone_shift_regime", "NEUTRAL"),
        "aspect_relevance": scored.get("aspect_relevance", {}),
    }
