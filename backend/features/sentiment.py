"""FinBERT sentiment scoring with class probabilities, rolling windows, momentum, and text hashing."""

import hashlib
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from backend.data.cache import get_disk_cache


class FinBertEngine:
    """Thread-safe singleton for ProsusAI/finbert with text-hash caching."""

    _instance: Optional["FinBertEngine"] = None
    _init_lock = threading.Lock()

    def __init__(self, model_name: str = "ProsusAI/finbert", cache_dir: str = ".cache"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.cache = get_disk_cache(cache_dir)
        self.id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}

        # Identify class index positions
        self.pos_idx = next((i for i, lbl in self.id2label.items() if lbl == "positive"), 0)
        self.neg_idx = next((i for i, lbl in self.id2label.items() if lbl == "negative"), 1)
        self.neu_idx = next((i for i, lbl in self.id2label.items() if lbl == "neutral"), 2)

    @classmethod
    def get_instance(cls, model_name: str = "ProsusAI/finbert") -> "FinBertEngine":
        """Thread-safe singleton accessor."""
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls(model_name=model_name)
        return cls._instance

    def score_single_headline(self, text: str) -> Dict[str, Any]:
        """Score a single headline with class probabilities, using text hash cache."""
        clean_text = text.strip()
        if not clean_text:
            return {
                "title": "",
                "prob_positive": 0.0,
                "prob_negative": 0.0,
                "prob_neutral": 1.0,
                "net_score": 0.0,
                "label": "neutral",
                "confidence": 1.0,
            }

        text_hash = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
        cache_key = f"finbert_score:{text_hash}"

        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        # Run inference
        inputs = self.tokenizer(
            clean_text,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

        p_pos = float(probs[self.pos_idx])
        p_neg = float(probs[self.neg_idx])
        p_neu = float(probs[self.neu_idx])
        net = round(p_pos - p_neg, 4)

        top_idx = int(np.argmax(probs))
        top_label = self.id2label.get(top_idx, "neutral")
        top_conf = round(float(probs[top_idx]), 4)

        result = {
            "title": clean_text,
            "prob_positive": round(p_pos, 4),
            "prob_negative": round(p_neg, 4),
            "prob_neutral": round(p_neu, 4),
            "net_score": net,
            "label": top_label,
            "confidence": top_conf,
        }

        # Cache permanently (deterministic for text)
        self.cache.set(cache_key, result, ttl_seconds=None)
        return result

    def score_headlines_batch(self, headlines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score an array of headline dicts."""
        scored = []
        for h in headlines:
            title = h.get("title", "")
            s = self.score_single_headline(title)
            # Merge with original item metadata
            merged = {**h, **s}
            scored.append(merged)
        return scored


def compute_sentiment_features(
    trading_dates: pd.DatetimeIndex,
    scored_headlines: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Map scored headlines onto trading dates without look-ahead bias.
    Computes rolling 3d, 7d, 14d mean sentiment, momentum, news count, and dispersion.
    """
    df_sent = pd.DataFrame(index=trading_dates)
    df_sent["sentiment_score_raw"] = 0.0
    df_sent["prob_positive_mean"] = 0.33
    df_sent["prob_negative_mean"] = 0.33
    df_sent["prob_neutral_mean"] = 0.34
    df_sent["news_count_daily"] = 0
    df_sent["sentiment_dispersion"] = 0.0

    if not scored_headlines:
        # Default empty features
        for w in [3, 7, 14]:
            df_sent[f"sentiment_rolling_{w}d"] = 0.0
        df_sent["sentiment_momentum_3_14"] = 0.0
        return df_sent

    # Aggregate by date
    daily_groups: Dict[str, List[Dict[str, Any]]] = {}
    for h in scored_headlines:
        d_str = h.get("published_date")
        if not d_str and h.get("published_dt"):
            d_str = h["published_dt"].strftime("%Y-%m-%d")

        if d_str:
            daily_groups.setdefault(d_str, []).append(h)

    # Populate daily points
    for idx_dt in trading_dates:
        d_key = idx_dt.strftime("%Y-%m-%d")
        items = daily_groups.get(d_key, [])
        if items:
            scores = [it.get("net_score", 0.0) for it in items]
            p_pos = [it.get("prob_positive", 0.33) for it in items]
            p_neg = [it.get("prob_negative", 0.33) for it in items]
            p_neu = [it.get("prob_neutral", 0.34) for it in items]

            df_sent.loc[idx_dt, "sentiment_score_raw"] = float(np.mean(scores))
            df_sent.loc[idx_dt, "prob_positive_mean"] = float(np.mean(p_pos))
            df_sent.loc[idx_dt, "prob_negative_mean"] = float(np.mean(p_neg))
            df_sent.loc[idx_dt, "prob_neutral_mean"] = float(np.mean(p_neu))
            df_sent.loc[idx_dt, "news_count_daily"] = len(items)
            df_sent.loc[idx_dt, "sentiment_dispersion"] = float(np.std(scores)) if len(scores) > 1 else 0.0

    # Exponential decay / rolling windows over trading days
    # Fill days without news with exponential decay towards 0
    decayed_series = df_sent["sentiment_score_raw"].ewm(span=5, adjust=False).mean()

    for w in [3, 7, 14]:
        df_sent[f"sentiment_rolling_{w}d"] = decayed_series.rolling(window=w, min_periods=1).mean()

    df_sent["sentiment_momentum_3_14"] = (
        df_sent["sentiment_rolling_3d"] - df_sent["sentiment_rolling_14d"]
    )

    return df_sent
