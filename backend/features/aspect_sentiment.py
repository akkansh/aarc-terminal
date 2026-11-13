"""Aspect-Based Sentiment Analysis (ABSA) and Tone Shift Index (TSI) for Quant Pipelines."""

import re
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd


ASPECT_LEXICONS = {
    "risk_solvency": {
        "keywords": [
            "debt", "covenant", "leverage", "default", "insolvency", "subpoena",
            "probe", "fraud", "litigation", "lawsuit", "penalty", "compliance",
            "downgrade", "credit risk", "rating watch", "restructure", "restructuring",
            "delinquency", "distress", "investigation", "allegation", "sanction",
        ],
        "weight": 0.40,
    },
    "liquidity_cashflow": {
        "keywords": [
            "cash flow", "fcf", "free cash flow", "liquidity", "burn rate",
            "working capital", "cash crunch", "credit line", "revolving credit",
            "debt service", "refinancing", "cash reserves", "fundraise", "capital call",
            "shortfall", "cash outflow", "outflows", "liquidity buffer",
        ],
        "weight": 0.35,
    },
    "margin_health": {
        "keywords": [
            "margin", "margins", "ebitda", "gross margin", "operating margin",
            "pricing power", "inflation", "input costs", "raw material", "price hike",
            "margin compression", "margin expansion", "cost pressure", "headwind",
            "profitability", "operating leverage", "margin squeeze",
        ],
        "weight": 0.25,
    },
}

POLARITY_LEXICON = {
    "positive": [
        "beat", "expansion", "record", "surpass", "bullish", "accretive", "dividend",
        "upgrade", "boost", "strong", "growth", "soar", "gain", "rebound", "resilient",
        "outperform", "healthy", "ample", "expand", "widening", "improving",
    ],
    "negative": [
        "impairment", "loss", "slump", "plummet", "fall", "warning", "distress",
        "plunge", "tumble", "weakness", "deteriorate", "squeezed", "curtail", "headwind",
        "drop", "deficit", "miss", "pessimistic", "eroded", "contracting", "narrowing",
        "default", "insolvency", "probe", "lawsuit", "penalty", "fraud", "risk", "crunch",
        "shortfall", "squeeze", "covenant",
    ],
}


class AspectBasedSentimentAnalyzer:
    """
    Extracts aspect-targeted sentiment across:
      1. Risk & Solvency
      2. Liquidity & Cash Flow
      3. Margin Health & Pricing Power
      Computes institutional Tone Shift Index (TSI) over temporal publication windows.
    """

    def __init__(self):
        self.aspects = list(ASPECT_LEXICONS.keys())

    def _score_text_for_aspect(self, text: str, aspect_key: str, base_score: float = 0.0) -> Tuple[float, int]:
        clean = text.lower()
        aspect_terms = ASPECT_LEXICONS[aspect_key]["keywords"]

        # Check if text mentions this aspect
        relevance_count = sum(1 for term in aspect_terms if term in clean)
        if relevance_count == 0:
            return 0.0, 0

        pos_count = sum(1 for p in POLARITY_LEXICON["positive"] if p in clean)
        neg_count = sum(1 for n in POLARITY_LEXICON["negative"] if n in clean)

        total_sentiment = pos_count + neg_count
        if total_sentiment == 0:
            # Fall back to FinBERT base score when specific polarity words are absent
            return float(np.clip(base_score, -1.0, 1.0)), relevance_count

        raw_polarity = (pos_count - neg_count) / total_sentiment
        # Blend lexicon polarity with FinBERT model score
        if abs(base_score) > 0.01:
            blended = (0.60 * raw_polarity) + (0.40 * base_score)
        else:
            blended = raw_polarity

        return float(np.clip(blended, -1.0, 1.0)), relevance_count

    def analyze_aspects_and_tone_shift(
        self,
        scored_headlines: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Ingests headlines with text and publication timestamps.
        Computes aspect scores and Tone Shift Index (TSI).
        """
        if not scored_headlines:
            return {
                "aspect_scores": {
                    "risk_solvency": 0.0,
                    "liquidity_cashflow": 0.0,
                    "margin_health": 0.0,
                },
                "composite_aspect_score": 0.0,
                "tone_shift_index": 0.0,
                "tone_shift_regime": "NEUTRAL",
                "aspect_breakdown": [],
            }

        aspect_accum = {a: {"scores": [], "weights": []} for a in self.aspects}
        temporal_scores = []  # list of (timestamp_rank, net_score)

        for i, item in enumerate(scored_headlines):
            title = item.get("title", "")
            base_score = float(item.get("net_polarity", item.get("score", 0.0)))
            if item.get("label") == "negative" and base_score > 0:
                base_score = -base_score

            temporal_scores.append(base_score)

            # Evaluate each aspect
            for a in self.aspects:
                pol, rel = self._score_text_for_aspect(title, a, base_score=base_score)
                if rel > 0:
                    aspect_accum[a]["scores"].append(pol)
                    aspect_accum[a]["weights"].append(rel)

        # Compute weighted average score per aspect
        aspect_results = {}
        for a in self.aspects:
            scores = aspect_accum[a]["scores"]
            weights = aspect_accum[a]["weights"]
            if scores:
                aspect_results[a] = round(float(np.average(scores, weights=weights)), 3)
            else:
                aspect_results[a] = 0.0

        # Composite aspect score
        composite_score = (
            ASPECT_LEXICONS["risk_solvency"]["weight"] * aspect_results["risk_solvency"]
            + ASPECT_LEXICONS["liquidity_cashflow"]["weight"] * aspect_results["liquidity_cashflow"]
            + ASPECT_LEXICONS["margin_health"]["weight"] * aspect_results["margin_health"]
        )

        # Exponential Moving Average (EMA) smoothing of sequential sentiment outputs (alpha=0.35)
        ema_scores = []
        alpha_ema = 0.35
        curr_ema = None
        # Forward pass: oldest to newest
        for s in reversed(temporal_scores):
            if curr_ema is None:
                curr_ema = s
            else:
                curr_ema = (alpha_ema * s) + ((1.0 - alpha_ema) * curr_ema)
            ema_scores.append(curr_ema)
        ema_smoothed_polarity = round(float(ema_scores[-1]), 3) if ema_scores else 0.0

        # Tone Shift Index (TSI): Recent 3 items vs. Remaining Baseline items
        # Headlines are typically ordered chronologically (newest first)
        n = len(temporal_scores)
        if n >= 4:
            recent = temporal_scores[:3]
            baseline = temporal_scores[3:]

            mean_recent = float(np.mean(recent))
            mean_baseline = float(np.mean(baseline))
            std_baseline = float(np.std(baseline)) if len(baseline) > 1 else 0.20

            # Z-score of velocity shift
            tsi = (mean_recent - mean_baseline) / (std_baseline + 0.05)
            tsi = float(np.clip(tsi, -3.5, 3.5))
        else:
            tsi = 0.0

        # Tone Shift Regime classification
        if tsi <= -1.20:
            regime = "SHARP_DETERIORATION"
        elif tsi >= 1.20:
            regime = "RAPID_IMPROVEMENT"
        else:
            regime = "STABLE"

        return {
            "aspect_scores": aspect_results,
            "composite_aspect_score": round(composite_score, 3),
            "ema_smoothed_sentiment": ema_smoothed_polarity,
            "sentiment_ema_trajectory": [round(float(x), 3) for x in ema_scores],
            "tone_shift_index": round(tsi, 2),
            "tone_shift_regime": regime,
            "aspect_relevance": {
                a: len(aspect_accum[a]["scores"]) for a in self.aspects
            },
        }
