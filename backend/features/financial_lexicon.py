"""Loughran-McDonald Financial Sentiment Lexicon and Heuristic Cross-Verification Gate."""

import re
from typing import Dict, Any, List, Tuple, Optional


# Curated domain-specific financial sentiment lexicons (Loughran & McDonald, 2011)
LM_POSITIVE_TERMS = {
    "beat", "beats", "beating", "exceed", "exceeded", "exceeds", "exceeding",
    "outperform", "outperformed", "outperforming", "outperformance",
    "dividend", "dividends", "accretive", "expansion", "expanding", "expanded",
    "record", "surpass", "surpassed", "surpasses", "surpassing",
    "bullish", "breakthrough", "breakthroughs", "upgrade", "upgraded", "upgrades",
    "soar", "soared", "soaring", "milestone", "milestones", "all-time high",
    "synergies", "synergy", "robust", "strong", "buyback", "buybacks",
    "windfall", "upside", "lucrative", "profitability", "profitable",
    "gain", "gains", "gained", "rally", "rallied", "rallying", "rebound", "rebounds",
}

LM_NEGATIVE_TERMS = {
    "impairment", "impairments", "headwind", "headwinds", "litigation", "litigations",
    "downgrade", "downgraded", "downgrades", "default", "defaulted", "defaults",
    "restructure", "restructuring", "breach", "breached", "subpoena", "subpoenas",
    "loss", "losses", "deficit", "deficits", "curtailment", "shortfall", "shortfalls",
    "insolvency", "insolvent", "plummet", "plummeted", "plummeting", "penalized",
    "slump", "slumped", "slumping", "investigation", "investigations", "distress",
    "distressed", "write-down", "writedown", "writedowns", "unpaid", "plunge",
    "plunged", "plunging", "layoff", "layoffs", "scandal", "scandals", "probe",
    "probes", "fraud", "fraudulent", "tumble", "tumbled", "tumbling", "weakness",
    "weaknesses", "warning", "warnings", "warns", "warned", "delinquent", "penalties",
}

LM_UNCERTAINTY_TERMS = {
    "speculation", "speculative", "pending", "scrutiny", "renegotiate",
    "renegotiation", "ambiguity", "contingent", "contingency", "fluctuate",
    "fluctuations", "volatile", "volatility", "uncertain", "uncertainty",
    "interim", "provisional", "cautious", "risk", "risks", "unpredictable",
}


class LoughranMcDonaldVerifier:
    """
    Heuristic rule-based cross-verification engine for quant and corporate jargon.
    Cross-checks ambiguous FinBERT sentiment outputs against Loughran-McDonald lexicons.
    """

    def __init__(self, confidence_threshold: float = 0.75):
        self.confidence_threshold = confidence_threshold

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Tokenize text and evaluate Loughran-McDonald sentiment polarity."""
        clean_text = text.lower()
        # Extract word tokens and bigrams
        tokens = set(re.findall(r"\b[a-z\-]{3,}\b", clean_text))

        pos_matches = [w for w in tokens if w in LM_POSITIVE_TERMS]
        neg_matches = [w for w in tokens if w in LM_NEGATIVE_TERMS]
        unc_matches = [w for w in tokens if w in LM_UNCERTAINTY_TERMS]

        # Multi-word checks
        if "all-time high" in clean_text or "all time high" in clean_text:
            pos_matches.append("all-time high")
        if "share buyback" in clean_text or "stock buyback" in clean_text:
            pos_matches.append("share buyback")
        if "profit warning" in clean_text or "revenue warning" in clean_text:
            neg_matches.append("profit warning")

        n_pos = len(pos_matches)
        n_neg = len(neg_matches)
        n_unc = len(unc_matches)

        total_informative = n_pos + n_neg + (0.5 * n_unc)
        if total_informative == 0:
            lm_polarity = 0.0
            lm_confidence = 0.50
        else:
            raw_net = (n_pos - n_neg) / total_informative
            lm_polarity = round(max(-1.0, min(1.0, raw_net)), 3)
            # More matching domain terms increase lexicon confidence
            lm_confidence = round(min(0.95, 0.50 + (0.15 * (n_pos + n_neg))), 2)

        return {
            "lm_polarity": lm_polarity,
            "lm_confidence": lm_confidence,
            "positive_matches": pos_matches,
            "negative_matches": neg_matches,
            "uncertainty_matches": unc_matches,
        }

    def verify_and_calibrate(
        self,
        headline: str,
        finbert_label: str,
        finbert_confidence: float,
        finbert_net_score: float,
    ) -> Dict[str, Any]:
        """
        Confidence-Weighting Gate:
        - If raw FinBERT confidence >= threshold (0.75): accept model judgment directly.
        - If raw FinBERT confidence < threshold (0.75): route through Loughran-McDonald
          keyword check, reconcile quant jargon, and blend sentiment scores.
        """
        # Analyze financial dictionary
        lm_res = self.analyze_text(headline)
        lm_polarity = lm_res["lm_polarity"]
        pos_m = lm_res["positive_matches"]
        neg_m = lm_res["negative_matches"]

        # High-confidence FinBERT pass
        if finbert_confidence >= self.confidence_threshold:
            # Check if there is a severe direct contradiction with explicit negative red flags
            if finbert_label == "positive" and len(neg_m) >= 2:
                # E.g., headline discusses an investigation or lawsuit while FinBERT saw positive wording
                blended_score = round(0.4 * finbert_net_score + 0.6 * lm_polarity, 3)
                calibrated_label = "negative" if blended_score < -0.10 else "neutral"
                return {
                    "final_score": blended_score,
                    "final_label": calibrated_label,
                    "calibrated_confidence": round((finbert_confidence + lm_res["lm_confidence"]) / 2.0, 3),
                    "routing": "OVERRIDE_RED_FLAG",
                    "reason": f"FinBERT high-confidence overridden due to critical red flags: {neg_m}",
                    "lm_matches": {"pos": pos_m, "neg": neg_m},
                }

            return {
                "final_score": finbert_net_score,
                "final_label": finbert_label,
                "calibrated_confidence": finbert_confidence,
                "routing": "FINBERT_DIRECT",
                "reason": f"FinBERT confidence ({finbert_confidence}) >= {self.confidence_threshold}",
                "lm_matches": {"pos": pos_m, "neg": neg_m},
            }

        # Sub-threshold: confidence < 0.75 -> Heuristic Cross-Verification Gate
        # Case A: LM found strong directional signal
        if len(pos_m) > 0 or len(neg_m) > 0:
            # Confidence weighting blend
            w_fb = max(0.20, finbert_confidence)
            w_lm = lm_res["lm_confidence"]
            total_w = w_fb + w_lm

            blended_score = round((w_fb * finbert_net_score + w_lm * lm_polarity) / total_w, 3)
            if blended_score > 0.15:
                calibrated_label = "positive"
            elif blended_score < -0.15:
                calibrated_label = "negative"
            else:
                calibrated_label = "neutral"

            # Boost confidence since two independent engines agreed or corrected
            boosted_conf = round(min(0.92, max(finbert_confidence, lm_res["lm_confidence"]) + 0.10), 3)

            return {
                "final_score": blended_score,
                "final_label": calibrated_label,
                "calibrated_confidence": boosted_conf,
                "routing": "HEURISTIC_CROSS_VERIFIED",
                "reason": f"Sub-threshold FinBERT ({finbert_confidence} < {self.confidence_threshold}) cross-verified with LM lexicon ({pos_m or neg_m})",
                "lm_matches": {"pos": pos_m, "neg": neg_m},
            }

        # Case B: No strong LM keywords found either -> Keep neutral / dampened
        dampened_score = round(finbert_net_score * 0.75, 3)
        return {
            "final_score": dampened_score,
            "final_label": finbert_label if abs(dampened_score) > 0.20 else "neutral",
            "calibrated_confidence": round(finbert_confidence * 0.90, 3),
            "routing": "DAMPENED_AMBIGUOUS",
            "reason": f"Sub-threshold FinBERT ({finbert_confidence}) with no LM keyword confirmation; score dampened towards zero",
            "lm_matches": {"pos": [], "neg": []},
        }
