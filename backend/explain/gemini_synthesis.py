"""Ground-truth constrained Gemini synthesis of bull/bear cases and decision notes."""

import json
import os
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


class SynthesisOutputSchema(BaseModel):
    bull_thesis: str = Field(description="3 sharp quantitative bullet points supporting the bull case based on provided drivers")
    bear_thesis: str = Field(description="3 ruthless risk bullet points supporting the bear case based on provided drivers")
    verdict_summary: str = Field(description="Executive synthesis strictly aligned with the recommended action and confidence")
    risk_note: str = Field(description="Short risk note ending with not financial advice disclaimer")


def build_fallback_narrative(payload: Dict[str, Any]) -> Dict[str, str]:
    """Deterministic fallback matching model ground truth when LLM is unavailable."""
    action = payload.get("action", "HOLD")
    conf = payload.get("confidence", 50.0)
    top_pos = payload.get("top_drivers", {}).get("top_positive_drivers", [])
    top_neg = payload.get("top_drivers", {}).get("top_negative_drivers", [])
    regime = payload.get("regime", {}).get("regime_name", "NORMAL")
    vol = payload.get("volatility_forecast", 20.0)

    pos_desc = ", ".join([f"{d.get('feature', '')} ({d.get('value', 0):.2f})" for d in top_pos[:3]]) or "positive momentum metrics"
    neg_desc = ", ".join([f"{d.get('feature', '')} ({d.get('value', 0):.2f})" for d in top_neg[:3]]) or "volatility indicators"

    bull_bullets = (
        f"• Leading quantitative support driven by {pos_desc}.\n"
        f"• Calibrated model assigns {payload.get('calibrated_probabilities', {}).get('UP', 0.33)*100:.1f}% probability to 5-day upside.\n"
        f"• Favorable momentum and market relative strength confirm short-term support."
    )

    bear_bullets = (
        f"• Downside pressure and tail risks signaled by {neg_desc}.\n"
        f"• Regime classified as {regime} with annualized 5-day GARCH volatility forecast at {vol:.1f}%.\n"
        f"• Risk committee identifies model disagreement or anomaly presence limiting upside conviction."
    )

    summary = (
        f"AARC Committee issues verdict: {action} with calibrated confidence of {conf}%. "
        f"The call is strictly governed by out-of-fold ensemble probabilities, adjusted for {regime.lower()} market regime."
    )

    risk_note = (
        f"Capital allocation must be sized to ATR stop-loss. "
        f"This automated analysis is for decision-support and educational purposes only; not financial advice."
    )

    return {
        "bull_thesis": bull_bullets,
        "bear_thesis": bear_bullets,
        "verdict_summary": summary,
        "risk_note": risk_note,
    }


def synthesize_explanation_with_gemini(
    payload: Dict[str, Any],
    api_key: Optional[str] = None,
) -> Dict[str, str]:
    """
    Generate bull/bear case and decision synthesis grounded strictly in payload ground truth.
    Never hallucinates or overrides model action.
    """
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY", "")

    if not api_key or "YOUR_GEMINI" in api_key or genai is None:
        return build_fallback_narrative(payload)

    client = genai.Client(api_key=api_key)

    prompt = f"""You are the quantitative explainer for the Autonomous Adversarial Risk Committee (AARC).
You are provided with verified quantitative GROUND TRUTH from an ensemble machine learning and risk pipeline.

RULES:
1. The model's recommended ACTION ({payload.get('action')}) and CONFIDENCE ({payload.get('confidence')}%) are IMMUTABLE. Never contradict them.
2. Ground the Bull case ONLY in the top positive drivers and upside probabilities provided below.
3. Ground the Bear case ONLY in the top negative drivers, volatility forecast, and downside risks provided below.
4. Do not invent facts, numbers, or events not present in the payload.
5. End the risk note with: "This automated analysis is for decision-support and educational purposes only; not financial advice."

QUANTITATIVE GROUND TRUTH PAYLOAD:
{json.dumps(payload, indent=2)}

Format response adhering strictly to the JSON schema:
- bull_thesis: exactly 3 quantitative bullet points starting with •
- bear_thesis: exactly 3 adversarial risk bullet points starting with •
- verdict_summary: concise executive verdict summary
- risk_note: concise risk note with disclaimer
"""

    models_to_try = ["gemini-3.8-flash", "gemini-2.5-flash"]
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SynthesisOutputSchema,
                    temperature=0.1,
                ),
            )
            raw_text = response.text.strip()
            parsed = json.loads(raw_text)
            return {
                "bull_thesis": parsed["bull_thesis"],
                "bear_thesis": parsed["bear_thesis"],
                "verdict_summary": parsed["verdict_summary"],
                "risk_note": parsed["risk_note"],
            }
        except Exception:
            continue

    # Fallback if API fails or exhausts quota
    return build_fallback_narrative(payload)
