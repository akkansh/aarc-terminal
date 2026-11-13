"""Explainability and LLM narrative package."""

from .shap_explainer import TreeShapExplainer
from .gemini_synthesis import synthesize_explanation_with_gemini, build_fallback_narrative

__all__ = [
    "TreeShapExplainer",
    "synthesize_explanation_with_gemini",
    "build_fallback_narrative",
]
