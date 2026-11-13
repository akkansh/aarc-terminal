"""SHAP value computation for LightGBM model drivers."""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import shap


class TreeShapExplainer:
    """Computes TreeSHAP explanations for LightGBM predictions."""

    def __init__(self, lgbm_model: Any, feature_names: List[str]):
        self.feature_names = feature_names
        # lgbm_model is LightGBMClassifierModel
        raw_lgb = lgbm_model.model if hasattr(lgbm_model, "model") else lgbm_model
        self.explainer = shap.TreeExplainer(raw_lgb)

    def explain_instance(self, sample_df: pd.DataFrame, target_class: int = 2) -> Dict[str, Any]:
        """
        Explain a single feature vector instance.
        target_class: 0 = DOWN, 1 = FLAT, 2 = UP.
        Returns top 5 positive drivers and top 5 negative drivers.
        """
        X_vec = sample_df[self.feature_names].fillna(0.0).iloc[[-1]]
        shap_values = self.explainer.shap_values(X_vec.values)

        # For multiclass, shap_values is a list of length 3 or array of shape (N, D, 3)
        if isinstance(shap_values, list):
            class_shap = shap_values[target_class][0]
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            class_shap = shap_values[0, :, target_class]
        else:
            class_shap = np.array(shap_values)[0]

        feature_vals = X_vec.iloc[0].values
        drivers = []
        for i, name in enumerate(self.feature_names):
            val = float(feature_vals[i])
            impact = float(class_shap[i])
            drivers.append({
                "feature": name,
                "value": round(val, 4),
                "shap_impact": round(impact, 4),
            })

        # Sort by impact
        sorted_drivers = sorted(drivers, key=lambda d: d["shap_impact"], reverse=True)
        top_positive = sorted_drivers[:5]
        top_negative = sorted_drivers[-5:][::-1]  # Most negative first

        return {
            "target_class": target_class,
            "top_positive_drivers": top_positive,
            "top_negative_drivers": top_negative,
        }
