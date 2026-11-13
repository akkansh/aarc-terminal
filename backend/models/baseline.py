"""L2-regularized Logistic Regression baseline model."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Optional


class BaselineLogisticModel:
    """Multiclass regularized Logistic Regression benchmark."""

    def __init__(self, c_penalty: float = 1.0, random_state: int = 42, max_iter: int = 1000):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=c_penalty,
            solver="lbfgs",
            max_iter=max_iter,
            random_state=random_state,
        )
        self.feature_names: list[str] = []
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaselineLogisticModel":
        self.feature_names = list(X.columns)
        X_clean = X.fillna(0.0).values
        y_clean = y.values.astype(int)

        X_scaled = self.scaler.fit_transform(X_clean)
        self.model.fit(X_scaled, y_clean)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        X_clean = X[self.feature_names].fillna(0.0).values
        X_scaled = self.scaler.transform(X_clean)
        return self.model.predict_proba(X_scaled)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
