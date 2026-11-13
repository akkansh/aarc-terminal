"""LightGBM multiclass classifier with Optuna walk-forward hyperparameter optimization."""

import logging
from typing import Any, Dict, List, Optional, Tuple
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss

# Suppress Optuna chatty logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


class LightGBMClassifierModel:
    """Primary multiclass tree-based model for 5-day directional return prediction."""

    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42,
    ):
        self.params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": random_state,
            "n_jobs": 1,
            "verbose": -1,
        }
        self.model: Optional[lgb.LGBMClassifier] = None
        self.feature_names: List[str] = []
        self.is_fitted = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "LightGBMClassifierModel":
        self.feature_names = list(X_train.columns)
        self.model = lgb.LGBMClassifier(**self.params)

        eval_set = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            eval_set = [(X_val[self.feature_names].values, y_val.values.astype(int))]

        callbacks = [lgb.early_stopping(stopping_rounds=25, verbose=False)] if eval_set else None

        self.model.fit(
            X_train.values,
            y_train.values.astype(int),
            eval_set=eval_set,
            callbacks=callbacks,
        )
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model is not fitted.")
        return self.model.predict_proba(X[self.feature_names].values)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def get_feature_importances(self) -> pd.Series:
        if not self.is_fitted or self.model is None:
            return pd.Series()
        return pd.Series(self.model.feature_importances_, index=self.feature_names).sort_values(ascending=False)


def tune_lightgbm_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    n_trials: int = 25,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Tune LightGBM hyperparameters with Optuna using strictly walk-forward cross-validation splits.
    Optimizes out-of-fold multi-class log loss.
    """
    if len(splits) == 0:
        return {}

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=25),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 40),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": random_state,
        }

        fold_losses = []
        for train_idx, val_idx in splits:
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

            clf = LightGBMClassifierModel(**params)
            clf.fit(X_tr, y_tr, X_va, y_va)
            preds = clf.predict_proba(X_va)
            # Evaluate log loss
            loss = log_loss(y_va.values.astype(int), preds, labels=[0, 1, 2])
            fold_losses.append(loss)

        return float(np.mean(fold_losses))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    return study.best_params
