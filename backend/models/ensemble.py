"""Stacking Ensemble Meta-Learner with walk-forward out-of-fold blending and versioned persistence."""

import datetime
import json
import os
import pickle
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, f1_score

from .baseline import BaselineLogisticModel
from .tree import LightGBMClassifierModel
from .sequence import SequenceGRUModel


def _get_git_hash() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


class StackingEnsemble:
    """
    Ensemble meta-learner blending out-of-fold probability predictions
    from Logistic Baseline, LightGBM, and optional Sequence GRU.
    """

    def __init__(
        self,
        use_sequence_model: bool = False,
        random_state: int = 42,
        artifacts_dir: str = "backend/artifacts",
    ):
        self.use_sequence_model = use_sequence_model
        self.random_state = random_state
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.baseline_model = BaselineLogisticModel(random_state=random_state)
        self.lgbm_model = LightGBMClassifierModel(random_state=random_state)
        self.sequence_model = SequenceGRUModel(seed=random_state) if use_sequence_model else None

        self.weights = np.array([0.5, 0.5])
        self.feature_names: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.is_fitted = False

    def _build_meta_features(
        self,
        p_base: np.ndarray,
        p_lgbm: np.ndarray,
        p_seq: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self.use_sequence_model and p_seq is not None:
            return np.hstack([p_base, p_lgbm, p_seq])
        return np.hstack([p_base, p_lgbm])

    def fit_walk_forward(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        splits: List[Tuple[np.ndarray, np.ndarray]],
        best_lgbm_params: Optional[Dict[str, Any]] = None,
    ) -> "StackingEnsemble":
        """
        Fit ensemble via purged/embargoed walk-forward cross validation.
        Collects out-of-fold base model predictions, then fits meta-learner.
        """
        self.feature_names = list(X.columns)
        n_samples = len(X)
        y_int = y.values.astype(int)

        oof_base = np.zeros((n_samples, 3))
        oof_lgbm = np.zeros((n_samples, 3))
        oof_seq = np.zeros((n_samples, 3)) if self.use_sequence_model else None
        eval_indices: List[int] = []

        if best_lgbm_params:
            self.lgbm_model = LightGBMClassifierModel(**best_lgbm_params)

        for train_idx, val_idx in splits:
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

            # Fit base models on fold
            b_clf = BaselineLogisticModel(random_state=self.random_state).fit(X_tr, y_tr)
            l_clf = LightGBMClassifierModel(random_state=self.random_state).fit(X_tr, y_tr, X_va, y_va)

            oof_base[val_idx] = b_clf.predict_proba(X_va)
            oof_lgbm[val_idx] = l_clf.predict_proba(X_va)

            if self.use_sequence_model and oof_seq is not None:
                s_clf = SequenceGRUModel(seed=self.random_state).fit(X_tr, y_tr)
                oof_seq[val_idx] = s_clf.predict_proba(X_va)

            eval_indices.extend(val_idx)

        eval_indices = sorted(list(set(eval_indices)))
        if not eval_indices:
            # Fallback: simple fit on full dataset
            self.baseline_model.fit(X, y)
            self.lgbm_model.fit(X, y)
            if self.sequence_model:
                self.sequence_model.fit(X, y)
            self.is_fitted = True
            return self

        # Optimize probability blending weights on out-of-fold predictions
        from scipy.optimize import minimize
        meta_y = y_int[eval_indices]

        def blend_objective(w):
            w_norm = np.array(w) / (np.sum(w) + 1e-12)
            if self.use_sequence_model and oof_seq is not None:
                p_blend = (
                    w_norm[0] * oof_base[eval_indices]
                    + w_norm[1] * oof_lgbm[eval_indices]
                    + w_norm[2] * oof_seq[eval_indices]
                )
            else:
                p_blend = w_norm[0] * oof_base[eval_indices] + w_norm[1] * oof_lgbm[eval_indices]
            p_blend = np.clip(p_blend, 1e-7, 1.0 - 1e-7)
            p_blend = p_blend / np.sum(p_blend, axis=1, keepdims=True)
            return log_loss(meta_y, p_blend, labels=[0, 1, 2])

        n_models = 3 if (self.use_sequence_model and oof_seq is not None) else 2
        init_w = np.ones(n_models) / n_models
        bnds = [(0.0, 1.0) for _ in range(n_models)]
        res = minimize(
            blend_objective,
            init_w,
            bounds=bnds,
            method="SLSQP",
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        )
        if res.success:
            self.weights = np.array(res.x) / np.sum(res.x)
        else:
            self.weights = init_w

        # Store the OOF predictions so callers can evaluate purely out-of-sample
        self.oof_predictions = np.zeros((n_samples, 3))
        if self.use_sequence_model and oof_seq is not None:
            oof_blend = (
                self.weights[0] * oof_base[eval_indices]
                + self.weights[1] * oof_lgbm[eval_indices]
                + self.weights[2] * oof_seq[eval_indices]
            )
        else:
            oof_blend = self.weights[0] * oof_base[eval_indices] + self.weights[1] * oof_lgbm[eval_indices]
        self.oof_predictions[eval_indices] = oof_blend / np.sum(oof_blend, axis=1, keepdims=True)

        # Retrain full base models on all available training data for future production predictions
        self.baseline_model.fit(X, y)
        self.lgbm_model.fit(X, y)
        if self.sequence_model:
            self.sequence_model.fit(X, y)

        # Compute strictly out-of-fold ensemble metrics
        oof_ens_preds = self.oof_predictions[eval_indices]
        oof_classes = np.argmax(oof_ens_preds, axis=1)

        b_loss = log_loss(meta_y, oof_base[eval_indices], labels=[0, 1, 2])
        l_loss = log_loss(meta_y, oof_lgbm[eval_indices], labels=[0, 1, 2])
        ens_loss = log_loss(meta_y, oof_ens_preds, labels=[0, 1, 2])
        ens_acc = accuracy_score(meta_y, oof_classes)
        ens_f1 = f1_score(meta_y, oof_classes, average="macro")

        self.metadata = {
            "trained_at": datetime.datetime.utcnow().isoformat() + "Z",
            "git_hash": _get_git_hash(),
            "n_samples": n_samples,
            "n_features": len(self.feature_names),
            "features": self.feature_names,
            "metrics": {
                "baseline_log_loss": round(float(b_loss), 4),
                "lightgbm_log_loss": round(float(l_loss), 4),
                "ensemble_log_loss": round(float(ens_loss), 4),
                "ensemble_accuracy": round(float(ens_acc), 4),
                "ensemble_macro_f1": round(float(ens_f1), 4),
            },
        }

        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated ensemble class probabilities [P(DOWN), P(FLAT), P(UP)]."""
        if not self.is_fitted:
            raise RuntimeError("Ensemble is not fitted.")

        p_base = self.baseline_model.predict_proba(X)
        p_lgbm = self.lgbm_model.predict_proba(X)
        if self.use_sequence_model and self.sequence_model:
            p_seq = self.sequence_model.predict_proba(X)
            probs = self.weights[0] * p_base + self.weights[1] * p_lgbm + self.weights[2] * p_seq
        else:
            probs = self.weights[0] * p_base + self.weights[1] * p_lgbm
        return probs / np.sum(probs, axis=1, keepdims=True)

    def get_ensemble_agreement(self, X: pd.DataFrame) -> np.ndarray:
        """
        Measure agreement between base models.
        Returns a score in [0.0, 1.0] where 1.0 means unanimous agreement.
        """
        p_base = self.baseline_model.predict_proba(X)
        p_lgbm = self.lgbm_model.predict_proba(X)

        # 1 - Total Variation Distance between probability vectors
        tvd = 0.5 * np.sum(np.abs(p_base - p_lgbm), axis=1)
        agreement = np.clip(1.0 - tvd, 0.0, 1.0)
        return agreement

    def save_artifacts(self, version: str = "v1") -> Path:
        """Save serialized model bundle and JSON metadata."""
        out_dir = self.artifacts_dir / version
        out_dir.mkdir(parents=True, exist_ok=True)

        bundle_path = out_dir / "ensemble_bundle.pkl"
        meta_path = out_dir / "metadata.json"

        with open(bundle_path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        return bundle_path

    def save_model_bundle(self, version: str = "v1") -> Path:
        """Alias for save_artifacts."""
        return self.save_artifacts(version)

    @classmethod
    def load_artifacts(cls, artifact_path: str) -> "StackingEnsemble":
        with open(artifact_path, "rb") as f:
            obj = pickle.load(f)
        return obj
