"""Benchmarking anomaly detectors on precision@k against large price/volume jumps."""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def benchmark_anomaly_detectors(
    returns: pd.Series,
    volume_zscore: pd.Series,
    k: int = 10,
) -> Dict[str, Any]:
    """
    Evaluates precision@k on identifying jump events (top extreme 3-sigma moves).
    Compares IsolationForest against Statistical Mahalanobis/Z-score detector.
    """
    valid = (~returns.isna()) & (~volume_zscore.isna())
    ret = returns[valid].values
    vol = volume_zscore[valid].values
    n = len(ret)

    if n < 30:
        return {"precision_at_k": 0.0, "chosen_model": "IsolationForest"}

    # Define ground truth event days: top extreme return moves (|return| >= 2.5 * std)
    threshold = 2.5 * np.std(ret)
    ground_truth_events = set(np.where(np.abs(ret) >= threshold)[0])

    # Model 1: IsolationForest
    X = np.column_stack([ret, vol])
    iso = IsolationForest(contamination=0.08, random_state=42)
    iso.fit(X)
    iso_scores = -iso.decision_function(X)  # Higher is more anomalous
    top_k_iso = set(np.argsort(iso_scores)[-k:])
    precision_iso = len(top_k_iso.intersection(ground_truth_events)) / max(1, k)

    # Model 2: Statistical 2D Mahalanobis / Z-score baseline
    cov = np.cov(X.T) + 1e-6 * np.eye(2)
    inv_cov = np.linalg.pinv(cov)
    diff = X - np.mean(X, axis=0)
    stat_scores = np.sum((diff @ inv_cov) * diff, axis=1)
    top_k_stat = set(np.argsort(stat_scores)[-k:])
    precision_stat = len(top_k_stat.intersection(ground_truth_events)) / max(1, k)

    return {
        "k": k,
        "n_events": len(ground_truth_events),
        "precision_at_k_isolation_forest": round(precision_iso, 3),
        "precision_at_k_statistical_baseline": round(precision_stat, 3),
        "chosen_model": "IsolationForest" if precision_iso >= precision_stat else "StatisticalDetector",
    }
