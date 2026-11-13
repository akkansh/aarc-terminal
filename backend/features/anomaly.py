"""Rolling Isolation Forest anomaly detection on strictly stationary features."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Optional, Tuple


def compute_rolling_anomalies(
    features_df: pd.DataFrame,
    rolling_window: int = 60,
    contamination: float = 0.05,
    random_state: int = 42,
    refit_step: int = 5,
) -> pd.DataFrame:
    """
    Train IsolationForest on a rolling past window of stationary features.
    Prevents look-ahead bias by training strictly on data up to bar t.
    Refits every `refit_step` bars for computational efficiency.
    Returns continuous anomaly score and a percentile-based boolean flag.
    """
    anomaly_cols = ["return_1d", "volume_zscore_20d", "vol_ratio_20_60", "hl_spread"]
    available_cols = [c for c in anomaly_cols if c in features_df.columns]

    if not available_cols:
        raise ValueError(f"Features DataFrame must contain stationary columns: {anomaly_cols}")

    X = features_df[available_cols].fillna(0.0).copy()
    n_rows = len(X)

    scores = np.zeros(n_rows, dtype=float)
    flags = np.zeros(n_rows, dtype=bool)

    min_train = min(30, rolling_window // 2)
    active_iso = None
    active_threshold = 0.0

    for i in range(n_rows):
        start_idx = max(0, i - rolling_window + 1)
        X_train = X.iloc[start_idx : i + 1]

        if len(X_train) < min_train:
            scores[i] = 0.0
            flags[i] = False
            continue

        # Refit every `refit_step` bars or if not yet initialized
        if active_iso is None or i % refit_step == 0:
            active_iso = IsolationForest(
                contamination=contamination,
                random_state=random_state,
                n_estimators=30,
                n_jobs=1,
            )
            active_iso.fit(X_train.values)
            train_scores = -active_iso.decision_function(X_train.values)
            active_threshold = float(np.percentile(train_scores, (1.0 - contamination) * 100))

        # Score current bar strictly with current model
        current_sample = X.iloc[[i]].values
        raw_df = active_iso.decision_function(current_sample)[0]
        score = -float(raw_df)
        scores[i] = score
        flags[i] = bool(score >= active_threshold)

    out = pd.DataFrame(index=features_df.index)
    out["anomaly_score"] = scores
    out["is_anomaly"] = flags
    # Rolling anomaly density in past 10 days
    out["anomaly_count_10d"] = out["is_anomaly"].astype(int).rolling(window=10, min_periods=1).sum()

    return out
