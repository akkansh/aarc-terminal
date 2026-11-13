"""Purged and embargoed walk-forward cross-validation splitter."""

from typing import Generator, List, Tuple
import numpy as np
import pandas as pd


class PurgedWalkForwardCV:
    """
    Purged and Embargoed Walk-Forward Cross Validation.
    Guarantees no label overlap or information leakage between train and test splits.
    embargo_days = prediction horizon H (default 5).
    """

    def __init__(
        self,
        n_splits: int = 5,
        horizon_days: int = 5,
        min_train_ratio: float = 0.50,
    ):
        self.n_splits = n_splits
        self.horizon_days = horizon_days
        self.min_train_ratio = min_train_ratio

    def split(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        if n_samples < 50:
            # Insufficient samples for multi-split
            split_point = int(n_samples * 0.70)
            train_idx = np.arange(0, split_point - self.horizon_days)
            val_idx = np.arange(split_point, n_samples)
            return [(train_idx, val_idx)]

        min_train = int(n_samples * self.min_train_ratio)
        test_size = (n_samples - min_train) // self.n_splits

        splits = []
        for i in range(self.n_splits):
            val_start = min_train + (i * test_size)
            val_end = min_train + ((i + 1) * test_size) if i < self.n_splits - 1 else n_samples

            # Embargo gap: train ends at val_start - horizon_days
            train_end = max(20, val_start - self.horizon_days)
            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_end)

            if len(train_idx) > 20 and len(val_idx) > 5:
                splits.append((train_idx, val_idx))

        return splits
