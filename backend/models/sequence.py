"""Sequence model (GRU) for sequential temporal representation."""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class PyTorchGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_classes: int = 3, num_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        # Take last time step
        last_step = out[:, -1, :]
        logits = self.fc(last_step)
        return logits


class SequenceGRUModel:
    """Optional sequence model processing rolling windows of daily features."""

    def __init__(self, seq_len: int = 10, hidden_dim: int = 32, epochs: int = 20, lr: float = 0.005, seed: int = 42):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.model: Optional[PyTorchGRU] = None
        self.feature_names: List[str] = []
        self.is_fitted = False
        torch.manual_seed(seed)

    def _create_sequences(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        n_samples = len(X)
        if n_samples < self.seq_len:
            return np.empty((0, self.seq_len, X.shape[1])), None

        X_seq = []
        y_seq = []
        for i in range(self.seq_len - 1, n_samples):
            X_seq.append(X[i - self.seq_len + 1 : i + 1])
            if y is not None:
                y_seq.append(y[i])

        X_arr = np.array(X_seq, dtype=np.float32)
        y_arr = np.array(y_seq, dtype=np.int64) if y is not None else None
        return X_arr, y_arr

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "SequenceGRUModel":
        self.feature_names = list(X_train.columns)
        X_arr = X_train.fillna(0.0).values
        y_arr = y_train.values.astype(int)

        X_seq, y_seq = self._create_sequences(X_arr, y_arr)
        if len(X_seq) < 20:
            return self

        input_dim = X_seq.shape[2]
        self.model = PyTorchGRU(input_dim=input_dim, hidden_dim=self.hidden_dim)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        dataset = TensorDataset(torch.from_numpy(X_seq), torch.from_numpy(y_seq))
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        self.model.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            # Fallback uniform
            return np.full((len(X), 3), 1.0 / 3.0)

        X_arr = X[self.feature_names].fillna(0.0).values
        X_seq, _ = self._create_sequences(X_arr)
        if len(X_seq) == 0:
            return np.full((len(X), 3), 1.0 / 3.0)

        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.from_numpy(X_seq)
            logits = self.model(tensor_x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        # Prepend uniform probabilities for warmup sequence period
        warmup_len = len(X) - len(probs)
        if warmup_len > 0:
            warmup_probs = np.full((warmup_len, 3), 1.0 / 3.0)
            probs = np.vstack([warmup_probs, probs])

        return probs
