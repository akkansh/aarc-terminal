"""PyTorch LSTM Autoencoder for Temporal Sequence Anomaly Detection."""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler


def _winsorize_series(s: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    """Clips extreme outlier spikes to protect neural sequence latent representations."""
    clean = s.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty or len(clean) < 10:
        return s.fillna(0.0)
    low_val = float(clean.quantile(lower_q))
    high_val = float(clean.quantile(upper_q))
    return s.clip(lower=low_val, upper=high_val).fillna(0.0)


class _LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, latent_dim: int = 16, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.to_latent = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Seq_Len, Input_Dim)
        out, (hn, cn) = self.lstm(x)
        # hn[-1]: (Batch, Hidden_Dim)
        latent = self.to_latent(hn[-1])
        return latent


class _LSTMDecoder(nn.Module):
    def __init__(self, seq_len: int, latent_dim: int = 16, hidden_dim: int = 32, output_dim: int = 5, num_layers: int = 2):
        super().__init__()
        self.seq_len = seq_len
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # latent: (Batch, Latent_Dim)
        hidden_init = self.from_latent(latent)  # (Batch, Hidden_Dim)
        # Repeat vector along sequence length
        repeated = hidden_init.unsqueeze(1).repeat(1, self.seq_len, 1)  # (Batch, Seq_Len, Hidden_Dim)
        out, _ = self.lstm(repeated)
        recon = self.out_proj(out)  # (Batch, Seq_Len, Output_Dim)
        return recon


class LSTMAutoencoderNet(nn.Module):
    def __init__(self, seq_len: int = 15, input_dim: int = 5, hidden_dim: int = 32, latent_dim: int = 16):
        super().__init__()
        self.encoder = _LSTMEncoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.decoder = _LSTMDecoder(seq_len=seq_len, latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon


class LSTMTemporalAnomalyDetector:
    """
    Ingests rolling multi-variable feature sequences (returns, volume profiles, parkinson vol, rsi).
    Trains an LSTM Autoencoder to reconstruct normal sequential dynamics.
    High reconstruction error signals temporal structural breaks and microstructural anomalies.
    """

    def __init__(
        self,
        seq_len: int = 15,
        hidden_dim: int = 32,
        latent_dim: int = 16,
        epochs: int = 20,
        lr: float = 0.005,
        random_state: int = 42,
    ):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scaler = RobustScaler()
        self.feature_names = [
            "return_1d",
            "volume_zscore",
            "parkinson_spread",
            "rsi_deviation",
            "macd_ratio",
        ]
        self.model: Optional[LSTMAutoencoderNet] = None
        self.error_threshold = 0.50

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        close = df["Close"].astype(float)

        feat["return_1d"] = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

        vol = df["Volume"].astype(float).replace(0, np.nan).ffill().fillna(1.0)
        vol_mean = vol.rolling(20, min_periods=5).mean()
        vol_std = vol.rolling(20, min_periods=5).std().replace(0, np.nan)
        feat["volume_zscore"] = ((vol - vol_mean) / vol_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        high, low = df["High"].astype(float), df["Low"].astype(float)
        feat["parkinson_spread"] = ((high - low) / (close + 1e-6)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        delta = close.diff()
        gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
        rs = gain.rolling(14, min_periods=5).mean() / loss.rolling(14, min_periods=5).mean().replace(0, 1e-6)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        feat["rsi_deviation"] = ((rsi - 50.0) / 25.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # MACD histogram / close
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        feat["macd_ratio"] = ((macd - signal) / (close + 1e-6) * 100.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Apply robust winsorization across all input features before RobustScaler
        for col in self.feature_names:
            feat[col] = _winsorize_series(feat[col], 0.01, 0.99)

        return feat[self.feature_names].fillna(0.0)

    def _create_sliding_windows(self, X: np.ndarray) -> np.ndarray:
        # X: (N, D) -> Windows: (N - seq_len + 1, seq_len, D)
        n, d = X.shape
        if n < self.seq_len:
            return np.empty((0, self.seq_len, d))
        windows = []
        for i in range(n - self.seq_len + 1):
            windows.append(X[i : i + self.seq_len])
        return np.array(windows)

    def fit_and_detect(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Fits LSTM Autoencoder on rolling historical sequences and computes reconstruction errors.
        Returns latest bar temporal anomaly telemetry.
        """
        raw_feats = self._extract_features(df)
        n_bars = len(raw_feats)

        if n_bars < self.seq_len + 15:
            return {
                "is_temporal_anomaly": False,
                "temporal_anomaly_score": 0.0,
                "reconstruction_error": 0.0,
                "threshold": 1.0,
                "feature_breakdown": {},
                "recent_10d_temporal_count": 0,
            }

        scaled = self.scaler.fit_transform(raw_feats)
        windows = self._create_sliding_windows(scaled)  # (N_win, seq_len, D)

        torch.manual_seed(self.random_state)
        X_tensor = torch.tensor(windows, dtype=torch.float32).to(self.device)

        # Initialize network
        input_dim = len(self.feature_names)
        self.model = LSTMAutoencoderNet(
            seq_len=self.seq_len,
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
        ).to(self.device)

        criterion = nn.SmoothL1Loss(reduction="none")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        # Fast training loop
        self.model.train()
        batch_size = min(64, len(windows))
        for epoch in range(self.epochs):
            perm = torch.randperm(len(windows))
            for i in range(0, len(windows), batch_size):
                b_idx = perm[i : i + batch_size]
                batch_x = X_tensor[b_idx]

                optimizer.zero_grad()
                recon_x = self.model(batch_x)
                loss = criterion(recon_x, batch_x).mean()
                loss.backward()
                optimizer.step()

        # Inference: compute reconstruction errors across all sequence windows
        self.model.eval()
        with torch.no_grad():
            recon_all = self.model(X_tensor)
            # Element-wise squared error: (N_win, seq_len, D)
            sq_err = ((recon_all - X_tensor) ** 2).cpu().numpy()
            # Mean error per window: (N_win,)
            window_errors = np.mean(sq_err, axis=(1, 2))

        # Dynamic threshold: Median + 2.5 * IQR
        q25 = float(np.percentile(window_errors, 25))
        q75 = float(np.percentile(window_errors, 75))
        iqr = max(1e-4, q75 - q25)
        median_err = float(np.median(window_errors))
        threshold = median_err + 2.5 * iqr
        self.error_threshold = threshold

        # Latest window error
        latest_err = float(window_errors[-1])
        is_anomaly = bool(latest_err > threshold)

        # Normalized anomaly score in [0.0, 1.0]
        score = float(np.clip(latest_err / (threshold * 1.5 + 1e-6), 0.0, 1.0))

        # Feature error breakdown on the latest bar of the sequence
        latest_feat_errors = sq_err[-1, -1, :] if len(sq_err) > 0 else np.zeros(input_dim)
        feature_breakdown = {
            self.feature_names[j]: round(float(latest_feat_errors[j]), 4)
            for j in range(input_dim)
        }

        # Trailing 10-day temporal anomalies count
        recent_10d_errors = window_errors[-10:] if len(window_errors) >= 10 else window_errors
        recent_10d_count = int(np.sum(recent_10d_errors > threshold))

        return {
            "is_temporal_anomaly": is_anomaly,
            "temporal_anomaly_score": round(score, 3),
            "reconstruction_error": round(latest_err, 4),
            "threshold": round(threshold, 4),
            "feature_breakdown": feature_breakdown,
            "recent_10d_temporal_count": recent_10d_count,
        }
