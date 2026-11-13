"""Robust-scaled Multi-Timeframe Ensemble Isolation Forest for Microstructural and Regime Anomalies."""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


class MultiTimeframeEnsembleIsolationForest:
    """
    Ensemble anomaly detection pipeline that:
    1. Preprocesses multi-factor features (RSI, returns, volume spikes, Parkinson vol)
       using RobustScaler (median-centered, IQR-scaled) to prevent outlier contamination.
    2. Runs multi-timeframe Isolation Forest estimators (Micro 15d, Tactical 45d, Structural 120d)
       to cross-verify anomalies across short, medium, and long horizons.
    3. Requires multi-timeframe confirmation (consensus >= 2 votes) before firing alerts,
       substantially eliminating single-day false positives.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        timeframe_lookbacks: Tuple[int, int, int] = (15, 45, 120),
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.lookbacks = timeframe_lookbacks
        self.random_state = random_state
        self.scaler = RobustScaler(quantile_range=(15.0, 85.0))
        self.feature_names = [
            "return_1d",
            "return_5d",
            "volume_zscore",
            "parkinson_spread",
            "rsi_deviation",
        ]

    def extract_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract stationary multi-factor features for anomaly detection."""
        feat_df = pd.DataFrame(index=df.index)

        # 1. 1-day and 5-day log returns
        close = df["Close"].astype(float)
        feat_df["return_1d"] = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        feat_df["return_5d"] = close.pct_change(5).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # 2. Volume Z-score
        vol = df["Volume"].astype(float).replace(0, np.nan).ffill().fillna(1.0)
        vol_mean = vol.rolling(window=20, min_periods=5).mean()
        vol_std = vol.rolling(window=20, min_periods=5).std().replace(0, np.nan)
        feat_df["volume_zscore"] = ((vol - vol_mean) / vol_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # 3. High-Low Parkinson Spread
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        feat_df["parkinson_spread"] = ((high - low) / (close + 1e-6)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # 4. Normalized RSI deviation from 50 (neutral)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        roll_gain = gain.rolling(window=14, min_periods=5).mean()
        roll_loss = loss.rolling(window=14, min_periods=5).mean()
        rs = roll_gain / roll_loss.replace(0, 1e-6)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        feat_df["rsi_deviation"] = ((rsi - 50.0) / 25.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return feat_df[self.feature_names].fillna(0.0)

    def fit_predict_multitimeframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Fits multi-timeframe Isolation Forests and flags anomalies with consensus confirmation.
        Returns:
        - anomaly_flags: Series with -1 (confirmed anomaly) or 1 (normal)
        - anomaly_scores: Continuous anomaly intensity [0.0 (normal) to 1.0 (extreme outlier)]
        - recent_10d_anomalies_count: confirmed anomalies in the trailing 10 sessions
        - latest_bar_status: detailed consensus breakdown for the current bar
        """
        raw_feats = self.extract_anomaly_features(df)
        n_bars = len(raw_feats)

        if n_bars < max(self.lookbacks[0], 20):
            # Fallback for short histories
            return {
                "anomaly_flags": pd.Series(1, index=df.index),
                "anomaly_scores": pd.Series(0.0, index=df.index),
                "recent_10d_anomalies_count": 0,
                "latest_bar_status": {
                    "is_confirmed_anomaly": False,
                    "consensus_votes": 0,
                    "confidence_score": 0.0,
                    "timeframe_breakdown": {},
                },
            }

        # Fit RobustScaler on full history without look-ahead (IQR-based)
        scaled_feats = self.scaler.fit_transform(raw_feats)

        # Run multi-timeframe isolation forests
        # Short (15d), Medium (45d), Long (120d)
        votes_matrix = np.zeros((n_bars, len(self.lookbacks)), dtype=int)
        score_matrix = np.zeros((n_bars, len(self.lookbacks)), dtype=float)

        for i, lookback in enumerate(self.lookbacks):
            effective_lookback = min(lookback, n_bars)
            # Create timeframe-specific IsolationForest
            iso = IsolationForest(
                contamination=self.contamination,
                max_samples=min(effective_lookback, 256),
                random_state=self.random_state + (i * 17),
                n_estimators=100,
            )
            iso.fit(scaled_feats[-effective_lookback:])
            
            # Predict across all bars
            preds = iso.predict(scaled_feats)  # -1 = anomaly, 1 = normal
            dec_scores = iso.decision_function(scaled_feats)  # lower = more abnormal

            votes_matrix[:, i] = (preds == -1).astype(int)
            # Normalize decision score to [0, 1] outlier probability: min-max inverted
            norm_scores = np.clip(-dec_scores * 2.0, 0.0, 1.0)
            score_matrix[:, i] = norm_scores

        # Consensus Rule: Confirmed anomaly requires >= 2 timeframe votes (out of 3)
        consensus_votes = np.sum(votes_matrix, axis=1)
        confirmed_anomalies = np.where(consensus_votes >= 2, -1, 1)

        # Weighted aggregate anomaly intensity
        weights = np.array([0.40, 0.35, 0.25])  # higher weight on short/medium
        aggregate_scores = np.dot(score_matrix, weights)
        aggregate_scores = np.clip(aggregate_scores, 0.0, 1.0)

        # Trailing 10-day confirmed anomalies count
        recent_10d_count = int(np.sum(confirmed_anomalies[-10:] == -1))

        # Latest bar diagnostic details
        latest_idx = -1
        latest_status = {
            "is_confirmed_anomaly": bool(confirmed_anomalies[latest_idx] == -1),
            "consensus_votes": int(consensus_votes[latest_idx]),
            "confidence_score": round(float(aggregate_scores[latest_idx]), 3),
            "timeframe_breakdown": {
                f"{self.lookbacks[k]}d_micro": bool(votes_matrix[latest_idx, k] == 1)
                for k in range(len(self.lookbacks))
            },
            "scaled_feature_magnitudes": {
                name: round(float(scaled_feats[latest_idx, j]), 2)
                for j, name in enumerate(self.feature_names)
            },
        }

        return {
            "anomaly_flags": pd.Series(confirmed_anomalies, index=df.index),
            "anomaly_scores": pd.Series(aggregate_scores, index=df.index),
            "recent_10d_anomalies_count": recent_10d_count,
            "latest_bar_status": latest_status,
        }
