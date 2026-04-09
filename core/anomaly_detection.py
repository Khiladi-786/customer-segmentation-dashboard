"""
Anomaly Detection using Isolation Forest.
Detects unusual customer behavior and flags outliers.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


class AnomalyDetector:
    """Isolation Forest based anomaly detection for customer data."""

    def __init__(self, contamination: float = 0.05):
        """
        Args:
            contamination: expected proportion of anomalies (default 5%).
        """
        self.contamination = contamination
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        self.is_fitted = False
        self.threshold: float = 0.0

    def fit(self, X: np.ndarray) -> "AnomalyDetector":
        self.model.fit(X)
        scores = self.model.decision_function(X)
        self.threshold = float(np.percentile(scores, self.contamination * 100))
        self.is_fitted = True
        logger.info(
            f"AnomalyDetector fitted: contamination={self.contamination}, "
            f"threshold={self.threshold:.4f}"
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns array: 1 = normal, -1 = anomaly."""
        return self.model.predict(X)

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Returns decision function scores.
        Lower (more negative) = more anomalous.
        """
        return self.model.decision_function(X)

    def flag_dataframe(self, df: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
        """Add IsAnomaly (bool) and AnomalyScore columns to df."""
        df = df.copy()
        preds  = self.predict(X)
        scores = self.anomaly_scores(X)
        df["IsAnomaly"]    = preds == -1
        df["AnomalyScore"] = np.round(scores, 4)
        # Normalize score to 0-100 risk scale (100 = most anomalous)
        min_s, max_s = scores.min(), scores.max()
        if max_s != min_s:
            df["AnomalyRisk"] = np.round(
                100 * (1 - (scores - min_s) / (max_s - min_s)), 1
            )
        else:
            df["AnomalyRisk"] = 0.0
        return df

    def anomaly_summary(self, df: pd.DataFrame) -> dict:
        """Return summary statistics about detected anomalies."""
        if "IsAnomaly" not in df.columns:
            return {}
        anomalies = df[df["IsAnomaly"]]
        normals   = df[~df["IsAnomaly"]]
        return {
            "total_customers":   len(df),
            "anomaly_count":     int(df["IsAnomaly"].sum()),
            "anomaly_pct":       round(df["IsAnomaly"].mean() * 100, 2),
            "avg_anomaly_risk":  round(anomalies["AnomalyRisk"].mean(), 1) if len(anomalies) else 0,
            "avg_income_anomaly": round(anomalies.get("Income", pd.Series([0])).mean(), 0),
            "avg_income_normal":  round(normals.get("Income", pd.Series([0])).mean(), 0),
        }

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> str:
        save_path = path or str(MODELS_DIR / "anomaly_detector.pkl")
        joblib.dump(self, save_path)
        logger.info(f"AnomalyDetector saved → {save_path}")
        return save_path

    @classmethod
    def load(cls, path: Optional[str] = None) -> "AnomalyDetector":
        load_path = path or str(MODELS_DIR / "anomaly_detector.pkl")
        return joblib.load(load_path)
