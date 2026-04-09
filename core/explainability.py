"""
Explainability Engine using SHAP.
Uses a Random Forest surrogate model so SHAP works with any clustering algorithm.
Provides: global summary, per-cluster heatmap, per-customer waterfall.

Fixed for SHAP >= 0.44 where shap_values() returns ndarray of shape
(n_samples, n_features, n_classes) instead of list[(n_samples, n_features)].
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


class ExplainabilityEngine:
    """SHAP-based XAI for unsupervised clustering via surrogate RF."""

    def __init__(self):
        self.explainer = None
        # shap_values shape: (n_samples, n_features, n_classes)
        self.shap_values: Optional[np.ndarray] = None
        self.surrogate = None
        self.feature_names: list[str] = []
        self._X_sample: Optional[np.ndarray] = None
        self._labels_sample: Optional[np.ndarray] = None

    # ── Fitting ─────────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        feature_names: list[str],
        n_background: int = 100,
    ) -> "ExplainabilityEngine":
        """
        Train a Random Forest surrogate on cluster labels, then build SHAP explainer.
        Handles both old SHAP (list output) and new SHAP (3-D ndarray output).
        """
        try:
            import shap
            from sklearn.ensemble import RandomForestClassifier

            self.feature_names = feature_names

            # ── Surrogate RF ───────────────────────────────────────────────
            self.surrogate = RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            )
            self.surrogate.fit(X, labels)

            # ── SHAP TreeExplainer ─────────────────────────────────────────
            self.explainer = shap.TreeExplainer(self.surrogate)

            # Pre-compute SHAP values (subsample for speed)
            sample_size = min(len(X), 500)
            idx = np.random.default_rng(42).choice(len(X), size=sample_size, replace=False)
            X_sample = X[idx]
            raw = self.explainer.shap_values(X_sample)

            # Normalise output to (n_samples, n_features, n_classes)
            if isinstance(raw, list):
                # Old SHAP: list of (n_samples, n_features) one per class
                # Stack → (n_classes, n_samples, n_features) then transpose
                stacked = np.stack(raw, axis=0)                  # (n_classes, n_s, n_f)
                self.shap_values = stacked.transpose(1, 2, 0)    # (n_s, n_f, n_classes)
            elif isinstance(raw, np.ndarray):
                if raw.ndim == 2:
                    # Binary: (n_samples, n_features) → expand dim
                    self.shap_values = raw[:, :, np.newaxis]
                elif raw.ndim == 3:
                    # New SHAP: already (n_samples, n_features, n_classes)
                    self.shap_values = raw
                else:
                    raise ValueError(f"Unexpected SHAP array ndim={raw.ndim}")
            else:
                raise TypeError(f"Unexpected SHAP output type: {type(raw)}")

            self._X_sample = X_sample
            self._labels_sample = labels[idx]
            logger.info(
                f"SHAP explainer fitted — shap_values.shape={self.shap_values.shape}"
            )

        except Exception as e:
            logger.warning(f"SHAP fitting failed: {e}")

        return self

    # ── Global Importance ────────────────────────────────────────────────────

    def global_feature_importance(self) -> pd.DataFrame:
        """Mean absolute SHAP values across all classes → global feature importance."""
        if self.shap_values is None:
            return pd.DataFrame()

        # shap_values: (n_samples, n_features, n_classes)
        # Mean |SHAP| over samples and classes → (n_features,)
        mean_abs = np.abs(self.shap_values).mean(axis=(0, 2))  # (n_features,)

        n_feats = len(self.feature_names)
        if len(mean_abs) != n_feats:
            logger.warning(
                f"SHAP mean_abs length ({len(mean_abs)}) ≠ feature_names length ({n_feats}). "
                "Truncating/padding."
            )
            # Align lengths
            mean_abs = mean_abs[:n_feats] if len(mean_abs) > n_feats else np.pad(
                mean_abs, (0, n_feats - len(mean_abs))
            )

        return (
            pd.DataFrame({"Feature": self.feature_names, "SHAP_Importance": mean_abs})
            .sort_values("SHAP_Importance", ascending=False)
            .reset_index(drop=True)
        )

    # ── Cluster Feature Impact ───────────────────────────────────────────────

    def cluster_feature_impact(self) -> pd.DataFrame:
        """
        Per-cluster mean SHAP values.
        Returns DataFrame: rows=cluster_labels, cols=features.
        """
        if self.shap_values is None or self._labels_sample is None:
            return pd.DataFrame()

        # shap_values: (n_samples, n_features, n_classes)
        clusters = sorted(set(self._labels_sample.tolist()))
        rows: dict[str, np.ndarray] = {}

        for c in clusters:
            if c == -1:
                continue
            mask = self._labels_sample == c
            c_idx = min(int(c), self.shap_values.shape[2] - 1)
            # Mean SHAP of the class-c explainer for samples in cluster c
            # shape: (n_masked_samples, n_features)
            cluster_shap = self.shap_values[mask, :, c_idx]
            rows[f"Cluster {c}"] = cluster_shap.mean(axis=0)  # (n_features,)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, index=self.feature_names).T
        return df

    # ── Customer-Level SHAP ──────────────────────────────────────────────────

    def customer_shap(self, customer_index: int) -> Optional[dict]:
        """Return SHAP values for a specific customer (by index in sample)."""
        if self.shap_values is None or self._labels_sample is None:
            return None
        if customer_index >= len(self._X_sample):
            return None

        label = int(self._labels_sample[customer_index])
        c_idx = min(label, self.shap_values.shape[2] - 1)

        # shap_values: (n_samples, n_features, n_classes)
        vals = self.shap_values[customer_index, :, c_idx]  # (n_features,)

        base = getattr(self.explainer, "expected_value", 0)
        if isinstance(base, (list, np.ndarray)):
            base = float(base[c_idx]) if c_idx < len(base) else float(base[0])

        return {
            "feature_names": self.feature_names,
            "shap_values":   vals.tolist(),
            "base_value":    float(base),
            "cluster":       label,
        }

    # ── Utilities ────────────────────────────────────────────────────────────

    def get_sample_size(self) -> int:
        return len(self._X_sample) if self._X_sample is not None else 0

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> str:
        save_path = path or str(MODELS_DIR / "explainability.pkl")
        joblib.dump(self, save_path)
        logger.info(f"ExplainabilityEngine saved → {save_path}")
        return save_path

    @classmethod
    def load(cls, path: Optional[str] = None) -> "ExplainabilityEngine":
        load_path = path or str(MODELS_DIR / "explainability.pkl")
        return joblib.load(load_path)
