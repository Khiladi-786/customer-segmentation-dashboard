"""
train.py ─ Standalone ML Training Pipeline
==========================================
Run this ONCE before launching Streamlit to pre-train and persist all models.
Subsequent Streamlit launches will load the saved .pkl files instantly.

Usage:
    python train.py            # Full AutoML (slow, ~3–5 min)
    python train.py --fast     # KMeans-only quick mode (~30s)
"""

from __future__ import annotations
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Setup ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def train(fast: bool = False) -> None:
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("SegmentAI — Training Pipeline")
    logger.info("=" * 60)

    # ── 1. Data Loading & Preprocessing ───────────────────────────────────────
    logger.info("Step 1/8 — Loading & preprocessing data …")
    from core.data_pipeline import DataPipeline
    pipeline = DataPipeline(scaler_type="standard")
    raw_df = pipeline.load_data(source="csv")
    X, proc_df = pipeline.preprocess(raw_df, fit=True)
    pipeline.save()
    logger.info(
        f"  ✓ Loaded {len(raw_df)} rows → {X.shape[0]} clean rows, {X.shape[1]} features"
    )

    # ── 2. Feature Engineering (RFM, CLV proxy, behavioral) ───────────────────
    logger.info("Step 2/8 — Feature engineering …")
    from core.feature_engineering import full_feature_pipeline, get_feature_importance
    feat_df = full_feature_pipeline(proc_df)
    logger.info(f"  ✓ Engineered {feat_df.shape[1]} features for {len(feat_df)} customers")

    # ── 3. Clustering ──────────────────────────────────────────────────────────
    if fast:
        logger.info("Step 3/8 — Clustering: KMeans quick mode …")
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from utils.metrics import compute_all_metrics
        from core.clustering import ClusteringEngine

        best_score = -9999
        best_k = 4
        best_labels = None
        best_model = None
        best_metrics = {}

        for k in [3, 4, 5]:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            lbls = km.fit_predict(X)
            m = compute_all_metrics(X, lbls)
            logger.info(f"    KMeans k={k}: silhouette={m['silhouette']:.4f}, composite={m['composite']:.4f}")
            if m["composite"] > best_score:
                best_score = m["composite"]
                best_k = k
                best_labels = lbls
                best_model = km
                best_metrics = m

        engine = ClusteringEngine()
        engine.best_model     = best_model
        engine.best_algorithm = "KMeans"
        engine.best_labels    = best_labels
        engine.best_metrics   = best_metrics
        engine.best_params    = {"n_clusters": best_k}
        engine.results        = [{
            "algorithm": "KMeans",
            "params":    {"n_clusters": best_k},
            "metrics":   best_metrics,
            "n_clusters": best_k,
            "labels":    best_labels,
            "model":     best_model,
        }]
        engine.pca_2d = PCA(n_components=2, random_state=42)
        engine.pca_2d.fit(X)
        engine.pca_3d = PCA(n_components=3, random_state=42)
        engine.pca_3d.fit(X)
        engine.save()
        labels = best_labels
        logger.info(f"  ✓ Best: KMeans k={best_k} | composite={best_score:.4f}")
    else:
        logger.info("Step 3/8 — Clustering: Full AutoML (KMeans + DBSCAN + Agglomerative + GMM) …")
        from core.clustering import ClusteringEngine
        engine = ClusteringEngine()
        best = engine.run_automl(X, log_mlflow=False)
        engine.save()
        labels = engine.best_labels
        logger.info(f"  ✓ Best: {engine.best_algorithm} | composite={engine.best_metrics['composite']:.4f}")

    # ── 4. Add cluster labels + PCA coords ────────────────────────────────────
    logger.info("Step 4/8 — Computing PCA projections …")
    feat_df["Cluster"] = labels[: len(feat_df)]
    coords_2d = engine.project_2d(X)
    coords_3d = engine.project_3d(X)
    feat_df["PCA1"] = coords_2d[: len(feat_df), 0]
    feat_df["PCA2"] = coords_2d[: len(feat_df), 1]
    feat_df["PCA3"] = (
        coords_3d[: len(feat_df), 2] if coords_3d.shape[1] > 2 else 0.0
    )
    logger.info(f"  ✓ PCA projections added")

    # ── 5. CLV Model ──────────────────────────────────────────────────────────
    logger.info("Step 5/8 — Training CLV prediction model …")
    from core.clv_model import CLVModel
    clv_model = CLVModel()
    clv_model.fit(feat_df)
    # Overwrite CLV_Score with model-predicted values (replaces the proxy)
    feat_df = clv_model.score_dataframe(feat_df)
    clv_model.save()
    logger.info(
        f"  ✓ CLV model trained — CV R²={clv_model.cv_score:.4f} | "
        f"avg CLV score={feat_df['CLV_Score'].mean():.1f}"
    )

    # ── 6. SHAP Explainability ─────────────────────────────────────────────────
    logger.info("Step 6/8 — Building SHAP explainability engine …")
    feat_names = pipeline.get_feature_names()
    from core.explainability import ExplainabilityEngine
    xai = ExplainabilityEngine()
    xai.fit(X, labels, feat_names)
    xai.save()
    if xai.shap_values is not None:
        logger.info(f"  ✓ SHAP fitted — values shape: {xai.shap_values.shape}")
    else:
        logger.warning("  ⚠ SHAP fitting failed — check SHAP installation")

    # ── 7. Feature Importance (RF surrogate) ──────────────────────────────────
    logger.info("Step 7/8 — Computing feature importance …")
    fi_df = get_feature_importance(X, labels, feat_names)
    fi_path = str(MODELS_DIR / "feature_importance.pkl")
    import joblib
    joblib.dump(fi_df, fi_path)
    logger.info(f"  ✓ Top feature: {fi_df.iloc[0]['Feature']} ({fi_df.iloc[0]['Importance']:.4f})")

    # ── 8. Anomaly Detection ──────────────────────────────────────────────────
    logger.info("Step 8/8 — Training anomaly detector …")
    from core.anomaly_detection import AnomalyDetector
    detector = AnomalyDetector(contamination=0.05)
    detector.fit(X)
    detector.save()
    feat_df = detector.flag_dataframe(feat_df, X[: len(feat_df)])
    n_anomalies = int(feat_df["IsAnomaly"].sum())
    logger.info(f"  ✓ {n_anomalies} anomalies detected ({n_anomalies/len(feat_df)*100:.1f}%)")

    # ── Save processed dataframe ───────────────────────────────────────────────
    feat_df_path = str(MODELS_DIR / "processed_df.pkl")
    import joblib
    joblib.dump(feat_df, feat_df_path)
    logger.info(f"  ✓ Processed DataFrame saved → {feat_df_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"✅ Training complete in {elapsed:.1f}s")
    logger.info(f"   Customers processed : {len(feat_df):,}")
    logger.info(f"   Segments found      : {feat_df['Cluster'].nunique()}")
    logger.info(f"   Best algorithm      : {engine.best_algorithm}")
    logger.info(f"   Silhouette score    : {engine.best_metrics.get('silhouette', 0):.4f}")
    logger.info(f"   Anomalies detected  : {n_anomalies}")
    logger.info(f"   Models saved to     : {MODELS_DIR}")
    logger.info("=" * 60)
    logger.info("👉 Now run:  streamlit run app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SegmentAI Training Pipeline")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Quick mode: KMeans only (skips full AutoML grid search)",
    )
    args = parser.parse_args()
    train(fast=args.fast)
