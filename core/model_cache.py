"""
Model Cache: centralized loading + optional training for all Streamlit pages.

Strategy:
  1. If pre-trained .pkl files exist (produced by train.py) → load instantly.
  2. If not → train on-the-fly with a fast KMeans-only pipeline.

All pages call get_models() to obtain a shared dict of fitted objects.
Uses st.cache_resource so training/loading happens once per Streamlit session.
"""

from __future__ import annotations
import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd
import streamlit as st
import joblib

logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent.parent
DATA_PATH  = BASE_DIR / "new.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

_REQUIRED_PKLS = [
    "pipeline.pkl",
    "clustering_engine.pkl",
    "clv_model.pkl",
    "anomaly_detector.pkl",
    "explainability.pkl",
    "feature_importance.pkl",
    "processed_df.pkl",
]


def _all_models_exist() -> bool:
    return all((MODELS_DIR / f).exists() for f in _REQUIRED_PKLS)


@st.cache_resource(show_spinner="🤖  Initialising AI models …")
def get_models() -> dict:
    """
    Load pre-trained models if available, otherwise train a fast pipeline.
    Returns a rich dict used by all Streamlit pages.
    """
    if _all_models_exist():
        return _load_from_disk()
    else:
        return _train_and_return()


# ── Load path ──────────────────────────────────────────────────────────────────

def _load_from_disk() -> dict:
    """Load all .pkl files produced by train.py."""
    t0 = time.time()
    logger.info("Loading pre-trained models from disk …")

    from core.data_pipeline    import DataPipeline
    from core.clustering       import ClusteringEngine
    from core.clv_model        import CLVModel
    from core.explainability   import ExplainabilityEngine
    from core.anomaly_detection import AnomalyDetector

    pipeline  = DataPipeline.load()
    engine    = ClusteringEngine.load()
    clv_model = CLVModel.load()
    detector  = AnomalyDetector.load()
    xai       = ExplainabilityEngine.load()
    fi_df     = joblib.load(str(MODELS_DIR / "feature_importance.pkl"))
    feat_df   = joblib.load(str(MODELS_DIR / "processed_df.pkl"))

    # Re-derive X for the Algorithms page (needs raw scaled features)
    raw_df = pipeline.load_data(source="csv")
    X, _   = pipeline.preprocess(raw_df, fit=False)

    # Elbow data (fast — uses saved pipeline, just KMeans fits)
    elbow = engine.elbow_data(X)

    # Cluster profiles
    profiles = engine.cluster_profiles(feat_df)

    # Customer evolution simulation
    evolution_df = _simulate_evolution(feat_df, engine, X)

    logger.info(f"Models loaded in {time.time()-t0:.1f}s")

    return _build_result_dict(
        pipeline=pipeline,
        engine=engine,
        clv_model=clv_model,
        detector=detector,
        xai=xai,
        fi_df=fi_df,
        feat_df=feat_df,
        X=X,
        elbow=elbow,
        profiles=profiles,
        evolution_df=evolution_df,
    )


# ── Train path ─────────────────────────────────────────────────────────────────

def _train_and_return() -> dict:
    """Fast on-the-fly training (KMeans only).  Saves models to disk for next time."""
    t0 = time.time()
    logger.info("No pre-trained models found — running fast training pipeline …")

    from core.data_pipeline       import DataPipeline
    from core.feature_engineering import full_feature_pipeline, get_feature_importance
    from core.clustering          import ClusteringEngine
    from core.clv_model           import CLVModel
    from core.explainability      import ExplainabilityEngine
    from core.anomaly_detection   import AnomalyDetector
    from sklearn.cluster          import KMeans
    from sklearn.decomposition    import PCA
    from utils.metrics            import compute_all_metrics

    # ──  1. Load & preprocess ───────────────────────────────────────────────
    pipeline = DataPipeline(scaler_type="standard")
    raw_df   = pipeline.load_data(source="csv")
    X, proc_df = pipeline.preprocess(raw_df, fit=True)
    pipeline.save()

    # ── 2. Feature engineering ─────────────────────────────────────────────
    feat_df = full_feature_pipeline(proc_df)

    # ── 3. Clustering (fast KMeans) ────────────────────────────────────────
    best_score, best_k, best_labels, best_model, best_metrics = -9999, 4, None, None, {}
    for k in [3, 4, 5]:
        km   = KMeans(n_clusters=k, n_init=10, random_state=42)
        lbls = km.fit_predict(X)
        m    = compute_all_metrics(X, lbls)
        if m["composite"] > best_score:
            best_score, best_k, best_labels, best_model, best_metrics = (
                m["composite"], k, lbls, km, m
            )

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
    feat_df["Cluster"] = labels[: len(feat_df)]
    coords_2d = engine.project_2d(X)
    coords_3d = engine.project_3d(X)
    feat_df["PCA1"] = coords_2d[: len(feat_df), 0]
    feat_df["PCA2"] = coords_2d[: len(feat_df), 1]
    feat_df["PCA3"] = (
        coords_3d[: len(feat_df), 2] if coords_3d.shape[1] > 2 else 0.0
    )

    # ── 4. CLV model ───────────────────────────────────────────────────────
    clv_model = CLVModel()
    clv_model.fit(feat_df)
    feat_df   = clv_model.score_dataframe(feat_df)
    clv_model.save()

    # ── 5. SHAP explainability ─────────────────────────────────────────────
    feat_names = pipeline.get_feature_names()
    xai = ExplainabilityEngine()
    xai.fit(X, labels, feat_names)
    xai.save()

    # ── 6. Feature importance ──────────────────────────────────────────────
    fi_df = get_feature_importance(X, labels, feat_names)
    joblib.dump(fi_df, str(MODELS_DIR / "feature_importance.pkl"))

    # ── 7. Anomaly detection ───────────────────────────────────────────────
    detector = AnomalyDetector(contamination=0.05)
    detector.fit(X)
    detector.save()
    feat_df = detector.flag_dataframe(feat_df, X[: len(feat_df)])

    # ── 8. Save processed DataFrame ────────────────────────────────────────
    joblib.dump(feat_df, str(MODELS_DIR / "processed_df.pkl"))

    # ── Derived artefacts ──────────────────────────────────────────────────
    elbow      = engine.elbow_data(X)
    profiles   = engine.cluster_profiles(feat_df)
    evolution_df = _simulate_evolution(feat_df, engine, X)

    logger.info(f"Fast training done in {time.time()-t0:.1f}s")

    return _build_result_dict(
        pipeline=pipeline,
        engine=engine,
        clv_model=clv_model,
        detector=detector,
        xai=xai,
        fi_df=fi_df,
        feat_df=feat_df,
        X=X,
        elbow=elbow,
        profiles=profiles,
        evolution_df=evolution_df,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_result_dict(
    *,
    pipeline,
    engine,
    clv_model,
    detector,
    xai,
    fi_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    X: np.ndarray,
    elbow: dict,
    profiles: pd.DataFrame,
    evolution_df: pd.DataFrame,
) -> dict:
    return {
        "pipeline":      pipeline,
        "engine":        engine,
        "clv_model":     clv_model,
        "xai":           xai,
        "detector":      detector,
        "df":            feat_df,
        "X":             X,
        "feature_names": pipeline.get_feature_names(),
        "fi_df":         fi_df,
        "elbow":         elbow,
        "profiles":      profiles,
        "all_results":   engine.results,
        "evolution_df":  evolution_df,
        "best_algo":     engine.best_algorithm,
        "best_metrics":  engine.best_metrics,
        "n_clusters":    int(feat_df["Cluster"].nunique()),
    }


def _simulate_evolution(
    df: pd.DataFrame,
    engine,
    X: np.ndarray,
) -> pd.DataFrame:
    """
    Simulate customer cluster evolution across 3 future periods by adding
    progressive Gaussian noise to the feature matrix.
    """
    rng  = np.random.default_rng(42)
    n    = min(len(df), len(X))
    rows = []

    # Period 0 = current
    for i in range(n):
        rows.append({"CustomerIdx": i, "Period": 0, "Cluster": int(df["Cluster"].iloc[i])})

    # Periods 1-3 with increasing drift
    for period in range(1, 4):
        X_period = X[:n].copy()
        noise    = rng.normal(0, 0.08 * period, X_period.shape)
        X_period = X_period + noise
        try:
            labels_period = engine.predict(X_period)
        except Exception:
            labels_period = engine.best_labels[:n]
        for i in range(n):
            rows.append({"CustomerIdx": i, "Period": period, "Cluster": int(labels_period[i])})

    return pd.DataFrame(rows)
