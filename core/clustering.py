"""
Multi-Algorithm Clustering Engine with AutoML hyperparameter tuning.

Algorithms: KMeans | DBSCAN | Agglomerative | Gaussian Mixture Models
AutoML: Grid search → best composite score (Silhouette + DB + CH)
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from utils.metrics import compute_all_metrics, elbow_inertias

logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ── Algorithm Definitions ────────────────────────────────────────────────────

ALGORITHMS = {
    "KMeans": {
        "class": KMeans,
        "automl_params": {
            "n_clusters": [3, 4, 5, 6, 7],
            "init": ["k-means++"],
            "n_init": [10],
            "random_state": [42],
        },
    },
    "DBSCAN": {
        "class": DBSCAN,
        "automl_params": {
            "eps": [0.3, 0.5, 0.7, 1.0],
            "min_samples": [5, 10, 15],
        },
    },
    "Agglomerative": {
        "class": AgglomerativeClustering,
        "automl_params": {
            "n_clusters": [3, 4, 5, 6],
            "linkage": ["ward", "average"],
        },
    },
    "GaussianMixture": {
        "class": GaussianMixture,
        "automl_params": {
            "n_components": [3, 4, 5, 6],
            "covariance_type": ["full", "diag"],
            "random_state": [42],
        },
    },
}


# ── ClusteringEngine ─────────────────────────────────────────────────────────

class ClusteringEngine:
    """Fit, evaluate, select, and persist clustering models."""

    def __init__(self):
        self.results: list[dict] = []
        self.best_model = None
        self.best_algorithm: str = ""
        self.best_params: dict = {}
        self.best_metrics: dict = {}
        self.best_labels: np.ndarray = np.array([])
        self.pca_2d: Optional[PCA] = None
        self.pca_3d: Optional[PCA] = None

    # ── AutoML ──────────────────────────────────────────────────────────────

    def run_automl(
        self,
        X: np.ndarray,
        algorithms: Optional[list[str]] = None,
        log_mlflow: bool = True,
    ) -> dict:
        """
        Grid-search all algorithms + hyperparams, select best composite score.
        Returns the best result dict.
        """
        algos = algorithms or list(ALGORITHMS.keys())
        self.results = []

        for algo_name in algos:
            logger.info(f"AutoML: evaluating {algo_name} …")
            best_for_algo = self._search_algo(X, algo_name)
            if best_for_algo:
                self.results.append(best_for_algo)

        if not self.results:
            raise RuntimeError("No valid clustering results — check data.")

        # Pick overall best
        best = max(self.results, key=lambda r: r["metrics"]["composite"])

        self.best_model     = best["model"]
        self.best_algorithm = best["algorithm"]
        self.best_params    = best["params"]
        self.best_metrics   = best["metrics"]
        self.best_labels    = best["labels"]

        # PCA projections
        self.pca_2d = PCA(n_components=2, random_state=42)
        self.pca_2d.fit(X)
        self.pca_3d = PCA(n_components=3, random_state=42)
        self.pca_3d.fit(X)

        if log_mlflow:
            self._log_all_runs()

        logger.info(
            f"Best: {self.best_algorithm} | composite={self.best_metrics['composite']:.4f}"
        )
        return best

    def _search_algo(self, X: np.ndarray, algo_name: str) -> Optional[dict]:
        """Grid-search one algorithm, return best result or None."""
        spec   = ALGORITHMS[algo_name]
        params = spec["automl_params"]
        param_combos = self._expand_params(params)

        best_result = None
        best_score  = -999

        for combo in param_combos:
            try:
                labels = self._fit_predict(algo_name, combo, X)
                n_valid = len(set(labels) - {-1})
                if n_valid < 2:
                    continue
                metrics = compute_all_metrics(X, labels)
                if metrics["composite"] > best_score:
                    best_score  = metrics["composite"]
                    best_result = {
                        "algorithm": algo_name,
                        "params":    combo,
                        "metrics":   metrics,
                        "n_clusters": n_valid,
                        "labels":    labels,
                        "model":     self._refit(algo_name, combo, X),
                    }
            except Exception as exc:
                logger.debug(f"{algo_name} {combo}: {exc}")

        return best_result

    def _fit_predict(self, algo_name: str, params: dict, X: np.ndarray) -> np.ndarray:
        cls = ALGORITHMS[algo_name]["class"]
        model = cls(**params)
        if algo_name == "GaussianMixture":
            model.fit(X)
            return model.predict(X)
        return model.fit_predict(X)

    def _refit(self, algo_name: str, params: dict, X: np.ndarray):
        cls = ALGORITHMS[algo_name]["class"]
        model = cls(**params)
        if algo_name == "GaussianMixture":
            model.fit(X)
        else:
            model.fit(X)
        return model

    @staticmethod
    def _expand_params(param_grid: dict) -> list[dict]:
        """Cartesian product of param_grid values."""
        import itertools
        keys   = list(param_grid.keys())
        values = list(param_grid.values())
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    # ── Prediction ──────────────────────────────────────────────────────────

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Assign new data to clusters using the best model."""
        if self.best_model is None:
            raise RuntimeError("No model fitted — run run_automl() first.")
        if self.best_algorithm in ("KMeans", "Agglomerative"):
            return self.best_model.predict(X_new)
        elif self.best_algorithm == "GaussianMixture":
            return self.best_model.predict(X_new)
        else:  # DBSCAN — use nearest-centroid fallback
            return self._dbscan_predict(X_new)

    def _dbscan_predict(self, X_new: np.ndarray) -> np.ndarray:
        from sklearn.metrics.pairwise import euclidean_distances
        core_pts = self.best_model.components_
        if len(core_pts) == 0:
            return np.zeros(len(X_new), dtype=int)
        dists   = euclidean_distances(X_new, core_pts)
        nearest = dists.argmin(axis=1)
        return self.best_model.labels_[self.best_model.core_sample_indices_[nearest]]

    # ── Projections ─────────────────────────────────────────────────────────

    def project_2d(self, X: np.ndarray) -> np.ndarray:
        if self.pca_2d is None:
            self.pca_2d = PCA(n_components=2, random_state=42)
            return self.pca_2d.fit_transform(X)
        return self.pca_2d.transform(X)

    def project_3d(self, X: np.ndarray) -> np.ndarray:
        if self.pca_3d is None:
            self.pca_3d = PCA(n_components=3, random_state=42)
            return self.pca_3d.fit_transform(X)
        return self.pca_3d.transform(X)

    # ── Elbow Curve ─────────────────────────────────────────────────────────

    def elbow_data(self, X: np.ndarray, k_range: range = range(2, 11)) -> dict:
        k_vals = list(k_range)
        inertias = elbow_inertias(X, k_range)
        return {"k": k_vals, "inertia": inertias}

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> str:
        save_path = path or str(MODELS_DIR / "clustering_engine.pkl")
        joblib.dump(self, save_path)
        logger.info(f"Engine saved → {save_path}")
        return save_path

    @classmethod
    def load(cls, path: Optional[str] = None) -> "ClusteringEngine":
        load_path = path or str(MODELS_DIR / "clustering_engine.pkl")
        return joblib.load(load_path)

    # ── Cluster Profiles ────────────────────────────────────────────────────

    def cluster_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return per-cluster mean statistics for the key columns."""
        if "Cluster" not in df.columns:
            df = df.copy()
            df["Cluster"] = self.best_labels[:len(df)]
        numeric = df.select_dtypes(include=[np.number])
        return numeric.groupby("Cluster").mean().round(2)

    # ── Private: MLflow ─────────────────────────────────────────────────────

    def _log_all_runs(self):
        try:
            from utils.mlflow_tracker import log_run
            for r in self.results:
                log_run(
                    algorithm=r["algorithm"],
                    params={**r["params"], "algorithm": r["algorithm"]},
                    metrics={**r["metrics"], "n_clusters": r["n_clusters"]},
                )
        except Exception as e:
            logger.warning(f"MLflow batch log failed: {e}")
