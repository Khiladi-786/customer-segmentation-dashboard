"""
Clustering evaluation metrics:
Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


def compute_all_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute all three cluster quality metrics.
    Returns a dict with scores and a composite 0-1 quality score.
    """
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters < 2:
        return {"silhouette": 0.0, "davies_bouldin": 99.0, "calinski_harabasz": 0.0, "composite": 0.0}

    sil   = float(silhouette_score(X, labels))
    db    = float(davies_bouldin_score(X, labels))
    ch    = float(calinski_harabasz_score(X, labels))

    # Composite score (higher = better)
    # Normalize each: sil ∈ [-1,1], db ∈ [0, ∞], ch ∈ [0, ∞]
    sil_norm = (sil + 1) / 2                          # 0→1
    db_norm  = 1 / (1 + db)                           # 0→1 (lower db = better)
    ch_norm  = min(ch / 10000, 1.0)                   # rough cap at 10k
    composite = 0.5 * sil_norm + 0.3 * db_norm + 0.2 * ch_norm

    return {
        "silhouette": round(sil, 4),
        "davies_bouldin": round(db, 4),
        "calinski_harabasz": round(ch, 2),
        "composite": round(composite, 4),
    }


def metrics_dataframe(results: list[dict]) -> pd.DataFrame:
    """
    Build a comparison DataFrame from a list of metric dicts.
    Each dict must contain 'algorithm' + metric keys.
    """
    rows = []
    for r in results:
        rows.append({
            "Algorithm": r.get("algorithm", "Unknown"),
            "Silhouette ↑": r.get("silhouette", 0),
            "Davies-Bouldin ↓": r.get("davies_bouldin", 0),
            "Calinski-Harabasz ↑": r.get("calinski_harabasz", 0),
            "Composite Score ↑": r.get("composite", 0),
            "N Clusters": r.get("n_clusters", 0),
        })
    df = pd.DataFrame(rows).sort_values("Composite Score ↑", ascending=False)
    return df.reset_index(drop=True)


def elbow_inertias(X: np.ndarray, k_range: range) -> list[float]:
    """Compute KMeans inertia for each k (used for Elbow plot)."""
    from sklearn.cluster import KMeans
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
    return inertias
