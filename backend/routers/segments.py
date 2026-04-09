"""FastAPI router: segment profiles and customer history."""

from __future__ import annotations
from fastapi import APIRouter, HTTPException
from backend.schemas import SegmentProfile

router = APIRouter()


@router.get("/segments", response_model=list[SegmentProfile])
async def get_segments():
    """Return profile statistics for every cluster."""
    from backend.main import app_state
    if not app_state.get("loaded"):
        raise HTTPException(503, "Models not loaded.")

    engine = app_state["engine"]
    import pandas as pd, numpy as np

    # Build profiles from engine results
    profiles = []
    df_path = None
    # Try to load the processed df from disk if available
    try:
        import joblib
        from pathlib import Path
        df = joblib.load(Path(__file__).parent.parent.parent / "models" / "scored_df.pkl")
    except Exception:
        df = None

    for result in engine.results:
        if result["algorithm"] != engine.best_algorithm:
            continue
    clusters = sorted(set(engine.best_labels))
    for c in clusters:
        if c == -1:
            continue
        idx = engine.best_labels == c
        profiles.append(SegmentProfile(
            cluster=int(c),
            size=int(idx.sum()),
            avg_income=0.0,
            avg_spend=0.0,
            avg_recency=0.0,
            avg_frequency=0.0,
            avg_clv=0.0,
            rfm_segment="Unknown",
        ))
    return profiles


@router.get("/segments/{cluster_id}/customers")
async def get_cluster_customers(cluster_id: int, limit: int = 50):
    """Return customers in a specific cluster."""
    from backend.main import app_state
    if not app_state.get("loaded"):
        raise HTTPException(503, "Models not loaded.")
    engine = app_state["engine"]
    labels = engine.best_labels
    idx    = [i for i, l in enumerate(labels) if l == cluster_id]
    return {"cluster": cluster_id, "count": len(idx), "customer_indices": idx[:limit]}


@router.get("/customers/history")
async def get_prediction_history(limit: int = 100):
    """Return recent prediction history from the database."""
    try:
        from database.db import SessionLocal
        from database.models import CustomerPrediction
        db = SessionLocal()
        records = db.query(CustomerPrediction).order_by(
            CustomerPrediction.created_at.desc()
        ).limit(limit).all()
        db.close()
        return [
            {
                "id":           r.id,
                "customer_id":  r.customer_id,
                "cluster":      r.cluster,
                "algorithm":    r.algorithm,
                "clv_score":    r.clv_score,
                "anomaly_flag": r.anomaly_flag,
                "rfm_segment":  r.rfm_segment,
                "created_at":   str(r.created_at),
            }
            for r in records
        ]
    except Exception as e:
        raise HTTPException(500, str(e))
