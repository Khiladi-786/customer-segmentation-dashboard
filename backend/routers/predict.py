"""FastAPI router: real-time prediction endpoints."""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from sqlalchemy.orm import Session

from backend.schemas import CustomerInput, PredictionOutput

router = APIRouter()
logger = logging.getLogger(__name__)


def _get_state(request: Request) -> dict:
    return request.app.extra.get("state", {}) or getattr(request.app, "state_data", {})


def _build_customer_dict(inp: CustomerInput) -> dict:
    return {
        "Year_Birth":          inp.year_birth,
        "Education":           inp.education,
        "Marital_Status":      inp.marital_status,
        "Income":              inp.income,
        "Kidhome":             inp.kidhome,
        "Teenhome":            inp.teenhome,
        "Recency":             inp.recency,
        "MntWines":            inp.mnt_wines,
        "MntFruits":           inp.mnt_fruits,
        "MntMeatProducts":     inp.mnt_meat,
        "MntFishProducts":     inp.mnt_fish,
        "MntSweetProducts":    inp.mnt_sweets,
        "MntGoldProds":        inp.mnt_gold,
        "NumDealsPurchases":   inp.num_deals,
        "NumWebPurchases":     inp.num_web,
        "NumCatalogPurchases": inp.num_catalog,
        "NumStorePurchases":   inp.num_store,
        "NumWebVisitsMonth":   inp.num_web_visits,
        "AcceptedCmp1": 0, "AcceptedCmp2": 0, "AcceptedCmp3": 0,
        "AcceptedCmp4": 0, "AcceptedCmp5": 0,
        "Complain": 0, "Z_CostContact": 3, "Z_Revenue": 11,
        "Response": inp.response,
    }


@router.post("/predict", response_model=PredictionOutput)
async def predict_customer(inp: CustomerInput, request: Request):
    """Predict segment, CLV, and anomaly status for a new customer in real-time."""
    from backend.main import app_state

    if not app_state.get("loaded"):
        raise HTTPException(503, "Models not loaded — run Streamlit app first to train.")

    pipeline = app_state["pipeline"]
    engine   = app_state["engine"]
    clv      = app_state.get("clv")
    anomaly  = app_state.get("anomaly")

    try:
        cust_dict = _build_customer_dict(inp)
        X_new     = pipeline.preprocess_single(cust_dict)

        cluster = int(engine.predict(X_new)[0])

        clv_score = 0.0
        clv_tier  = "Unknown"
        if clv and clv.is_fitted:
            import pandas as pd
            df_single = pd.DataFrame([cust_dict])
            df_scored = clv.score_dataframe(df_single)
            clv_score = float(df_scored["CLV_Score"].iloc[0])
            clv_tier  = str(df_scored["CLV_Tier"].iloc[0])

        anom_flag = False
        anom_risk = 0.0
        if anomaly and anomaly.is_fitted:
            pred_a    = anomaly.predict(X_new)
            anom_flag = bool(pred_a[0] == -1)
            score     = float(anomaly.anomaly_scores(X_new)[0])
            anom_risk = round(max(0, min(100, 100 * (1 - (score + 1) / 2))), 1)

        # RFM segment estimate
        total_spend = (
            inp.mnt_wines + inp.mnt_fruits + inp.mnt_meat +
            inp.mnt_fish + inp.mnt_sweets + inp.mnt_gold
        )
        frequency = inp.num_web + inp.num_catalog + inp.num_store
        rfm_score = (
            (5 if inp.recency < 30 else 3 if inp.recency < 60 else 1) +
            (5 if frequency >= 15 else 3 if frequency >= 8 else 1) +
            (5 if total_spend >= 1000 else 3 if total_spend >= 400 else 1)
        )
        rfm_seg = "Champions" if rfm_score >= 13 else "Loyal Customers" if rfm_score >= 10 else "Potential Loyalists" if rfm_score >= 7 else "At Risk" if rfm_score >= 5 else "Lost / Inactive"

        # Persist prediction
        try:
            from database.db import SessionLocal
            from database.models import CustomerPrediction
            db = SessionLocal()
            db.add(CustomerPrediction(
                customer_id=inp.customer_id,
                cluster=cluster,
                algorithm=engine.best_algorithm,
                clv_score=clv_score,
                anomaly_flag=anom_flag,
                anomaly_risk=anom_risk,
                rfm_segment=rfm_seg,
            ))
            db.commit()
            db.close()
        except Exception as e:
            logger.warning(f"DB persist failed: {e}")

        return PredictionOutput(
            customer_id=inp.customer_id,
            cluster=cluster,
            algorithm=engine.best_algorithm,
            clv_score=round(clv_score, 2),
            clv_tier=clv_tier,
            anomaly_flag=anom_flag,
            anomaly_risk=anom_risk,
            rfm_segment=rfm_seg,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(500, f"Prediction failed: {str(e)}")
