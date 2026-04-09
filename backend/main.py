"""
FastAPI Backend — Real-time Customer Segmentation API.

Start: uvicorn backend.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database.db import init_db
from backend.routers import predict, segments, reports

logger = logging.getLogger(__name__)

# ── App State ─────────────────────────────────────────────────────────────────
app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    logger.info("🚀 Starting Customer Segmentation API …")
    init_db()
    try:
        from core.data_pipeline  import DataPipeline
        from core.clustering     import ClusteringEngine
        from core.clv_model      import CLVModel
        from core.anomaly_detection import AnomalyDetector

        MODELS_DIR = Path(__file__).parent.parent / "models"

        pipeline = DataPipeline.load() if (MODELS_DIR / "pipeline.pkl").exists() else None
        engine   = ClusteringEngine.load() if (MODELS_DIR / "clustering_engine.pkl").exists() else None
        clv      = CLVModel.load() if (MODELS_DIR / "clv_model.pkl").exists() else None
        anomaly  = AnomalyDetector.load() if (MODELS_DIR / "anomaly_detector.pkl").exists() else None

        if pipeline and engine:
            app_state.update({
                "pipeline": pipeline,
                "engine":   engine,
                "clv":      clv,
                "anomaly":  anomaly,
                "loaded":   True,
            })
            logger.info(f"✅ Models loaded: {engine.best_algorithm} ({engine.best_metrics.get('composite', 0):.4f})")
        else:
            logger.warning("⚠️  No saved models found — run the Streamlit app first to train.")
            app_state["loaded"] = False

    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        app_state["loaded"] = False

    yield
    logger.info("🛑 API shutting down.")


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Segmentation API",
    description="Real-time AI-powered customer segmentation, CLV prediction, and anomaly detection.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(predict.router,  prefix="/api/v1", tags=["Prediction"])
app.include_router(segments.router, prefix="/api/v1", tags=["Segments"])
app.include_router(reports.router,  prefix="/api/v1", tags=["Reports"])


@app.get("/health", tags=["Health"])
async def health():
    from backend.schemas import HealthResponse
    loaded = app_state.get("loaded", False)
    engine = app_state.get("engine")
    return HealthResponse(
        status="healthy" if loaded else "degraded — models not loaded",
        algorithm=engine.best_algorithm if engine else "none",
        n_clusters=int(engine.best_labels.max() + 1) if engine and len(engine.best_labels) else 0,
        model_loaded=loaded,
    )


@app.get("/", tags=["Health"])
async def root():
    return {"message": "Customer Segmentation API v2.0 — visit /docs for API reference."}
