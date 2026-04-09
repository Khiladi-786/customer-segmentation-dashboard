"""
MLflow Experiment Tracker — log clustering runs and retrieve best models.
"""

from __future__ import annotations
import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", str(Path(__file__).parent.parent / "mlruns"))
EXPERIMENT_NAME = "customer_segmentation"


def _get_client():
    try:
        import mlflow
        mlflow.set_tracking_uri(TRACKING_URI)
        return mlflow
    except ImportError:
        logger.warning("MLflow not installed — tracking disabled.")
        return None


def log_run(
    algorithm: str,
    params: dict,
    metrics: dict,
    model_path: Optional[str] = None,
    tags: Optional[dict] = None,
) -> Optional[str]:
    """Log a training run to MLflow. Returns run_id or None."""
    mlflow = _get_client()
    if mlflow is None:
        return None
    try:
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name=algorithm) as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            if tags:
                mlflow.set_tags(tags)
            if model_path and Path(model_path).exists():
                mlflow.log_artifact(model_path)
            run_id = run.info.run_id
            logger.info(f"MLflow run logged: {algorithm} → {run_id}")
            return run_id
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")
        return None


def get_best_run() -> Optional[dict]:
    """Retrieve the best run (highest composite score) from MLflow."""
    mlflow = _get_client()
    if mlflow is None:
        return None
    try:
        mlflow.set_experiment(EXPERIMENT_NAME)
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            return None
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.composite DESC"],
            max_results=1,
        )
        if not runs:
            return None
        best = runs[0]
        return {
            "run_id": best.info.run_id,
            "algorithm": best.data.params.get("algorithm", "Unknown"),
            "params": best.data.params,
            "metrics": best.data.metrics,
        }
    except Exception as e:
        logger.warning(f"MLflow retrieval failed: {e}")
        return None


def get_all_runs() -> list[dict]:
    """Retrieve all logged runs sorted by composite score."""
    mlflow = _get_client()
    if mlflow is None:
        return []
    try:
        mlflow.set_experiment(EXPERIMENT_NAME)
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            return []
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.composite DESC"],
        )
        return [
            {
                "run_id": r.info.run_id,
                "algorithm": r.data.params.get("algorithm", "Unknown"),
                "params": r.data.params,
                "metrics": r.data.metrics,
                "start_time": r.info.start_time,
            }
            for r in runs
        ]
    except Exception as e:
        logger.warning(f"MLflow retrieval failed: {e}")
        return []
