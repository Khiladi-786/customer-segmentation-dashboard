"""ORM models for persisting predictions and customer evolution history."""

from datetime import datetime

from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime

from database.db import Base


class CustomerPrediction(Base):
    __tablename__ = "customer_predictions"

    id            = Column(Integer, primary_key=True, index=True)
    customer_id   = Column(String, index=True, nullable=True)
    cluster       = Column(Integer, nullable=False)
    algorithm     = Column(String, nullable=False)
    clv_score     = Column(Float, default=0.0)
    anomaly_flag  = Column(Boolean, default=False)
    anomaly_risk  = Column(Float, default=0.0)
    rfm_segment   = Column(String, nullable=True)
    created_at    = Column(DateTime, default=datetime.utcnow)


class CustomerEvolution(Base):
    __tablename__ = "customer_evolution"

    id            = Column(Integer, primary_key=True, index=True)
    customer_id   = Column(String, index=True, nullable=False)
    from_cluster  = Column(Integer, nullable=False)
    to_cluster    = Column(Integer, nullable=False)
    period_from   = Column(String, nullable=True)
    period_to     = Column(String, nullable=True)
    recorded_at   = Column(DateTime, default=datetime.utcnow)
