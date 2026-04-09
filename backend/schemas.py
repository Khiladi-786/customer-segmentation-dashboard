"""Pydantic schemas for the FastAPI backend."""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    customer_id:    Optional[str]   = Field(None, description="Optional customer ID for tracking")
    year_birth:     int             = Field(1980, ge=1900, le=2010)
    education:      str             = Field("Graduation")
    marital_status: str             = Field("Single")
    income:         float           = Field(50000.0, ge=0)
    kidhome:        int             = Field(0, ge=0, le=3)
    teenhome:       int             = Field(0, ge=0, le=3)
    recency:        int             = Field(30, ge=0, le=365)
    mnt_wines:      float           = Field(0.0, ge=0)
    mnt_fruits:     float           = Field(0.0, ge=0)
    mnt_meat:       float           = Field(0.0, ge=0)
    mnt_fish:       float           = Field(0.0, ge=0)
    mnt_sweets:     float           = Field(0.0, ge=0)
    mnt_gold:       float           = Field(0.0, ge=0)
    num_deals:      int             = Field(0, ge=0)
    num_web:        int             = Field(0, ge=0)
    num_catalog:    int             = Field(0, ge=0)
    num_store:      int             = Field(0, ge=0)
    num_web_visits: int             = Field(0, ge=0)
    response:       int             = Field(0, ge=0, le=1)


class PredictionOutput(BaseModel):
    customer_id:   Optional[str]
    cluster:       int
    algorithm:     str
    clv_score:     float
    clv_tier:      str
    anomaly_flag:  bool
    anomaly_risk:  float
    rfm_segment:   str
    confidence:    Optional[float] = None


class SegmentProfile(BaseModel):
    cluster:        int
    size:           int
    avg_income:     float
    avg_spend:      float
    avg_recency:    float
    avg_frequency:  float
    avg_clv:        float
    rfm_segment:    str


class HealthResponse(BaseModel):
    status: str
    algorithm: str
    n_clusters: int
    model_loaded: bool
