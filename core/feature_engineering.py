"""
Advanced Feature Engineering:
- RFM Analysis (Recency, Frequency, Monetary with 1-5 scoring)
- Customer Lifetime Value proxy
- Behavioral engagement features
- Feature importance via Random Forest surrogate
"""

from __future__ import annotations
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


# ── RFM Analysis ─────────────────────────────────────────────────────────────

def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RFM scores from the existing dataset columns.

    Recency  → days since last purchase (column already exists)
    Frequency → sum of all purchase-channel counts
    Monetary  → sum of all Mnt* spend columns
    """
    df = df.copy()

    # Monetary
    mnt_cols = [c for c in df.columns if c.startswith("Mnt")]
    df["Monetary"] = df[mnt_cols].sum(axis=1) if mnt_cols else 0.0

    # Frequency
    purch_cols = [c for c in df.columns if c.startswith("Num") and "Purchase" in c]
    df["Frequency"] = df[purch_cols].sum(axis=1) if purch_cols else 1.0

    # R Score (lower recency = better → s 5)
    try:
        df["R_Score"] = pd.qcut(
            df["Recency"], 5, labels=[5, 4, 3, 2, 1], duplicates="drop"
        ).astype(float)
    except Exception:
        df["R_Score"] = 3.0

    # F Score
    try:
        df["F_Score"] = pd.qcut(
            df["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5], duplicates="drop"
        ).astype(float)
    except Exception:
        df["F_Score"] = 3.0

    # M Score
    try:
        df["M_Score"] = pd.qcut(
            df["Monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5], duplicates="drop"
        ).astype(float)
    except Exception:
        df["M_Score"] = 3.0

    df["RFM_Score"] = df["R_Score"] + df["F_Score"] + df["M_Score"]

    # Human-readable segment based on RFM_Score
    def _rfm_segment(score: float) -> str:
        if score >= 13:
            return "Champions"
        elif score >= 10:
            return "Loyal Customers"
        elif score >= 7:
            return "Potential Loyalists"
        elif score >= 5:
            return "At Risk"
        else:
            return "Lost / Inactive"

    df["RFM_Segment"] = df["RFM_Score"].apply(_rfm_segment)
    return df


# ── Behavioral Features ───────────────────────────────────────────────────────

def compute_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive advanced behavioral and engagement features."""
    df = df.copy()

    # Campaign acceptance rate
    cmp_cols = [c for c in df.columns if c.startswith("AcceptedCmp")]
    if cmp_cols:
        df["CampaignAcceptRate"] = df[cmp_cols].sum(axis=1) / len(cmp_cols)

    # Web engagement score
    if "NumWebPurchases" in df.columns and "NumWebVisitsMonth" in df.columns:
        web_max = df["NumWebVisitsMonth"].max() or 1
        df["WebEngagementScore"] = (
            df["NumWebPurchases"] * 2 + df["NumWebVisitsMonth"]
        ) / (web_max * 3 + 1)

    # Spend diversity (how many categories a customer spends in)
    mnt_cols = [c for c in df.columns if c.startswith("Mnt")]
    if mnt_cols:
        df["SpendDiversity"] = (df[mnt_cols] > 0).sum(axis=1)

    # Premium product ratio
    if "MntGoldProds" in df.columns and "Monetary" in df.columns:
        df["PremiumRatio"] = df["MntGoldProds"] / (df["Monetary"] + 1)

    # Deal sensitivity
    if "NumDealsPurchases" in df.columns and "Frequency" in df.columns:
        df["DealSensitivity"] = df["NumDealsPurchases"] / (df["Frequency"] + 1)

    # Online ratio (web vs. catalog+store)
    if "NumWebPurchases" in df.columns:
        total = df.get("TotalPurchases", 1)
        df["OnlineRatio"] = df["NumWebPurchases"] / (total + 1)

    return df


# ── CLV Proxy ─────────────────────────────────────────────────────────────────

def compute_clv_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a Customer Lifetime Value proxy score (0–1000)."""
    df = df.copy()

    mnt_cols = [c for c in df.columns if c.startswith("Mnt")]
    df["TotalSpend"] = df[mnt_cols].sum(axis=1) if mnt_cols else df.get("Monetary", 0)

    freq = df.get("Frequency", 1).replace(0, 1)
    recency_weight = 1 / (1 + df.get("Recency", 30) / 30.0)
    tenure = df.get("CustomerTenureDays", 365).replace(0, 1) / 365.0

    df["CLV_Proxy"] = df["TotalSpend"] * freq * recency_weight * (1 + tenure * 0.2)

    scaler = MinMaxScaler(feature_range=(0, 1000))
    df["CLV_Score"] = scaler.fit_transform(df[["CLV_Proxy"]]).flatten()

    df["CLV_Tier"] = pd.cut(
        df["CLV_Score"],
        bins=[-1, 200, 500, 750, 1001],
        labels=["Low", "Medium", "High", "Premium"],
    )
    return df


# ── Feature Importance ────────────────────────────────────────────────────────

def get_feature_importance(
    X: np.ndarray, labels: np.ndarray, feature_names: list
) -> pd.DataFrame:
    """
    Compute feature importance using a Random Forest surrogate trained
    on cluster labels (standard approach for unsupervised XAI).
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, labels)
    return (
        pd.DataFrame({"Feature": feature_names, "Importance": rf.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def full_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the complete feature engineering pipeline."""
    df = compute_rfm(df)
    df = compute_behavioral_features(df)
    df = compute_clv_proxy(df)
    return df
