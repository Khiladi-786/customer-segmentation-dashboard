"""
Customer Lifetime Value Prediction using GradientBoostingRegressor.
Predicts future total spend → ranks customers by profitability.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


class CLVModel:
    """Predict Customer Lifetime Value with GradientBoosting."""

    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, random_state=42, subsample=0.8,
        )
        self.scaler = MinMaxScaler(feature_range=(0, 1000))
        self.feature_cols: list[str] = []
        self.cv_score: float = 0.0
        self.is_fitted = False
        self._train_min: float = 0.0
        self._train_max: float = 1.0

    # ── Training ────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "CLVModel":
        """
        Train on raw customer DataFrame.
        Target: TotalSpend (sum of all Mnt* columns).
        """
        df = df.copy()
        target_col = self._build_target(df)

        feature_df = self._build_features(df)
        self.feature_cols = feature_df.columns.tolist()

        X = feature_df.values
        y = df[target_col].values

        # Cross-validate
        cv = cross_val_score(self.model, X, y, cv=5, scoring="r2")
        self.cv_score = float(np.mean(cv))
        logger.info(f"CLV CV R² = {self.cv_score:.4f}")

        self.model.fit(X, y)
        self.is_fitted = True

        # Capture training range for consistent scoring later
        raw_train_preds = self.model.predict(X)
        self._train_min = float(raw_train_preds.min())
        self._train_max = float(raw_train_preds.max())
        self.scaler.fit(raw_train_preds.reshape(-1, 1))  # fit once on training preds
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return raw predicted spend values."""
        feature_df = self._build_features(df)
        for col in self.feature_cols:
            if col not in feature_df.columns:
                feature_df[col] = 0.0
        return self.model.predict(feature_df[self.feature_cols].values)

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add CLV_Predicted, CLV_Score (0-1000), CLV_Tier, Rank to df.
        Uses the scaler fitted during training for consistent 0-1000 range.
        """
        df = df.copy()
        raw_preds = self.predict(df)
        df["CLV_Predicted"] = raw_preds
        # Use transform (not fit_transform) to stay on the training scale
        df["CLV_Score"] = np.clip(
            self.scaler.transform(raw_preds.reshape(-1, 1)).flatten(), 0, 1000
        )
        df["CLV_Tier"] = pd.cut(
            df["CLV_Score"],
            bins=[-1, 200, 500, 750, 1001],
            labels=["Low", "Medium", "High", "Premium"],
        ).astype(str)
        df["CLV_Rank"] = df["CLV_Score"].rank(ascending=False).astype(int)
        return df

    def predict_single_clv(self, record: dict) -> tuple[float, str]:
        """
        Score a single customer record dict → (clv_score, clv_tier).
        Uses training-fitted scaler for correct range.
        """
        df_single = pd.DataFrame([record])
        raw = self.predict(df_single)
        score = float(np.clip(
            self.scaler.transform(raw.reshape(-1, 1)).flatten()[0], 0, 1000
        ))
        if score >= 750:
            tier = "Premium"
        elif score >= 500:
            tier = "High"
        elif score >= 200:
            tier = "Medium"
        else:
            tier = "Low"
        return score, tier

    # ── Feature Importance ──────────────────────────────────────────────────

    def feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted:
            return pd.DataFrame()
        return (
            pd.DataFrame({
                "Feature":    self.feature_cols,
                "Importance": self.model.feature_importances_,
            })
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )

    # ── Private ─────────────────────────────────────────────────────────────

    @staticmethod
    def _build_target(df: pd.DataFrame) -> str:
        mnt_cols = [c for c in df.columns if c.startswith("Mnt")]
        if mnt_cols and "TotalSpend" not in df.columns:
            df["TotalSpend"] = df[mnt_cols].sum(axis=1)
        return "TotalSpend"

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and derive predictive features."""
        mnt_cols   = [c for c in df.columns if c.startswith("Mnt")]
        purch_cols = [c for c in df.columns if c.startswith("Num") and "Purchase" in c]
        cmp_cols   = [c for c in df.columns if c.startswith("AcceptedCmp")]

        feat = pd.DataFrame(index=df.index)

        # Demographics
        if "Age" in df.columns:
            feat["Age"] = df["Age"]
        elif "Year_Birth" in df.columns:
            feat["Age"] = 2024 - df["Year_Birth"]

        if "Income" in df.columns:
            feat["Income"] = df["Income"].fillna(df["Income"].median())
        if "TotalChildren" in df.columns:
            feat["TotalChildren"] = df["TotalChildren"]

        # Behavioral
        feat["Recency"]   = df.get("Recency", 30)
        feat["Frequency"] = df[purch_cols].sum(axis=1) if purch_cols else 1
        feat["CampaignRate"] = (
            df[cmp_cols].sum(axis=1) / len(cmp_cols) if cmp_cols else 0
        )
        if "NumWebVisitsMonth" in df.columns:
            feat["WebVisits"] = df["NumWebVisitsMonth"]
        if "NumDealsPurchases" in df.columns:
            feat["DealsUsed"] = df["NumDealsPurchases"]

        # Partial spend (exclude TotalSpend to avoid leakage)
        if "MntWines" in df.columns:
            feat["MntWines"] = df["MntWines"]
        if "MntMeatProducts" in df.columns:
            feat["MntMeat"] = df["MntMeatProducts"]
        if "MntGoldProds" in df.columns:
            feat["MntGold"] = df["MntGoldProds"]
        if "CustomerTenureDays" in df.columns:
            feat["Tenure"] = df["CustomerTenureDays"]

        feat["Response"] = df.get("Response", 0)
        return feat.fillna(0)

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> str:
        save_path = path or str(MODELS_DIR / "clv_model.pkl")
        joblib.dump(self, save_path)
        logger.info(f"CLV model saved → {save_path}")
        return save_path

    @classmethod
    def load(cls, path: Optional[str] = None) -> "CLVModel":
        load_path = path or str(MODELS_DIR / "clv_model.pkl")
        return joblib.load(load_path)
