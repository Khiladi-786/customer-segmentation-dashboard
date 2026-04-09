"""
Data Pipeline: Ingestion, preprocessing, and feature scaling.
Supports CSV files, API endpoints, and SQLite/PostgreSQL sources.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "new.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


class DataPipeline:
    """End-to-end data processing pipeline with persistence."""

    def __init__(self, scaler_type: str = "standard"):
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == "standard" else RobustScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_columns: list = []
        self.drop_cols: list = []
        self.is_fitted = False

    # ── Source Loading ──────────────────────────────────────────────────────

    def load_data(self, source: str = "csv", path: Optional[str] = None) -> pd.DataFrame:
        """Load data from CSV, API, or database."""
        if source == "csv":
            file_path = path or str(DATA_PATH)
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} rows from {file_path}")
            return df
        elif source == "api":
            import requests
            resp = requests.get(path, timeout=30)
            resp.raise_for_status()
            return pd.DataFrame(resp.json())
        elif source == "database":
            from database.db import engine
            return pd.read_sql("SELECT * FROM customers", engine)
        else:
            raise ValueError(f"Unknown source: {source}")

    # ── Full Pipeline ───────────────────────────────────────────────────────

    def preprocess(
        self, df: pd.DataFrame, fit: bool = True
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Full preprocessing: clean → engineer → encode → impute → scale.
        Returns (X_scaled, processed_df).
        """
        df = df.copy()
        df = self._clean_data(df)
        df = self._engineer_basic_features(df)
        df = self._encode_categoricals(df, fit=fit)

        # Select numeric features, drop ID / constant cols
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        if fit:
            self.drop_cols = [
                c for c in numeric_df.columns
                if c.upper() in ("ID", "Z_COSTCONTACT", "Z_REVENUE")
                or numeric_df[c].nunique() <= 1
            ]
        numeric_df.drop(columns=self.drop_cols, inplace=True, errors="ignore")
        self.feature_columns = list(numeric_df.columns)

        # Impute
        if fit:
            X_imputed = self.imputer.fit_transform(numeric_df)
        else:
            X_imputed = self.imputer.transform(numeric_df)

        # Remove outliers (IQR × 3) during training only
        final_df = df.copy()
        if fit:
            mask = self._iqr_mask(X_imputed, factor=3.0)
            X_imputed = X_imputed[mask]
            final_df = df.iloc[mask].reset_index(drop=True)

        # Scale
        if fit:
            X_scaled = self.scaler.fit_transform(X_imputed)
            self.is_fitted = True
        else:
            X_scaled = self.scaler.transform(X_imputed)

        return X_scaled, final_df

    def preprocess_single(self, record: dict) -> np.ndarray:
        """Transform a single customer dict into a scaled feature vector."""
        df = pd.DataFrame([record])
        df = self._clean_data(df)
        df = self._engineer_basic_features(df)
        df = self._encode_categoricals(df, fit=False)

        numeric_df = df.select_dtypes(include=[np.number]).copy()
        numeric_df.drop(columns=self.drop_cols, inplace=True, errors="ignore")

        # Align columns
        missing = set(self.feature_columns) - set(numeric_df.columns)
        for col in missing:
            numeric_df[col] = 0.0
        numeric_df = numeric_df[self.feature_columns]

        X_imputed = self.imputer.transform(numeric_df)
        return self.scaler.transform(X_imputed)

    # ── Private Helpers ─────────────────────────────────────────────────────

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df.drop_duplicates(inplace=True)
        if "Dt_Customer" in df.columns:
            df["Dt_Customer"] = pd.to_datetime(
                df["Dt_Customer"], dayfirst=True, errors="coerce"
            )
        return df

    def _engineer_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive Age, TotalChildren, CustomerTenureDays."""
        if "Year_Birth" in df.columns:
            df["Age"] = 2024 - df["Year_Birth"]
        if "Kidhome" in df.columns and "Teenhome" in df.columns:
            df["TotalChildren"] = df["Kidhome"] + df["Teenhome"]
        if "Dt_Customer" in df.columns:
            ref_date = pd.Timestamp("2014-12-31")
            df["CustomerTenureDays"] = (
                (ref_date - df["Dt_Customer"]).dt.days.clip(lower=0)
            )
        # Total spend
        mnt_cols = [c for c in df.columns if c.startswith("Mnt")]
        if mnt_cols:
            df["TotalSpend"] = df[mnt_cols].sum(axis=1)
        # Total purchases
        purch_cols = [c for c in df.columns if c.startswith("Num") and "Purchase" in c]
        if purch_cols:
            df["TotalPurchases"] = df[purch_cols].sum(axis=1)
        # Campaign acceptance rate
        cmp_cols = [c for c in df.columns if c.startswith("AcceptedCmp")]
        if cmp_cols:
            df["CampaignAcceptRate"] = df[cmp_cols].sum(axis=1) / len(cmp_cols)
        return df

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        cat_cols = [
            c for c in df.select_dtypes(include=["object", "category"]).columns
            if c != "Dt_Customer"
        ]
        for col in cat_cols:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].fillna("Unknown").astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    df[col] = df[col].fillna("Unknown").astype(str).map(
                        lambda x, _le=le: int(_le.transform([x])[0])
                        if x in _le.classes_ else -1
                    )
        return df

    @staticmethod
    def _iqr_mask(X: np.ndarray, factor: float = 3.0) -> np.ndarray:
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        lower, upper = Q1 - factor * IQR, Q3 + factor * IQR
        return np.all((X >= lower) & (X <= upper), axis=1)

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> str:
        save_path = path or str(MODELS_DIR / "pipeline.pkl")
        joblib.dump(self, save_path)
        logger.info(f"Pipeline saved → {save_path}")
        return save_path

    @classmethod
    def load(cls, path: Optional[str] = None) -> "DataPipeline":
        load_path = path or str(MODELS_DIR / "pipeline.pkl")
        return joblib.load(load_path)

    def get_feature_names(self) -> list:
        return self.feature_columns
