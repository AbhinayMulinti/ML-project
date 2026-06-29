"""
preprocessing.py
Bridge Condition Predictor – Sprint 4
Handles all data transformation steps: encoding, feature engineering, scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────────
NUMERICAL_COLS     = ["Age_of_Bridge", "Traffic_Volume"]
CATEGORICAL_COLS   = ["Material_Type", "Maintenance_Level"]
TARGET_COL         = "Bridge_Condition"

SCALE_COLS = [
    "Age_of_Bridge", "Traffic_Volume",
    "Traffic_per_Year", "Age_Squared",
    "Age_Bucket", "Concrete_Age", "Steel_Traffic", "Neglect_Score",
]

VALID_MATERIALS    = ["Concrete", "Steel"]
VALID_MAINTENANCE  = ["Annual", "Bi-Annual", "No-Maintainance"]


# ── Step 1: Encode categoricals ────────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode Material_Type and Maintenance_Level."""
    df = df.copy()
    df = pd.get_dummies(df, columns=["Material_Type"], prefix="Material_Type")
    df = pd.get_dummies(df, columns=["Maintenance_Level"], prefix="Maintenance_Level")

    # Ensure all expected dummy columns exist (handles unseen categories at inference)
    expected_dummies = [
        "Material_Type_Concrete", "Material_Type_Steel",
        "Maintenance_Level_Annual", "Maintenance_Level_Bi-Annual",
        "Maintenance_Level_No-Maintainance",
    ]
    for col in expected_dummies:
        if col not in df.columns:
            df[col] = 0

    logger.info("Categorical encoding complete.")
    return df


# ── Step 2: Feature engineering ────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-driven features from base columns."""
    df = df.copy()

    df["Traffic_per_Year"] = df["Traffic_Volume"] / (df["Age_of_Bridge"] + 1)
    df["Age_Squared"]      = df["Age_of_Bridge"] ** 2

    df["Age_Bucket"] = pd.cut(
        df["Age_of_Bridge"],
        bins=[0, 25, 60, 100],
        labels=[0, 1, 2],
    ).astype(int)

    age_thresh     = df["Age_of_Bridge"].median()
    traffic_thresh = df["Traffic_Volume"].median()
    df["High_Stress"] = (
        (df["Age_of_Bridge"] > age_thresh) &
        (df["Traffic_Volume"] > traffic_thresh)
    ).astype(int)

    if "Material_Type_Concrete" in df.columns:
        df["Concrete_Age"]  = df["Material_Type_Concrete"] * df["Age_of_Bridge"]
        df["Steel_Traffic"] = df["Material_Type_Steel"]    * df["Traffic_Volume"]
    else:
        df["Concrete_Age"]  = 0
        df["Steel_Traffic"] = 0

    no_maint_col = "Maintenance_Level_No-Maintainance"
    if no_maint_col in df.columns:
        df["Neglect_Score"] = df[no_maint_col] * df["Age_of_Bridge"]
    else:
        df["Neglect_Score"] = 0

    logger.info("Feature engineering complete. New cols: Traffic_per_Year, Age_Squared, "
                "Age_Bucket, High_Stress, Concrete_Age, Steel_Traffic, Neglect_Score")
    return df


# ── Step 3: Scale numerical features ──────────────────────────────────────────
def scale_features(df: pd.DataFrame, scaler: StandardScaler,
                   fit: bool = False) -> pd.DataFrame:
    """Scale numerical columns with the provided StandardScaler."""
    df = df.copy()
    cols_present = [c for c in SCALE_COLS if c in df.columns]

    if fit:
        df[cols_present] = scaler.fit_transform(df[cols_present])
        logger.info("Scaler fitted and applied.")
    else:
        df[cols_present] = scaler.transform(df[cols_present])
        logger.info("Scaler applied (transform-only).")
    return df


# ── Full pipeline ──────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame, scaler: StandardScaler = None,
               fit_scaler: bool = False):
    """
    Run the complete preprocessing pipeline.

    Parameters
    ----------
    df         : raw DataFrame (must contain Age_of_Bridge, Traffic_Volume,
                 Material_Type, Maintenance_Level)
    scaler     : an existing StandardScaler (pass None to create a new one)
    fit_scaler : if True, fit the scaler on this data (training mode)

    Returns
    -------
    df_processed, scaler
    """
    if scaler is None:
        scaler = StandardScaler()
        fit_scaler = True

    df = encode_categoricals(df)
    df = engineer_features(df)
    df = scale_features(df, scaler, fit=fit_scaler)
    return df, scaler


# ── Inference helper ───────────────────────────────────────────────────────────
def preprocess_single(age: float, traffic: float,
                      material: str, maintenance: str,
                      scaler_path: str = "models/final_bridge_scaler.pkl",
                      features_path: str = "models/final_bridge_features.pkl") -> pd.DataFrame:
    """
    Preprocess a single bridge record for real-time inference.

    Parameters
    ----------
    age         : Age of bridge in years
    traffic     : Annual traffic volume
    material    : 'Concrete' or 'Steel'
    maintenance : 'Annual', 'Bi-Annual', or 'No-Maintainance'

    Returns
    -------
    DataFrame ready for model.predict()
    """
    if material not in VALID_MATERIALS:
        raise ValueError(f"material must be one of {VALID_MATERIALS}")
    if maintenance not in VALID_MAINTENANCE:
        raise ValueError(f"maintenance must be one of {VALID_MAINTENANCE}")

    row = pd.DataFrame([{
        "Age_of_Bridge"    : age,
        "Traffic_Volume"   : traffic,
        "Material_Type"    : material,
        "Maintenance_Level": maintenance,
    }])

    scaler   = joblib.load(scaler_path)
    features = joblib.load(features_path)

    row, _ = preprocess(row, scaler=scaler, fit_scaler=False)
    row = row[features]   # select only the model's expected features

    logger.info(f"Single record preprocessed. Features: {features}")
    return row


if __name__ == "__main__":
    # Quick smoke-test
    sample = pd.DataFrame([{
        "Age_of_Bridge": 45, "Traffic_Volume": 80000,
        "Material_Type": "Steel", "Maintenance_Level": "No-Maintainance",
    }])
    scaler   = joblib.load("models/final_bridge_scaler.pkl")
    features = joblib.load("models/final_bridge_features.pkl")
    out, _   = preprocess(sample, scaler=scaler, fit_scaler=False)
    print(out[features])
