"""
predict.py
Bridge Condition Predictor – Sprint 4
Loads the serialized model/scaler/features and exposes a predict() function.
Also logs every prediction to logs/predictions.log for monitoring.
"""

import os
import json
import logging
import datetime
import joblib
import pandas as pd

from preprocessing import preprocess_single

# ── Logging setup ──────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("predict")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler("logs/predictions.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(fh)

MODEL_PATH    = os.path.join(os.path.dirname(__file__), "..", "models", "final_bridge_model.pkl")
SCALER_PATH   = os.path.join(os.path.dirname(__file__), "..", "models", "final_bridge_scaler.pkl")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "final_bridge_features.pkl")

_model    = None
_scaler   = None
_features = None


def _load_artifacts():
    """Lazy-load model artifacts once (avoids re-reading disk on every call)."""
    global _model, _scaler, _features
    if _model is None:
        _model    = joblib.load(MODEL_PATH)
        _scaler   = joblib.load(SCALER_PATH)
        _features = joblib.load(FEATURES_PATH)
    return _model, _scaler, _features


def predict(age: float, traffic: float, material: str, maintenance: str) -> dict:
    """
    Predict bridge condition for a single bridge.

    Parameters
    ----------
    age         : Age_of_Bridge in years (e.g. 45)
    traffic     : Traffic_Volume, vehicles/year (e.g. 80000)
    material    : 'Concrete' or 'Steel'
    maintenance : 'Annual', 'Bi-Annual', or 'No-Maintainance'

    Returns
    -------
    dict with keys: prediction ('Good/Safe' | 'Poor/Unsafe'),
                     probability_poor (float 0-1), input echo, timestamp
    """
    model, scaler, features = _load_artifacts()

    X = preprocess_single(
        age=age, traffic=traffic, material=material, maintenance=maintenance,
        scaler_path=SCALER_PATH, features_path=FEATURES_PATH,
    )

    pred  = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0, 1])
    label = "Poor/Unsafe" if pred == 1 else "Good/Safe"

    result = {
        "prediction"      : label,
        "prediction_code" : pred,
        "probability_poor": round(proba, 4),
        "input": {
            "Age_of_Bridge"     : age,
            "Traffic_Volume"    : traffic,
            "Material_Type"     : material,
            "Maintenance_Level" : maintenance,
        },
        "timestamp": datetime.datetime.now().isoformat(),
    }

    # Monitoring: log every prediction (input + output) as one JSON line
    logger.info(json.dumps(result))

    return result


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict for a batch of bridges.

    Parameters
    ----------
    df : DataFrame with columns Age_of_Bridge, Traffic_Volume, Material_Type, Maintenance_Level

    Returns
    -------
    df with two new columns: Predicted_Condition, Probability_Poor
    """
    results = df.apply(
        lambda row: predict(
            row["Age_of_Bridge"], row["Traffic_Volume"],
            row["Material_Type"], row["Maintenance_Level"]
        ),
        axis=1,
    )
    df = df.copy()
    df["Predicted_Condition"] = results.apply(lambda r: r["prediction"])
    df["Probability_Poor"]    = results.apply(lambda r: r["probability_poor"])
    return df


if __name__ == "__main__":
    # Quick smoke-test
    out = predict(age=45, traffic=80000, material="Steel", maintenance="No-Maintainance")
    print(json.dumps(out, indent=2))

    out2 = predict(age=5, traffic=10000, material="Concrete", maintenance="Annual")
    print(json.dumps(out2, indent=2))
