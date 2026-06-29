"""
train.py
Bridge Condition Predictor – Sprint 4
End-to-end training pipeline: load → preprocess → tune → evaluate → save.
Run: python src/train.py
"""

import os
import json
import logging
import datetime
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)
from preprocessing import preprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_PATH       = "data/bridge_data.csv"
MODEL_OUT       = "models/final_bridge_model.pkl"
SCALER_OUT      = "models/final_bridge_scaler.pkl"
FEATURES_OUT    = "models/final_bridge_features.pkl"
METRICS_OUT     = "logs/metrics.json"
TARGET          = "Bridge_Condition"

SELECTED_FEATURES = [
    "Age_of_Bridge", "Traffic_Volume", "Traffic_per_Year",
    "High_Stress", "Concrete_Age", "Steel_Traffic", "Neglect_Score",
]

BEST_PARAMS = {
    "n_estimators"     : 150,
    "max_depth"        : None,
    "max_features"     : "log2",
    "min_samples_split": 2,
    "min_samples_leaf" : 2,
    "class_weight"     : "balanced",
    "random_state"     : 42,
}


def load_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {df.shape[0]} rows × {df.shape[1]} cols")
    return df


def split_data(df: pd.DataFrame):
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def evaluate(model, X_test, y_test, label="Test") -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy" : round(accuracy_score(y_test, y_pred),           4),
        "precision": round(precision_score(y_test, y_pred,           zero_division=0), 4),
        "recall"   : round(recall_score(y_test, y_pred,              zero_division=0), 4),
        "f1"       : round(f1_score(y_test, y_pred,                  zero_division=0), 4),
        "roc_auc"  : round(roc_auc_score(y_test, y_proba),           4),
    }
    logger.info(f"[{label}] " + " | ".join(f"{k}={v}" for k, v in metrics.items()))
    print(classification_report(y_test, y_pred,
                                 target_names=["Good (0)", "Poor (1)"],
                                 zero_division=0))
    return metrics


def train():
    logger.info("=" * 60)
    logger.info("Bridge Condition Predictor — Training Pipeline")
    logger.info("=" * 60)

    # ── 1. Load ────────────────────────────────────────────────────────────────
    raw = load_data(DATA_PATH)

    # Sprint 1 columns: raw CSV has original categories
    # Sprint 2/3 data: already encoded — detect and branch
    if "Material_Type" in raw.columns:
        # Raw data — needs full preprocessing
        X_train_raw, X_test_raw, y_train, y_test = split_data(raw)

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

        X_train_pp, scaler = preprocess(X_train_raw, scaler=scaler, fit_scaler=True)
        X_test_pp,  _      = preprocess(X_test_raw,  scaler=scaler, fit_scaler=False)
    else:
        # Already-encoded data (from Sprint 3 output)
        from sklearn.preprocessing import StandardScaler
        from preprocessing import engineer_features, scale_features
        scaler = StandardScaler()
        X_train_pp, X_test_pp, y_train, y_test = split_data(raw)
        # engineer + scale
        X_train_pp = engineer_features(X_train_pp)
        X_test_pp  = engineer_features(X_test_pp)
        X_train_pp = scale_features(X_train_pp, scaler, fit=True)
        X_test_pp  = scale_features(X_test_pp,  scaler, fit=False)

    # ── 2. Select features ─────────────────────────────────────────────────────
    feats_present = [f for f in SELECTED_FEATURES if f in X_train_pp.columns]
    X_train = X_train_pp[feats_present]
    X_test  = X_test_pp[feats_present]
    logger.info(f"Using features: {feats_present}")

    # ── 3. Train ───────────────────────────────────────────────────────────────
    logger.info("Training Random Forest with best hyperparameters …")
    model = RandomForestClassifier(**BEST_PARAMS)
    model.fit(X_train, y_train)

    cv_f1 = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
    logger.info(f"CV F1 (5-fold): {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    # ── 4. Evaluate ────────────────────────────────────────────────────────────
    metrics = evaluate(model, X_test, y_test)
    metrics["cv_f1_mean"] = round(cv_f1.mean(), 4)
    metrics["cv_f1_std"]  = round(cv_f1.std(),  4)
    metrics["timestamp"]  = datetime.datetime.now().isoformat()
    metrics["params"]     = BEST_PARAMS

    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved → {METRICS_OUT}")

    # ── 5. Serialize ───────────────────────────────────────────────────────────
    joblib.dump(model,         MODEL_OUT)
    joblib.dump(scaler,        SCALER_OUT)
    joblib.dump(feats_present, FEATURES_OUT)
    logger.info(f"Model saved   → {MODEL_OUT}")
    logger.info(f"Scaler saved  → {SCALER_OUT}")
    logger.info(f"Features saved→ {FEATURES_OUT}")
    logger.info("Training complete.")
    return model, scaler, feats_present, metrics


if __name__ == "__main__":
    train()
