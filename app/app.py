"""
app.py
Bridge Condition Predictor – Streamlit Frontend (Sprint 4)
Run: streamlit run app/app.py
"""

import os
import sys
import json
import datetime
import joblib
import pandas as pd
import streamlit as st

# Make src/ importable
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing import preprocess_single  # noqa: E402

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH    = os.path.join(BASE_DIR, "models", "final_bridge_model.pkl")
SCALER_PATH   = os.path.join(BASE_DIR, "models", "final_bridge_scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "final_bridge_features.pkl")
LOG_PATH      = os.path.join(BASE_DIR, "logs", "predictions.log")

os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bridge Condition Predictor",
    page_icon="🌉",
    layout="centered",
)

# ── Custom styling: structural blueprint theme ─────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --blueprint-navy: #0b2545;
    --blueprint-navy-deep: #07172e;
    --steel-blue: #1d4e89;
    --rivet-line: rgba(255,255,255,0.10);
    --safety-amber: #e8a93b;
    --safe-green: #3bb273;
    --unsafe-red: #d6533c;
    --paper: #f4f1ea;
}

html, body, [class*="css"] { font-family: 'Oswald', sans-serif; }

/* App background: blueprint grid over deep navy gradient */
.stApp {
    background:
        repeating-linear-gradient(0deg, var(--rivet-line) 0px, var(--rivet-line) 1px, transparent 1px, transparent 28px),
        repeating-linear-gradient(90deg, var(--rivet-line) 0px, var(--rivet-line) 1px, transparent 1px, transparent 28px),
        linear-gradient(160deg, var(--blueprint-navy-deep) 0%, var(--blueprint-navy) 55%, var(--steel-blue) 100%);
    background-attachment: fixed;
}

/* Title block styled like a structural drawing title card */
h1 {
    font-family: 'Oswald', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em;
    color: var(--paper) !important;
    text-transform: uppercase;
    border-left: 6px solid var(--safety-amber);
    padding-left: 0.6em;
}

h2, h3 { color: var(--paper) !important; font-family: 'Oswald', sans-serif !important; }

p, span, label, .stMarkdown, .stCaption { color: #d9e2ec; }

/* Card-like containers for inputs */
div[data-testid="stVerticalBlock"] > div:has(> div.stNumberInput),
div[data-testid="stVerticalBlock"] > div:has(> div.stSelectbox) {
    background: rgba(255,255,255,0.04);
}

/* Inputs themselves */
.stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
    background-color: rgba(244,241,234,0.95) !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
}

/* Primary predict button: riveted steel-plate look */
.stButton button[kind="primary"] {
    background: linear-gradient(180deg, var(--safety-amber) 0%, #c8862a 100%) !important;
    color: #1a1a1a !important;
    font-family: 'Oswald', sans-serif !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    border: none !important;
    border-radius: 4px !important;
    box-shadow: 0 3px 0 #8a5e1c, inset 0 1px 0 rgba(255,255,255,0.3);
    transition: transform 0.08s ease;
}
.stButton button[kind="primary"]:hover { transform: translateY(-1px); }
.stButton button[kind="primary"]:active { transform: translateY(2px); box-shadow: 0 1px 0 #8a5e1c; }

/* Secondary buttons (delete / clear) */
.stButton button[kind="secondary"] {
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 4px !important;
}

/* Result banners */
div[data-testid="stAlert"] {
    border-radius: 6px !important;
    font-family: 'Oswald', sans-serif;
    font-weight: 500;
    border-left: 5px solid currentColor;
}

/* Metric numbers in technical monospace */
div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--safety-amber) !important;
}
div[data-testid="stMetricLabel"] { color: #d9e2ec !important; }

/* Sidebar: rolled-steel panel look */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--blueprint-navy-deep), #0a1f3d) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] * { color: #d9e2ec !important; }

/* Log rows in the monitoring panel */
.log-row {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    padding: 6px 8px;
    margin-bottom: 4px;
    background: rgba(255,255,255,0.04);
    border-radius: 4px;
    border-left: 3px solid rgba(255,255,255,0.15);
}
.log-ts { color: #8aa0b8; }

/* Dividers as technical rule lines */
hr { border-color: rgba(255,255,255,0.12) !important; }

/* Expander headers */
details summary { color: var(--paper) !important; font-family: 'Oswald', sans-serif; }
</style>
""", unsafe_allow_html=True)


# ── Cached resource loading ───────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = joblib.load(MODEL_PATH)
    scaler   = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, scaler, features


def log_prediction(record: dict):
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_log() -> list:
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH) as f:
        return [json.loads(l) for l in f if l.strip()]


def delete_log_row(index_from_end: int):
    """Delete one entry counting from the end of the file (0 = most recent)."""
    records = read_log()
    if not records:
        return
    pos = len(records) - 1 - index_from_end
    if 0 <= pos < len(records):
        records.pop(pos)
    with open(LOG_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🌉 Bridge Condition Predictor")
st.caption(
    "Predicts whether a bridge is in **Good/Safe** or **Poor/Unsafe** condition "
    "using a tuned Random Forest model trained on historical bridge inspection data."
)

model, scaler, features = load_artifacts()

with st.expander("ℹ️ About this model"):
    st.markdown(
        "- **Algorithm:** Random Forest (tuned via RandomizedSearchCV + GridSearchCV)\n"
        "- **Features used:** " + ", ".join(features) + "\n"
        "- **Class imbalance handling:** `class_weight='balanced'`\n"
        "- **Target:** 0 = Good/Safe, 1 = Poor/Unsafe\n\n"
        "**Training data ranges** (predictions outside these ranges are "
        "extrapolations and may be unreliable):\n"
        "- Age of Bridge: 1–99 years\n"
        "- Traffic Volume: 51–4,994 vehicles/day"
    )

st.divider()

# ── User Inputs ───────────────────────────────────────────────────────────────
st.subheader("Bridge Details")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input(
        "Age of Bridge (years)", min_value=0, max_value=150, value=30, step=1,
        help="Model was trained on values roughly between 1 and 99."
    )
    material = st.selectbox("Material Type", ["Concrete", "Steel"])

with col2:
    traffic = st.number_input(
        "Daily Traffic Volume (vehicles/day)",
        min_value=0, max_value=5000, value=2500, step=50,
        help="Model was trained on values roughly between 50 and 5,000."
    )
    maintenance = st.selectbox(
        "Maintenance Level", ["Annual", "Bi-Annual", "No-Maintainance"]
    )

predict_clicked = st.button("🔍 Predict Bridge Condition", type="primary", use_container_width=True)

st.divider()

# ── Prediction & Display ─────────────────────────────────────────────────────
if predict_clicked:
    try:
        X = preprocess_single(
            age=age, traffic=traffic, material=material, maintenance=maintenance,
            scaler_path=SCALER_PATH, features_path=FEATURES_PATH,
        )
        pred  = int(model.predict(X)[0])
        proba = float(model.predict_proba(X)[0, 1])
        label = "Poor / Unsafe ⚠️" if pred == 1 else "Good / Safe ✅"

        st.subheader("Prediction Result")
        if pred == 1:
            st.error(f"**{label}**")
        else:
            st.success(f"**{label}**")

        st.metric("Probability of Poor/Unsafe Condition", f"{proba:.1%}")
        st.progress(min(max(proba, 0.0), 1.0))

        with st.expander("Show input & raw model output"):
            st.json({
                "input": {
                    "Age_of_Bridge": age, "Traffic_Volume": traffic,
                    "Material_Type": material, "Maintenance_Level": maintenance,
                },
                "prediction_code": pred,
                "probability_poor": round(proba, 4),
            })

        # ── Logging / Monitoring ──────────────────────────────────────────────
        log_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "input": {
                "Age_of_Bridge": age, "Traffic_Volume": traffic,
                "Material_Type": material, "Maintenance_Level": maintenance,
            },
            "prediction": label,
            "probability_poor": round(proba, 4),
        }
        log_prediction(log_record)

    except Exception as e:
        st.exception(e)

# ── Sidebar: Monitoring dashboard ────────────────────────────────────────────
st.sidebar.header("📊 Monitoring")
records = read_log()
if records:
    hist_df = pd.DataFrame(records)
    st.sidebar.metric("Total Predictions Logged", len(hist_df))
    poor_rate = (hist_df["prediction"].str.contains("Poor")).mean()
    st.sidebar.metric("Share Predicted Poor/Unsafe", f"{poor_rate:.1%}")

    with st.sidebar.expander("Recent predictions", expanded=True):
        recent = list(enumerate(records[::-1][:10]))  # (index_from_end, record)
        for idx, rec in recent:
            ts_short = rec["timestamp"][11:19] if len(rec["timestamp"]) > 19 else rec["timestamp"]
            badge = "🔴" if "Poor" in rec["prediction"] else "🟢"
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(
                    f"<div class='log-row'>{badge} <b>{rec['prediction'].split(' ')[0]}</b> "
                    f"&nbsp;·&nbsp; {rec['probability_poor']:.0%} risk "
                    f"&nbsp;·&nbsp; <span class='log-ts'>{ts_short}</span></div>",
                    unsafe_allow_html=True,
                )
            with c2:
                if st.button("✕", key=f"del_{idx}", help="Remove this entry", type="secondary"):
                    delete_log_row(idx)
                    st.rerun()

        st.markdown("")
        if st.button("🗑️ Clear all history", use_container_width=True, type="secondary"):
            open(LOG_PATH, "w").close()
            st.rerun()
else:
    st.sidebar.info("No predictions logged yet.")
