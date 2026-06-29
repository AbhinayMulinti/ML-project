# 🌉 Bridge Condition Predictor

A machine learning project that predicts whether a bridge is in **Good/Safe**
or **Poor/Unsafe** structural condition based on its age, traffic load,
material, and maintenance history. Built across four sprints — from raw data
to a deployed, monitored Streamlit application.

---

## 1. Problem Statement

Civil infrastructure agencies need to prioritize inspection and maintenance
budgets across thousands of bridges. Manual inspection is slow and expensive,
and deteriorating bridges that are missed pose serious safety risks.

**Goal:** build a classification model that flags bridges likely to be in
**Poor/Unsafe** condition using easily available structural and operational
data (age, traffic volume, material, maintenance level), so inspectors can
prioritize their attention.

This is a **binary classification** problem with a **class-imbalanced
target** (~83% Good, ~17% Poor) — missing a "Poor" bridge (a false negative)
is far more costly than a false alarm, so the project optimizes for
**recall / F1 on the minority class**, not raw accuracy.

---

## 2. Approach

The project was built over four sprints:

### Sprint 1 — Data Collection & EDA
- Loaded `bridge_data.csv` (720 rows → 592 after removing 128 duplicates).
- No missing values. 5 columns: `Age_of_Bridge`, `Traffic_Volume`,
  `Material_Type`, `Maintenance_Level`, `Bridge_Condition`.
- Univariate/bivariate/multivariate EDA revealed: target imbalance, right
  skew in bridge age, and visibly higher "Poor" rates among Steel bridges and
  Bi-Annual maintenance bridges.

### Sprint 2 — Baseline & Model Comparison
- Established a Logistic Regression baseline.
- Trained and compared 6 models: Logistic Regression, Decision Tree, Random
  Forest, SVM, Naive Bayes, Gradient Boosting.
- Evaluated via accuracy, precision, recall, F1, and confusion matrices, and
  checked train-vs-test accuracy gaps for over/underfitting.
- **Random Forest and Gradient Boosting** captured the minority ("Poor")
  class best.

### Sprint 3 — Optimization & Final Model
- **Feature engineering:** `Traffic_per_Year`, `Age_Squared`, `Age_Bucket`,
  `High_Stress` (old + high traffic flag), `Concrete_Age`, `Steel_Traffic`,
  `Neglect_Score` (no-maintenance × age).
- **Feature selection:** correlation analysis, Random Forest importance, and
  RFE narrowed the feature set to 7: `Age_of_Bridge`, `Traffic_Volume`,
  `Traffic_per_Year`, `High_Stress`, `Concrete_Age`, `Steel_Traffic`,
  `Neglect_Score`.
- **Hyperparameter tuning:** RandomizedSearchCV (wide sweep, 30 iterations)
  followed by GridSearchCV (fine-tuning), optimizing for F1 with 5-fold CV.
  `class_weight='balanced'` was used throughout to address class imbalance.
- **Final model:** tuned Random Forest, serialized as `final_bridge_model.pkl`
  along with its `StandardScaler` and selected feature list.

### Sprint 4 — Deployment & MLOps (this sprint)
- Wrapped preprocessing + inference into a reusable pipeline (`src/`).
- Built an interactive **Streamlit** UI for real-time predictions.
- Added structured **logging** of every prediction for monitoring.
- Organized the project for **version control** (Git/GitHub-ready) and
  reproducible training (`train.py` regenerates model + metrics).

---

## 3. Results

Final tuned Random Forest, evaluated on a held-out 20% test set
(119 bridges):

| Metric              | Score  |
|---------------------|--------|
| Accuracy            | 0.857  |
| Precision (Poor)    | 0.556  |
| Recall (Poor)       | 0.750  |
| F1 Score (Poor)     | 0.638  |
| ROC-AUC             | 0.919  |
| CV F1 (5-fold)      | 0.704 ± 0.049 |

*(Exact numbers will vary slightly run-to-run if you retrain via `train.py`;
the values above are from the reproducible pipeline run, see
`logs/metrics.json`. The originally delivered `models/*.pkl` files are the
exact Sprint-3 artifacts and are what the Streamlit app uses by default.)*

**Key insights:**
- `Age_of_Bridge` is the single strongest predictor — older bridges are
  significantly more likely to be Poor/Unsafe.
- `Traffic_Volume` and the engineered `Traffic_per_Year` show that
  high-traffic bridges deteriorate faster.
- `Neglect_Score` (no maintenance × age) is a strong risk multiplier —
  unmaintained old bridges are the highest-risk group.
- Random Forest outperformed Logistic Regression, SVM, and Naive Bayes by
  capturing non-linear interactions between age, traffic, and maintenance.

---

## 4. Project Structure

```
bridge_condition_predictor/
├── data/
│   └── bridge_data.csv              # Cleaned, encoded dataset (592 × 8)
├── notebooks/
│   ├── Sprint-1_EDA.ipynb           # Data collection, cleaning, EDA
│   ├── Sprint-2_Modeling.ipynb      # Baseline + multi-model comparison
│   └── Sprint-3_Optimization.ipynb  # Feature engineering, tuning, final model
├── src/
│   ├── preprocessing.py             # Encoding, feature engineering, scaling
│   ├── train.py                     # End-to-end training pipeline
│   └── predict.py                   # Inference + prediction logging
├── models/
│   ├── final_bridge_model.pkl       # Tuned Random Forest
│   ├── final_bridge_scaler.pkl      # Fitted StandardScaler
│   └── final_bridge_features.pkl    # Selected feature list (order matters)
├── app/
│   └── app.py                       # Streamlit frontend
├── logs/
│   ├── metrics.json                 # Latest training run metrics
│   ├── training.log                 # Training run logs (generated on run)
│   └── predictions.log              # Every prediction made via the app (generated on run)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 5. How to Run

### 5.1 Setup

```bash
git clone <your-repo-url>
cd bridge_condition_predictor
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 5.2 Retrain the model (optional)

The repo ships with trained artifacts in `models/`, so this step is optional.
To retrain from scratch using the pipeline:

```bash
cd src
python train.py
```

This reads `data/bridge_data.csv`, runs the full preprocessing →
feature-engineering → training pipeline, evaluates on a held-out test split,
and overwrites `models/*.pkl` plus `logs/metrics.json`.

### 5.3 Run a single prediction from the command line

```bash
cd src
python predict.py
```

Or import it directly:

```python
from predict import predict
result = predict(age=45, traffic=80000, material="Steel", maintenance="No-Maintainance")
print(result)
```

### 5.4 Launch the Streamlit app

```bash
streamlit run app/app.py
```

Open the URL Streamlit prints (typically `http://localhost:8501`). Enter
bridge details in the form, click **Predict Bridge Condition**, and view the
predicted label, probability, and a live monitoring panel in the sidebar
showing prediction history.

### 5.5 Deploy

**Local:** the `streamlit run` command above is a fully working local
deployment.

**Cloud (Streamlit Community Cloud):**
1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect your
   GitHub account, and select this repo.
3. Set the main file path to `app/app.py`.
4. Deploy — Streamlit Cloud installs `requirements.txt` automatically.

**Other cloud options:** the same `app/app.py` + `requirements.txt` can be
containerized (Dockerfile) and deployed to Render, Railway, AWS
Elastic Beanstalk, or Azure App Service with minimal changes.

---

## 6. MLOps Practices

- **Version control:** project structured for Git; `.gitignore` excludes
  logs, virtual environments, and cache files so only code, notebooks,
  data, and model artifacts are tracked.
- **Experiment tracking:** `train.py` writes every run's hyperparameters and
  metrics (accuracy, precision, recall, F1, ROC-AUC, CV scores) to
  `logs/metrics.json` with a timestamp, giving a lightweight audit trail of
  model performance over time.
- **Logging & monitoring:** `predict.py` and the Streamlit app log every
  prediction (input features, predicted class, probability, timestamp) as a
  JSON line in `logs/predictions.log`. The Streamlit sidebar surfaces this
  as a live monitoring dashboard — total predictions served, the share
  flagged Poor/Unsafe, and a "Recent predictions" list where any single
  entry can be removed (✕) or the whole history cleared in one click — useful
  for spotting **prediction drift** over time (e.g., if the proportion of
  "Poor" predictions suddenly spikes, that may signal a shift in the
  population of bridges being queried, warranting model retraining).

---

## 7. Notes / Limitations

- **Input ranges matter.** The model was trained on `Traffic_Volume` values
  between **51 and 4,994** (likely vehicles/day, not vehicles/year) and
  `Age_of_Bridge` between **1 and 99 years**. Feeding values far outside
  these ranges (e.g. traffic = 200,000) puts the input far outside anything
  the Random Forest's trees ever split on, so predictions stop being
  meaningful and tend to collapse toward "Good/Safe" almost regardless of
  the other inputs. The Streamlit app's input widgets are capped to the
  trained ranges to prevent this.
- The dataset is small (592 rows after deduplication), so metrics have
  meaningful variance across random seeds/splits — treat reported scores as
  indicative, not precise.
- The model's recall on the "Poor" class (0.75) means roughly 1 in 4 unsafe
  bridges may still be missed; this tool should support, not replace,
  professional inspection judgment.
- The Streamlit app does basic input validation (dropdowns constrain
  Material_Type / Maintenance_Level to known categories, numeric inputs are
  range-capped) but does not yet include authentication or rate limiting,
  which would be needed for a production multi-user deployment.
