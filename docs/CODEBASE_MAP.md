# Codebase Map: Smartphone Addiction Prediction

## 1. Project Overview & Context
- **Repository**: `SmartphoneAddictionPrediction` (Kaggle Playground Series S6E8)
- **Objective**: Predict smartphone addiction risk from behavioral and demographic metrics using tabular ML.
- **Current State**: Advanced competitive ML models trained to benchmark $>0.964$ ROC AUC, with a legacy single-file Streamlit prototype (`app/app.py`).
- **Target Evolution**: Transition from Streamlit to a production FastAPI service serving an informational and analytical Web Dashboard with live model inference and cohort analytics.

---

## 2. Technology Stack & Dependencies
- **Core Language**: Python (>=3.11, running in virtual environment)
- **Machine Learning & Tabular**:
  - `lightgbm` (>=4.0.0)
  - `xgboost` (>=2.0.0)
  - `catboost` (>=1.2.0)
  - `torch` (PyTorch for Tabular ResNet & Deep MLP)
  - `scikit-learn` (>=1.4.0)
  - `scipy` (>=1.12.0)
- **Data Engineering**:
  - `pandas` (>=2.2.0)
  - `numpy` (>=1.26.0)
  - `joblib` (>=1.3.0)
- **Backend & API**:
  - `fastapi` (0.115.6)
  - `uvicorn` (0.32.1)
  - `pydantic` (v2)
- **Frontend / UI**:
  - Legacy: Streamlit (in `app/app.py`, to be replaced)
  - Target: Bespoke HTML5 / CSS3 / Vanilla JS analytical dashboard with IBM Plex / Source Serif typography, responsive data views, Plotly / Chart.js interactive charts.
- **Testing**:
  - `pytest` (8.3.4)

---

## 3. Directory Layout & Architecture
```
SmartphoneAddictionPrediction/
├── app/
│   ├── app.py                     # Legacy Streamlit app (to be replaced)
│   ├── assets/                    # Static assets
│   ├── components/                # UI components (.keep)
│   └── pages/                     # Multi-page directory (.keep)
├── data/
│   ├── train.csv                  # 691,369 training samples (~45MB)
│   ├── test.csv                   # 296,302 test samples (~19MB)
│   ├── sample_submission.csv
│   └── submission_*.csv
├── docs/                          # Living documentation suite
├── models/
│   ├── lgb_fold_1..5.joblib       # 5-fold trained LightGBM models (~50MB each)
│   ├── xgb_fold_1..5.joblib       # 5-fold trained XGBoost models (~9.6MB each)
│   ├── cat_fold_1..5.joblib       # 5-fold trained CatBoost models (~3.8MB each)
│   ├── nn_fold_1..5.pt            # 5-fold PyTorch models (~1.2MB each)
│   └── oof_predictions.npz        # Out-of-fold probability distributions
├── notebooks/
│   └── SmartphoneAddictionPrediction.ipynb
├── src/
│   ├── features.py                # 74-column feature engineering pipeline
│   ├── train.py                   # 5-fold CV training pipeline
│   ├── ensemble.py                # SLSQP logit optimization & stacking
│   └── nn_model.py                # Tabular ResNet & MLP architectures
├── requirements.txt
└── README.md
```

---

## 4. UI Inventory (Legacy Streamlit)
- **Source**: `app/app.py`
- **Inputs**:
  - Demographic: `Age` (18-35), `Gender` (Male, Female, Other), `Stress Level` (Low, Medium, High), `Impact on Work/Academics` (No, Yes).
  - Time Budget: `Daily Screen Time`, `Social Media`, `Gaming`, `Work/Study`, `Weekend Daily Screen Time`, `Sleep Duration`.
  - Engagement: `Notifications / Day`, `App Opens / Day`.
  - Decision threshold: `Classification Threshold (tau)` (0.10 to 0.90).
- **Current Behavior**:
  - Uses a hardcoded manual linear heuristic instead of calling trained models.
  - Computes basic ratios (`screen_to_sleep`, `recreational_to_screen`, `weekend_diff`, `avg_unlock_minutes`).
  - Displays risk tier (High, Moderate, Healthy) and static benchmark bar chart.
- **Decommission Plan**:
  - Remove Streamlit dependency and single-file heuristic.
  - Implement FastAPI backend with `/api/predict`, `/api/analytics/cohorts`, `/api/analytics/what-if`, and `/api/health`.
  - Serve custom analytical web dashboard with responsive tabs, rich charts, and live inference.

---

## 5. Coding & Naming Conventions
- **Python modules**: `snake_case.py` (e.g. `features.py`, `inference.py`).
- **Classes**: `PascalCase` (e.g. `TabularResNet`, `FeaturePipeline`).
- **Functions & variables**: `snake_case` (e.g. `engineer_features()`, `predict_risk()`).
- **Constants**: `UPPER_SNAKE_CASE` (e.g. `FEATURE_NAMES`, `BENCHMARK_MEANS`).
- **Living Docs**: `UPPER_SNAKE_CASE.md` in `docs/`.
- **Commits**: Conventional Commits (`feat(scope): ...`, `fix(scope): ...`, `docs(scope): ...`).

---

## 6. Test Suite & Baseline Status
- **Current Test Suite**: No existing test files in repository (`tests/` directory did not exist).
- **Baseline Command**: `pytest`
- **Baseline Result**: Exit code 5 (no tests collected).
- **Resolution Plan**: Introduce automated test suite in M1 covering:
  - Feature transformation pipeline integrity (`tests/test_features.py`)
  - Inference service validation with LightGBM fold 1 (`tests/test_inference.py`)
  - FastAPI endpoints and response schemas (`tests/test_api.py`)
