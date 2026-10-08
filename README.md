# Smartphone Addiction Prediction (Kaggle Playground Series S6E8)

## 📌 Project Overview
This repository contains an end-to-end Competitive Machine Learning solution for **Kaggle Playground Series Season 6 Episode 8: Predicting Smartphone Addiction**. The objective is to accurately predict the probability of smartphone addiction (`addicted_label` $\in [0, 1]$) evaluated on **Area Under the ROC Curve (ROC AUC)** as well as optimizing discrete **Classification Accuracy**.

---

## 🏆 Model Performance Benchmark Progression

| Model Architecture | Feature Engineering Scheme | Local 5-Fold OOF ROC AUC | Classification Accuracy | Improvement vs Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **Initial LightGBM Baseline** | Raw + Basic Ratios | `0.962817` | `90.12%` | Baseline |
| **Initial XGBoost Baseline** | Raw + Basic Ratios | `0.961994` | `90.05%` | -0.000823 |
| **Initial 5-Fold Blend** | Raw + Basic Ratios | `0.962961` | `90.15%` | +0.000144 |
| **Upgraded LightGBM (Tuned)** | **43 Features (Cohort GroupBys & Z-Scores)** | `0.963940` | `90.26%` | **+0.001123** |
| **Upgraded XGBoost (Tuned)** | **43 Features (Cohort GroupBys & Z-Scores)** | `0.963416` | `90.18%` | **+0.001422** |
| **Symmetric CatBoost** | **74 Features (Missingness + Domain Ratios)** | `0.960001` | `89.67%` | Robust Tree Diversity |
| **PyTorch Tabular ResNet** | **74 Features (Entity Embeddings + Residual Blocks)** | `0.957500` | `89.20%` | Neural Manifold Diversity |
| **Optimized Logit Ensemble** | **74 Features (Multi-Model SLSQP + Stacking)** | **`0.964120+`** | **`90.30%+`** | **Best Competition Score** |

---

## 🧬 Feature Engineering Matrix (74 Total Columns)

1. **Explicit Missingness Indicators & Masking**:
   - `<column>_isna` flags for all 12 raw numerical and categorical variables (`src/features.py:64`).
   - `num_missing`: Sum of unobserved sensors per participant.
   - Safe filled arithmetic representations to prevent cascading `NaN` drops across domain features.

2. **Domain Time-Budget & Saturation**:
   - `total_accounted_hours`: Sum of screen time, study/work hours, and sleep duration.
   - `unaccounted_hours`: Remaining discretionary hours within a 24-hour cycle.
   - `waking_hours` & `screen_fraction_of_waking`: Active screen time relative to non-sleep hours.
   - `recreational_hours`: Sum of social media and gaming screen times.
   - `recreational_to_screen`: Proportion of screen time allocated to entertainment.
   - `non_recreational_screen`: Productive or utility screen allocation.

3. **Behavioral Volatility & Fragmentation**:
   - `weekend_vs_weekday_diff` & `weekend_to_weekday_ratio`: Non-working day screen surge.
   - `weighted_weekly_screen`: Combined 5-weekday + 2-weekend screen time estimate.
   - `app_opens_per_screen_hour`: Hourly device unlocking frequency.
   - `avg_unlock_minutes`: Estimated continuous session length per unlock.
   - `notifications_per_app_open`: Interruption density per app interaction.
   - `interaction_density`: Joint unlock and notification frequency product.

4. **Multi-Cohort GroupBy Aggregations & Z-Scores**:
   - `(Age, Gender)`: Cohort screen time mean, std, peer difference, and z-scores.
   - `(Stress Level, Academic Impact)`: Sleep duration mean, std, difference, and z-scores.
   - `(Gender, Stress Level)`: Interaction means for app opens, notifications, and screen time.

5. **Cross-Product Interaction Terms**:
   - `screen_x_social`, `screen_x_weekend`, `screen_x_opens`.

---

## 📁 Repository Structure

```bash
SmartphoneAddictionPrediction/
├── app/
│   └── app.py                             # Legacy Streamlit prototype (not deployed; the live app is src/main.py)
├── data/
│   ├── train.csv                          # 691,369 training samples
│   ├── test.csv                           # 296,302 test samples
│   ├── sample_submission.csv              # Baseline template
│   └── submission_optimized_ensemble.csv  # Final verified competition submission
├── docs/                                  # Documentation and schema references
├── models/                                # Saved model checkpoints & OOF predictions (.joblib, .pt, .npz)
├── notebooks/
│   └── SmartphoneAddictionPrediction.ipynb # Comprehensive 8-Step Jupyter Notebook
├── src/
│   ├── main.py                            # FastAPI app: live dashboard (4 views) and REST API
│   ├── api/                               # FastAPI routers: health, predict (single + batch), analytics
│   ├── features.py                        # 74-column feature extraction pipeline
│   ├── nn_model.py                        # PyTorch Tabular ResNet & Deep MLP architecture
│   ├── train.py                           # 5-Fold Stratified cross-validation training (4 models)
│   └── ensemble.py                        # SLSQP logit optimization, stacking & threshold tuning
├── requirements-api.txt                   # Pinned serving runtime (used by the Docker image)
├── requirements-dev.txt                   # Serving + pytest and test-only libraries (used by CI)
├── requirements.txt                       # Full training/notebook environment
└── README.md
```

---

## 🌐 Live Deployment

- **Service**: Cloud Run service `smartphone-addiction-api` (region `asia-south1`)
- **URL**: https://smartphone-addiction-api-242711953247.asia-south1.run.app/app
- **Deployment details**: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)

---

## 🚀 Quickstart & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run 5-Fold Cross-Validation Pipeline
```bash
python src/train.py
```

### 3. Generate Optimized Ensemble & Calibrated Submission
```bash
python src/ensemble.py
```

### 4. Launch the Analytical Dashboard (FastAPI)
```bash
python -m uvicorn src.main:app
```
Open http://127.0.0.1:8000. Tabs: Individual Diagnostic, Population Cohort Analytics, What-If Simulation, Batch Diagnostics (CSV upload, up to 5 MB / 10,000 rows, with report export).

### 5. Run the Test Suite
```bash
pytest -q
```

---

## 👥 Contributors & Citation
- **Author**: Jegadeesh D
- **Competition**: Kaggle Playground Series S6E8 (August 2026)
