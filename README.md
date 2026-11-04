# Smartphone Addiction Prediction

Kaggle Playground Series S6E8 solution plus a diagnostic dashboard. It predicts the probability of smartphone addiction (`addicted_label`, scored by ROC AUC and accuracy) with a tuned ensemble, and serves single-user diagnostics, cohort analytics, what-if simulation and batch scoring through a FastAPI app.

Live app: not currently deployed (Cloud Run service removed Nov 2026; redeploy with `.github/workflows/deploy.yml`, which is disabled in GitHub Actions)

## Features

- **Individual diagnostic:** risk gauge, behavioral ratio cards and interventions for one person.
- **Population cohort analytics:** demographic cohorts, a screen-versus-sleep density heatmap and personal quantile benchmarks.
- **What-if simulation:** change habits and see the predicted risk move.
- **Batch diagnostics:** upload a CSV (up to 5 MB and 10,000 rows) and export a scored report.
- **Training pipeline:** 74-column feature extraction, 5-fold cross-validation for four model families, and an optimized logit ensemble.

## Model performance

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

## Feature engineering (74 columns)

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

## Quick start

Requires Python 3.13.

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows Git Bash; use .venv/bin/activate elsewhere
pip install -r requirements-dev.txt  # serving runtime plus test libraries
cp .env.example .env                 # then edit the values
python -m uvicorn src.main:app
```

Open http://127.0.0.1:8000. The trained checkpoint (`models/lgb_fold_1.joblib`) and the Kaggle CSVs are not tracked in git. Set `USE_MOCK_MODEL=True` in `.env` to run the API on the calibrated heuristic engine without the checkpoint.

## Usage

To retrain, place the Kaggle files in `data/` and install the full environment (`pip install -r requirements.txt`):

```bash
python src/train.py      # 5-fold cross-validation for four models
python src/ensemble.py   # optimized ensemble and calibrated submission
```

The dashboard has four tabs: Individual Diagnostic, Population Cohort Analytics, What-If Simulation and Batch Diagnostics.

## Running tests

```bash
pytest -q                          # full suite
pytest tests/test_features.py -q   # one file
```

Last full run: 331 passed, 2 skipped (the skipped tests need PyTorch, which the serving runtime does not install).

## Configuration

Copy `.env.example` to `.env`.

| Variable | Purpose |
|---|---|
| `APP_HOST` | Network host address for the FastAPI backend |
| `APP_PORT` | Port for the Uvicorn server (default 8000) |
| `APP_DEBUG` | Verbose ASGI logging flag |
| `MODEL_PATH` | Path to the serialized LightGBM Fold 1 model |
| `PRIORS_PATH` | Path to the pre-computed population priors JSON |
| `COHORT_CACHE_PATH` | Path to the pre-aggregated cohort summary JSON |
| `USE_MOCK_MODEL` | Use the calibrated heuristic mock inference engine |
| `CORS_ORIGINS` | Allowed CORS origins, comma-separated or a JSON list |

## Project structure

```
app/          legacy Streamlit prototype (not deployed)
data/         priors and cohort summary (Kaggle CSVs are local only)
docs/         specification, architecture, decisions and deployment notes
models/       trained checkpoints and out-of-fold predictions (local only)
notebooks/    the 8-step analysis notebook
scripts/      cohort cache and priors generators
src/          FastAPI app (main.py, api/), features, training, ensemble, inference
tests/        pytest suite
```

## Documentation

- [docs/README.md](docs/README.md): index of the documents
- [docs/SPEC.md](docs/SPEC.md): user journeys and acceptance criteria
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): components, data models and API contracts
- [docs/DECISIONS.md](docs/DECISIONS.md): architecture decision records
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md): Cloud Run pipeline and runbook
- [CHANGELOG.md](CHANGELOG.md): history of changes

## How this was built

Designed and built by jegadeesh17 with [SoftwareEngineeringAgents](https://github.com/jegadeesh17/SoftwareEngineeringAgents), a multi-agent workflow for Claude Code. I reviewed and tested all code.

## License

[MIT](LICENSE)
