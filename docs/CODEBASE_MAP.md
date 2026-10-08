# Codebase Map: Smartphone Addiction Prediction

Comprehensive architectural map of the existing codebase, runtime stack, coding conventions, UI inventory, and test baseline for Smartphone Addiction Prediction.

*Date: 2026-10-06*

---

## 1. Summary
The project is a competitive tabular machine learning system developed for the **Kaggle Playground Series Season 6 Episode 8: Predicting Smartphone Addiction**. Its core objective is predicting the probability of smartphone addiction (`addicted_label` $\in [0, 1]$) from participant demographic and digital habit metrics. The repository contains:
- Complete exploratory analysis and competitive feature engineering pipelines producing 74 domain, interaction, and cohort-level features.
- A 5-fold stratified cross-validation training driver supporting four model families: LightGBM, XGBoost, CatBoost, and a PyTorch Tabular ResNet, achieving an out-of-fold benchmark $>0.964$ ROC AUC and $>90.3\%$ accuracy.
- An SLSQP continuous logit ensembling and threshold calibration engine exporting competition-ready submissions.
- An interactive single-page Streamlit diagnostic web application ([app/app.py](../app/app.py)) presenting individual behavioral risk assessments, ratio diagnostics, and population comparisons.

---

## 2. Stack
- **Language**: Python 3.13.2
- **Package Manager**: `pip` (manifest: [requirements.txt](../requirements.txt))
- **Machine Learning & Gradient Boosting**:
  - `lightgbm` 4.7.0 (manifest: `>=4.0.0`)
  - `xgboost` 3.2.0 (manifest: `>=2.0.0`)
  - `catboost` 1.2.10 (manifest: `>=1.2.0`)
  - `scikit-learn` 1.8.0 (manifest: `>=1.4.0`)
  - `torch` 2.12.1+cpu (PyTorch tabular neural network; used in [src/nn_model.py](../src/nn_model.py) and [src/train.py](../src/train.py), omitted from [requirements.txt](../requirements.txt))
  - `scipy` 1.17.1 (manifest: `>=1.12.0`)
  - `joblib` 1.4.2 (manifest: `>=1.3.0`)
- **Data Manipulation**:
  - `pandas` 2.2.3 (manifest: `>=2.2.0`)
  - `numpy` 2.2.3 (manifest: `>=1.26.0`)
- **Web UI & Serving**:
  - `streamlit` 1.40.2 (manifest: `>=1.30.0`)
  - `fastapi` 0.115.6 (installed in runtime environment)
  - `uvicorn` 0.32.1 (installed in runtime environment)
- **Testing**:
  - `pytest` 8.3.4 (with plugins: `pytest-asyncio` 0.24.0, `anyio` 4.13.0, `pytest-cov` 7.1.0)
- **Manifest File**: [requirements.txt](../requirements.txt)

---

## 3. Layout
```
SmartphoneAddictionPrediction/
├── app/                                       # Web application prototype layer
│   ├── app.py                                 # Streamlit interactive diagnostic dashboard
│   ├── assets/                                # Static visual assets directory (.keep)
│   ├── components/                            # Modular UI components placeholder (.keep)
│   └── pages/                                 # Multi-page application views placeholder (.keep)
├── catboost_info/                             # CatBoost training telemetry and event logs
├── data/                                      # Competition datasets and submissions
│   ├── train.csv                              # 691,369 labeled training records (14 columns, ~45.3MB)
│   ├── test.csv                               # 296,302 unlabeled test records (13 columns, ~19.3MB)
│   ├── sample_submission.csv                  # Competition submission template
│   ├── submission_lgb_xgb_blend.csv           # Baseline dual-tree submission
│   └── submission_optimized_ensemble.csv      # Champion multi-model ensemble submission
├── docs/                                      # Living project documentation suite
│   ├── CODEBASE_MAP.md                        # This codebase architecture map
│   ├── PROJECT_MENTAL_MODEL.md                # Mental model, architectural invariants, and roadmap
│   ├── PROJECT_STATUS.md                      # Phase workflow progress checklist
│   └── README.md                              # Living documentation suite table of contents
├── models/                                    # Serialized model checkpoints and prediction matrices (22 files)
│   ├── cat_fold_1.joblib .. cat_fold_5.joblib # 5-fold CatBoost classifiers (~3.8MB each)
│   ├── lgb_fold_1.joblib .. lgb_fold_5.joblib # 5-fold LightGBM classifiers (~50MB each)
│   ├── nn_fold_1.pt .. nn_fold_5.pt           # 5-fold PyTorch Tabular ResNet weights (~1.2MB each)
│   ├── oof_predictions.npz                    # Compressed out-of-fold and test predictions (22.7MB)
│   └── xgb_fold_1.joblib .. xgb_fold_5.joblib # 5-fold XGBoost classifiers (~9.6MB each)
├── notebooks/                                 # Jupyter prototyping notebooks
│   └── SmartphoneAddictionPrediction.ipynb    # 10-step exploratory and modeling notebook
├── src/                                       # Core machine learning pipeline modules
│   ├── ensemble.py                            # SLSQP logit optimization, rank blending, and threshold tuning
│   ├── features.py                            # 74-column feature extraction pipeline with cohort aggregations
│   ├── nn_model.py                            # PyTorch Tabular ResNet & Deep MLP architectures
│   └── train.py                               # 5-fold Stratified cross-validation training driver
├── .env                                       # Local environment configuration (placeholder only)
├── .gitignore                                 # Git ignore patterns (virtualenvs, data, models, logs)
├── README.md                                  # Repository overview and benchmark documentation
└── requirements.txt                           # Core dependencies manifest
```

### Entry Points
- **Streamlit Web Application**: `streamlit run app/app.py` ([app/app.py](../app/app.py))
- **Cross-Validation Training Driver**: `python src/train.py` ([src/train.py](../src/train.py#L220-L222))
- **Ensemble Optimization & Submission**: `python src/ensemble.py` ([src/ensemble.py](../src/ensemble.py#L193-L195))

---

## 4. Conventions
- **Naming Conventions**:
  - Python files: lowercase `snake_case.py` (e.g., [src/features.py](../src/features.py), [src/nn_model.py](../src/nn_model.py), [src/ensemble.py](../src/ensemble.py)).
  - Classes: `PascalCase` (e.g., `TabularDataset` at [src/nn_model.py#L14](../src/nn_model.py#L14), `ResNetBlock` at [src/nn_model.py#L29](../src/nn_model.py#L29), `TabularResNet` at [src/nn_model.py#L47](../src/nn_model.py#L47)).
  - Functions: lowercase `snake_case` (e.g., `create_features()` at [src/features.py#L12](../src/features.py#L12), `train_pipeline()` at [src/train.py#L35](../src/train.py#L35), `optimize_and_submit()` at [src/ensemble.py#L23](../src/ensemble.py#L23)).
  - Variables: lowercase `snake_case` (e.g., `daily_screen_time_hours`, `recreational_hours`, `feat_cols`).
  - Documentation files: `UPPER_SNAKE_CASE.md` in `docs/` (e.g., [docs/CODEBASE_MAP.md](../docs/CODEBASE_MAP.md), [docs/PROJECT_MENTAL_MODEL.md](../docs/PROJECT_MENTAL_MODEL.md)).
- **Formatting and Lint Configuration**:
  - 4-space indentation across all Python modules.
  - No `.flake8`, `.black`, or `ruff.toml` config file currently present.
  - Multi-line module docstrings describe functionality at the head of every source file (e.g., [src/features.py#L1-L6](../src/features.py#L1-L6)).
  - Explicit console encoding handling for cross-platform Windows compatibility:
    ```python
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ```
    (Found in [src/train.py#L28-L29](../src/train.py#L28-L29) and [src/ensemble.py#L19-L20](../src/ensemble.py#L19-L20)).
- **Typing**:
  - Python standard type hints are partially adopted in function signatures (e.g., `def create_features(df_train: pd.DataFrame, df_test: pd.DataFrame = None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:` in [src/features.py#L12](../src/features.py#L12)).
  - Local variables and helper sub-routines generally omit type hints.
- **Error Handling**:
  - Suppressive error handling in training driver: `warnings.filterwarnings('ignore')` in [src/train.py#L25](../src/train.py#L25).
  - Defensive missing-value imputation prevents NaN propagation in feature generation (`.fillna(0.0)`, `.fillna(median)`, and `.clip(lower=1.0)` in [src/features.py#L53-L69](../src/features.py#L53-L69)).
  - Explicit existence verification on required disk assets before processing:
    ```python
    oof_path = os.path.join(models_dir, 'oof_predictions.npz')
    if not os.path.exists(oof_path):
        print(f"Error: {oof_path} not found. Please run train.py first.")
        return
    ```
    (Found in [src/ensemble.py#L28-L31](../src/ensemble.py#L28-L31)).
- **Module Organization**:
  - Single-responsibility scripts: data transformation ([src/features.py](../src/features.py)), neural architecture definition ([src/nn_model.py](../src/nn_model.py)), training orchestration ([src/train.py](../src/train.py)), meta-learning ensembling ([src/ensemble.py](../src/ensemble.py)), and web presentation ([app/app.py](../app/app.py)).

---

## 5. Tests
- **Test Runner**: `pytest` (version 8.3.4 detected in runtime environment).
- **Test Location**: Currently no `tests/` directory exists in the repository.
- **Naming Convention**: `test_*.py` / `*_test.py` (standard pytest convention).
- **Test Execution Commands**:
  - Run all tests: `pytest`
  - Run a specific test file: `pytest tests/<filename>.py`
  - Run with verbose output: `pytest -v`
- **Markers**: No custom markers (e.g., `@pytest.mark.slow`) are configured in any `pytest.ini` or `pyproject.toml`.

---

## 6. Baseline
- **Command Run**: `pytest`
- **Exit Code**: `1`
- **Summary Line**: `collected 0 items / no tests ran in 0.09s`
- **Failure Details**: No tests found. The test runner discovered zero test files because the `tests/` directory has not yet been introduced.

---

## 7. Existing Behavior
The existing user-visible journeys, public interfaces, and data transformation routines comprise:

1. **Interactive Streamlit Diagnostic Dashboard** ([app/app.py](../app/app.py)):
   - **Input Profile Form**: Sidebar widgets capturing 4 demographic and 8 behavioral indicators:
     - `Age` (18–35), `Gender` (Male, Female, Other), `Stress Level` (Low, Medium, High), `Academic / Work Impact` (No, Yes).
     - `Daily Screen Time` (0.5–15.0h), `Social Media` (0.0–10.0h), `Gaming` (0.0–8.0h), `Work / Study` (0.0–12.0h), `Weekend Daily Screen Time` (0.5–17.5h), `Sleep Duration` (3.0–12.0h).
     - `Notifications Received / Day` (10–250), `App Opens / Day` (15–180).
     - `Classification Threshold (τ)` (0.10–0.90, default 0.50).
   - **Risk Evaluation Output**: Computes probability score using a synthetic linear logit heuristic:
     - Progress bar displaying estimated probability percentage.
     - Severity banner: High ($\ge 70\%$), Moderate ($\ge 40\%$), or Healthy ($< 40\%$).
     - Classification status: `ADDICITON DETECTED` (source string typo preserved) or `HEALTHY / BORDERLINE` / `HEALTHY`.
   - **Behavioral Ratios & Intervention Triggers**:
     - Calculates `screen_to_sleep`, `recreational_to_screen`, `avg_unlock_minutes`, `weekend_diff`.
     - Displays warnings for sleep protection (`screen_to_sleep > 1.2`), recreation audit (`recreational_to_screen > 0.60`), notification hygiene (`app_opens_per_day > 120`), and weekend disconnect (`weekend_diff > 2.5`).
   - **Population Benchmark Comparison**: Bar chart comparing user metrics against population means (Daily Screen: 7.64h, Weekend Screen: 9.48h, Social Media: 2.47h, Sleep: 6.80h, App Opens: 10.26/10).
   - **Multi-Model Benchmark Display**: Read-only comparison table displaying local 5-fold CV ROC AUC and accuracy.
   - *Test Coverage*: Untested (0% automated test coverage).

2. **Feature Engineering Pipeline** ([src/features.py:create_features](../src/features.py#L12)):
   - Ingests raw train and optional test datasets; concatenates with `is_train` partition flags.
   - Generates 12 missingness flags (`*_isna`) and total missing sensor count (`num_missing`).
   - Generates composite categorical encodings (`gender_stress_code`, `stress_impact_code`).
   - Performs frequency encodings across 8 columns.
   - Derives domain time budgets, weekend surge metrics, session unlock fragmentation, and interaction cross-products.
   - Merges cohort aggregations (mean, std, z-scores) across `(age, gender)`, `(stress_level, academic_work_impact)`, and `(gender, stress_level)`.
   - Drops non-feature columns (`id`, `is_train`, `addicted_label`, raw categoricals) and outputs 74 feature columns.
   - *Test Coverage*: Untested.

3. **Stratified 5-Fold Training Pipeline** ([src/train.py:train_pipeline](../src/train.py#L35)):
   - Loads `data/train.csv` (691,369 rows) and `data/test.csv` (296,302 rows).
   - Trains LightGBM (191 leaves), XGBoost (hist), CatBoost (symmetric), and PyTorch Tabular ResNet across 5 stratified folds.
   - Saves 20 trained model checkpoint files into `models/` directory.
   - Serializes out-of-fold and test predictions to `models/oof_predictions.npz`.
   - *Test Coverage*: Untested.

4. **SLSQP Logit Ensembling & Submission Export** ([src/ensemble.py:optimize_and_submit](../src/ensemble.py#L23)):
   - Loads `models/oof_predictions.npz`.
   - Computes probability-space and logit-space continuous blend weights using SciPy SLSQP optimization.
   - Computes rank averaging and logistic regression stacking meta-learner.
   - Calibrates optimal decision thresholds over grid $[0.05, 0.95]$ for accuracy and F1 score.
   - Generates and exports verified submission to `data/submission_optimized_ensemble.csv`.
   - *Test Coverage*: Untested.

---

## 8. Setup
- **System Requirements**:
  - Python (>=3.11, verified on 3.13.2)
  - Operating System: Windows / Linux / macOS
- **Environment Variables**:
  - No required runtime environment variables.
  - A `.env.example` configuration template is tracked at [.env.example](../.env.example); the local `.env` is git-ignored.
- **External Services**:
  - None required. All computation, training, and inference operate against local file storage (`data/` and `models/`).

---

## 9. Risks
1. **Disconnected Model Inference in Legacy Streamlit Prototype**:
   - The prototype [app/app.py](../app/app.py) imports `joblib` but never invokes any trained model from `models/`. Instead, it uses a hand-coded linear logit equation (`logit_score = -4.15 + 0.38 * daily_screen_time ...`). User predictions from the prototype do not reflect the actual competitive ML pipeline ($>0.964$ ROC AUC).
2. **Dynamic Cohort GroupBy Leakage & Single-Sample Inference Failure**:
   - In [src/features.py](../src/features.py), cohort means, standard deviations, and frequency encodings are computed on the passed dataframe via `df_all.groupby(...)` and `df_all[c].value_counts(...)`.
   - Passing a single sample (as in interactive single-user inference) produces `NaN` for standard deviations, `1.0` for frequency encodings, and malformed category codes. Real-time inference requires extracting and caching population-level reference statistics (means, standard deviations, frequencies) from `data/train.csv`.
3. **Missing Dependency in Manifest**:
   - `torch` is imported and used in [src/train.py](../src/train.py) and [src/nn_model.py](../src/nn_model.py), and PyTorch model checkpoints exist in `models/`, but `torch` is not listed in [requirements.txt](../requirements.txt). Clean environment setups will fail without installing PyTorch.
4. **Complete Absence of Automated Tests**:
   - No test suite exists. Refactoring feature generation, model loading, or API routing carries high regression risk unless automated test coverage is established.
5. **Memory Footprint of Multi-Fold Ensembles**:
   - The 20 saved models in `models/` total ~360MB (LightGBM models are ~50MB each). Loading all 20 models into memory concurrently for interactive web serving incurs significant RAM overhead (>2GB) and initial cold-start latency. Serving LightGBM Fold 1 as the primary interactive inference engine provides $<20\text{ms}$ latency with minimal memory footprint.
6. **Typographical Defect in User-Facing String**:
   - [app/app.py#L136](../app/app.py#L136) contains `'ADDICITON DETECTED'`. Any migration or replacement must correct this to `'ADDICTION DETECTED'`.

---

## 10. Documentation
- **Root Repository Overview**: [README.md](../README.md) (109 lines detailing competition framing, 5-fold benchmarks, 74-column feature matrix, and execution steps).
- **Living Documentation Index**: [docs/README.md](../docs/README.md) (index linking living documents).
- **Mental Model & Invariants**: [docs/PROJECT_MENTAL_MODEL.md](../docs/PROJECT_MENTAL_MODEL.md) (vision, technical decisions, persona, milestone roadmap).
- **Project Workflow Status**: [docs/PROJECT_STATUS.md](../docs/PROJECT_STATUS.md) (phase completion checklist).
- **Exploratory Jupyter Notebook**: [notebooks/SmartphoneAddictionPrediction.ipynb](../notebooks/SmartphoneAddictionPrediction.ipynb) (comprehensive 10-step EDA and cross-validation walkthrough).
- **CHANGELOG**: None (no `CHANGELOG.md` present).
- **ADR Location & Format**: Projected living document at `docs/DECISIONS.md`.
- **Planned Living Documents**: `docs/SPEC.md`, `docs/ARCHITECTURE.md`, `docs/TASKS.json`.

---

## 11. UI Inventory
- **Frontend Framework**: Streamlit (version 1.40.2) in [app/app.py](../app/app.py).
- **Styling Approach**: Custom CSS block embedded directly in Python via `st.markdown("<style>...</style>", unsafe_allow_html=True)` ([app/app.py#L16-L53](../app/app.py#L16-L53)).
- **Component Library**: Streamlit built-in widgets: `st.sidebar.slider`, `st.sidebar.selectbox`, `st.sidebar.subheader`, `st.metric`, `st.progress`, `st.bar_chart`, `st.dataframe`, `st.warning`, `st.info`.
- **Colors, Fonts, and Spacing Definitions**:
  - Color Palette ([app/app.py#L16-L52](../app/app.py#L16-L52)):
    - Primary text / headers: `#1E293B` (slate 800)
    - Subtitle / captions: `#64748B` (slate 500)
    - Metric card background: `#F8FAFC` (slate 50)
    - Metric card border: `#E2E8F0` (slate 200)
    - High-risk severity: `#DC2626` (red 600)
    - Moderate-risk severity: `#D97706` (amber 600)
    - Low-risk / healthy severity: `#16A34A` (green 600)
  - Typography: System default sans-serif font stack. Header font-size `2.1rem` (font-weight 700), sub-header `1.0rem`.
  - Spacing & Geometry: Metric cards have `20px` padding, `12px` border-radius, `15px` margin-bottom, with light drop shadow `0 4px 6px -1px rgba(0, 0, 0, 0.05)`.
- **Screens and Routes**:
  - Route `/`: Single-page dashboard divided into two responsive layout columns:
    - Left Column (`1.2` width): Diagnostic Risk Assessment progress meter, classification banner, 4 granular behavioral diagnostic metrics (`screen_to_sleep`, `recreational_to_screen`, `avg_unlock_minutes`, `weekend_diff`), and dynamic rule-based intervention callouts.
    - Right Column (`1.0` width): Population comparison benchmark bar chart and multi-model benchmark performance table.
- **UI Launch Command & URL**:
  - Command: `streamlit run app/app.py`
  - URL: `http://localhost:8501`
- **Root PRODUCT.md and DESIGN.md**: None (neither `PRODUCT.md` nor `DESIGN.md` exists in the repository root).
