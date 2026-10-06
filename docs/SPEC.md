# Product Specification: Smartphone Addiction Prediction Platform

This document defines the functional requirements, user journeys, parity baselines, regression safeguards, and testable acceptance criteria for the Smartphone Addiction Prediction analytical dashboard platform across Milestones M1, M2, and M3.

*Date: 2026-10-06*  
*Related Documents: [mental model](PROJECT_MENTAL_MODEL.md) | [codebase map](CODEBASE_MAP.md)*

---

## 1. Summary

- **Vision**: Transition the Smartphone Addiction Prediction platform from an untracked, disconnected Streamlit script into a production-grade FastAPI service and bespoke, responsive analytical dashboard. The platform pairs a competitive gradient-boosted tabular ML model ($<20\text{ms}$ live inference latency) with hybrid cohort analytics, physiological validation, and counterfactual what-if habit simulations.
- **Posture**: `Production`
- **Target Persona**: Digital well-being analysts, clinical behavioral researchers, and individuals seeking diagnostic risk clarity with actionable habit interventions.
- **Change Type**: `Redesign` & `Feature` (Redesign of the primary diagnostic dashboard; Feature additions of Population Cohort Analytics Explorer, Real-Time What-If Simulation, and Batch Diagnostic Scoring).

---

## 2. User Journeys

### Journey 1: Executive Overview & Live Individual Risk Diagnostic (M1 - Redesign)
*Starts from*: The legacy single-page Streamlit diagnostic screen ([app/app.py](file:///c:/Users/jegad/projects/SmartphoneAddictionPrediction/app/app.py)), which used heuristic mock logit scoring disconnected from trained ML checkpoints.

1. **Dashboard Access**: The user opens `http://localhost:8000/app` (or `/`). The application renders a polished editorial header, system health/latency telemetry badges, and a tabbed navigation bar (`Individual Diagnostic`, `Population Cohort Analytics`, `What-If Simulation`).
2. **Behavioral Profile Configuration**: In the "Individual Diagnostic" view, the user interacts with two structured input cards:
   - *Demographics & Context*: `Age` (18–35), `Gender` (`Male`, `Female`, `Other`), `Stress Level` (`Low`, `Medium`, `High`), and `Academic / Work Impact` (`No`, `Yes`).
   - *Daily Digital Time Budget*: `Daily Screen Time` (0.0–24.0h), `Social Media` (0.0–12.0h), `Gaming` (0.0–12.0h), `Work / Study` (0.0–16.0h), `Weekend Daily Screen Time` (0.0–24.0h), and `Sleep Duration` (1.0–18.0h).
   - *Habit Intensity*: `Notifications Received / Day` (0–500), `App Opens / Day` (0–500), and `Classification Threshold (τ)` (0.10–0.90, default 0.50).
3. **Physiological Boundary Feedback**: As the user adjusts sliders, the UI calculates the sum of daily screen time, work/study hours, and sleep duration. If the sum exceeds 24.0 hours, an inline amber warning appears (`"Total accounted time exceeds 24h per day"`) without locking exploratory controls.
4. **Live Diagnostic Risk Evaluation**: Without manual page reloads or submit buttons, input changes trigger an asynchronous `<20ms` call to `/api/predict`. The user sees:
   - A reactive circular/linear addiction risk probability gauge ($P \in [0.0\%, 100.0\%]$).
   - A severity tier badge: `High Addiction Probability` ($P \ge 70\%$), `Moderate / At-Risk Usage` ($40\% \le P < 70\%$), or `Healthy Usage Pattern` ($P < 40\%$).
   - A definitive decision classification badge: `ADDICTION DETECTED` if $P \ge \tau$, else `HEALTHY` (or `HEALTHY / BORDERLINE`).
   - Four granular ratio diagnostic cards comparing personal ratios to healthy targets:
     - *Screen / Sleep Ratio*: $\text{Screen} / \text{Sleep}$ (Target $< 1.0\text{x}$)
     - *Recreational Share*: $(\text{Social} + \text{Gaming}) / \text{Screen}$ (Target $< 50.0\%$)
     - *Avg Session Length*: $(\text{Screen} \times 60) / \text{Opens}$ (Target $> 5.0\text{ min}$)
     - *Weekend Surge*: $\text{Weekend Screen} - \text{Daily Screen}$ (Target $< 2.0\text{h}$)
5. **Actionable Intervention Guidance**: When behavioral ratios breach healthy parameters, rule-based clinical recommendation callouts dynamically appear:
   - *Sleep Protection*: Triggered if $\text{Screen / Sleep} > 1.2$.
   - *Recreation Audit*: Triggered if $\text{Recreational Share} > 60.0\%$.
   - *Notification Hygiene*: Triggered if $\text{App Opens / Day} > 120$.
   - *Weekend Disconnect*: Triggered if $\text{Weekend Surge} > 2.5\text{h}$.

---

### Journey 2: Population Cohort Analytics Explorer & Demographic Slicing (M2 - Feature)
*Starts from*: The legacy static 5-metric benchmark bar chart comparing individual sliders against hard-coded means.

1. **Cohort Explorer Navigation**: The user clicks the "Population Cohort Analytics" tab in the main header. The dashboard transitions smoothly to the population exploration view.
2. **Cohort Slicing Controls**: The user selects demographic segmentation filters:
   - Primary slicing dimension: `Age Bracket` (`18-21`, `22-25`, `26-30`, `31-35`), `Gender` (`Male`, `Female`, `Other`), or `Stress Level` (`Low`, `Medium`, `High`).
   - Secondary grouping: `Academic / Work Impact` (`Yes` vs `No`).
3. **Cohort Benchmark Inspection**: The UI queries `/api/analytics/cohorts` and renders responsive visualizations powered by cached statistics of the 691,369 Kaggle population records:
   - Addiction prevalence rate comparison across cohorts (% addicted).
   - Mean digital time budget breakdowns (recreational vs productive vs sleep).
   - Interaction intensity boxplots/distributions (app opens and notifications).
4. **2D Joint Density Heatmap**: The user views a screen-time vs sleep-duration distribution heatmap illustrating where addiction risk density concentrates across the population.
5. **Personal Quantile Overlay**: The UI queries `/api/analytics/distributions` with the user's active profile, displaying where their individual habits place them relative to 691k participants (e.g., `"92nd percentile in screen time"`, `"18th percentile in sleep duration"`).

---

### Journey 3: Real-Time What-If Behavioral Habit Simulation (M3 - Feature)
*Starts from*: The static diagnostic output in M1 where users cannot directly model habit modifications side-by-side.

1. **Simulator Activation**: The user clicks the "What-If Simulation" tab. The dashboard presents a split comparison layout: "Current Baseline Profile" vs "Simulated Counterfactual".
2. **Interactive Intervention Adjustments**: The user manipulates target behavioral levers:
   - `Recreational Reduction`: Adjust slider to subtract hours from social media or gaming.
   - `Sleep Extension`: Adjust slider to add sleep duration.
   - `Notification Batching`: Adjust slider to reduce app open frequency and alert volume.
3. **Instant Counterfactual Scoring**: The dashboard queries `/api/analytics/what-if`, computing simulated probability $P_{\text{sim}}$ and risk delta $\Delta P = P_{\text{sim}} - P_{\text{baseline}}$.
4. **Visual Risk Delta Feedback**: The user sees:
   - A delta badge showing the risk reduction (e.g., `"-28.4% Addiction Risk"`, styled in emerald green).
   - A threshold transition indicator showing whether the proposed habit shift flips the status from `ADDICTION DETECTED` to `HEALTHY`.
   - Side-by-side behavioral ratio updates showing the simulated screen-to-sleep ratio and recreational share.
5. **Optimal Intervention Pathway**: The user clicks `"Generate Optimal Habit Target"`. The system computes and highlights the minimal habit modification required to bring $P_{\text{sim}} < \tau$ with minimal lifestyle disruption.

---

### Journey 4: Batch Population CSV Diagnostic Upload & Export (M3 - Feature)
*Starts from*: Single-user manual slider input in M1.

1. **Batch Mode Access**: A clinical researcher or analyst clicks "Batch Diagnostics" in the dashboard utility navigation.
2. **File Selection & Upload**: The user uploads a CSV file containing participant records (up to 10,000 rows with required demographic and digital usage columns) to `/api/predict/batch`.
3. **Schema Validation & Preview**: The UI parses and validates the uploaded file:
   - Displays total row count, valid rows, and invalid rows (e.g., missing columns or out-of-bounds physiological values).
   - Highlights row-level validation errors in a collapsible preview table.
4. **Batch Inference Execution**: The user confirms execution. The backend scores the records in parallel batches via LightGBM Fold 1.
5. **Population Diagnostic Summary & CSV Download**: The user sees an aggregate cohort summary banner (Overall Addiction Rate, Mean Screen/Sleep Ratio, High-Risk Percentage) and clicks `"Export Full Diagnostic CSV"`, downloading an enriched CSV containing original records alongside predicted probabilities, classifications, behavioral ratios, and recommended interventions.

---

## 3. Acceptance Criteria

### Milestone 1: Executive Overview, Live Inference & Risk Diagnostic

#### AC-1.1: Live Inference Endpoint Contract & Response Structure
- **Given** a valid participant payload containing:
  - `age`: 23
  - `gender`: `"Female"`
  - `stress_level`: `"High"`
  - `academic_work_impact`: `"Yes"`
  - `daily_screen_time_hours`: 8.5
  - `social_media_hours`: 3.5
  - `gaming_hours`: 1.5
  - `work_study_hours`: 4.0
  - `weekend_screen_time`: 11.0
  - `sleep_hours`: 5.5
  - `notifications_per_day`: 140
  - `app_opens_per_day`: 95
  - `threshold`: 0.50
- **When** the client submits a POST request to `/api/predict`
- **Then** the service must respond with HTTP `200 OK` and a JSON payload containing:
  - `probability`: a float in range $[0.0, 1.0]$
  - `prediction`: an integer $\in \{0, 1\}$ equal to $1$ if `probability` $\ge \text{threshold}$ else $0$
  - `classification`: `"ADDICTION DETECTED"` if `prediction == 1` else `"HEALTHY"`
  - `severity`: `"High"` if `probability >= 0.70`, `"Moderate"` if `0.40 <= probability < 0.70`, else `"Healthy"`
  - `ratios`: object containing `screen_to_sleep`, `recreational_to_screen`, `avg_unlock_minutes`, and `weekend_diff`
  - `interventions`: list of actionable recommendation strings
  - `latency_ms`: a positive float

#### AC-1.2: Inference Latency Performance SLA
- **Given** the FastAPI service is running with pre-loaded LightGBM Fold 1 model weights and cached population priors
- **When** 50 sequential POST requests are sent to `/api/predict`
- **Then** the 95th percentile latency ($p95$) must be $< 20\text{ms}$.

#### AC-1.3: Decision Threshold Responsiveness
- **Given** an inference payload whose predicted probability is $0.58$
- **When** the client submits `/api/predict` with `threshold = 0.50`
- **Then** `prediction` is $1$ and `classification` is `"ADDICTION DETECTED"`;
- **When** the client submits `/api/predict` with `threshold = 0.65` for the identical profile
- **Then** `prediction` is $0$ and `classification` is `"HEALTHY"` (or `"HEALTHY / BORDERLINE"`).

#### AC-1.4: Ratio Calculation Precision
- **Given** an inference payload with `daily_screen_time_hours = 8.0`, `sleep_hours = 6.0`, `social_media_hours = 3.0`, `gaming_hours = 1.0`, `app_opens_per_day = 80`, and `weekend_screen_time = 11.0`
- **When** `/api/predict` processes the request
- **Then** the returned `ratios` must exactly match:
  - `screen_to_sleep`: $8.0 / 6.0 \approx 1.3333 \pm 0.001$
  - `recreational_to_screen`: $(3.0 + 1.0) / 8.0 = 0.5000 \pm 0.001$
  - `avg_unlock_minutes`: $(8.0 \times 60) / 80 = 6.0000 \pm 0.001$
  - `weekend_diff`: $11.0 - 8.0 = 3.0000 \pm 0.001$

#### AC-1.5: Rule-Based Intervention Trigger Rules
- **Given** an inference request where:
  - `screen_to_sleep = 1.35` ($> 1.2$)
  - `recreational_to_screen = 0.65` ($> 0.60$)
  - `app_opens_per_day = 135` ($> 120$)
  - `weekend_diff = 2.8` ($> 2.5$)
- **When** `/api/predict` evaluates the payload
- **Then** the `interventions` list must contain all 4 recommendations:
  - Sleep Protection
  - Recreation Audit
  - Notification Hygiene
  - Weekend Disconnect

#### AC-1.6: Physiological Boundary Validation Rejection
- **Given** an inference request with any of the following physiological violations:
  - `sleep_hours`: $0.5$ (below minimum $1.0$) or $19.0$ (above maximum $18.0$)
  - `daily_screen_time_hours`: $-1.0$ or $25.0$ (above $24.0$)
  - `notifications_per_day`: $-5$ (negative count)
  - `age`: $15$ or $45$ (outside supported demographic cohort 18–35)
- **When** the client posts to `/api/predict`
- **Then** the service must respond with HTTP `422 Unprocessable Entity` containing field-specific validation error details.

#### AC-1.7: Total Accounted Hours UI Warning Indicator
- **Given** the user inputs `daily_screen_time_hours = 12.0`, `work_study_hours = 8.0`, and `sleep_hours = 6.0` (sum = $26.0\text{h} > 24.0\text{h}$)
- **When** the inputs are rendered on the web dashboard
- **Then** an inline physiological alert (`"Accounted hours exceed 24h/day"`) must be displayed, while the live prediction continues to compute without crashing.

#### AC-1.8: System Health & Model Readiness Endpoint
- **Given** the backend application is initialized
- **When** the client submits a GET request to `/api/health`
- **Then** the service must respond with HTTP `200 OK` and a payload containing:
  - `status`: `"healthy"`
  - `model_loaded`: `true`
  - `model_family`: `"LightGBM"`
  - `fold`: 1
  - `cohort_priors_cached`: `true`
  - `version`: `"1.0.0"`

#### AC-1.9: Graceful Fail-Fast Startup on Missing Artifacts
- **Given** the model checkpoint `models/lgb_fold_1.joblib` or required cohort prior files are missing or unreadable
- **When** the FastAPI application is initialized
- **Then** the server must log a fatal error explaining the missing file and return HTTP `503 Service Unavailable` on `/api/predict` and `/api/health` instead of crashing silently.

---

### Milestone 2: Population Cohort Analytics Explorer

#### AC-2.1: Population Cohort Metrics Retrieval
- **Given** the pre-aggregated population cache from `data/train.csv` (691,369 records)
- **When** the client submits a GET request to `/api/analytics/cohorts` with query parameters `dimension=age_bracket`
- **Then** the service responds with HTTP `200 OK` and an array of cohort summaries, each containing:
  - `cohort_name`: string (e.g., `"18-21"`, `"22-25"`)
  - `sample_count`: integer $> 0$
  - `addiction_prevalence`: float $\in [0.0, 1.0]$
  - `mean_screen_time`: float
  - `mean_sleep_hours`: float
  - `mean_app_opens`: float

#### AC-2.2: Demographic Cohort Slicing by Gender and Stress Level
- **Given** the cohort endpoint `/api/analytics/cohorts`
- **When** the client submits a request with `dimension=gender` and `filter_stress=High`
- **Then** the response contains distinct summaries for each gender (`Female`, `Male`, `Other`) conditioned on high stress, with correct relative addiction rates.

#### AC-2.3: Joint Density Screen-vs-Sleep Heatmap Matrix
- **When** the client requests GET `/api/analytics/distributions/screen-sleep-matrix`
- **Then** the service responds with HTTP `200 OK` and a 2D grid payload containing:
  - `screen_bins`: array of bin edges (e.g., $[0, 2, 4, 6, 8, 10, 12, 14, 16]$)
  - `sleep_bins`: array of bin edges (e.g., $[3, 4, 5, 6, 7, 8, 9, 10, 12]$)
  - `density_matrix`: 2D array of normalized participant counts
  - `addiction_rate_matrix`: 2D array of addiction prevalence percentages per bin cell

#### AC-2.4: Personal Quantile Benchmark Overlay Calculation
- **Given** a personal profile with `daily_screen_time_hours = 9.5` and `sleep_hours = 5.0`
- **When** the client posts this profile to `/api/analytics/distributions/benchmark-overlay`
- **Then** the service responds with HTTP `200 OK` and quantile rank metrics against 691k records:
  - `screen_time_percentile`: float $\in [0.0, 100.0]$ (e.g., $> 80.0$)
  - `sleep_duration_percentile`: float $\in [0.0, 100.0]$ (e.g., $< 20.0$)
  - `population_mean_screen`: $7.64 \pm 0.05$
  - `population_mean_sleep`: $6.80 \pm 0.05$

#### AC-2.5: Cohort Filter Error Handling
- **Given** an invalid dimension query parameter `dimension=invalid_dimension`
- **When** the client calls `/api/analytics/cohorts`
- **Then** the service responds with HTTP `422 Unprocessable Entity` or `400 Bad Request` with an explicit list of permissible dimensions (`age_bracket`, `gender`, `stress_level`, `academic_work_impact`).

---

### Milestone 3: Real-Time What-If Simulation, Batch Scoring & Craft Polish

#### AC-3.1: Counterfactual What-If Scenario Evaluation
- **Given** a baseline profile with addiction probability $P_{\text{base}} = 0.78$
- **When** the client posts a what-if request to `/api/analytics/what-if` with modifications:
  - `delta_social_media_hours`: $-2.0$
  - `delta_sleep_hours`: $+1.5$
  - `delta_app_opens_per_day`: $-30$
- **Then** the service responds with HTTP `200 OK` containing:
  - `baseline_probability`: $0.78 \pm 0.01$
  - `simulated_probability`: float $< 0.78$
  - `risk_delta`: float $< 0.0$ (equal to $\text{simulated\_probability} - \text{baseline\_probability}$)
  - `simulated_classification`: string (`"ADDICTION DETECTED"` or `"HEALTHY"`)
  - `baseline_ratios`: dictionary of original ratios
  - `simulated_ratios`: dictionary of updated ratios

#### AC-3.2: Automated Optimal Habit Target Recommendation
- **Given** an at-risk profile where $P_{\text{base}} = 0.68$ and $\tau = 0.50$
- **When** the client requests POST `/api/analytics/what-if/optimize`
- **Then** the service returns a recommended habit reduction pathway detailing:
  - Target daily screen time reduction (in hours)
  - Target sleep duration increase (in hours)
  - Projected new probability $< 0.50$

#### AC-3.3: Batch CSV Scoring Execution
- **Given** a valid CSV file containing 100 participant records with mandatory feature columns
- **When** the client uploads the file via multipart form-data to `/api/predict/batch`
- **Then** the service responds with HTTP `200 OK` and a payload containing:
  - `total_records`: 100
  - `processed_records`: 100
  - `addiction_count`: integer
  - `addiction_prevalence_pct`: float $\in [0.0, 100.0]$
  - `download_token` or inline JSON rows with predictions and probability scores

#### AC-3.4: Batch Upload Header Validation Error Handling
- **Given** an uploaded CSV file missing mandatory columns (e.g., missing `daily_screen_time_hours`)
- **When** the client posts to `/api/predict/batch`
- **Then** the service responds with HTTP `422 Unprocessable Entity` detailing the specific missing header names.

#### AC-3.5: Batch Upload Row Limit Enforcement
- **Given** an uploaded CSV file containing 15,000 rows ($> 10,000$ limit)
- **When** the client posts to `/api/predict/batch`
- **Then** the service responds with HTTP `413 Payload Too Large` with the message `"Batch upload exceeds maximum limit of 10,000 rows"`.

#### AC-3.6: Full Diagnostic CSV Export Download
- **Given** a completed batch scoring session
- **When** the client requests GET `/api/predict/batch/export?token={download_token}`
- **Then** the service returns HTTP `200 OK` with header `Content-Disposition: attachment; filename="diagnostic_report.csv"` containing all original input columns plus `predicted_probability`, `classification`, `screen_to_sleep_ratio`, and `primary_intervention`.

---

## 4. Parity Criteria (Redesigned Diagnostic View)

To ensure zero feature regression when replacing the legacy Streamlit prototype ([app/app.py](file:///c:/Users/jegad/projects/SmartphoneAddictionPrediction/app/app.py)), the redesigned Individual Diagnostic view must satisfy the following parity requirements:

- **PAR-1: Input Parameter Completeness**:
  - *Given* the legacy Streamlit form captured 12 input features (`age`, `gender`, `stress_level`, `academic_work_impact`, `daily_screen_time`, `social_media_hours`, `gaming_hours`, `work_study_hours`, `weekend_screen_time`, `sleep_hours`, `notifications_per_day`, `app_opens_per_day`) plus decision threshold $\tau$,
  - *When* a user views the redesigned diagnostic dashboard,
  - *Then* all 12 input features and threshold slider $\tau$ must be present, interactive, and defaulted to equivalent midpoint baseline values.
- **PAR-2: Behavioral Ratio Formula Parity**:
  - *Given* input values $S_{\text{daily}}, S_{\text{weekend}}, S_{\text{social}}, S_{\text{gaming}}, T_{\text{sleep}}, N_{\text{opens}}$,
  - *When* ratio metrics are computed in the backend or frontend,
  - *Then* calculations must exactly match legacy formulas:
    - $\text{Screen / Sleep} = S_{\text{daily}} / (T_{\text{sleep}} + 10^{-5})$
    - $\text{Recreational Share} = (S_{\text{social}} + S_{\text{gaming}}) / (S_{\text{daily}} + 10^{-5})$
    - $\text{Avg Session Length} = (S_{\text{daily}} \times 60) / (N_{\text{opens}} + 10^{-5})$
    - $\text{Weekend Surge} = S_{\text{weekend}} - S_{\text{daily}}$
- **PAR-3: Threshold Adjustment Behavior Parity**:
  - *Given* the threshold $\tau$ slider configured between $0.10$ and $0.90$ with step $0.01$ and default $0.50$,
  - *When* $\tau$ is altered,
  - *Then* the classification badge must immediately re-evaluate against the newly selected $\tau$ without altering the underlying probability score.
- **PAR-4: Severity Classification Tier Parity**:
  - *Given* predicted probability $P$,
  - *When* rendered in the diagnostic card,
  - *Then* the visual severity tier must map identically:
    - $P \ge 0.70 \implies \text{High Addiction Probability}$
    - $0.40 \le P < 0.70 \implies \text{Moderate / At-Risk Usage}$
    - $P < 0.40 \implies \text{Healthy Usage Pattern}$
- **PAR-5: Intervention Trigger Rule Parity**:
  - *Given* ratio values exceeding the legacy thresholds ($S/\text{Sleep} > 1.2$, $\text{Rec Share} > 0.60$, $\text{Opens} > 120$, $\text{Weekend Surge} > 2.5$),
  - *When* the diagnostic recommendations render,
  - *Then* the exact corresponding clinical advisories (Sleep Protection, Recreation Audit, Notification Hygiene, Weekend Disconnect) must be triggered.
- **PAR-6: Typographical Defect Correction**:
  - *Given* the legacy Streamlit code displayed the misspelled string `'ADDICITON DETECTED'`,
  - *When* the redesigned UI renders an addicted state,
  - *Then* it must display the correctly spelled string `'ADDICTION DETECTED'`.
- **PAR-7: Multi-Model Benchmark Table Parity**:
  - *Given* the competition multi-model benchmark table in the legacy right column,
  - *When* the user views the benchmark documentation section,
  - *Then* the 5-fold CV ROC AUC and accuracy metrics for LightGBM (0.96394), XGBoost (0.96342), CatBoost (0.96000), PyTorch Tabular ResNet (0.95750), and the Logit Ensemble (0.96410+) must remain accessible.

---

## 5. Regression Criteria (Existing ML & Data Pipeline)

The introduction of the FastAPI backend and cohort cache must not regress or mutate existing training and feature generation routines documented in [docs/CODEBASE_MAP.md](file:///c:/Users/jegad/projects/SmartphoneAddictionPrediction/docs/CODEBASE_MAP.md):

- **REG-1: Feature Engineering Pipeline Integrity**:
  - *Given* raw dataframes loaded from `data/train.csv` and `data/test.csv`,
  - *When* `src/features.py:create_features(df_train, df_test)` is executed,
  - *Then* it must produce transformed dataframes with exactly 74 feature columns and zero unexpected NaN values.
- **REG-2: Model Checkpoint Deserialization**:
  - *Given* serialized LightGBM checkpoints in `models/lgb_fold_1.joblib` through `models/lgb_fold_5.joblib`,
  - *When* loaded using `joblib.load()`,
  - *Then* each model must load without `AttributeError` or deserialization errors and provide `.predict_proba()` producing valid 2-class probability distributions.
- **REG-3: Immutability of Raw Datasets**:
  - *Given* competition datasets `data/train.csv` (691,369 rows) and `data/test.csv` (296,302 rows),
  - *When* cohort pre-aggregation scripts or analytical caches are generated,
  - *Then* the source data files in `data/` must not be overwritten, truncated, or modified in place.
- **REG-4: Ensemble Optimization Driver Stability**:
  - *Given* existing scripts [src/ensemble.py](file:///c:/Users/jegad/projects/SmartphoneAddictionPrediction/src/ensemble.py) and [src/train.py](file:///c:/Users/jegad/projects/SmartphoneAddictionPrediction/src/train.py),
  - *When* executed or imported,
  - *Then* their public signatures and command-line execution interfaces must remain fully operational.

---

## 6. Non-Goals

1. **Full 5-Fold Multi-Model Ensemble Live Scoring**:
   - Running live inference through all 5 folds of LightGBM, XGBoost, CatBoost, and PyTorch Tabular ResNet concurrently during real-time slider dragging is out of scope. Option A (Single Fold LightGBM Fold 1 with $<20\text{ms}$ latency) satisfies live user interaction requirements.
2. **User Authentication & Persistent Database Accounts**:
   - User account creation, OAuth/SSO, hashed passwords, and individual user profile persistence in SQL/NoSQL databases are excluded. The application runs locally as a self-contained analytical tool.
3. **Automated Online Model Retraining from UI**:
   - Triggering 5-fold cross-validation or hyperparameter optimization from the web browser is out of scope. Retraining remains an offline script execution workflow via `src/train.py`.
4. **Native Mobile Applications**:
   - Native iOS or Android mobile applications are excluded. The interface is optimized for responsive modern desktop and tablet web browsers.
5. **Real-Time Mobile OS Telemetry Streaming**:
   - Integrating live OS background daemons or Screen Time APIs to stream device metrics in real-time is out of scope. Input is driven by interactive sliders and batch CSV uploads.
6. **Maintenance of Legacy Streamlit Runtime**:
   - Streamlit runtime dependencies and [app/app.py](file:///c:/Users/jegad/projects/SmartphoneAddictionPrediction/app/app.py) will be retired in favor of the unified FastAPI service and bespoke HTML5/CSS3/JS interface.

---

## 7. Open Questions & Assumptions

1. **Cohort Cache Generation & Storage**:
   - *Question*: Should the hybrid cohort analytics cache be generated dynamically on first server start or generated ahead of time as a standalone cache artifact?
   - *Assumption*: Provide a standalone cache builder utility (`src/cohort_cache.py` or similar) that saves pre-aggregated distributions to `data/cohort_cache.json` (or `.parquet`). If the file is not present when FastAPI starts, compute it in-memory once and write the cache to disk.
2. **Web Frontend Asset Delivery**:
   - *Question*: Should the web dashboard rely on a Node.js / React build pipeline (e.g., Vite/Webpack) or static vanilla assets mounted directly by FastAPI?
   - *Assumption*: Serve vanilla HTML5, modern CSS3, and JavaScript directly via FastAPI's `StaticFiles` mount under `app/static/` and `app/templates/`. This guarantees zero external Node/npm dependencies, works immediately in any Python environment, and facilitates rapid inspection.
3. **Batch Upload Size Limits**:
   - *Question*: What is the appropriate upper threshold for batch diagnostic CSV uploads on local hardware?
   - *Assumption*: Cap batch CSV uploads at 10,000 records per request to maintain sub-second scoring throughput and prevent memory exhaustion on consumer workstations.
