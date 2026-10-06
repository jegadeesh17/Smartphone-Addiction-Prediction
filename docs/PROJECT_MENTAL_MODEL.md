# Project Mental Model: Smartphone Addiction Prediction Platform

## 1. Project Context & Objectives
- **Project**: Smartphone Addiction Prediction Analytical Dashboard
- **Vision & Change**: Replace the legacy single-file Streamlit script with a production FastAPI backend and a bespoke, responsive web dashboard with live tabular ML inference, hybrid cohort analytics, what-if behavioral simulation, and editorial design inspired by the reference web app.
- **Change Type**: `Redesign` & `Feature`
- **Posture**: `Production`
- **Target Persona**: Digital well-being analysts, clinical researchers, and individuals seeking clear behavioral risk diagnostics with actionable habit interventions.

---

## 2. Technical Decisions & Invariants
- **Inference Strategy**: Option A (Fast Single Fold - LightGBM fold 1 from `models/lgb_fold_1.joblib` via `src/features.py`), achieving $<20\text{ms}$ live inference latency.
- **Data Architecture**: Option 2 Hybrid (Pre-aggregated cohort summary metrics and distributions from `data/train.csv` stored in a fast lightweight analytical cache, with on-demand sample data for deep slicing).
- **Backend Architecture**: FastAPI with Pydantic v2 validation, modular routers (`/api/predict`, `/api/analytics/cohorts`, `/api/analytics/what-if`, `/api/health`, `/api/predict/batch`).
- **Frontend Architecture**: Bespoke HTML5 / CSS3 / Vanilla JS single-page application with responsive tabs, IBM Plex Sans / Source Serif typography, responsive metric cards, Chart.js / SVG data visualizations, and instant client-side updates without page reloads.

---

## 3. Repository & Git Metadata
- **Repository Remote**: `https://github.com/jegadeesh17/Smartphone-Addiction-Prediction.git`
- **Working Branch**: `feature/analytical-dashboard`
- **Base Commit**: `b8b89352ab428366beca33997ef54f1d2e2225cc`
- **Codebase Map**: [docs/CODEBASE_MAP.md](file:///c:/Users/jegad/projects/SmartphoneAddictionPrediction/docs/CODEBASE_MAP.md)

---

## 4. Project Facts (Routing Flags)
- `ui`: **yes** (FastAPI-served web UI with interactive data visualizations)
- `sensitive-data`: **no** (anonymized/synthetic behavioral competition data)
- `deploy`: **no** (local execution with standard Uvicorn server, container-ready)
- `existing-code`: **yes** (existing ML models, training pipelines, and Kaggle dataset)

---

## 5. Milestone Roadmaps
- **M1: Modular FastAPI Backend, Live Inference & Risk Diagnosis Engine**
  - Implement FastAPI application layout and Pydantic schemas.
  - Implement inference service loading LightGBM fold 1 and running 74-feature transformation.
  - Generate pre-aggregated cohort summaries from `data/train.csv` for hybrid fast loading.
  - Build initial web dashboard view: Executive Overview & Risk Assessment with live model scoring.
  - Setup automated pytest suite (`tests/test_features.py`, `tests/test_inference.py`, `tests/test_api.py`).
- **M2: Population Cohort Analytics Explorer & Interactive Visualizations**
  - Implement `/api/analytics/cohorts` and `/api/analytics/distributions` endpoints.
  - Build Population Analytics tab with demographic cohort comparisons, screen-vs-sleep heatmaps, and unlock habit distributions.
  - Add benchmark overlays comparing personal metrics to 691k population quantiles.
- **M3: Real-Time What-If Simulation, Batch Scoring & Craft Polish**
  - Implement `/api/analytics/what-if` and rule-based behavioral interventions.
  - Build interactive What-If scenario simulator with real-time risk delta computation.
  - Implement batch CSV diagnostic upload and report export.
  - Polish editorial typography, responsive design, and status telemetry.

---

## Part 1: Requirements and UX (Spec Interview)
- **Primary Journey**:
  1. User accesses the dashboard at `http://localhost:8000/app` or `/`.
  2. Page renders an Executive Header with editorial styling, followed by two core analytical zones:
     - **Zone 1: Behavioral Profile Controls**: Sliders and segmented controls for demographics (`Age`, `Gender`, `Stress Level`, `Academic/Work Impact`), screen allocations (`Daily Screen`, `Social Media`, `Gaming`, `Weekend Screen`, `Sleep`), and unlock patterns (`App Opens`, `Notifications`).
     - **Zone 2: Live Diagnostic Card & Population Benchmarks**: Instant reactive risk gauge displaying addiction probability calculated via LightGBM Fold 1 using cached population priors (<20ms latency), dynamic classification badge (`ADDICTION DETECTED` vs `HEALTHY` at adjustable threshold $\tau$), core behavioral ratios (`Screen/Sleep Ratio`, `Recreational Share`, `Avg Session Length`, `Weekend Surge`), and rule-based digital hygiene recommendations.
  3. Header tab switcher enables flipping between Individual Diagnostic, Population Analytics (M2), and What-If Simulation (M3).
- **Errors and Edge Cases**:
  - API validation via Pydantic v2 enforces physiological limits (`sleep_hours`: 1.0–18.0, `daily_screen_time`: 0.0–24.0, non-negative counts).
  - If total allocated hours (screen + sleep) exceed 24h, UI displays an inline visual constraint indicator warning without blocking exploratory input.
  - If model artifacts or priors fail to load at startup, application fails fast with structured error logging.
- **Non-Goals (Out of Scope for M1)**:
  - Full 5-fold heavy ensemble evaluation during live slider interactions (Option A LightGBM fold 1 is used).
  - What-If scenario simulations and batch CSV scoring (deferred to M3).
  - User authentication, session databases, or external cloud deployments.
  - Streamlit runtime (Streamlit is completely removed and replaced with FastAPI + custom web UI).

