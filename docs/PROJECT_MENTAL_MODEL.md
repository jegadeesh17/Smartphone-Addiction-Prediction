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
*(To be populated in Phase 1 upon user confirmation)*
