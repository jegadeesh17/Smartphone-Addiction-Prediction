# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Setup test runner, execution markers (`slow`), and environment manifest (`pytest.ini`, `tests/conftest.py`, `tests/test_baseline.py`) (M1-TASK-01).
- Implemented Pydantic v2 domain schemas and clinical behavioral recommendation rules (`src/schemas.py`, `src/recommendations.py`) (M1-TASK-02).
- Generated population priors cache and single-row 74-feature vectorized transformer (`scripts/generate_priors.py`, `data/priors.json`, `src/priors.py`) (M1-TASK-03).
- Implemented LightGBM Fold 1 live inference engine with <2.5ms latency and calibrated mock adapter (`src/inference.py`, `src/mock_adapter.py`) (M1-TASK-04).
- Built modular FastAPI asynchronous application with `/health` and `/api/predict` endpoints (`src/main.py`, `src/config.py`, `src/api/health.py`, `src/api/predict.py`) (M1-TASK-05).
- Established editorial design tokens, responsive app shell, and multi-model 5-fold CV benchmark table (`src/static/css/tokens.css`, `src/static/css/style.css`, `src/static/css/app.css`, `src/templates/index.html`) (M1-TASK-06).
- Built Individual Diagnostic view with reactive 220-degree SVG risk gauge, ratio cards, dynamic interventions, and 24h physiological limit warning alert (`src/static/js/app.js`, `src/static/js/charts.js`) (M1-TASK-07).
- Pre-aggregated population cohort distributions and 2D density grid cache from 691,369 participant records (`scripts/generate_cohort_cache.py`, `data/cohort_summary.json`) (M2-TASK-01).
- Implemented in-memory Cohort Analytics Service and empirical personal quantile overlay logic (`src/cohort_service.py`, `src/schemas.py`) (M2-TASK-02).
- Implemented Cohort Analytics API router and REST endpoints (`/api/analytics/cohorts`, `/api/analytics/distributions/screen-sleep-matrix`, `/api/analytics/distributions/benchmark-overlay`) with 422 input validation (`src/api/analytics.py`) (M2-TASK-03).
- Built Population Cohort Analytics Explorer UI view, 2D Screen-vs-Sleep joint density heatmap, and personal quantile benchmark progress tracks (`src/templates/index.html`, `src/static/js/app.js`, `src/static/js/charts.js`, `src/static/css/style.css`) (M2-TASK-04).

- Implemented counterfactual What-If simulation engine and habit optimizer (M3-TASK-01).
- Implemented batch CSV diagnostic scoring, validation (row-level 422 errors, physiological bounds, 5 MB / 10,000-row caps) and report export with spreadsheet-formula sanitising (M3-TASK-02).
- Built What-If Scenario Simulator and Batch Diagnostics UI views, including template CSV download and mobile card layout (M3-TASK-03).
- Frontend craft polish pass, responsive hardening, 44px touch targets and accessibility compliance (M3-TASK-04).
- Comprehensive regression safeguards and full pipeline verification (M3-TASK-05).

### Changed
- Brand accent changed from golden yellow to deep indigo (`--gold` token values in `tokens.css`; token name kept).

### Fixed
- Decoupled classification threshold ($\tau$) from intrinsic severity and risk tier mapping per SPEC AC-1.1, PAR-3, and PAR-4.
- Rendered single authoritative diagnostic verdict (`ADDICTION DETECTED` vs `HEALTHY PATTERN`) in UI status pill and narrative summary per ADR-0006.
- Aligned behavioral ratio target thresholds for Avg Session Length (> 5.0 min) and Weekend Surge (< 2.0h) per SPEC Journey 1.
- Batch scoring no longer returns HTTP 500 on non-numeric or infinite cells; invalid rows return a readable 422 naming rows and columns.
