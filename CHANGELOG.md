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

### Fixed
- Decoupled classification threshold ($\tau$) from intrinsic severity and risk tier mapping per SPEC AC-1.1, PAR-3, and PAR-4.
- Rendered single authoritative diagnostic verdict (`ADDICTION DETECTED` vs `HEALTHY PATTERN`) in UI status pill and narrative summary per ADR-0006.
- Aligned behavioral ratio target thresholds for Avg Session Length (> 5.0 min) and Weekend Surge (< 2.0h) per SPEC Journey 1.
