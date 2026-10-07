# Adversarial Reviews

## Milestone M1: Initial Review (Attempt 1)
- **Date**: 2026-10-06
- **Verdict**: REJECTED
- **Test Command**: `pytest -q` -> exit code 0 (164 passed)

### Summary of Evaluation
Milestone M1 establishes the core foundational infrastructure:
- Modular FastAPI server architecture, Pydantic v2 schemas with physiological validation bounds, static asset delivery, and responsive editorial layout.
- High-performance LightGBM Fold 1 single-row inference engine with cached population priors meeting the <20ms p95 latency SLA.
- Immutable data boundaries preserving raw datasets (`train.csv`, `test.csv`) and model weights.
- 164 passing tests across unit, integration, and feature math suites.

However, the audit revealed two critical defects involving contract drift, broken diagnostic logic, and violations of SPEC.md (AC-1.1, PAR-3, PAR-4, PAR-6) and DECISIONS.md (ADR-0006):
1. Conflation of Decision Threshold ($\tau$) with Intrinsic Risk Severity: `severity` / `risk_tier` was incorrectly coupled to whether probability >= threshold.
2. Missing Authoritative Classification Verdict in UI: Status pill omitted definitive binary classification.

---

## Milestone M1: Re-Review (Attempt 2)
- **Date**: 2026-10-06
- **Verdict**: APPROVED
- **Test Command**: `pytest -q` -> exit code 0 (164 passed in 18.70s)

### Summary of Evaluation of Fixes
A comprehensive adversarial re-audit of Milestone M1 was conducted following the defect resolution in commit `92ad2b0`:
1. **Decoupled Classification Threshold ($\tau$) from Intrinsic Severity & Risk Tier**:
   - In `src/inference.py` and `src/mock_adapter.py`, `risk_tier` and `severity` are derived strictly from predicted probability $P$ (`classify_risk_tier` and `classify_severity`: $P \ge 0.70 \implies \text{High/HIGH}$, $0.40 \le P < 0.70 \implies \text{Moderate/MODERATE}$, $P < 0.40 \implies \text{Healthy/LOW}$).
   - Decision threshold $\tau$ strictly governs binary diagnosis (`is_addicted = prob >= threshold`, `classification = "ADDICTION DETECTED" | "HEALTHY"`).
   - Adjusting $\tau$ dynamically flips diagnostic classification without mutating intrinsic severity, satisfying SPEC AC-1.1, AC-1.3, PAR-3, and PAR-4.
2. **Authoritative Diagnostic Verdict in UI**:
   - In `src/static/js/charts.js` and `src/static/js/app.js`, the single authoritative status pill (`#status-pill`, `#badge-status`) renders the consolidated verdict (`"ADDICTION DETECTED • <TIER> RISK"` or `"HEALTHY PATTERN • <TIER> RISK"`), driven directly by `PredictionResponse.status_label`.
   - The diagnostic summary text (`#classification-summary`) renders explicit narrative decisions including threshold $\tau$.
3. **Behavioral Ratio Target Alignment**:
   - Target criteria across `src/templates/index.html` and `src/static/js/app.js` strictly match SPEC Journey 1:
     - `Avg Session Length`: Target $> 5.0\text{ min}$
     - `Weekend Surge`: Target $< 2.0\text{h}$
     - `Screen / Sleep Ratio`: Target $< 1.0\text{x}$
     - `Recreational Share`: Target $< 50.0\%$

### Test Suite Verification Note
- Full test suite execution: `pytest -q` exited with code 0 (`164 passed in 18.70s`).
- Sub-20ms inference latency benchmark confirmed: LightGBM Fold 1 p95 latency is $< 2.5\text{ms}$.
- Zero regression against raw datasets (`data/train.csv` and `data/test.csv` immutable).

---

## Milestone M2: Population Cohort Analytics & 2D Density Heatmap
- **Date**: 2026-10-07
- **Verdict**: APPROVED
- **Test Command**: `pytest -q` -> exit code 0 (224 passed in 41.56s)

### Summary of Evaluation
Milestone M2 was audited across all specification requirements (Journey 2, AC-2.1 through AC-2.5), architectural constraints (ADR-0003 hybrid in-memory analytical cache), schema contracts, data immutability, error handling, performance SLAs, and regression invariants.
1. **Journey 2 & Acceptance Criteria (AC-2.1 – AC-2.5)**:
   - `GET /api/analytics/cohorts` returns HTTP 200 with all 4 primary demographic dimensions (`age_bracket`, `gender`, `stress_level`, `academic_work_impact`) with full sample metrics.
   - Conditioned slicing on `filter_stress` and `filter_gender` verified against 691,369 training records.
   - `GET /api/analytics/distributions/screen-sleep-matrix` delivers 6x5 matrix (30 cells) with joint density and addiction rates summing to ~100%.
   - `POST /api/analytics/distributions/benchmark-overlay` delivers personal empirical quantile rankings against 691k population curves.
   - Invalid dimension query parameters return HTTP 422 Unprocessable Entity with explicit list of permissible dimensions.
2. **Data Immutability (REG-3)**:
   - Zero modifications to source datasets (`data/train.csv` and `data/test.csv`).
3. **Performance Invariants (ADR-0003)**:
   - In-memory lookups execute in ~0.05ms (exceeding sub-2ms SLA).
   - In-memory client caching eliminates redundant network traffic.
4. **M1 Regression Check**:
   - Zero regressions on M1 endpoints and test suites; full test suite (224 tests) passes with exit code 0.

---

## Milestone M3: What-If Simulation, Batch Scoring & Craft Polish
- **Date**: 2026-10-07
- **Verdict**: APPROVED (after 1 rejected review)
- **Test Command**: `pytest -q` -> exit code 0 (326 passed)

### Review 1: REJECTED
- Batch CSV with non-numeric or infinite cells returned HTTP 500 instead of a row-level 422.
- No server-side physiological bounds on batch rows (negative or zero sleep accepted), inconsistent with `BehavioralProfileInput`.
- Upload fully read and parsed before any byte cap.

### Review 2: APPROVED
- All three defects fixed in `src/api/predict.py` and verified by probe: 422 with row/column detail, bounds mirror the single-profile schema, 5 MB cap returns 413 before parsing.
- Export cells starting with `=`, `+`, `-`, `@` are prefixed with `'`.
- No regressions (326 tests pass).
- Residual low-severity items (non-blocking): header-name and tab/CR formula injection not neutralised; categorical batch columns (`gender`, `stress_level`, `academic_work_impact`) not validated; export cache is per-process in-memory.
