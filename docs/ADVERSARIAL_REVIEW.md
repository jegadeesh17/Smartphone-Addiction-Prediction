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
