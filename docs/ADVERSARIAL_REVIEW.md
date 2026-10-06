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
1. **Conflation of Decision Threshold ($\tau$) with Intrinsic Risk Severity**: In `src/inference.py` and `src/mock_adapter.py`, `severity` / `risk_tier` is incorrectly assigned to `"HIGH"` whenever `probability >= threshold`, rather than mapping strictly to probability $P$ ($P \ge 0.70 \to \text{High}$, $0.40 \le P < 0.70 \to \text{Moderate}$, $P < 0.40 \to \text{Healthy}$). Consequently, an average user with $P = 52.8\%$ is misclassified as `"HIGH"` risk, and adjusting the threshold slider mutates the user's intrinsic severity tier.
2. **Missing Authoritative Classification Verdict in UI**: In `src/static/js/charts.js` and `src/static/js/app.js`, the UI ignores `status_label` from the API and never renders `"ADDICTION DETECTED"` or `"HEALTHY"` anywhere in the diagnostic assessment card, breaking ADR-0006 and PAR-6.

---

### Critical Defects
1. `src/inference.py` & `src/mock_adapter.py`: Conflation of classification decision threshold with intrinsic risk severity tier violating SPEC AC-1.1 and PAR-4.
2. `src/static/js/charts.js`, `src/static/js/app.js`, & `src/templates/index.html`: UI status pill omits definitive classification verdict ("ADDICTION DETECTED" / "HEALTHY"), breaking ADR-0006 and PAR-6.

---

### Major Defects
1. `src/static/js/app.js` & `src/templates/index.html`: Behavioral ratio target discrepancy against SPEC Journey 1 (Avg Session Length target should be >5.0 min, Weekend Surge target should be <2.0h).
2. `tests/test_inference.py`: Unit tests asserted threshold-severity coupling instead of independent severity evaluation per SPEC criteria.

