# User Feedback Log

## Round 1 (Prototype Review)
- **Date**: 2026-10-06
- **Status**: Resolved via Impeccable Polish Pass
- **Items**:
  1. **UI - Gauge Geometry & Text Collision**: [RESOLVED] Adjusted `.gauge-container` height to 125px, increased label margin (`14px auto 14px`), and vertically centered readout. Clean clearance between arc feet and label.
  2. **UI - Redundant Status Badges**: [RESOLVED] Replaced dual `badge-severity` and `badge-classification` pills with a single authoritative status pill (`badge-status`) mapped to a 3-tier clinical spectrum (`Balanced Habit Profile`, `Compensatory Usage Pattern`, `Elevated Risk Tier`).
  3. **Behavior - Probability Calibration**: [RESOLVED] Re-calibrated client fallback logistic weights in `app.js` to match backend model priors, generating a grounded risk score (~45-55% for average habits).

