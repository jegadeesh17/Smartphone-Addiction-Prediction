# Architecture Decision Records
Living record of architectural and design decisions for the Smartphone Addiction Prediction analytical platform.

| ID | Title | Status | Date |
| :--- | :--- | :--- | :--- |
| [ADR-0001](#adr-0001-replacement-of-legacy-streamlit-prototype-with-fastapi-and-bespoke-editorial-web-ui) | Replacement of Legacy Streamlit Prototype with FastAPI and Bespoke Editorial Web UI | Accepted | 2026-10-06 |
| [ADR-0002](#adr-0002-real-time-tabular-inference-via-lightgbm-fold-1-and-cached-population-priors) | Real-Time Tabular Inference via LightGBM Fold 1 and Cached Population Priors | Accepted | 2026-10-06 |
| [ADR-0003](#adr-0003-hybrid-in-memory-analytical-cache-for-population-cohort-exploration) | Hybrid In-Memory Analytical Cache for Population Cohort Exploration | Accepted | 2026-10-06 |
| [ADR-0004](#adr-0004-automated-testing-framework-with-pytest-async-testclient-and-execution-markers) | Automated Testing Framework with Pytest, Async TestClient, and Execution Markers | Accepted | 2026-10-06 |
| [ADR-0005](#adr-0005-decoupled-mock-adapter-for-frontend-prototyping-and-isolated-testing) | Decoupled Mock Adapter for Frontend Prototyping and Isolated Testing | Accepted | 2026-10-06 |
| [ADR-0006](#adr-0006-consolidation-of-single-authoritative-diagnostic-status-pill) | Consolidation of Single Authoritative Diagnostic Status Pill | Accepted | 2026-10-06 |
| [ADR-0007](#adr-0007-baseline-centered-heuristic-probability-calibration-for-mock-adapter) | Baseline-Centered Heuristic Probability Calibration for Mock Adapter | Accepted | 2026-10-06 |
| [ADR-0008](#adr-0008-svg-risk-gauge-geometry-and-text-clearance-layout-contract) | SVG Risk Gauge Geometry and Text Clearance Layout Contract | Accepted | 2026-10-06 |

---

## ADR-0001: Replacement of Legacy Streamlit Prototype with FastAPI and Bespoke Editorial Web UI
Date: 2026-10-06  
Status: Accepted

### Context
The initial application prototype was built as a single Python script using Streamlit (`app/app.py`). While Streamlit enables rapid exploratory prototyping, its reactive execution model re-executes the entire Python script upon every user slider change. This causes noticeable interaction latency (>150ms), prevents sub-20ms responsiveness during continuous slider adjustments, and restricts layout, typography, and styling customization. Furthermore, the Streamlit prototype did not invoke the trained machine learning models, relying instead on a hardcoded linear equation.

### Decision
Replace the Streamlit runtime with a production-grade FastAPI asynchronous backend and a bespoke, responsive web dashboard built with HTML5, CSS3, modern vanilla JavaScript, and Chart.js. The legacy Streamlit script is preserved in `app/app.py` for historical baseline comparison but will not be used in production serving.

### Alternatives considered
1. **Retain and optimize Streamlit**: Streamlit's architecture fundamentally couples widget interaction with full-script re-runs, making true sub-20ms slider latency and bespoke editorial layout unattainable.
2. **Heavy Single-Page App (React / Next.js with Node.js runtime)**: Introducing Node.js, package managers, and a frontend build step adds operational complexity, slower build cycles, and external runtime requirements for an application whose interactivity is fully achievable with vanilla JavaScript.

### Consequences
- **Positive**: Enables instant (<20ms) slider responsiveness, editorial typography control (IBM Plex Sans & Source Serif 4), clear separation between API and UI layers, and zero Node build dependencies for Python deployments.
- **Negative**: Requires writing standard HTML/CSS templates and JavaScript client logic instead of using high-level Python widget abstractions.

---

## ADR-0002: Real-Time Tabular Inference via LightGBM Fold 1 and Cached Population Priors
Date: 2026-10-06  
Status: Accepted

### Context
The competitive machine learning pipeline trained a 20-model ensemble across 5 folds (LightGBM, XGBoost, CatBoost, and PyTorch Tabular ResNet). Loading all 20 models into memory requires >2GB of RAM and takes 250–400ms to compute an ensembled prediction. For an interactive diagnostic dashboard where users adjust sliders continuously, this latency causes stutter. In contrast, LightGBM Fold 1 (`models/lgb_fold_1.joblib`) independently achieves an exceptional ROC AUC of >0.963 while requiring less than 50MB of memory.

However, the feature extraction module (`src/features.py`) derives 74 domain and cohort features using dataset-level aggregations (group means, standard deviations, and frequency encodings across all 691,369 participants). When evaluated on a single incoming user profile, these group operations fail: standard deviations become `NaN`, frequency encodings collapse to 1.0, and category codes produce arbitrary mappings.

### Decision
Serve LightGBM Fold 1 as the primary live diagnostic model, paired with a pre-computed population priors cache (`data/priors.json`). The priors cache stores population medians, category frequency mappings, and cohort group statistics extracted from the 691k training dataset. The single-row inference engine uses these cached priors to calculate the exact 74 features in under 1ms, feeding them directly into LightGBM Fold 1.

### Alternatives considered
1. **Full 20-model ensemble evaluation during live slider interactions**: Results in severe latency (>300ms) and large server memory overhead (>2GB), harming the user experience.
2. **Re-calculating cohort statistics dynamically against `data/train.csv` per request**: Causes heavy disk reading and high CPU usage on every slider event, resulting in ~400ms latency.
3. **Retraining a stripped-down model without cohort interactions**: Sacrifices predictive performance and clinical fidelity to work around a feature engineering challenge that is cleanly solved by cached priors.

### Consequences
- **Positive**: Reduces single-user inference latency to under 10ms (well within the <20ms target), lowers memory usage to ~150MB, and faithfully mirrors the feature distribution used during model training.
- **Negative**: Requires an offline script (`scripts/generate_priors.py`) to extract and update `data/priors.json` whenever the base training data changes.

---

## ADR-0003: Hybrid In-Memory Analytical Cache for Population Cohort Exploration
Date: 2026-10-06  
Status: Accepted

### Context
A core feature of the platform is exploring smartphone dependency across demographic and psychological cohorts (Age, Gender, Stress Level, Academic Impact) based on 691,369 participant records in `data/train.csv` (~45MB). If the backend scanned and filtered the raw CSV file on every HTTP request to `/api/analytics/cohorts`, response times would exceed 300–800ms and cause CPU spikes under concurrent user traffic.

### Decision
Adopt a hybrid in-memory analytical cache architecture:
1. **Pre-aggregated cohort summary cache (`data/cohort_summary.json`)**: Pre-compute distribution histograms, percentiles (10th, 25th, 50th, 75th, 90th), and addiction prevalence rates across primary demographic cross-sections. This is loaded into memory on server startup, providing sub-2ms response times.
2. **On-demand sample fallback**: For multi-parameter edge queries not covered by the pre-aggregated summary, an in-memory stratified sample (or Arrow table) resolves ad-hoc queries in under 15ms.

### Alternatives considered
1. **Querying raw CSV directly on every request**: High latency, high CPU consumption, and poor concurrent scaling.
2. **Provisioning an external database (e.g., PostgreSQL or ClickHouse)**: Adds infrastructure overhead, network hops, container management, and configuration complexity for static analytical data.
3. **Downloading the full dataset to the client browser**: Impractical for mobile users and low-bandwidth connections (~45MB download).

### Consequences
- **Positive**: Eliminates external database dependencies, guarantees sub-2ms responses for standard cohort analytics views, and minimizes memory footprint (~5MB JSON in RAM).
- **Negative**: Requires an offline script (`scripts/generate_cohort_cache.py`) to regenerate `data/cohort_summary.json` if the underlying population dataset is updated.

---

## ADR-0004: Automated Testing Framework with Pytest, Async TestClient, and Execution Markers
Date: 2026-10-06  
Status: Accepted

### Context
The repository previously contained zero automated tests; running `pytest` exited with code 1 due to zero collected tests. Transitioning from an exploratory prototype to a production platform requires comprehensive automated verification covering Pydantic schemas, single-row feature math, model inference accuracy, cohort cache queries, and HTTP endpoints. At the same time, running tests that scan large datasets or benchmark disk I/O would slow down developer feedback loops.

### Decision
Establish an automated test suite using `pytest`, `pytest-asyncio`, and `httpx.TestClient`. Configure `pytest.ini` with a custom `@pytest.mark.slow` marker. The fast test suite (`pytest -q -m "not slow"`) executes all unit, schema, and API integration tests in under 5 seconds and runs after every development task. Slow tests (full training set re-processing, model cold starts, large batch benchmarks) run selectively via `pytest -q`.

### Alternatives considered
1. **Python standard `unittest` library**: Lacks concise fixture syntax, rich parameterization, and native async support found in pytest.
2. **Single monolithic test suite without markers**: Forces developers to wait for heavy dataset parsing on simple code edits, discouraging frequent local testing.

### Consequences
- **Positive**: Provides rapid, automated feedback (<5s), protects against mathematical and API regressions, and gives clear verification gates before code commits.
- **Negative**: Developers must remember to add `@pytest.mark.slow` to long-running tests to preserve fast test suite execution speed.

---

## ADR-0005: Decoupled Mock Adapter for Frontend Prototyping and Isolated Testing
Date: 2026-10-06  
Status: Accepted

### Context
Iterating on the frontend dashboard layout, testing responsive design across viewports (1440px and 390px), and running automated Playwright visual screenshot verifications should not depend on loading a 50MB LightGBM model file into memory. In addition, automated CI environments or lightweight test runners benefit from verifying HTTP routes and validation rules without requiring large binary model checkpoints.

### Decision
Implement a decoupled `MockInferenceEngine` in `src/mock_adapter.py` that implements the exact same Python interface as the production inference engine. It returns realistic behavioral metrics, simulated risk probabilities, and rule-based recommendations in under 1ms without loading LightGBM. The application toggles between real and mock modes using the environment variable `USE_MOCK_MODEL=true`.

### Alternatives considered
1. **Requiring model weights in all environments**: Increases cold-start time and memory consumption during visual layout development and simple frontend testing.
2. **Ad-hoc monkeypatching inside test files**: Results in scattered, duplicated mock logic and does not allow running the live web dashboard in mock mode.

### Consequences
- **Positive**: Enables instant startup during frontend UI styling and screenshot captures, eliminates model dependencies in lightweight test environments, and ensures zero drift between mock and real interfaces.
- **Negative**: The mock adapter logic must be maintained in sync with any future updates to the real prediction interface.

---

## ADR-0006: Consolidation of Single Authoritative Diagnostic Status Pill
Date: 2026-10-06  
Status: Accepted

### Context
During Round 1 prototype user review, feedback highlighted that presenting two distinct visual status badges side-by-side (`HIGH ADDICTION PROBABILITY` and `ADDICTION DETECTED`) created visual clutter and cognitive redundancy. Users require an immediate, unambiguous diagnostic verdict that conveys both the threshold decision and severity tier without competing UI elements.

### Decision
Consolidate the diagnostic status display into a single authoritative status pill component (`div.status-pill`) driven by a unified `status_label` field in `PredictionResponse`. The pill communicates both decision and risk level (e.g., `ADDICTION DETECTED • HIGH RISK`, `HEALTHY PATTERN • LOW RISK`, or `BORDERLINE PATTERN • MODERATE RISK`) using semantic color tokens.

### Alternatives considered
1. **Retain dual badges side-by-side**: Causes layout clutter, especially on compact mobile viewports (390px), and forces users to parse two overlapping status indicators.
2. **Display only binary threshold badge (`ADDICTION DETECTED`)**: Omits critical risk severity nuance (differentiating high-risk from moderate/borderline risk).

### Consequences
- **Positive**: Unambiguous diagnostic presentation, cleaner visual hierarchy, reduced cognitive friction, and streamlined mobile layout.
- **Negative**: Frontend rendering logic and API schemas must maintain the combined `status_label` string in sync with classification status and tier.

---

## ADR-0007: Baseline-Centered Heuristic Probability Calibration for Mock Adapter
Date: 2026-10-06  
Status: Accepted

### Context
In Round 1 prototype testing, the baseline heuristic scoring formula produced an extreme `99.0%` addiction probability when evaluated against population midpoint inputs (7.5h daily screen time, 6.8h sleep, 140 notifications, 100 app opens). Assignating an extreme 99% risk score to an average user profile distorts clinical perception and impairs prototype review feedback.

### Decision
Calibrate the mock inference engine using a centered logit equation where each input feature is centered against its population median (screen time: 7.5h, sleep: 6.8h, notifications: 140, app opens: 100, weekend screen: 9.5h, social media: 3.5h, gaming: 1.2h). At median habits, the logit evaluates to $\approx 0.02$, yielding a calibrated baseline risk probability of $\approx 50.5\%$ (within the target $45\% - 55\%$ range). Deviations above or below baseline smoothly scale the probability across $[0.02, 0.98]$.

### Alternatives considered
1. **Uncentered additive scoring**: Highly susceptible to saturation at normal bounds, leading directly to the extreme 99% baseline defect.
2. **Hardcoded static return for default profile**: Produces an unnatural plateau that does not respond realistically when users make small slider adjustments around the baseline.

### Consequences
- **Positive**: Guarantees clinically plausible baseline risk scoring (~50%) for median habits while maintaining responsive, proportional sensitivity as sliders are manipulated.
- **Negative**: Mock adapter implementation requires maintaining population median constants alongside the scoring coefficients.

---

## ADR-0008: SVG Risk Gauge Geometry and Text Clearance Layout Contract
Date: 2026-10-06  
Status: Accepted

### Context
Round 1 visual evaluation revealed that the caption label `ADDICTION RISK PROBABILITY` collided with and overlapped the lower terminal feet of the circular arc gauge. This collision was caused by negative CSS margins pulling the caption into the SVG bounding box, coupled with inadequate vertical canvas padding.

### Decision
Establish an explicit visual and layout contract for the SVG Risk Gauge component:
1. Render the SVG gauge with a $220^\circ$ sweep in an explicit viewBox of `0 0 240 170`, ensuring the arc feet terminate at $Y \approx 149.1\text{px}$, leaving over $20\text{px}$ of internal canvas headroom.
2. Render the sub-gauge label `ADDICTION RISK PROBABILITY` in a dedicated HTML container (`div.gauge-caption`) located entirely outside the SVG element.
3. Enforce positive top margin (`margin-top: 1rem;`) and strictly prohibit negative margin offsets, guaranteeing at least $32\text{px}$ of visual vertical clearance between the arc feet and caption text.

### Alternatives considered
1. **Embedding caption text inside SVG as an SVG `<text>` node**: Creates cross-browser typography rendering inconsistencies, complicates responsive font scaling, and hampers screen reader accessibility.
2. **Semi-transparent overlay banner**: Clutters the circular arc visual and violates WCAG contrast guidelines.

### Consequences
- **Positive**: Pixel-perfect rendering across desktop (1440px) and mobile (390px) viewports with guaranteed clearance preventing text collision.
- **Negative**: Layout requires strict synchronization between SVG canvas height and external container margin styling.
