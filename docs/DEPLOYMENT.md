# Deployment — Smartphone Addiction Analytical Platform

The service runs on **Google Cloud Run** (`smartphone-addiction-api`, region `asia-south1`), deployed automatically via GitHub Actions.

> **Continuous Delivery**: A push to `main` (outside docs/markdown) runs tests, builds the production container, pushes it to Google Artifact Registry, and deploys to Cloud Run.

---

## 1. Pipeline Overview

| Workflow | Trigger | What It Does |
| :--- | :--- | :--- |
| `.github/workflows/ci.yml` | Every push and pull request | Installs Python 3.11 and `requirements-dev.txt`, and runs fast tests (`pytest -q -m "not slow"`) |
| `.github/workflows/deploy.yml` | Push to `main`, or manual `workflow_dispatch` | Verifies tests, builds multi-stage Docker container, pushes to Artifact Registry, and deploys to Cloud Run |

### Deployment Steps:
1. **Automated Test Validation**: GitHub Actions runs the fast pytest suite to verify all APIs and schemas pass before building.
2. **Authenticate with GCP**: Authenticates via `google-github-actions/auth` using the repository secret `GCP_SA_KEY`.
3. **Container Build & Push**: Builds the multi-stage Docker image and pushes it to Artifact Registry:
   `asia-south1-docker.pkg.dev/<GCP_PROJECT_ID>/ml-apis/smartphone-addiction-api:<git sha>`
4. **Deploy to Cloud Run**: Deploys `smartphone-addiction-api` with:
   - **Memory**: 1Gi
   - **CPU**: 1, with `--cpu-boost` (extra CPU during startup to shorten cold starts)
   - **Instances**: min 0 / **max 1** (scales to zero when idle; see below)
   - **Port**: 8080 (injected automatically)
   - **Access**: `--allow-unauthenticated` (public web UI & API)

### Dependency Files
- `requirements-api.txt`: pinned serving runtime only. The Docker image installs this file.
- `requirements-dev.txt`: `requirements-api.txt` plus pytest, pytest-asyncio, httpx, xgboost and catboost (`tests/test_regression.py` imports `src/train.py`, which imports xgboost and catboost). CI and the deploy test job install this file.
- `requirements.txt`: `requirements-dev.txt` plus the training and notebook libraries (matplotlib, seaborn, streamlit).

The image copies only `src/`, `data/priors.json`, `data/cohort_summary.json` and `models/lgb_fold_1.joblib`. The model file is not in git; the workflows download it from the `v1.0.0-artifacts` release before building. Expected image size is under 0.5 GB (not measured yet).

### Why `--max-instances 1`
The batch export cache (`src/api/predict.py`) lives in process memory, and a download token only exists on the instance that scored the batch. With two or more instances, an export request routed to another instance returns 404. One instance keeps export tokens consistent.

Trade-offs: the service does not scale out, so concurrent requests share one instance. The cache is also lost when an instance restarts or scales to zero (`--min-instances 0`). If a batch export returns 404, re-run the batch scoring.

---

## 2. GitHub Secrets (Settings → Secrets and variables → Actions)

| Secret | Purpose |
| :--- | :--- |
| `GCP_SA_KEY` | Service-account JSON key for Google Cloud authentication |
| `GCP_PROJECT_ID` | GCP Project ID owning Artifact Registry and Cloud Run (`ml-portfolio-501915`) |

---

## 3. Local Verification

To run the container locally:

```bash
docker build -t smartphone-addiction-api .
docker run -p 8080:8080 smartphone-addiction-api
```

Check the image size (expected under 0.5 GB):
```bash
docker images smartphone-addiction-api
```

Test health readiness (returns 503 until the LightGBM engine has loaded):
```bash
curl http://localhost:8080/api/health
```

---

## 4. Rollback

To list past revisions and route traffic back to an earlier stable deployment:

```bash
gcloud run revisions list --service smartphone-addiction-api --region asia-south1
gcloud run services update-traffic smartphone-addiction-api --region asia-south1 --to-revisions <REVISION_NAME>=100
```

