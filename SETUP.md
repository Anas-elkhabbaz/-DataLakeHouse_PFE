# Setup Guide — Spark Issue Triage Platform

Complete step-by-step guide to go from a fresh clone to a fully running project in Snowflake.

**Estimated total time:** ~2–3 hours (mostly waiting for data upload and dbt runs)

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone & Configure](#2-clone--configure)
3. [Snowflake Infrastructure](#3-snowflake-infrastructure)
4. [Load Source Data (Bronze)](#4-load-source-data-bronze)
5. [dbt Transformations (Silver + Gold)](#5-dbt-transformations-silver--gold)
6. [ML Pipeline](#6-ml-pipeline)
7. [Deploy to Snowflake](#7-deploy-to-snowflake)
8. [Run Applications Locally (Docker)](#8-run-applications-locally-docker)
9. [Verification Checklist](#9-verification-checklist)

---

## 1. Prerequisites

### Required

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11+ | Load scripts, ML pipeline |
| dbt-snowflake | 1.11+ | dbt transformations |
| Snowflake account | Trial or paid | Data warehouse |
| Source CSVs | ~8 GB total | Raw JIRA data from Kaggle |

### Optional

| Tool | Purpose |
|------|---------|
| Docker Desktop | Run apps locally without installing dependencies |
| Anthropic API key | LLM analysis in the inference app |
| Kaggle account + GPU | Re-train the DeBERTa model from scratch |

### Python packages (load scripts)

```bash
pip install snowflake-connector-python python-dotenv pandas numpy scikit-learn \
            sentence-transformers transformers torch requests tqdm
```

### dbt setup

```bash
# In the dbt_project/ directory
pip install dbt-snowflake==1.11.*
# Or using uv:
uv venv .venv
uv pip install dbt-snowflake
```

### Source CSV files

Download the 4 JIRA Apache Spark CSV files from Kaggle and place them in `data/`:

```
data/
├── issues.csv       (~2 GB)
├── comments.csv     (~3 GB)
├── changelog.csv    (~2 GB)
└── issuelinks.csv   (~300 MB)
```

> These files are not in the repository (too large). They come from the Kaggle dataset:
> **Apache Spark JIRA Issues** (March 2025, ~49 832 SPARK-* tickets).

---

## 2. Clone & Configure

```bash
git clone https://github.com/Anas-elkhabbaz/DataLakeHouse_PFE.git
cd DataLakeHouse_PFE
```

Copy and fill in the environment file:

```bash
cp .env.example .env
```

Edit `.env` with your Snowflake credentials:

```env
SNOWFLAKE_ACCOUNT=your-account-identifier   # e.g. xy12345.eu-west-1
SNOWFLAKE_USER=your-username
SNOWFLAKE_PASSWORD=your-password
SNOWFLAKE_ROLE=SYSADMIN
SNOWFLAKE_WAREHOUSE=PFE_WH
SNOWFLAKE_DATABASE=PFE_SPARK
```

> Find your **account identifier** in Snowsight → bottom-left corner → hover over your account name.
> Format is typically `ORGNAME-ACCOUNTNAME` (e.g. `TFBQJMT-BTC11286`).

---

## 3. Snowflake Infrastructure

### 3.1 Create database, schemas and warehouse

```bash
python load/run_phase1.py
```

Creates:
- Database `PFE_SPARK`
- Schemas: `RAW`, `STAGING`, `INTERMEDIATE`, `MARTS_ML`, `MARTS_ANALYTICS`, `PREDICTIONS`, `ML_MODELS`
- Warehouse `PFE_WH` (X-Small, auto-suspend 60s)
- Internal stage `RAW.CSV_STAGE`

### 3.2 Write the dbt profile

```bash
python load/write_profiles.py
```

Reads `.env` and writes `~/.dbt/profiles.yml` with the correct Snowflake connection.

---

## 4. Load Source Data (Bronze)

### 4.1 Inspect CSV headers (optional but recommended)

```bash
python load/inspect_headers.py
```

Prints column positions for each CSV. Verify the mapping in `load/04_copy_into_raw.sql` matches before continuing.

### 4.2 Upload CSVs to Snowflake stage

```bash
python load/03_put_files.py
```

PUTs the 4 CSV files to `@RAW.CSV_STAGE`. This compresses and uploads ~8 GB.
**Duration: 30–90 minutes depending on your upload speed.**

### 4.3 Load into raw tables

```bash
python load/run_phase4.py
```

Runs `COPY INTO` for each table and verifies row counts:

| Table | Expected rows |
|-------|--------------|
| RAW.ISSUES | ~1 149 321 |
| RAW.COMMENTS | ~5 047 714 |
| RAW.CHANGELOG | ~9 653 526 |
| RAW.ISSUELINKS | ~390 063 |

---

## 5. dbt Transformations (Silver + Gold)

All commands run from inside `dbt_project/`:

```bash
cd dbt_project
```

### 5.1 Install dbt packages

```bash
dbt deps
```

Installs `dbt-utils` (declared in `packages.yml`).

### 5.2 Load seed tables

```bash
dbt seed
```

Loads `issuetype_mapping.csv` and `resolution_mapping.csv` into Snowflake.

### 5.3 Run all models

```bash
dbt run
```

Builds the full medallion pipeline:

| Layer | Models | Schema |
|-------|--------|--------|
| Staging | `stg_issues`, `stg_comments`, `stg_changelog`, `stg_issuelinks` | STAGING |
| Intermediate | `int_spark_issues`, `int_comments_aggregated`, `int_issuelinks_features`, `int_train_val_split`, `int_issues_analytics`, `int_daily_activity`, `int_fix_versions`, `int_status_transitions` | INTERMEDIATE |
| Marts ML | `mart_ml` | MARTS_ML |
| Marts Analytics | `mart_analytics_ops`, `mart_analytics_workload`, `mart_analytics_links`, `mart_analytics_transitions_sankey`, `mart_analytics_transitions_time`, `mart_analytics_versions`, `mart_analytics_calendar` | MARTS_ANALYTICS |

### 5.4 Run tests

```bash
dbt test
```

Expected output: **PASS=44 WARN=2 ERROR=0**

> The 2 warnings are on `accepted_values` tests for resolution `"Won't Fix"` — the apostrophe
> triggers a SQL warning in some dbt versions. This is harmless.

---

## 6. ML Pipeline

### 6.1 Get the DeBERTa model

The fine-tuned DeBERTa-v3-base model (`results/deberta_v3_parent.zip`, ~570 MB) is not included in the repository due to GitHub's 100 MB file limit.

**Option A — Use the pre-trained model (recommended)**

Download `deberta_v3_parent.zip` from the Snowflake internal stage:

```sql
-- In Snowsight SQL worksheet:
GET @ML_MODELS.APP_STAGE/deberta/ file:///path/to/local/folder/;
```

Or ask the project maintainer for the zip file and place it at `results/deberta_v3_parent.zip`.

**Option B — Re-train from scratch**

Open `results/notebooks/deberta_fine_tuning_v3.ipynb` on **Kaggle** (requires a free account + GPU enabled):

1. Upload the notebook to Kaggle
2. Enable GPU accelerator (Tesla T4)
3. Run all cells (~2 hours training)
4. Download `deberta_v3_parent.zip` from the output
5. Place it in `results/deberta_v3_parent.zip`

### 6.2 Fetch parent keys from JIRA API

```bash
python fetch_parent_keys.py
```

Calls the public Apache JIRA REST API to retrieve `has_parent` for all ~42K SPARK tickets.
Output: `results/spark_parent_keys.csv`
**Duration: ~15–20 minutes (10 parallel threads).**

> Skip this step if `results/spark_parent_keys.csv` already exists.

### 6.3 Migrate to Snowflake and run dbt mart_ml

```bash
python migrate_to_snowflake.py
```

Does three things:
1. Uploads `spark_parent_keys.csv` → `RAW.SPARK_PARENT_KEYS` (42 083 rows)
2. Runs `dbt run --select int_spark_issues mart_ml` to rebuild with `has_parent`
3. Verifies `MARTS_ML.MART_ML` row count

### 6.4 Run the full ML pipeline

```bash
python load/run_ml_pipeline.py
```

- Loads `MARTS_ML.MART_ML` from Snowflake
- Fine-tuned DeBERTa v3 → predicts `issuetype` on 3 809 validation tickets
- Sentence-transformers KNN → predicts `resolution`
- Evaluates and saves metrics to `results/`
- Uploads predictions → `PREDICTIONS.MART_PREDICTIONS`

**Duration: ~20–40 minutes (CPU-only inference).**

Expected metrics:

| Model | Metric | Value |
|-------|--------|-------|
| DeBERTa v3 (issuetype) | Accuracy | 79.6% |
| DeBERTa v3 (issuetype) | Macro-F1 | 73.63% |
| KNN resolution | Accuracy | 91.52% |

---

## 7. Deploy to Snowflake

### 7.1 Deploy the Streamlit inference app

```bash
python deploy_streamlit_snowflake.py
```

Does:
1. Creates schema `ML_MODELS` and stage `APP_STAGE` (if not already present)
2. Uploads `apps/inference/environment.yml` (Python packages for Snowflake)
3. Uploads sklearn models (`results/sklearn_models/`)
4. Extracts and uploads DeBERTa model files from `results/deberta_v3_parent.zip`
5. Creates or replaces the Streamlit app in Snowflake

Access the app in Snowsight: **Streamlit → SPARK_TRIAGE_APP**

### 7.2 Deploy notebooks to Snowflake

```bash
python deploy_notebooks.py
```

Uploads and creates two notebooks in Snowflake:
- `deberta_fine_tuning_v3` — DeBERTa v3 training notebook
- `sklearn_resolution_training` — sklearn KNN resolution model

Access in Snowsight: **Projects → Notebooks**

---

## 8. Run Applications Locally (Docker)

The simplest way to run both apps without installing all dependencies:

```bash
# Make sure .env is configured
docker-compose up --build
```

| App | URL | Description |
|-----|-----|-------------|
| Inference | http://localhost:8501 | DeBERTa ticket classification |
| Analytics | http://localhost:8502 | 5-page analytics dashboard |

First build downloads `microsoft/deberta-v3-base` (~370 MB). Subsequent starts are instant.

**Manual launch (without Docker):**

```bash
# Inference app
pip install -r apps/inference/requirements.txt
streamlit run apps/inference/inference_app.py

# Analytics dashboard
pip install -r apps/analytics/requirements.txt
streamlit run apps/analytics/analytics_app.py
```

---

## 9. Verification Checklist

Run these checks in Snowsight (SQL worksheet) to confirm everything is in place:

```sql
-- Bronze layer
SELECT COUNT(*) FROM PFE_SPARK.RAW.ISSUES;       -- 1 149 321
SELECT COUNT(*) FROM PFE_SPARK.RAW.COMMENTS;     -- 5 047 714
SELECT COUNT(*) FROM PFE_SPARK.RAW.CHANGELOG;    -- 9 653 526
SELECT COUNT(*) FROM PFE_SPARK.RAW.ISSUELINKS;   -- 390 063
SELECT COUNT(*) FROM PFE_SPARK.RAW.SPARK_PARENT_KEYS; -- 42 083

-- Gold ML
SELECT COUNT(*) FROM PFE_SPARK.MARTS_ML.MART_ML; -- 42 083
SELECT COUNT(DISTINCT has_parent) FROM PFE_SPARK.MARTS_ML.MART_ML; -- 2 (0 and 1)

-- Predictions
SELECT COUNT(*) FROM PFE_SPARK.PREDICTIONS.MART_PREDICTIONS; -- 3 809

-- Analytics marts
SELECT COUNT(*) FROM PFE_SPARK.MARTS_ANALYTICS.MART_ANALYTICS_OPS; -- ~49 833
SELECT COUNT(*) FROM PFE_SPARK.MARTS_ANALYTICS.MART_ANALYTICS_WORKLOAD;
SELECT COUNT(*) FROM PFE_SPARK.MARTS_ANALYTICS.MART_ANALYTICS_LINKS;

-- Models in stage
LIST @PFE_SPARK.ML_MODELS.APP_STAGE;
-- Should show: environment.yml, sklearn_models/, deberta/ (12+ files)

-- Streamlit app
SHOW STREAMLITS IN SCHEMA PFE_SPARK.ML_MODELS;
-- Should show: SPARK_TRIAGE_APP
```

---

## Snowflake Schema Map

```
PFE_SPARK/
├── RAW/               Bronze — raw CSV data + SPARK_PARENT_KEYS
├── STAGING/           Silver — 4 cleaned views (1:1 with raw tables)
├── INTERMEDIATE/      Silver — 8 feature tables (NLP, aggregations, splits)
├── MARTS_ML/          Gold  — MART_ML (42 083 rows, train+val split)
├── MARTS_ANALYTICS/   Gold  — 7 analytics tables (Power BI source)
├── PREDICTIONS/       Gold  — MART_PREDICTIONS (3 809 validation predictions)
└── ML_MODELS/         — APP_STAGE (model files), Streamlit app, Notebooks
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'joblib'` in Streamlit in Snowflake**
→ Run `deploy_streamlit_snowflake.py` again — it uploads `environment.yml` which declares all required packages.

**`dbt run` fails with `invalid identifier 'IL.N_CONTAINER'`**
→ The intermediate table is stale. Run: `dbt run --select int_issuelinks_features mart_ml`

**DeBERTa loading fails in Streamlit with `'list' object has no attribute 'keys'`**
→ The `streamlit_in_snowflake.py` must use `torch_dtype=torch.float32` (not `dtype=`). Check the current version is deployed.

**`UnicodeEncodeError` in deploy scripts on Windows**
→ Open PowerShell and run: `$env:PYTHONIOENCODING = "utf-8"` before running the script.

**Snowflake trial account — Cortex functions blocked**
→ Expected. The pipeline uses local `sentence-transformers` instead. Cortex is not required.

**`PUT` upload very slow**
→ Normal for ~8 GB. Use a wired connection if possible. The stage auto-compresses files.
