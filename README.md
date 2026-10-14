# Pristyn Care Health — Patient Funnel Analysis

**Cynaris Solutions Data Analyst Internship** · Yashwanth J V Nayak
Client: Pristyn Care Health (HealthTech, elective surgery provider)

## Project Brief

Perform a patient funnel analysis from awareness to surgery conversion. Map the
7-stage funnel (ad click → enquiry → consultation → diagnostics → booking →
surgery → follow-up), calculate drop-off rates at each stage, and model the
revenue impact of a 5% conversion lift at the top-3 leakage points.

> **Scope note (see `docs/MILESTONE1_FINDINGS.md`):** the provided data packs
> are generic multi-category retail transaction data and contain no
> funnel-stage fields. This pipeline currently builds a **post-purchase
> retention & revenue baseline** (acquisition → first purchase → repeat
> orders → churn) as the closest available proxy, pending stakeholder
> confirmation of true funnel-stage data.

## Tech Stack

- Python 3.11, pandas, matplotlib
- pytest for unit tests
- Jupyter notebooks for exploration (production logic lives in `src/`, not notebooks)

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Pipeline

```bash
python -m src.pipeline
```

This loads the three raw data packs from `data/raw/`, cleans and validates
them, checks referential integrity, computes baseline metrics, and writes
processed outputs to `data/processed/`. The pipeline is idempotent — re-running
it always reproduces the same outputs from the same raw inputs.

## Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
pristyn-care-health-analysis/
├── data/
│   ├── raw/              # original data packs (untouched)
│   └── processed/        # pipeline outputs (cleaned data + metrics)
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
├── src/
│   ├── config.py          # all paths and config — nothing hardcoded elsewhere
│   ├── utils.py            # loading, cleaning, referential integrity checks
│   ├── metrics.py          # baseline metric calculations
│   └── pipeline.py         # main entry point (python -m src.pipeline)
├── tests/
│   └── test_pipeline.py
├── docs/
│   └── MILESTONE1_FINDINGS.md
├── requirements.txt
└── README.md
```

## Baseline Metrics (Checkpoint 2)

| Metric | Value |
|---|---|
| Total customers | 200 |
| Overall churn rate | 3.0% |
| Repeat customer rate | 96.5% |
| Avg. orders per customer | 11.54 |
| Avg. revenue per customer | ₹24,963 |
| Total transactions | 1,000 |
| Total net revenue | ₹80,21,851 |
| Discount % ↔ return-flag correlation | -0.018 (negligible) |

Full channel- and category-level breakdowns are in
`data/processed/channel_metrics.csv` and `data/processed/category_metrics.csv`,
and reproduced in `notebooks/01_exploratory_analysis.ipynb`.

## Checkpoint 3 — Deep-Dive Analysis

New data packs: `da_healthcare_appointments.csv` (2,000 appointments), `sunrise_patient_satisfaction.csv`, `sunrise_resource_utilisation.csv`.

```bash
python -m src.deep_dive      # funnel, segment tests, specialty table, monthly trend
pytest tests/ -v             # 11 tests
```

Key results (full write-up: `docs/Milestone3_DeepDive.docx`):

| Finding | Result |
|---|---|
| Overall no-show | 16.2% (324 of 2,000) |
| Only significant driver | Specialty (chi-square p = 0.00003) |
| Top-3 leakage | General Medicine 22.5%, Dermatology 20.5%, Ophthalmology 19.9% (39% of bookings, 50.3% of no-shows) |
| Not significant | Department, reason, insurance, age, lead time, hour, wait time |
| Anomaly | May 2025 no-show 26.0% (p = 0.0002) |

Limits: only booking -> attended -> consultation stages exist in the data; the resource file has impossible values in 336 of 400 rows and is not analysed; no revenue field exists, so the Excel revenue model needs an assumed revenue per consultation.

## AI Tool Usage

Portions of this pipeline (project scaffolding, docstrings, and test cases)
were drafted with AI assistance (Claude) and reviewed/understood before
committing, per Cynaris AI-use documentation standards.
