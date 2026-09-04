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

## AI Tool Usage

Portions of this pipeline (project scaffolding, docstrings, and test cases)
were drafted with AI assistance (Claude) and reviewed/understood before
committing, per Cynaris AI-use documentation standards.
