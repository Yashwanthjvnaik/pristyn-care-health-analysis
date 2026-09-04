# Architecture

## Data Flow

```
data/raw/*.csv
      │
      ▼
src/utils.py  (load_raw, clean_sales, clean_customers, check_referential_integrity)
      │
      ▼
src/metrics.py  (channel_conversion_metrics, category_performance,
                  customer_lifecycle_baseline, build_baseline_summary)
      │
      ▼
src/pipeline.py  (orchestrates the above, logs progress, writes outputs)
      │
      ▼
data/processed/*.csv  (sales_clean, customers_clean, baseline_metrics,
                        channel_metrics, category_metrics)
```

## Module Responsibilities

- **`src/config.py`** — single source of truth for all file paths. No other
  module hardcodes a path.
- **`src/utils.py`** — I/O and data-quality layer. Loads raw CSVs, deduplicates,
  coerces types, drops non-physical values (negative units, out-of-range
  discounts), derives `Net_Revenue`, and checks referential integrity between
  the sales, customer, and product tables.
- **`src/metrics.py`** — pure functions that take cleaned DataFrames and return
  metric DataFrames/dicts. No I/O — keeps metric logic independently testable.
- **`src/pipeline.py`** — orchestration layer. Calls `utils` then `metrics`,
  logs each stage, and writes final CSVs to `data/processed/`.

## Data Sources

Three CSVs provided by Cynaris as part of the Pristyn Care Health project
brief: `da_sales_transactions.csv` (transaction-level), `da_customer_data.csv`
(customer master), `da_product_catalogue.csv` (product master). No external
APIs or databases are used in this milestone.

## Known Limitation (tracked, not yet resolved)

The data does not contain patient-funnel-stage fields as described in the
brief. See `docs/MILESTONE1_FINDINGS.md` for the full data inventory and the
recommendation escalated to the stakeholder/mentor. `src/metrics.py` currently
computes a retention/revenue baseline as the closest defensible proxy; funnel
drop-off metrics will be added once stage-level data is confirmed or supplied.

## No Database in Use (Yet)

Checkpoint 2 uses flat-file CSVs only. If a database migration or SQL layer
becomes part of Checkpoint 3, it will be documented here and in
`DEPLOYMENT.md`.
