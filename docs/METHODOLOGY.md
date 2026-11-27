# Checkpoint 3 — Methodology, Assumptions & Reproducibility

## 1. Data and preprocessing
- Source: `data/raw/da_healthcare_appointments.csv` — 2,000 appointments, 15 columns, 200 patients, 100 doctors.
- Checks: 0 duplicate `appointment_id`, 0 missing values, lead time 1–14 days (no negative values), hours within clinic hours (booking 8–18, appointment 9–17).
- Derived fields: `lead_days` (appointment − booking date), `lead_bucket` (0-1d, 2-3d, 4-7d, 8-14d, 15d+), `hour_bucket` (9-11, 12-14, 15-17), `month`, `weekday`.
- Raw files are never modified; all outputs are regenerated into `data/processed/`.
- `wait_time_mins` and `consultation_mins` are **post-outcome** fields (consultation is 0 for every no-show by construction), so they are used only for the funnel stage and are **excluded from the predictive model** to avoid target leakage.

## 2. Statistical testing
| Step | Method | Why |
|---|---|---|
| Segment vs no-show | Pearson chi-square test of independence (7 categorical segments) | Categorical predictors, binary outcome |
| Assumption check | Minimum expected cell count ≥ 5 (all cells ≥ 24; 0% below 5) | Chi-square validity |
| Effect size | Cramér's V (specialty 0.128 = small; others ≤ 0.06) | p-values alone don't show magnitude |
| Multiple testing | Bonferroni and Holm across the 7 tests | 7 tests inflate false positives; specialty survives (Holm p = 0.0002), nothing else is close |
| Specialty rates | Wilson 95% CIs; each specialty vs rest with Bonferroni across 8 specialties | Small per-group n |
| Adjusted effect | Logistic regression (statsmodels), odds ratios vs Cardiology, controlling for age group, insurance, department, reason, lead days, hour | Checks specialty isn't confounded |
| Joint test | Likelihood-ratio test of specialty block: χ² = 34.0, df = 7, p = 0.00002 | |
| Anomaly | May 2025 vs rest chi-square (p = 0.0002); exploratory, 1 of 12 months examined post-hoc | Flagged as exploratory |

## 3. Predictive model (`src/predictive_model.py`)
- Target: `no_show`. Features (known at booking): specialty, department, reason, insurance type, age group, weekday, lead days, appointment hour, booking hour.
- Models: (a) no-skill baseline, (b) L2 logistic regression (C = 0.5), (c) gradient boosting (100 trees, depth 2, lr 0.05, subsample 0.8).
- Validation: repeated stratified 5-fold CV × 5 repeats; one-hot encoding and scaling sit inside the sklearn Pipeline, fit on training folds only.
- Significance: label-permutation test (200 permutations) on the logistic model.
- Interpretability: permutation importance (30 repeats, held-out 25%), calibration by quintile.

| Model | ROC AUC (mean ± SD) | PR AUC | Brier | Top-decile lift |
|---|---|---|---|---|
| Baseline | 0.500 | 0.162 | 0.1358 | n/a |
| Logistic regression | 0.563 ± 0.034 | 0.200 | 0.1365 | 1.14× |
| Gradient boosting | 0.536 ± 0.032 | 0.187 | 0.1369 | 1.10× |

Permutation test: observed AUC 0.555 vs null mean 0.499 (95th pct 0.531), p = 0.01.

**Interpretation:** the model has statistically detectable but practically weak discrimination. Specialty is the only feature with a meaningful importance (AUC drop 0.065); weekday is a weak secondary signal (0.013) that was not significant in univariate testing and should be treated as a hypothesis. The model is not accurate enough for patient-level targeting (Brier no better than baseline); the actionable use is **specialty-level** reminder/overbooking policy. The gradient-boosting model did not beat the simpler logistic model, consistent with a small dataset (n = 2,000, 324 events) and a single dominant signal.

## 4. Assumptions and limitations
1. Appointments are treated as independent, but 200 patients appear multiple times (and 100 doctors). Standard errors may be slightly understated; a patient-clustered model is a next step.
2. Data cover only booking → attended → consultation; no revenue, ad-click, diagnostics, surgery or follow-up fields, so revenue impact needs an explicit assumed revenue per consultation.
3. Observational data: associations are not causal.
4. `sunrise_resource_utilisation.csv` has physically impossible values in 336 of 400 rows and was excluded.
5. Small cells: specialty n per group ≈ 250, so CIs are wide (see `specialty_noshow.csv`).

## 5. Reproducibility
- Python 3.11+ (tested on 3.13), pinned libraries in `requirements.txt`, global `RANDOM_STATE = 42`.
- Steps:
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m src.pipeline            # Checkpoint 2 baseline
python -m src.deep_dive           # Checkpoint 3 funnel + tests
python -m src.predictive_model    # Checkpoint 3 revision: assumptions, odds ratios, model
pytest tests/ -v
```
- Outputs: `data/processed/segment_test_assumptions.csv`, `specialty_adjusted_odds_ratios.csv`, `model_cv_metrics.csv`, `model_feature_importance.csv`, `model_calibration.csv`, `reports/figures/predictive_model.png`.
