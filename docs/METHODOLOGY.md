# Checkpoint 3 — Methodology, Assumptions & Reproducibility

## 1. Data and preprocessing
- Source: `data/raw/da_healthcare_appointments.csv` — 2,000 appointment rows and 15 columns, covering 1,800 unique patients and 100 doctors.
- Validation checks: appointment IDs are unique; the supplied file has no missing values; lead time is non-negative; booking and appointment hours fall within the documented clinic-hour ranges.
- Derived fields: `lead_days` (appointment date minus booking date), lead-time and hour bands, month, and weekday.
- Raw files are never modified. Scripts write derived tables and figures to `data/processed/` and `reports/figures/`.
- `wait_time_mins` and `consultation_mins` are post-booking/outcome fields. Consultation duration is zero for no-shows by construction. Both are excluded from prediction to prevent target leakage.

## 2. Statistical testing
| Step | Method | Rationale / check |
|---|---|---|
| Segment vs no-show | Pearson chi-square test of independence | Both variables are categorical; expected cell counts are exported for every test. |
| Assumptions | Minimum expected count and percentage of expected cells below 5 | If sparse expected counts occur, combine substantively similar levels or use an exact/Monte Carlo alternative rather than relying on the asymptotic test. |
| Effect size | Cramér's V | Complements p-values with association magnitude. |
| Multiplicity | Bonferroni and Holm corrections across the seven segment tests | Controls family-wise error. Holm is the primary step-down interpretation. |
| Specialty rates | Wilson 95% confidence intervals | More suitable than a normal approximation for proportions. |
| Adjusted association | Logistic regression with specialty, age group, insurance, department, reason, lead days and appointment hour | Reports adjusted odds ratios and confidence intervals; association is not causation. |
| Joint specialty contribution | Likelihood-ratio test comparing full and reduced logistic models | Tests the specialty block jointly. |

The chi-square independence assumption applies to appointment rows. Since 200 patients have repeat appointments, the appointment-level chi-square analysis does not account for within-patient dependence. Interpret its p-values as exploratory; a patient-clustered or mixed-effects inferential model is a useful extension.

## 3. Predictive model (`src/predictive_model.py`)
- Target: `no_show` (1 = did not attend).
- Booking-time features: specialty, department, visit reason, insurance type, age group, weekday, lead days, appointment hour and booking hour.
- Excluded leakage fields: `wait_time_mins` and `consultation_mins`, because they are not reliably available at booking and consultation duration is structurally related to attendance.
- Models: DummyClassifier prior baseline, L2 Logistic Regression, and Gradient Boosting.
- Preprocessing: one-hot encoding for categorical fields and standardisation for numeric fields, both inside an sklearn Pipeline so transformations are fitted only on each training fold.
- **Validation split:** repeated 5-fold `StratifiedGroupKFold` (five repeats, seeds 42–46), grouping on `patient_id`. All appointments for a patient remain in one fold; this evaluates generalisation to patients not seen during training. A separate single grouped 5-fold out-of-fold pass supplies calibration and confusion-matrix outputs.
- Metrics: ROC-AUC, average precision (PR-AUC), Brier score, precision, recall and F1 at a predeclared 0.16 probability threshold (approximately the observed event prevalence), plus top-decile lift. The threshold is a reporting convention, not a clinically or operationally optimised threshold. It is deliberately below 0.50 because the no-show prevalence is approximately 16%; a 0.50 threshold produced no positive predictions in this dataset. Any deployment threshold should be selected using validation data and the relative costs of missed no-shows and unnecessary outreach.
- Baseline lift: constant-probability predictions have no ranking information; their top-decile lift is defined as 1.0 (random-selection expectation), not calculated from arbitrary tie ordering.
- Significance check: label-permutation test with grouped CV (49 permutations; 3-fold grouped CV; seed 42; minimum attainable one-sided p-value is 0.02). This is a diagnostic test, not proof of out-of-sample business value.
- Interpretability: permutation importance is calculated on a patient-disjoint grouped holdout. Calibration is reported by out-of-fold probability quintile.

### Outputs
- `model_cv_metrics.csv`: grouped CV means and standard deviations.
- `model_confusion_matrix.csv`: logistic regression confusion matrix from one grouped out-of-fold pass at threshold 0.16.
- `model_feature_importance.csv`: permutation importance on a patient-disjoint holdout.
- `model_calibration.csv`: observed versus predicted risk by out-of-fold quintile.
- `segment_test_assumptions.csv` and `specialty_adjusted_odds_ratios.csv`: inferential analysis outputs.
- `reports/figures/predictive_model.png`: model comparison, importance and calibration plots.

## 4. Interpretation and limitations
Latest grouped-CV run: logistic regression ROC-AUC 0.5634 (SD 0.0295), PR-AUC 0.2005 and Brier score 0.1363; gradient boosting ROC-AUC 0.5395 (SD 0.0307), PR-AUC 0.1924 and Brier score 0.1365. Logistic precision/recall/F1 at threshold 0.16 are 0.1860/0.5452/0.2772. The one-pass grouped out-of-fold confusion matrix is TN=912, FP=764, FN=151, TP=173. The grouped label-permutation diagnostic returned p=0.02 with 49 permutations (coarse resolution). These figures are specific to this supplied dataset and validation design. The model results must be read from the regenerated CSV outputs rather than copied from an earlier run. If ROC-AUC is only modestly above 0.50, the model has weak ranking ability even if a permutation test finds a statistically detectable difference. Statistical significance does not establish useful precision, recall, calibration, fairness, or financial value. The dataset contains only 2,000 appointments and 324 no-shows; the estimates may be unstable and may not generalise to other clinics or time periods.

The data support a partial booking → attendance → consultation funnel only. There are no ad-click, enquiry, diagnostics, surgery, follow-up conversion or revenue fields. The resource-utilisation file contains many physically impossible values and is excluded pending source validation. The satisfaction file relates to a different organisation and has no appointment-level join key, so it is not used as evidence about these no-shows. Observational associations are not causal.

## 5. Reproducibility
Use Python 3.11; install the exact direct-dependency versions in `requirements.txt` for a repeatable environment. Random seeds are fixed at 42 (and 42–46 for repeated grouped folds).

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.pipeline
python -m src.deep_dive
python -m src.predictive_model
pytest tests/ -v
```

Run commands from the repository root. The raw data files are inputs and are not overwritten. Re-running the scripts regenerates the derived outputs.
