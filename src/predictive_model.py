"""
Checkpoint 3 (revision): predictive no-show model, assumption checks and validation.

Run with:
    python -m src.predictive_model

Design decisions (documented in docs/METHODOLOGY.md):
  * Target: no_show (1 = patient did not attend).
  * Features are limited to what is KNOWN AT BOOKING TIME. `wait_time_mins` and
    `consultation_mins` are recorded after the visit outcome (consultation_mins is
    0 for every no-show by construction), so using them would be target leakage.
  * Every stochastic step uses RANDOM_STATE so results reproduce exactly.
  * Models are compared with repeated stratified 5-fold cross-validation against a
    no-skill baseline; the preprocessing lives inside the sklearn Pipeline so it is
    fit on training folds only.
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DATA_PROCESSED_DIR, FIGURES_DIR
from src.deep_dive import SEGMENT_COLUMNS, load_appointments

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
N_SPLITS, N_REPEATS = 5, 5

CATEGORICAL = ["specialty", "department", "reason", "insurance_type", "patient_age_group", "weekday"]
NUMERIC = ["lead_days", "appointment_hour", "booking_hour"]
# Deliberately excluded (post-outcome / leakage): wait_time_mins, consultation_mins
EXCLUDED_LEAKAGE = ["wait_time_mins", "consultation_mins"]


def build_frame() -> pd.DataFrame:
    df = load_appointments()
    df["weekday"] = df["appointment_date"].dt.day_name()
    return df


# --------------------------------------------------------------------------- #
# 1. Test-assumption checks
# --------------------------------------------------------------------------- #
def chi_square_assumptions(df: pd.DataFrame) -> pd.DataFrame:
    """Chi-square validity: smallest expected count and share of cells < 5, plus
    Bonferroni / Holm-adjusted p-values across the family of segment tests."""
    rows = []
    for col in SEGMENT_COLUMNS:
        table = pd.crosstab(df[col], df["no_show"])
        chi2, p, dof, expected = stats.chi2_contingency(table)
        n = table.values.sum()
        k = min(table.shape) - 1
        rows.append({
            "segment": col, "chi2": round(chi2, 2), "dof": dof, "p_value": p,
            "min_expected_count": round(float(expected.min()), 1),
            "pct_cells_expected_lt5": round(float((expected < 5).mean() * 100), 1),
            "cramers_v": round(float(np.sqrt(chi2 / (n * k))), 3),
        })
    out = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    m = len(out)
    out["p_bonferroni"] = (out["p_value"] * m).clip(upper=1.0)
    # Holm step-down
    holm, running = [], 0.0
    for i, p in enumerate(out["p_value"]):
        running = max(running, min(1.0, p * (m - i)))
        holm.append(running)
    out["p_holm"] = holm
    out["significant_after_holm"] = out["p_holm"] < 0.05
    return out


# --------------------------------------------------------------------------- #
# 2. Adjusted odds ratios (inference)
# --------------------------------------------------------------------------- #
def adjusted_odds_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Logistic regression with booking-time covariates; odds ratios vs Cardiology."""
    formula = (
        "no_show ~ C(specialty, Treatment('Cardiology')) + C(patient_age_group)"
        " + C(insurance_type) + C(department) + C(reason) + lead_days + appointment_hour"
    )
    res = smf.logit(formula, data=df).fit(disp=0)
    ci = res.conf_int()
    out = pd.DataFrame({
        "term": res.params.index, "odds_ratio": np.exp(res.params.values),
        "ci_low": np.exp(ci[0].values), "ci_high": np.exp(ci[1].values), "p_value": res.pvalues.values,
    })
    out = out[out["term"].str.contains("specialty")].copy()
    out["term"] = out["term"].str.extract(r"\[T\.(.*)\]")[0]
    # Likelihood-ratio test: does specialty add anything beyond other covariates?
    reduced = smf.logit(
        "no_show ~ C(patient_age_group) + C(insurance_type) + C(department) + C(reason)"
        " + lead_days + appointment_hour", data=df).fit(disp=0)
    lr = 2 * (res.llf - reduced.llf)
    ddf = res.df_model - reduced.df_model
    out.attrs["lr_stat"], out.attrs["lr_df"] = lr, int(ddf)
    out.attrs["lr_p"] = float(stats.chi2.sf(lr, ddf))
    return out.sort_values("odds_ratio", ascending=False).round(4).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 3. Predictive models with cross-validation
# --------------------------------------------------------------------------- #
def _preprocess() -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ])


def make_models() -> dict[str, Pipeline]:
    return {
        "Baseline (predict prevalence)": Pipeline([
            ("prep", _preprocess()), ("clf", DummyClassifier(strategy="prior"))]),
        "Logistic regression (L2)": Pipeline([
            ("prep", _preprocess()),
            ("clf", LogisticRegression(C=0.5, max_iter=1000, random_state=RANDOM_STATE))]),
        "Gradient boosting": Pipeline([
            ("prep", _preprocess()),
            ("clf", GradientBoostingClassifier(
                n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.8,
                random_state=RANDOM_STATE))]),
    }


def top_decile_lift(y: np.ndarray, p: np.ndarray) -> float:
    """No-show rate among the 10% highest-risk bookings divided by overall rate."""
    k = max(1, int(len(y) * 0.10))
    idx = np.argsort(-p)[:k]
    return float(y[idx].mean() / y.mean())


def evaluate_models(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    X, y = df[CATEGORICAL + NUMERIC], df["no_show"].to_numpy()
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    rows, oof = [], {}
    for name, model in make_models().items():
        aucs, aps, briers, lifts = [], [], [], []
        for train, test in cv.split(X, y):
            model.fit(X.iloc[train], y[train])
            p = model.predict_proba(X.iloc[test])[:, 1]
            aucs.append(roc_auc_score(y[test], p))
            aps.append(average_precision_score(y[test], p))
            briers.append(brier_score_loss(y[test], p))
            lifts.append(top_decile_lift(y[test], p))
        rows.append({
            "model": name,
            "roc_auc_mean": np.mean(aucs), "roc_auc_sd": np.std(aucs),
            "pr_auc_mean": np.mean(aps), "brier_mean": np.mean(briers),
            "top_decile_lift_mean": np.mean(lifts),
        })
        oof[name] = cross_val_predict(
            model, X, y, cv=__import__("sklearn.model_selection", fromlist=["StratifiedKFold"])
            .StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE), method="predict_proba")[:, 1]
    return pd.DataFrame(rows).round(4), oof


def permutation_test_auc(df: pd.DataFrame, n_perm: int = 200) -> dict:
    """Is the logistic model's CV AUC better than chance? Permute labels and refit."""
    from sklearn.model_selection import StratifiedKFold
    X, y = df[CATEGORICAL + NUMERIC], df["no_show"].to_numpy()
    model = make_models()["Logistic regression (L2)"]
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    def cv_auc(labels):
        p = cross_val_predict(model, X, labels, cv=cv, method="predict_proba")[:, 1]
        return roc_auc_score(labels, p)

    observed = cv_auc(y)
    rng = np.random.default_rng(RANDOM_STATE)
    null = np.array([cv_auc(rng.permutation(y)) for _ in range(n_perm)])
    return {"observed_auc": round(observed, 4), "null_mean": round(float(null.mean()), 4),
            "null_95th": round(float(np.quantile(null, 0.95)), 4),
            "p_value": round(float((1 + (null >= observed).sum()) / (n_perm + 1)), 4)}


def feature_importance(df: pd.DataFrame) -> pd.DataFrame:
    """Permutation importance (AUC drop) on a held-out split for the logistic model."""
    from sklearn.model_selection import train_test_split
    X, y = df[CATEGORICAL + NUMERIC], df["no_show"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
    model = make_models()["Logistic regression (L2)"].fit(Xtr, ytr)
    r = permutation_importance(model, Xte, yte, scoring="roc_auc", n_repeats=30,
                               random_state=RANDOM_STATE)
    out = pd.DataFrame({"feature": X.columns, "auc_drop_mean": r.importances_mean,
                        "auc_drop_sd": r.importances_std})
    return out.sort_values("auc_drop_mean", ascending=False).round(4).reset_index(drop=True)


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 5) -> pd.DataFrame:
    q = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False)
    out = pd.DataFrame({"bin": q, "pred": p, "obs": y}).groupby("bin").agg(
        n=("obs", "size"), mean_predicted=("pred", "mean"), observed_rate=("obs", "mean"))
    return out.round(4).reset_index()


def plot_results(metrics: pd.DataFrame, importance: pd.DataFrame, calib: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].barh(metrics["model"], metrics["roc_auc_mean"], xerr=metrics["roc_auc_sd"], color="#0e7c86")
    ax[0].axvline(0.5, color="grey", ls="--")
    ax[0].set_xlim(0.4, 0.7)
    ax[0].set_title("Cross-validated ROC AUC (mean ± SD)")
    imp = importance.iloc[::-1]
    ax[1].barh(imp["feature"], imp["auc_drop_mean"], xerr=imp["auc_drop_sd"], color="#0e7c86")
    ax[1].set_title("Permutation importance (AUC drop)")
    ax[2].plot([0, 0.4], [0, 0.4], "--", color="grey")
    ax[2].plot(calib["mean_predicted"], calib["observed_rate"], "o-", color="#0e7c86")
    ax[2].set_xlabel("Mean predicted risk"); ax[2].set_ylabel("Observed no-show rate")
    ax[2].set_title("Calibration (quintiles, out-of-fold)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "predictive_model.png", dpi=150)
    plt.close(fig)


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    df = build_frame()

    assumptions = chi_square_assumptions(df)
    odds = adjusted_odds_ratios(df)
    metrics, oof = evaluate_models(df)
    perm = permutation_test_auc(df)
    importance = feature_importance(df)
    calib = calibration_table(df["no_show"].to_numpy(), oof["Logistic regression (L2)"])

    assumptions.to_csv(DATA_PROCESSED_DIR / "segment_test_assumptions.csv", index=False)
    odds.to_csv(DATA_PROCESSED_DIR / "specialty_adjusted_odds_ratios.csv", index=False)
    metrics.to_csv(DATA_PROCESSED_DIR / "model_cv_metrics.csv", index=False)
    importance.to_csv(DATA_PROCESSED_DIR / "model_feature_importance.csv", index=False)
    calib.to_csv(DATA_PROCESSED_DIR / "model_calibration.csv", index=False)
    plot_results(metrics, importance, calib)

    pd.set_option("display.width", 200)
    print("\n== Chi-square assumptions & multiplicity ==\n", assumptions.round(4).to_string(index=False))
    print("\n== Adjusted specialty odds ratios (ref: Cardiology) ==\n", odds.to_string(index=False))
    print(f"\nLR test for specialty: chi2={odds.attrs['lr_stat']:.2f}, df={odds.attrs['lr_df']}, p={odds.attrs['lr_p']:.5f}")
    print("\n== CV metrics (5-fold x 5 repeats) ==\n", metrics.to_string(index=False))
    print("\n== Label-permutation test (logistic) ==\n", perm)
    print("\n== Permutation importance ==\n", importance.to_string(index=False))
    print("\n== Calibration ==\n", calib.to_string(index=False))


if __name__ == "__main__":
    run()
