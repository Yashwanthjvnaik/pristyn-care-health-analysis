"""
Checkpoint 3 deep-dive: appointment no-show segmentation and statistical testing.

Run with:
    python -m src.deep_dive

Scope note: the appointments file supports a booking -> attended -> consultation
funnel only. Ad click, enquiry, diagnostics, surgery and follow-up stages are
not present as fields (follow-up exists only as a visit reason).
"""

from __future__ import annotations

import logging

import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint

from src.config import RAW_FILES, DATA_PROCESSED_DIR

logger = logging.getLogger(__name__)

SEGMENT_COLUMNS = [
    "specialty", "department", "reason", "insurance_type",
    "patient_age_group", "lead_bucket", "hour_bucket",
]


def load_appointments() -> pd.DataFrame:
    """Load appointments and derive lead time, buckets and month."""
    df = pd.read_csv(RAW_FILES["appointments"], parse_dates=["booking_date", "appointment_date"])
    df = df.drop_duplicates(subset="appointment_id").copy()
    df["lead_days"] = (df["appointment_date"] - df["booking_date"]).dt.days
    df["lead_bucket"] = pd.cut(
        df["lead_days"], [-1, 1, 3, 7, 14, 1000], labels=["0-1d", "2-3d", "4-7d", "8-14d", "15d+"]
    )
    df["hour_bucket"] = pd.cut(
        df["appointment_hour"], [8, 11, 14, 17], labels=["9-11", "12-14", "15-17"]
    )
    df["month"] = df["appointment_date"].dt.to_period("M").astype(str)
    return df


def appointment_funnel(df: pd.DataFrame) -> pd.DataFrame:
    """Booked -> attended -> consultation completed, with stage drop-off %."""
    booked = len(df)
    attended = int((df["no_show"] == 0).sum())
    consulted = int(((df["no_show"] == 0) & (df["consultation_mins"] > 0)).sum())
    stages = [("Booked", booked), ("Attended", attended), ("Consultation completed", consulted)]
    out = pd.DataFrame(stages, columns=["stage", "patients"])
    out["drop_off_pct"] = (1 - out["patients"] / out["patients"].shift(1)).mul(100).round(1)
    out["cumulative_pct_of_booked"] = (out["patients"] / booked * 100).round(1)
    return out


def segment_noshow_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Chi-square test of no-show vs each categorical segment."""
    rows = []
    for col in SEGMENT_COLUMNS:
        table = pd.crosstab(df[col], df["no_show"])
        chi2, p, dof, _ = stats.chi2_contingency(table)
        rows.append({"segment": col, "chi2": round(chi2, 2), "dof": dof, "p_value": p,
                     "significant_5pct": p < 0.05})
    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


def specialty_noshow_table(df: pd.DataFrame) -> pd.DataFrame:
    """No-show rate per specialty with Wilson CI and Bonferroni-adjusted p vs rest."""
    grouped = df.groupby("specialty")["no_show"].agg(["sum", "count"])
    n_tests = len(grouped)
    rows = []
    for specialty, r in grouped.iterrows():
        lo, hi = proportion_confint(r["sum"], r["count"], method="wilson")
        a = df.loc[df["specialty"] == specialty, "no_show"]
        b = df.loc[df["specialty"] != specialty, "no_show"]
        table = [[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]]
        p = stats.chi2_contingency(table, correction=False)[1]
        rows.append({
            "specialty": specialty, "appointments": int(r["count"]), "no_shows": int(r["sum"]),
            "no_show_rate_pct": round(r["sum"] / r["count"] * 100, 1),
            "ci_low_pct": round(lo * 100, 1), "ci_high_pct": round(hi * 100, 1),
            "p_vs_rest": p, "p_bonferroni": min(1.0, p * n_tests),
        })
    return pd.DataFrame(rows).sort_values("no_show_rate_pct", ascending=False).reset_index(drop=True)


def leakage_summary(df: pd.DataFrame, top_n: int = 3) -> dict:
    """Share of bookings vs share of no-shows in the top-N leakage specialties."""
    rates = df.groupby("specialty")["no_show"].mean().sort_values(ascending=False)
    top = list(rates.index[:top_n])
    mask = df["specialty"].isin(top)
    return {
        "top_specialties": top,
        "share_of_bookings_pct": round(mask.mean() * 100, 1),
        "share_of_no_shows_pct": round(df.loc[mask, "no_show"].sum() / df["no_show"].sum() * 100, 1),
    }


def monthly_noshow(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly no-show rate and appointment volume."""
    out = df.groupby("month")["no_show"].agg(["mean", "count"]).reset_index()
    out["no_show_rate_pct"] = (out["mean"] * 100).round(1)
    return out.rename(columns={"count": "appointments"}).drop(columns="mean")


def month_vs_rest_p(df: pd.DataFrame, month: str) -> float:
    """Chi-square p-value for one month's no-show rate vs all other months."""
    m = df[df["month"] == month]["no_show"]
    r = df[df["month"] != month]["no_show"]
    table = [[m.sum(), len(m) - m.sum()], [r.sum(), len(r) - r.sum()]]
    return float(stats.chi2_contingency(table, correction=False)[1])


def adjusted_logit(df: pd.DataFrame):
    """Logistic regression of no-show on specialty, controlling for other factors."""
    formula = (
        "no_show ~ C(specialty, Treatment('Cardiology')) + C(patient_age_group)"
        " + C(insurance_type) + C(department) + lead_days + wait_time_mins"
    )
    return smf.logit(formula, data=df).fit(disp=0)


def resource_integrity(path=None) -> dict:
    """Count physically impossible rows in the resource-utilisation file."""
    res = pd.read_csv(path or RAW_FILES["resources"])
    beds = res["occupied_beds"] > res["total_beds"]
    icu = res["icu_beds_occupied"] > res["icu_beds_total"]
    surg = res["surgeries_completed"] > res["surgeries_scheduled"]
    return {
        "rows": len(res),
        "occupied_gt_total_beds": int(beds.sum()),
        "icu_occupied_gt_total": int(icu.sum()),
        "surgeries_completed_gt_scheduled": int(surg.sum()),
        "rows_with_any_issue": int((beds | icu | surg).sum()),
        "valid_rows": int((~(beds | icu | surg)).sum()),
    }


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    df = load_appointments()
    outputs = {
        "funnel_appointments.csv": appointment_funnel(df),
        "segment_tests.csv": segment_noshow_tests(df),
        "specialty_noshow.csv": specialty_noshow_table(df),
        "monthly_noshow.csv": monthly_noshow(df),
    }
    for name, table in outputs.items():
        table.to_csv(DATA_PROCESSED_DIR / name, index=False)
        logger.info("Wrote %s", name)
    print(outputs["funnel_appointments.csv"].to_string(index=False))
    print(outputs["segment_tests.csv"].to_string(index=False))
    print(outputs["specialty_noshow.csv"].round(4).to_string(index=False))
    print(leakage_summary(df))
    print("May-2025 vs rest p =", month_vs_rest_p(df, "2025-05"))
    print(resource_integrity())


if __name__ == "__main__":
    run()
