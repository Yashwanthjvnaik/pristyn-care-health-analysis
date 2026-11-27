"""Pristyn Care patient funnel analysis (Cynaris Data Analyst project).

Usage:
    python funnel_analysis.py <folder with the three CSVs> [output.json]

Reads the three data-pack files (matched by name: *appointments*, *satisfaction*,
*resource*) and writes every number used on the dashboard to one JSON file.

Definitions
-----------
Booked            every row in the appointments file (2,000)
Attended          no_show == 0  (identical to consultation_mins > 0 in this data)
Follow-up proxy   an attended patient who has a LATER appointment row. The data
                  has no follow-up flag, so this is a proxy and it is right-censored:
                  appointments end 12 Jan 2026, median gap between visits is 98 days.
Tests             chi-square (omnibus), two-proportion z-test of one group vs all
                  others, Wilson 95% intervals, Bonferroni correction across the
                  8 specialty comparisons.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

folder = sys.argv[1] if len(sys.argv) > 1 else "."
out_path = sys.argv[2] if len(sys.argv) > 2 else "metrics.json"


def find(pattern):
    hits = sorted(glob.glob(os.path.join(folder, pattern)))
    if not hits:
        sys.exit(f"No file matching {pattern} in {folder}")
    return hits[0]


appt = pd.read_csv(find("*appointments*.csv"), parse_dates=["booking_date", "appointment_date"])
sat = pd.read_csv(find("*satisfaction*.csv"), parse_dates=["survey_date"])
res = pd.read_csv(find("*resource*.csv"), parse_dates=["date"])


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def chi2_p(a, b):
    return float(stats.chi2_contingency(pd.crosstab(a, b))[1])


def two_prop(k1, n1, k2, n2):
    pp = (k1 + k2) / (n1 + n2)
    z = (k1 / n1 - k2 / n2) / np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    return float(z), float(2 * (1 - stats.norm.cdf(abs(z))))


out = {}
N = len(appt)
K = int(appt.no_show.sum())
appt["lead"] = (appt.appointment_date - appt.booking_date).dt.days
appt["month"] = appt.appointment_date.dt.to_period("M").astype(str)

out["overview"] = dict(
    appointments=N, no_shows=K, attended=N - K, no_show_rate=K / N,
    no_show_ci=wilson(K, N), patients=int(appt.patient_id.nunique()),
    doctors=int(appt.doctor_id.nunique()),
    avg_wait_mins=float(appt.wait_time_mins.mean()),
    avg_consult_mins=float(appt[appt.no_show == 0].consultation_mins.mean()),
    booking_window=[str(appt.booking_date.min().date()), str(appt.booking_date.max().date())],
    appointment_window=[str(appt.appointment_date.min().date()), str(appt.appointment_date.max().date())],
    consult_zero_equals_no_show=bool(((appt.consultation_mins == 0) == (appt.no_show == 1)).all()),
)

# --- specialty leakage ------------------------------------------------------
rows = []
for spec, g in appt.groupby("specialty"):
    rest = appt[appt.specialty != spec]
    k1, n1, k2, n2 = int(g.no_show.sum()), len(g), int(rest.no_show.sum()), len(rest)
    z, p = two_prop(k1, n1, k2, n2)
    lo, hi = wilson(k1, n1)
    rows.append(dict(specialty=spec, n=n1, no_shows=k1, rate=k1 / n1, ci=[lo, hi],
                     z=z, p=p, p_bonferroni=min(1.0, p * 8), visits_per_5pt=n1 * 0.05))
rows.sort(key=lambda r: -r["rate"])
out["specialty"] = rows
chi = stats.chi2_contingency(pd.crosstab(appt.specialty, appt.no_show))
out["specialty_omnibus"] = dict(chi2=float(chi[0]), dof=int(chi[2]), p=float(chi[1]))
top3 = [r["specialty"] for r in rows[:3]]
t = appt.specialty.isin(top3)
out["top3"] = dict(names=top3, rate=float(appt[t].no_show.mean()), n=int(t.sum()),
                   rest_rate=float(appt[~t].no_show.mean()),
                   p=float(stats.chi2_contingency(pd.crosstab(t, appt.no_show), correction=False)[1]),
                   visits_per_5pt=float(sum(r["visits_per_5pt"] for r in rows[:3])))

# --- other drivers (all tested, all reported) --------------------------------
appt["lead_bucket"] = pd.cut(appt.lead, [0, 1, 3, 7, 14], labels=["1 day", "2-3 days", "4-7 days", "8-14 days"])
appt["wait_bucket"] = pd.cut(appt.wait_time_mins, [0, 20, 30, 40, 50, 60], labels=["<=20", "21-30", "31-40", "41-50", "51-60"])
appt["weekday"] = appt.appointment_date.dt.day_name()
drivers = {}
for label, col in [("Department", "department"), ("Visit reason", "reason"), ("Insurance type", "insurance_type"),
                   ("Patient age group", "patient_age_group"), ("Booking lead time", "lead_bucket"),
                   ("Wait time at clinic", "wait_bucket"), ("Appointment weekday", "weekday"),
                   ("Doctor (100 doctors)", "doctor_id")]:
    tab = appt.groupby(col, observed=True).no_show.mean()
    drivers[label] = dict(p=chi2_p(appt[col], appt.no_show), groups=int(tab.shape[0]),
                          low=[str(tab.idxmin()), float(tab.min())], high=[str(tab.idxmax()), float(tab.max())])
out["drivers"] = drivers

# --- time ---------------------------------------------------------------------
m = appt.groupby("month").no_show.agg(["count", "sum", "mean"]).reset_index()
out["monthly"] = [dict(month=r["month"], n=int(r["count"]), no_shows=int(r["sum"]), rate=float(r["mean"])) for _, r in m.iterrows()]
may, oth = appt[appt.month == "2025-05"], appt[appt.month != "2025-05"]
ct = [[int(may.no_show.sum()), len(may) - int(may.no_show.sum())], [int(oth.no_show.sum()), len(oth) - int(oth.no_show.sum())]]
out["may_spike"] = dict(rate=float(may.no_show.mean()), n=len(may), others_rate=float(oth.no_show.mean()),
                        p=float(stats.chi2_contingency(ct, correction=False)[1]))
appt["t"] = (appt.appointment_date - appt.appointment_date.min()).dt.days / 30.4
lr = stats.linregress(appt.t, appt.no_show)
core = appt[(appt.month != "2025-05") & (appt.month != "2026-01")]
lr2 = stats.linregress(core.t, core.no_show)
h1, h2 = appt[appt.appointment_date < "2025-07-01"], appt[appt.appointment_date >= "2025-07-01"]
out["trend"] = dict(slope_pp_per_month=float(lr.slope * 100), p=float(lr.pvalue),
                    slope_excl_may_jan26=float(lr2.slope * 100), p_excl_may_jan26=float(lr2.pvalue),
                    h1_rate=float(h1.no_show.mean()), h2_rate=float(h2.no_show.mean()),
                    h1h2_p=float(stats.chi2_contingency(
                        [[int(h1.no_show.sum()), len(h1) - int(h1.no_show.sum())],
                         [int(h2.no_show.sum()), len(h2) - int(h2.no_show.sum())]], correction=False)[1]))

# --- follow-up proxy ----------------------------------------------------------
appt = appt.sort_values(["patient_id", "appointment_date", "appointment_id"])
appt["has_next"] = appt.groupby("patient_id").appointment_date.shift(-1).notna()
att = appt[appt.no_show == 0].copy()
att["half"] = np.where(att.appointment_date < "2025-07-01", "H1", "H2")
gaps = appt.groupby("patient_id").appointment_date.diff().dt.days.dropna()
out["follow_up_proxy"] = dict(
    attended=len(att), returned=int(att.has_next.sum()), rate=float(att.has_next.mean()),
    rate_h1=float(att[att.half == "H1"].has_next.mean()), n_h1=int((att.half == "H1").sum()),
    rate_h2=float(att[att.half == "H2"].has_next.mean()), n_h2=int((att.half == "H2").sum()),
    h1h2_p=float(stats.chi2_contingency(pd.crosstab(att.half, att.has_next), correction=False)[1]),
    median_gap_days=float(gaps.median()))

# --- Sunrise files (different hospital group, 2023-24, no join key) -----------
groups = [g.overall_satisfaction.values for _, g in sat.groupby("hospital")]
out["satisfaction"] = dict(
    surveys=len(sat), overall=float(sat.overall_satisfaction.mean()),
    would_recommend={k: float((sat.would_recommend == k).mean()) for k in ["Yes", "Maybe", "No"]},
    corr_wait_vs_overall=float(sat.waiting_time_mins.corr(sat.overall_satisfaction)),
    by_hospital={k: float(v) for k, v in sat.groupby("hospital").overall_satisfaction.mean().sort_values().items()},
    hospital_anova_p=float(stats.f_oneway(*groups)[1]),
    window=[str(sat.survey_date.min().date()), str(sat.survey_date.max().date())])
out["resource"] = dict(
    rows=len(res), beds_occupied_over_total=int((res.occupied_beds > res.total_beds).sum()),
    icu_occupied_over_total=int((res.icu_beds_occupied > res.icu_beds_total).sum()),
    surgeries_completed_over_scheduled=int((res.surgeries_completed > res.surgeries_scheduled).sum()),
    doctor_attendance=float(res.doctor_attendance_pct.mean()), nurse_attendance=float(res.nurse_attendance_pct.mean()),
    window=[str(res.date.min().date()), str(res.date.max().date())])

with open(out_path, "w") as f:
    json.dump(out, f, indent=1)
print(f"wrote {out_path}")
