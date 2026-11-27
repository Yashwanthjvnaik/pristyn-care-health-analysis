"""Unit tests for deep-dive functions. Run with: pytest"""

import pandas as pd
import pytest

from src.deep_dive import (
    appointment_funnel, leakage_summary, segment_noshow_tests, specialty_noshow_table,
)


@pytest.fixture
def appts():
    return pd.DataFrame({
        "specialty": ["A"] * 10 + ["B"] * 10,
        "department": ["X", "Y"] * 10,
        "reason": ["r"] * 20,
        "insurance_type": ["i"] * 20,
        "patient_age_group": ["a"] * 20,
        "lead_bucket": ["0-1d"] * 20,
        "hour_bucket": ["9-11"] * 20,
        "no_show": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0] + [0] * 9 + [1],
        "consultation_mins": [0, 0, 0, 0, 20, 20, 20, 20, 20, 20] + [20] * 9 + [0],
    })


def test_funnel_counts(appts):
    f = appointment_funnel(appts)
    assert list(f["patients"]) == [20, 15, 15]
    assert f.loc[1, "drop_off_pct"] == 25.0


def test_leakage_summary_picks_highest_rate(appts):
    s = leakage_summary(appts, top_n=1)
    assert s["top_specialties"] == ["A"]
    assert s["share_of_no_shows_pct"] == 80.0


def test_specialty_table_sorted_and_bonferroni(appts):
    t = specialty_noshow_table(appts)
    assert t.iloc[0]["specialty"] == "A"
    assert (t["p_bonferroni"] >= t["p_vs_rest"]).all()


def test_segment_tests_returns_all_segments(appts):
    t = segment_noshow_tests(appts.assign(lead_bucket=["0-1d", "2-3d"] * 10))
    assert {"segment", "p_value"} <= set(t.columns)
