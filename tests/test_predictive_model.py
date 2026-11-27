"""Tests for the predictive no-show model helpers."""
import numpy as np

from src import predictive_model as pm


def test_no_leakage_features():
    used = set(pm.CATEGORICAL + pm.NUMERIC)
    assert not used & set(pm.EXCLUDED_LEAKAGE)


def test_top_decile_lift_perfect_ranking():
    y = np.array([1] * 10 + [0] * 90)
    assert pm.top_decile_lift(y, y.astype(float)) == 10.0


def test_assumptions_table_shape_and_holm():
    df = pm.build_frame()
    t = pm.chi_square_assumptions(df)
    assert len(t) == len(pm.SEGMENT_COLUMNS)
    assert (t["min_expected_count"] >= 5).all()
    assert (t["p_holm"] >= t["p_value"]).all()
    assert t.loc[0, "segment"] == "specialty" and t.loc[0, "significant_after_holm"]


def test_models_cover_baseline_and_are_seeded():
    assert pm.RANDOM_STATE == 42
    assert "Baseline (predict prevalence)" in pm.make_models()


def test_constant_score_lift_is_random_baseline():
    y = np.array([1] * 10 + [0] * 90)
    assert pm.top_decile_lift(y, np.full(100, 0.162)) == 1.0

def test_patient_groups_never_cross_fold_boundary():
    df = pm.build_frame()
    groups = df["patient_id"].to_numpy()
    for train, test in pm.grouped_splits(df, repeats=1):
        assert set(groups[train]).isdisjoint(set(groups[test]))

def test_grouped_splits_cover_every_row_once_per_repeat():
    df = pm.build_frame()
    seen = np.zeros(len(df), dtype=int)
    for _, test in pm.grouped_splits(df, repeats=1):
        seen[test] += 1
    assert np.all(seen == 1)
