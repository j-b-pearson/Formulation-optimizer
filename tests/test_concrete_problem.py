"""Concrete data cleaning, emulator validation helpers and the concrete problem.

Uses the 40-row fake frame and a tiny ExtraTrees emulator from conftest (no download).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

from formopt.data import INGREDIENTS, SHORT_NAMES, clean_concrete, rename_concrete_columns
from formopt.design import sample_known_feasible
from formopt.emulator import grouped_cv, make_features, water_binder
from tests.conftest import N_FAKE_DUPLICATES, N_FAKE_MIXES


def test_load_and_mix_id(fake_concrete_raw: pd.DataFrame) -> None:
    assert len(fake_concrete_raw) == 40
    renamed = rename_concrete_columns(fake_concrete_raw)
    assert list(renamed.columns) == SHORT_NAMES

    df, n_dropped = clean_concrete(fake_concrete_raw)
    assert n_dropped == N_FAKE_DUPLICATES
    assert len(df) == 40 - N_FAKE_DUPLICATES
    assert df["mix_id"].nunique() == N_FAKE_MIXES
    # The same recipe at two ages shares one mix_id.
    per_mix_ages = df.groupby("mix_id")["age_days"].nunique()
    assert (per_mix_ages == 2).all()
    # Zeros are real values, not missing.
    assert not df[INGREDIENTS].isna().any().any()
    assert (df["slag"] == 0).any()


def test_grouped_cv_keeps_mixes_apart(fake_concrete_raw: pd.DataFrame) -> None:
    df, _ = clean_concrete(fake_concrete_raw)
    X, y, groups = make_features(df), df["strength_mpa"].to_numpy(), df["mix_id"].to_numpy()
    for train, test in GroupKFold(n_splits=3).split(X, y, groups):
        assert not set(groups[train]) & set(groups[test])
    report = grouped_cv(
        lambda: ExtraTreesRegressor(n_estimators=20, random_state=0), X, y, groups, 3
    )
    assert {"r2", "rmse", "mae", "residual_sd", "folds", "oof_pred"} <= set(report)
    assert len(report["folds"]) == 3 and report["oof_pred"].shape == y.shape


def test_domain_check(tiny_concrete_problem) -> None:
    domain = tiny_concrete_problem.emulator.domain
    mixes = tiny_concrete_problem.emulator.unique_mixes
    assert domain.inside(mixes).mean() >= 0.9
    far = mixes.mean(axis=0) + 10 * mixes.std(axis=0)
    assert not domain.inside(far[None, :])[0]


def test_sampled_candidates_pass_known_constraints(tiny_concrete_problem) -> None:
    problem = tiny_concrete_problem
    emu = problem.emulator
    X = sample_known_feasible(problem, 200, np.random.default_rng(0))
    assert X.shape == (200, 7)
    total = X.sum(axis=1)
    wb = water_binder(X)
    assert np.all((total >= emu.total_mass_range[0]) & (total <= emu.total_mass_range[1]))
    assert np.all((wb >= emu.wb_range[0]) & (wb <= emu.wb_range[1]))
    assert emu.domain.inside(X).all()
    assert problem.in_bounds(X).all()
    # Age is fixed at 28 days in the problem.
    assert problem.cfg.age_days == 28
    np.testing.assert_allclose(problem.true_outputs(X)["strength_mpa"], emu.predict(X, 28.0))


def test_cost_per_m3(tiny_concrete_problem) -> None:
    problem = tiny_concrete_problem
    x = np.array([300.0, 100.0, 50.0, 180.0, 5.0, 1000.0, 750.0])
    p = problem.prices
    expected = (
        300 * p["cement"]
        + 100 * p["slag"]
        + 50 * p["fly_ash"]
        + 180 * p["water"]
        + 5 * p["superplasticizer"]
        + 1000 * p["coarse_agg"]
        + 750 * p["fine_agg"]
    )
    assert abs(problem.material_cost(x[None, :])[0] - expected) < 1e-9
    rc = problem.cfg.run_cost
    assert abs(problem.run_cost(x[None, :])[0] - (rc.fixed + rc.batch_m3 * expected)) < 1e-9
    shocked = problem.with_prices({"cement": p["cement"] * 1.3})
    assert shocked.material_cost(x[None, :])[0] > problem.material_cost(x[None, :])[0]


def test_observe_adds_configured_noise(tiny_concrete_problem) -> None:
    problem = tiny_concrete_problem
    x = problem.emulator.unique_mixes[0]
    mean = problem.true_outputs(x[None, :])["strength_mpa"][0]
    draws = np.array(
        [problem.observe(x, np.random.default_rng([3, i]))["strength_mpa"] for i in range(800)]
    )
    assert abs(draws.mean() - mean) < 0.5
    assert abs(draws.std(ddof=1) - problem.noise_sd) / problem.noise_sd < 0.2
