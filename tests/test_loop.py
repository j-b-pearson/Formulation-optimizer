"""Budgeted loop: budget never exceeded, only known-feasible runs, determinism, toy optimum."""

from __future__ import annotations

import numpy as np
import pytest

from formopt.estimate import expected_spend_to_spec
from formopt.loop import noise_rng, run_campaign
from formopt.strategies import (
    ConstrainedBO,
    FactorialPlan,
    GPSettings,
    LHSPlan,
    RandomSearch,
    fit_surrogates,
)
from tests.conftest import TOY_OPT_COST, ToyProblem

BUDGET = 60.0


def _strategies() -> list:
    return [
        RandomSearch(batch=200),
        LHSPlan(),
        FactorialPlan("a b", (0.2, 0.8), 3),
        ConstrainedBO("bo_plain", "none", n_init=6, n_candidates=500),
        ConstrainedBO("bo_cost", "fixed", n_init=6, n_candidates=500),
        ConstrainedBO("bo_cost_cooled", "cooled", n_init=6, n_candidates=500),
    ]


@pytest.mark.parametrize("index", range(6))
def test_budget_never_exceeded_and_only_known_feasible_runs(index: int) -> None:
    problem = ToyProblem()
    seeds = (0, 1) if index < 3 else (0,)  # BO campaigns are slower; one seed each
    for seed in seeds:
        result = run_campaign(problem, _strategies()[index], BUDGET, seed)
        assert result.n_experiments > 5
        assert result.cumulative_spend()[-1] <= BUDGET
        assert problem.known_feasible(result.X()).all()
        # The loop stops only when no affordable run is left or the strategy gives up.
        assert BUDGET - result.cumulative_spend()[-1] < 5.0 + 1e-9


def test_factorial_points_are_repaired() -> None:
    problem = ToyProblem()
    strategy = FactorialPlan("a b", (0.2, 0.8), 3)
    strategy.reset(problem, BUDGET, np.random.default_rng(0))
    planned = np.array(strategy.queue[: strategy.n_planned])
    assert planned.shape == (7, 2)
    assert problem.known_feasible(planned).all()  # the (0.8, 0.8) corner was moved


@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_same_seed_same_campaign(index: int) -> None:
    problem = ToyProblem()
    budget = BUDGET if index < 3 else 30.0
    a = run_campaign(problem, _strategies()[index], budget, seed=3)
    b = run_campaign(problem, _strategies()[index], budget, seed=3)
    np.testing.assert_array_equal(a.X(), b.X())
    assert [o.outputs for o in a.observations] == [o.outputs for o in b.observations]


def test_bo_variants_share_initial_design() -> None:
    problem = ToyProblem()
    runs = [run_campaign(problem, s, 25.0, seed=5) for s in _strategies()[3:]]
    for r in runs[1:]:
        np.testing.assert_array_equal(r.X()[:6], runs[0].X()[:6])


def test_noise_depends_only_on_seed_and_step() -> None:
    a = noise_rng(7, 3).normal(size=4)
    b = noise_rng(7, 3).normal(size=4)
    c = noise_rng(7, 4).normal(size=4)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_bo_finds_toy_optimum_in_most_seeds() -> None:
    problem = ToyProblem()
    hits = 0
    for seed in range(5):
        result = run_campaign(
            problem, ConstrainedBO("bo_plain", "none", n_init=6, n_candidates=500), BUDGET, seed
        )
        best = result.best_feasible_cost_trace()[-1]
        hits += best <= TOY_OPT_COST * 1.10
    assert hits >= 3


def test_bo_records_predictions_for_calibration() -> None:
    result = run_campaign(
        ToyProblem(), ConstrainedBO("bo_cost", "fixed", n_init=6, n_candidates=300), BUDGET, 0
    )
    assert all(o.predicted is None for o in result.observations[:6])
    assert all(o.predicted is not None for o in result.observations[6:])


def test_expected_spend_to_spec_is_finite_with_real_gps() -> None:
    problem = ToyProblem()
    result = run_campaign(problem, LHSPlan(), 20.0, seed=0)
    gps = fit_surrogates(problem, result.observations, GPSettings(), seed=0)
    cand = np.random.default_rng(0).random((300, 2))
    cand = cand[problem.known_feasible(cand)]
    estimate = expected_spend_to_spec(problem, gps, cand, k=10)
    assert np.isfinite(estimate) and estimate > 0
