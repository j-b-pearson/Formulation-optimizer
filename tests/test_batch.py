"""Batch mode (exploratory): q-batch BO with the kriging-believer heuristic, on the toy lab."""

from __future__ import annotations

import math

import numpy as np

from formopt.batch import BatchBO, plan_batch
from formopt.benchmark import make_strategy
from formopt.config import load_config
from formopt.loop import make_observation, noise_rng, run_campaign
from formopt.strategies import GPSettings


def batch_strategy(q: int = 3) -> BatchBO:
    return BatchBO("bo_plain_q3", alpha_mode="none", q=q, n_init=5, n_candidates=300)


def test_batch_campaign_respects_budget_and_known_constraints(toy_problem) -> None:
    for seed in (0, 1):
        strategy = batch_strategy()
        result = run_campaign(toy_problem, strategy, 40.0, seed)
        assert result.cumulative_spend()[-1] <= 40.0 + 1e-9
        assert toy_problem.known_feasible(result.X()).all()
        n_bo = result.n_experiments - strategy.n_initial_run
        assert strategy.n_batches == math.ceil(n_bo / strategy.q)
        assert strategy.n_cycles == math.ceil(5 / 3) + strategy.n_batches
        assert strategy.n_cycles < result.n_experiments  # fewer oven cycles than runs


def test_plan_batch_returns_q_distinct_affordable_points(toy_problem) -> None:
    rng = np.random.default_rng(0)
    X = rng.random((6, 2)) * 0.7
    history = [make_observation(toy_problem, x, noise_rng(0, i)) for i, x in enumerate(X)]
    planned, _ = plan_batch(
        toy_problem, history, 0.0, 100.0, rng, 300, lambda spent: 0.0, GPSettings(), q=4
    )
    xs = np.array([x for x, _ in planned])
    assert xs.shape == (4, 2) and len(np.unique(xs, axis=0)) == 4
    assert toy_problem.known_feasible(xs).all()
    tight, _ = plan_batch(
        toy_problem, history, 0.0, 2.0, rng, 300, lambda spent: 0.0, GPSettings(), q=4
    )
    assert sum(float(toy_problem.run_cost(x[None, :])[0]) for x, _ in tight) <= 2.0


def test_same_seed_same_batch_campaign(toy_problem) -> None:
    a = run_campaign(toy_problem, batch_strategy(), 30.0, 3).X()
    b = run_campaign(toy_problem, batch_strategy(), 30.0, 3).X()
    assert np.array_equal(a, b)


def test_make_strategy_builds_batch_variants() -> None:
    cfg = load_config()
    s = make_strategy("bo_cost_cooled_q4", cfg, "enamel")
    assert isinstance(s, BatchBO) and s.q == 4 and s.alpha_mode == "cooled"
    assert make_strategy("bo_plain_q4", cfg, "enamel").alpha_mode == "none"
