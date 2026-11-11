"""The budgeted experiment loop: suggest, run, charge, record, repeat until the money is gone."""

from __future__ import annotations

import logging
import time

import numpy as np

from formopt.problems.base import Problem, passes_specs
from formopt.strategies import Strategy
from formopt.types import CampaignResult, Observation

logger = logging.getLogger(__name__)
MAX_STEPS = 10_000  # safety stop; real campaigns end on budget long before this


def noise_rng(seed: int, step: int) -> np.random.Generator:
    """Measurement-noise generator for experiment `step` of campaign `seed`.

    It depends only on (seed, step), never on the method, so two methods that run the same
    recipe at the same step see the same noise (common random numbers). This is the noise
    child stream of `SeedSequence(seed).spawn(2)`, indexed by step.
    """
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1, step)))


def make_observation(
    problem: Problem,
    x: np.ndarray,
    rng: np.random.Generator,
    predicted: dict[str, tuple[float, float]] | None = None,
) -> Observation:
    """Run one experiment on the virtual lab and record everything about it."""
    x = np.asarray(x, dtype=float)
    outputs = problem.observe(x, rng)
    truth = {k: float(v[0]) for k, v in problem.true_outputs(x[None, :]).items()}
    return Observation(
        x=x,
        outputs=outputs,
        run_cost=float(problem.run_cost(x[None, :])[0]),
        material_cost=float(problem.material_cost(x[None, :])[0]),
        passed_spec=bool(passes_specs(outputs, problem.specs)),
        truly_feasible=bool(passes_specs(truth, problem.specs)),
        true_outputs=truth,
        predicted=predicted,
    )


def run_campaign(
    problem: Problem, strategy: Strategy, budget: float, seed: int, verbose: bool = False
) -> CampaignResult:
    """Run one campaign until the strategy stops or the next run would exceed the budget.

    Two RNG streams come from `SeedSequence(seed).spawn(2)`: the first drives the strategy
    (designs, candidate sampling, GP restarts), the second the measurement noise.
    """
    strategy_seq, _ = np.random.SeedSequence(seed).spawn(2)
    strategy.reset(problem, budget, np.random.default_rng(strategy_seq))
    result = CampaignResult(problem=problem.name, method=strategy.name, seed=seed, budget=budget)
    spent = 0.0
    start = time.perf_counter()
    for step in range(MAX_STEPS):
        x = strategy.suggest(result.observations, spent)
        if x is None:
            break
        x = np.asarray(x, dtype=float)
        if not problem.known_feasible(x[None, :])[0]:
            raise RuntimeError(f"{strategy.name} proposed a known-infeasible point: {x}")
        cost = float(problem.run_cost(x[None, :])[0])
        if spent + cost > budget + 1e-9:
            break
        obs = make_observation(
            problem, x, noise_rng(seed, step), getattr(strategy, "last_prediction", None)
        )
        result.observations.append(obs)
        spent += cost
        if verbose:
            logger.info(
                "%s seed %d step %d: spent %.0f, cost %.4f, pass %s, true %s",
                strategy.name,
                seed,
                step,
                spent,
                obs.material_cost,
                obs.passed_spec,
                obs.truly_feasible,
            )
    result.wall_time_s = time.perf_counter() - start
    result.n_gp_warnings = int(getattr(strategy, "n_gp_warnings", 0))
    return result
