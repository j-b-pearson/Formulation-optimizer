"""Strategies: the rule a lab uses to pick its next experiment.

Baselines (random search, Latin hypercube plan, fractional factorial plan) and Bayesian
optimization (constrained BO that ignores run cost, and cost-aware BO with a fixed or cooled
alpha). Every strategy respects the remaining budget and only proposes known-feasible points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from formopt.acquisition import (
    constrained_cost_acquisition,
    cooled_alpha,
    cost_aware,
    probability_of_feasibility,
    probability_of_feasibility_per_spec,
)
from formopt.design import (
    fractional_factorial,
    lhs_design,
    repair_to_feasible,
    sample_known_feasible,
)
from formopt.gp import FittedGP, fit_gp
from formopt.problems.base import Problem
from formopt.types import Observation

POOL_SIZE = 4000  # known-feasible points used to repair planned designs


class Strategy(Protocol):
    name: str

    def reset(self, problem: Problem, budget: float, rng: np.random.Generator) -> None: ...

    def suggest(self, history: list[Observation], spent: float) -> np.ndarray | None: ...


@dataclass
class GPSettings:
    """GP fitting settings shared by the BO strategies (from `config.yaml` `bo:`)."""

    restarts: int = 2
    length_scale_bounds: tuple[float, float] = (1e-2, 10.0)
    constant_bounds: tuple[float, float] = (1e-3, 1e3)
    noise_bounds: tuple[float, float] = (1e-6, 1.0)


@dataclass
class Suggestion:
    """One BO recommendation with everything the app and the calibration analysis need."""

    x: np.ndarray
    gps: dict[str, FittedGP]
    candidates: np.ndarray
    scores: np.ndarray
    pof: np.ndarray  # PoF of every candidate
    pred_latent: dict[str, tuple[float, float]]  # at x, GP space, true response
    pred_noisy: dict[str, tuple[float, float]]  # at x, GP space, a new measurement
    pof_per_spec: dict[str, float]  # at x
    pof_at_x: float
    run_cost: float
    material_cost: float
    alpha: float
    n_gp_warnings: int = 0
    extra: dict = field(default_factory=dict)


def best_passed_cost(history: list[Observation]) -> float | None:
    """Cheapest material cost among runs the lab saw pass (all a real lab would know)."""
    costs = [o.material_cost for o in history if o.passed_spec]
    return min(costs) if costs else None


def fit_surrogates(
    problem: Problem, history: list[Observation], settings: GPSettings, seed: int
) -> dict[str, FittedGP]:
    """Fit one GP per unknown spec output on the full history (never on cost)."""
    X = np.vstack([o.x for o in history])
    gps = {}
    for spec in problem.specs:
        y = np.array([o.outputs[spec.output] for o in history])
        gps[spec.output] = fit_gp(
            X,
            y,
            problem.bounds,
            transform=spec.transform,
            n_restarts=settings.restarts,
            seed=seed,
            length_scale_bounds=settings.length_scale_bounds,
            constant_bounds=settings.constant_bounds,
            noise_bounds=settings.noise_bounds,
        )
    return gps


def bo_step(
    problem: Problem,
    history: list[Observation],
    spent: float,
    budget: float,
    rng: np.random.Generator,
    n_candidates: int,
    alpha: float,
    settings: GPSettings,
) -> Suggestion | None:
    """One Bayesian optimization step in spec mode.

    Fit GPs, sample known-feasible candidates, score each by
    max(best passed cost - cost, 0) x PoF / run_cost ** alpha, drop unaffordable ones,
    return the best. Returns None if nothing affordable is left.
    """
    gps = fit_surrogates(problem, history, settings, seed=int(rng.integers(2**31 - 1)))
    cand = sample_known_feasible(problem, n_candidates, rng)
    preds = {o: gp.predict(cand) for o, gp in gps.items()}
    pof = probability_of_feasibility(preds, problem.specs)
    cost = problem.material_cost(cand)
    run_cost = problem.run_cost(cand)
    best = best_passed_cost(history)
    affordable = run_cost <= budget - spent
    if not affordable.any():
        return None
    score = cost_aware(constrained_cost_acquisition(cost, pof, best), run_cost, alpha)
    if best is not None and not np.any(score[affordable] > 0):
        # No candidate is both cheaper and plausibly feasible (numerically zero). Fall back to
        # the most likely feasible cheaper candidate, then to the most likely feasible one.
        cheaper = cost < best
        score = cost_aware(pof * cheaper, run_cost, alpha)
        if not np.any(score[affordable] > 0):
            score = cost_aware(pof, run_cost, alpha)
    score = np.where(affordable, score, -np.inf)
    i = int(np.argmax(score))
    x = cand[i]
    pred_latent = {o: (float(m[i]), float(s[i])) for o, (m, s) in preds.items()}
    pred_noisy = {}
    for o, gp in gps.items():
        m, s = gp.predict(x[None, :], include_noise=True)
        pred_noisy[o] = (float(m[0]), float(s[0]))
    per_spec = probability_of_feasibility_per_spec(
        {o: (np.array([m]), np.array([s])) for o, (m, s) in pred_latent.items()}, problem.specs
    )
    return Suggestion(
        x=x,
        gps=gps,
        candidates=cand,
        scores=score,
        pof=pof,
        pred_latent=pred_latent,
        pred_noisy=pred_noisy,
        pof_per_spec={o: float(v[0]) for o, v in per_spec.items()},
        pof_at_x=float(pof[i]),
        run_cost=float(run_cost[i]),
        material_cost=float(cost[i]),
        alpha=float(alpha),
        n_gp_warnings=sum(gp.n_warnings for gp in gps.values()),
    )


class _BaseStrategy:
    """Shared bookkeeping: problem, budget, RNG and the planned-points queue."""

    name = "base"

    def reset(self, problem: Problem, budget: float, rng: np.random.Generator) -> None:
        self.problem = problem
        self.budget = budget
        self.rng = rng
        self.queue: list[np.ndarray] = []
        self.last_prediction: dict[str, tuple[float, float]] | None = None
        self.n_gp_warnings = 0

    def _pop_affordable(self, spent: float) -> np.ndarray | None:
        """Next planned point that still fits the budget; unaffordable ones are skipped."""
        remaining = self.budget - spent
        while self.queue:
            x = self.queue.pop(0)
            if self.problem.run_cost(x[None, :])[0] <= remaining:
                return x
        return None

    def _random_affordable(self, spent: float, batch: int = 1000) -> np.ndarray | None:
        """A uniform random known-feasible point that fits the remaining budget."""
        X = sample_known_feasible(self.problem, batch, self.rng)
        ok = self.problem.run_cost(X) <= self.budget - spent
        return X[int(np.argmax(ok))] if ok.any() else None

    def _repaired(self, X: np.ndarray) -> np.ndarray:
        """Move planned points that break known constraints to the nearest feasible ones."""
        pool = sample_known_feasible(self.problem, POOL_SIZE, self.rng)
        return repair_to_feasible(X, pool, self.problem.known_feasible(X), self.problem.bounds)

    def _median_run_cost(self) -> float:
        pool = sample_known_feasible(self.problem, POOL_SIZE, self.rng)
        return float(np.median(self.problem.run_cost(pool)))


class RandomSearch(_BaseStrategy):
    """Uniform random experiments over the known-feasible region."""

    name = "random"

    def __init__(self, batch: int = 1000) -> None:
        self.batch = batch

    def suggest(self, history: list[Observation], spent: float) -> np.ndarray | None:
        self.last_prediction = None
        return self._random_affordable(spent, self.batch)


class LHSPlan(_BaseStrategy):
    """Space-filling DOE: an LHS of n = floor(budget / median run cost) points planned up
    front, run in shuffled order. If budget is left after the plan, random points fill it."""

    name = "lhs"

    def reset(self, problem: Problem, budget: float, rng: np.random.Generator) -> None:
        super().reset(problem, budget, rng)
        n = max(int(budget // self._median_run_cost()), 1)
        plan = self._repaired(lhs_design(problem.bounds, n, rng))
        self.queue = list(plan[rng.permutation(n)])
        self.n_planned = n

    def suggest(self, history: list[Observation], spent: float) -> np.ndarray | None:
        self.last_prediction = None
        x = self._pop_affordable(spent)
        return x if x is not None else self._random_affordable(spent)


class FactorialPlan(_BaseStrategy):
    """Classic DOE: 2-level fractional factorial at 20/80 % levels plus centre points (run in
    randomized order), then an LHS fill sized to the expected remaining budget."""

    name = "factorial"

    def __init__(self, generators: str, levels: tuple[float, float], n_center: int) -> None:
        self.generators = generators
        self.levels = levels
        self.n_center = n_center

    def reset(self, problem: Problem, budget: float, rng: np.random.Generator) -> None:
        super().reset(problem, budget, rng)
        design = fractional_factorial(problem.bounds, self.generators, self.levels, self.n_center)
        design = self._repaired(design)[rng.permutation(len(design))]
        plan_cost = float(problem.run_cost(design).sum())
        n_fill = int(max(budget - plan_cost, 0.0) // self._median_run_cost())
        fill = self._repaired(lhs_design(problem.bounds, n_fill, rng)) if n_fill else design[:0]
        self.queue = list(design) + list(fill)
        self.n_planned = len(design)

    def suggest(self, history: list[Observation], spent: float) -> np.ndarray | None:
        self.last_prediction = None
        x = self._pop_affordable(spent)
        return x if x is not None else self._random_affordable(spent)


class ConstrainedBO(_BaseStrategy):
    """Bayesian optimization in spec mode.

    `alpha_mode="none"` is plain constrained BO (ignores run cost), `"fixed"` divides by
    run_cost ** alpha, `"cooled"` lowers alpha linearly from 1 to 0 as the budget is spent.
    All BO variants with the same seed share the same initial LHS design.
    """

    def __init__(
        self,
        name: str,
        alpha_mode: str = "none",
        alpha: float = 1.0,
        n_init: int = 6,
        n_candidates: int = 4000,
        gp: GPSettings | None = None,
    ) -> None:
        if alpha_mode not in ("none", "fixed", "cooled"):
            raise ValueError(f"Unknown alpha_mode {alpha_mode!r}")
        self.name = name
        self.alpha_mode = alpha_mode
        self.alpha = alpha
        self.n_init = n_init
        self.n_candidates = n_candidates
        self.gp = gp or GPSettings()

    def reset(self, problem: Problem, budget: float, rng: np.random.Generator) -> None:
        super().reset(problem, budget, rng)
        # Drawn first from the strategy stream, so every BO variant gets the same design.
        self.queue = list(self._repaired(lhs_design(problem.bounds, self.n_init, rng)))
        self.last_suggestion: Suggestion | None = None

    def current_alpha(self, spent: float) -> float:
        if self.alpha_mode == "none":
            return 0.0
        if self.alpha_mode == "fixed":
            return self.alpha
        return self.alpha * cooled_alpha(spent, self.budget)

    def suggest(self, history: list[Observation], spent: float) -> np.ndarray | None:
        self.last_prediction = None
        if self.queue:
            x = self._pop_affordable(spent)
            if x is not None:
                return x
        if len(history) < 2:
            return self._random_affordable(spent)
        sug = bo_step(
            self.problem,
            history,
            spent,
            self.budget,
            self.rng,
            self.n_candidates,
            self.current_alpha(spent),
            self.gp,
        )
        if sug is None:
            return None
        self.n_gp_warnings += sug.n_gp_warnings
        self.last_prediction = sug.pred_noisy
        self.last_suggestion = sug
        return sug.x


class PlainConstrainedBO(ConstrainedBO):
    """Standard constrained BO: improvement x PoF, run cost ignored (alpha = 0)."""

    def __init__(self, name: str = "bo_plain", **kwargs) -> None:
        super().__init__(name, alpha_mode="none", **kwargs)


class CostAwareBO(ConstrainedBO):
    """Cost-aware BO: improvement x PoF / run_cost ** alpha, alpha fixed or cooled to 0."""

    def __init__(self, name: str = "bo_cost", alpha_mode: str = "fixed", **kwargs) -> None:
        if alpha_mode == "none":
            raise ValueError("CostAwareBO needs alpha_mode 'fixed' or 'cooled'")
        super().__init__(name, alpha_mode=alpha_mode, **kwargs)
