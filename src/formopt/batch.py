"""Batch suggestions (exploratory): q experiments per oven cycle.

Kriging believer heuristic (Ginsbourger et al. 2010): pick the best candidate with the usual
acquisition, pretend the lab measured exactly the GP mean there, update the GP posterior with
that "believed" result (hyperparameters kept fixed, so no new marginal-likelihood fit), and
pick again, q times. All q recipes of a cycle are chosen with the same real information,
which is what happens when four trials share one oven cycle.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from formopt.acquisition import (
    constrained_cost_acquisition,
    cost_aware,
    probability_of_feasibility,
)
from formopt.design import sample_known_feasible, to_unit
from formopt.gp import FittedGP
from formopt.problems.base import Problem, passes_specs
from formopt.strategies import (
    ConstrainedBO,
    GPSettings,
    best_passed_cost,
    fit_surrogates,
)
from formopt.types import Observation


def believer_refit(gp: FittedGP, X: np.ndarray, y_raw: np.ndarray) -> FittedGP:
    """GP posterior on (X, y) with the already-fitted kernel (no hyperparameter search)."""
    y_fit = np.log(y_raw) if gp.transform == "log" else np.asarray(y_raw, dtype=float)
    model = GaussianProcessRegressor(kernel=gp.model.kernel_, normalize_y=True, optimizer=None)
    model.fit(to_unit(X, gp.bounds), y_fit)
    return FittedGP(model=model, bounds=gp.bounds, transform=gp.transform, n_warnings=0)


def plan_batch(
    problem: Problem,
    history: list[Observation],
    spent: float,
    budget: float,
    rng: np.random.Generator,
    n_candidates: int,
    alpha_at: Callable[[float], float],
    settings: GPSettings,
    q: int,
) -> tuple[list[tuple[np.ndarray, dict]], int]:
    """Choose up to q recipes for one cycle; returns [(x, predicted at x)] and GP warnings.

    Each pick uses the same score as sequential BO (improvement x PoF / run_cost ** alpha).
    After a pick, the believed outcome (GP mean) is added to the training data; if that
    believed outcome passes every spec and is cheaper, it also becomes the incumbent.
    """
    gps = fit_surrogates(problem, history, settings, seed=int(rng.integers(2**31 - 1)))
    n_warn = sum(gp.n_warnings for gp in gps.values())
    X = np.vstack([o.x for o in history])
    Y = {s.output: np.array([o.outputs[s.output] for o in history]) for s in problem.specs}
    cand = sample_known_feasible(problem, n_candidates, rng)
    cost, run_cost = problem.material_cost(cand), problem.run_cost(cand)
    best = best_passed_cost(history)
    taken = np.zeros(len(cand), dtype=bool)
    planned: list[tuple[np.ndarray, dict]] = []
    planned_cost = 0.0
    for _ in range(q):
        affordable = (run_cost <= budget - spent - planned_cost) & ~taken
        if not affordable.any():
            break
        preds = {o: gp.predict(cand) for o, gp in gps.items()}
        pof = probability_of_feasibility(preds, problem.specs)
        alpha = alpha_at(spent + planned_cost)
        score = cost_aware(constrained_cost_acquisition(cost, pof, best), run_cost, alpha)
        if best is not None and not np.any(score[affordable] > 0):
            score = cost_aware(pof * (cost < best), run_cost, alpha)
            if not np.any(score[affordable] > 0):
                score = cost_aware(pof, run_cost, alpha)
        i = int(np.argmax(np.where(affordable, score, -np.inf)))
        x = cand[i]
        pred = {}
        for o, gp in gps.items():
            m, s = gp.predict(x[None, :], include_noise=True)
            pred[o] = (float(m[0]), float(s[0]))
        planned.append((x, pred))
        planned_cost += float(run_cost[i])
        taken[i] = True
        believed = {
            spec.output: float(spec.inverse_transform(preds[spec.output][0][i]))
            for spec in problem.specs
        }
        if passes_specs(believed, problem.specs) and (best is None or cost[i] < best):
            best = float(cost[i])
        X = np.vstack([X, x])
        for spec in problem.specs:
            Y[spec.output] = np.append(Y[spec.output], believed[spec.output])
            gps[spec.output] = believer_refit(gps[spec.output], X, Y[spec.output])
    return planned, n_warn


class BatchBO(ConstrainedBO):
    """Constrained (optionally cost-aware) BO that plans q experiments per oven cycle.

    The initial LHS design is the same as for sequential BO with the same seed (run q per
    cycle). After that, each cycle's q recipes come from `plan_batch`.
    """

    def __init__(self, name: str, alpha_mode: str = "none", q: int = 4, **kwargs) -> None:
        super().__init__(name, alpha_mode=alpha_mode, **kwargs)
        self.q = q

    def reset(self, problem: Problem, budget: float, rng: np.random.Generator) -> None:
        super().reset(problem, budget, rng)
        self.batch: list[tuple[np.ndarray, dict]] = []
        self.n_initial_run = 0
        self.n_batches = 0

    @property
    def n_cycles(self) -> int:
        """Oven cycles used so far: initial design q at a time, then one per planned batch."""
        return math.ceil(self.n_initial_run / self.q) + self.n_batches

    def suggest(self, history: list[Observation], spent: float) -> np.ndarray | None:
        self.last_prediction = None
        if self.queue:
            x = self._pop_affordable(spent)
            if x is not None:
                self.n_initial_run += 1
                return x
        if len(history) < 2:
            return self._random_affordable(spent)
        if not self.batch:
            self.batch, n_warn = plan_batch(
                self.problem,
                history,
                spent,
                self.budget,
                self.rng,
                self.n_candidates,
                self.current_alpha,
                self.gp,
                self.q,
            )
            self.n_gp_warnings += n_warn
            if not self.batch:
                return None
            self.n_batches += 1
        x, pred = self.batch.pop(0)
        self.last_prediction = pred
        return x
