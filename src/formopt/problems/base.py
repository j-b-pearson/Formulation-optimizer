"""The Problem protocol (a "virtual lab"), spec checks, and the reference optimum.

A problem knows four things: its input box and known constraints, the material cost of a
recipe (a formula), the run cost of an experiment (a formula), and how to "run" an
experiment (noisy outputs). Benchmarks also use `true_outputs` (noise-free) to judge whether
a recipe really passes spec.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from formopt.config import ReferenceCfg
from formopt.design import from_unit, sample_known_feasible, to_unit
from formopt.types import Spec, Variable


@runtime_checkable
class Problem(Protocol):
    """What every virtual lab must provide."""

    name: str
    variables: list[Variable]
    specs: list[Spec]
    output_names: list[str]

    @property
    def bounds(self) -> np.ndarray: ...

    def known_feasible(self, X: np.ndarray) -> np.ndarray: ...

    def material_cost(self, X: np.ndarray) -> np.ndarray: ...

    def run_cost(self, X: np.ndarray) -> np.ndarray: ...

    def true_outputs(self, X: np.ndarray) -> dict[str, np.ndarray]: ...

    def observe(self, x: np.ndarray, rng: np.random.Generator) -> dict[str, float]: ...

    def with_prices(self, prices: dict[str, float]) -> Problem: ...

    def fingerprint(self) -> str: ...


class BaseProblem:
    """Shared helpers for concrete Problem classes."""

    name: str
    variables: list[Variable]
    specs: list[Spec]

    @property
    def output_names(self) -> list[str]:
        return [s.output for s in self.specs]

    @property
    def var_names(self) -> list[str]:
        return [v.name for v in self.variables]

    @property
    def bounds(self) -> np.ndarray:
        return np.array([[v.low, v.high] for v in self.variables], dtype=float)

    def in_bounds(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        b = self.bounds
        tol = 1e-9 * (b[:, 1] - b[:, 0])
        return np.all((X >= b[:, 0] - tol) & (X <= b[:, 1] + tol), axis=1)

    def cost_cap(self) -> float:
        """Highest material cost in the box (`C_none`): used before any feasible recipe exists.

        Material costs here are linear or bilinear in the inputs, so the maximum is at a corner.
        """
        corners = np.array(list(itertools.product(*[(v.low, v.high) for v in self.variables])))
        return float(np.max(self.material_cost(corners)))

    def as_dict(self, x: np.ndarray) -> dict[str, float]:
        """Recipe as {variable name: value}."""
        return {n: float(v) for n, v in zip(self.var_names, np.ravel(x), strict=True)}


def passes_specs(outputs: dict[str, float | np.ndarray], specs: list[Spec]) -> bool | np.ndarray:
    """True where every spec passes. Scalars in, bool out; arrays in, bool array out."""
    result = None
    for spec in specs:
        ok = spec.passes(outputs[spec.output])
        result = ok if result is None else result & ok
    if result is None:
        raise ValueError("No specs given")
    return bool(result) if np.ndim(result) == 0 else result


def spec_relative_distances(outputs: dict[str, float], specs: list[Spec]) -> dict[str, float]:
    """Relative distance |value - limit| / |limit| for every spec limit (raw units)."""
    out: dict[str, float] = {}
    for spec in specs:
        y = float(outputs[spec.output])
        if spec.lower is not None:
            out[f"{spec.output}>={spec.lower:g}"] = abs(y - spec.lower) / abs(spec.lower)
        if spec.upper is not None:
            out[f"{spec.output}<={spec.upper:g}"] = abs(spec.upper - y) / abs(spec.upper)
    return out


def _truly_feasible_cost(problem: Problem, X: np.ndarray) -> np.ndarray:
    """Material cost where the recipe is known-feasible and truly passes spec, else inf."""
    ok = problem.known_feasible(X) & passes_specs(problem.true_outputs(X), problem.specs)
    cost = problem.material_cost(X)
    return np.where(ok, cost, np.inf)


def polish(
    problem: Problem, x0: np.ndarray, rng: np.random.Generator, cfg: ReferenceCfg
) -> tuple[np.ndarray, float]:
    """Random local search that lowers material cost while staying truly feasible.

    Each move samples `polish_batch` Gaussian perturbations (in unit-cube coordinates, clipped
    to the box so bounds are reachable), keeps the cheapest truly feasible one if it improves,
    and shrinks the step size when no perturbation helps.
    """
    bounds = problem.bounds
    u = to_unit(x0, bounds)
    best = float(_truly_feasible_cost(problem, x0[None, :])[0])
    for radius in cfg.polish_radii:
        for _ in range(cfg.polish_max_moves):
            U = np.clip(u + rng.normal(0.0, radius, (cfg.polish_batch, len(u))), 0.0, 1.0)
            cost = _truly_feasible_cost(problem, from_unit(U, bounds))
            j = int(np.argmin(cost))
            if cost[j] < best - 1e-12:
                u, best = U[j], float(cost[j])
            else:
                break
    return from_unit(u, bounds), best


def reference_optimum(
    problem: Problem,
    cfg: ReferenceCfg,
    cache_dir: Path | None = None,
    refresh: bool = False,
) -> dict:
    """Estimate the true cheapest feasible recipe: dense sampling, then local polish.

    Returns a dict with `x`, `cost`, `feasible_fraction` and run-cost statistics. The result is
    cached in `cache_dir/reference_<problem>.json`, keyed by the problem fingerprint so any
    change to the simulator, prices or emulator invalidates it.
    """
    key = hashlib.sha1(
        (problem.fingerprint() + json.dumps(cfg.model_dump(), sort_keys=True)).encode()
    ).hexdigest()[:16]
    suffix = getattr(problem, "reference_cache_suffix", "")
    name = f"reference_{problem.name}{suffix}.json"
    cache = None if cache_dir is None else Path(cache_dir) / name
    if cache is not None and cache.exists() and not refresh:
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if cached.get("key") == key:
            cached["x"] = np.array(cached["x"])
            return cached

    rng = np.random.default_rng(cfg.seed)
    X = sample_known_feasible(problem, cfg.n_samples, rng)
    truth = problem.true_outputs(X)
    feasible = passes_specs(truth, problem.specs)
    cost = np.where(feasible, problem.material_cost(X), np.inf)
    run_cost = problem.run_cost(X)
    if not feasible.any():
        raise RuntimeError(f"No truly feasible point among {len(X)} samples for {problem.name}")
    starts = np.argsort(cost)[: cfg.polish_starts]
    best_x, best_cost = X[starts[0]], float(cost[starts[0]])
    sample_best = best_cost
    for i in starts:
        if not np.isfinite(cost[i]):
            continue
        x, c = polish(problem, X[i], rng, cfg)
        if c < best_cost:
            best_x, best_cost = x, c
    result = {
        "key": key,
        "problem": problem.name,
        "x": best_x,
        "cost": best_cost,
        "sample_best_cost": sample_best,
        "n_samples": int(len(X)),
        "feasible_fraction": float(feasible.mean()),
        "run_cost_min": float(run_cost.min()),
        "run_cost_max": float(run_cost.max()),
        "run_cost_median": float(np.median(run_cost)),
    }
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(result, x=[float(v) for v in best_x])
        cache.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return result


def calibration_report(problem: Problem, ref: dict, near_limit: float = 0.02) -> dict:
    """The DOMAIN_PRIMER calibration checks for a problem, as a JSON-ready dict."""
    x = np.asarray(ref["x"], dtype=float)
    truth = {k: float(v[0]) for k, v in problem.true_outputs(x[None, :]).items()}
    distances = spec_relative_distances(truth, problem.specs)
    bounds = problem.bounds
    rel_pos = (x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    at_bound = {
        v.name: bool(min(p, 1 - p) < 0.01) for v, p in zip(problem.variables, rel_pos, strict=True)
    }
    ratio = ref["run_cost_max"] / ref["run_cost_min"]
    noise = problem.noise_summary() if hasattr(problem, "noise_summary") else {}
    return {
        "problem": problem.name,
        "feasible_fraction": ref["feasible_fraction"],
        "feasible_fraction_ok": 0.05 <= ref["feasible_fraction"] <= 0.35,
        "n_samples": ref["n_samples"],
        "reference_cost": ref["cost"],
        "reference_cost_before_polish": ref["sample_best_cost"],
        "reference_recipe": {v.name: float(xi) for v, xi in zip(problem.variables, x, strict=True)},
        "true_outputs_at_reference": truth,
        "spec_relative_distance_at_reference": distances,
        "active_specs_within_2pct": [k for k, d in distances.items() if d <= near_limit],
        "active_spec_ok": any(d <= near_limit for d in distances.values()),
        "variable_at_bound": at_bound,
        "optimum_at_corner_on_every_variable": all(at_bound.values()),
        "run_cost_min": ref["run_cost_min"],
        "run_cost_max": ref["run_cost_max"],
        "run_cost_median": ref["run_cost_median"],
        "run_cost_ratio_max_min": ratio,
        "run_cost_ratio_ok": ratio >= 3.0,
        "cost_cap_if_none_feasible": problem.cost_cap(),
        "noise": noise,
    }
