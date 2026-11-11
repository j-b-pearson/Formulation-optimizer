"""Experimental designs: random, Latin hypercube, full and fractional factorial, and helpers.

All designs work in natural units with `bounds` of shape (d, 2). Factorial levels sit at 20
and 80 percent of each range (not the extremes) because corners often break known
constraints; points that still break them are repaired to the nearest known-feasible point.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import qmc

if TYPE_CHECKING:
    from formopt.problems.base import Problem


def to_unit(X: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Scale natural-unit inputs to the unit cube."""
    bounds = np.asarray(bounds, dtype=float)
    return (np.asarray(X, dtype=float) - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


def from_unit(U: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Scale unit-cube inputs back to natural units."""
    bounds = np.asarray(bounds, dtype=float)
    return bounds[:, 0] + np.asarray(U, dtype=float) * (bounds[:, 1] - bounds[:, 0])


def random_design(bounds: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random points inside the box."""
    d = np.asarray(bounds).shape[0]
    return from_unit(rng.random((n, d)), bounds)


def lhs_design(bounds: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Latin hypercube: each variable's range is cut into n strata, one point per stratum."""
    d = np.asarray(bounds).shape[0]
    sampler = qmc.LatinHypercube(d=d, rng=rng)
    bounds = np.asarray(bounds, dtype=float)
    return qmc.scale(sampler.random(n), bounds[:, 0], bounds[:, 1])


def fractional_factorial_coded(generators: str) -> np.ndarray:
    """Two-level (fractional) factorial design in coded -1/+1 units.

    `generators` uses standard notation: single letters are base factors (full factorial in
    them) and multi-letter words are products of base columns. Example: "a b c d abcd" is the
    2^(5-1) design with E = ABCD (16 runs, resolution V).
    """
    terms = generators.split()
    base = [t for t in terms if len(t) == 1]
    if not base:
        raise ValueError("Generator string needs at least one single-letter base factor")
    full = np.array(list(itertools.product([-1, 1], repeat=len(base))), dtype=int)
    columns = []
    for term in terms:
        missing = [ch for ch in term if ch not in base]
        if missing:
            raise ValueError(f"Generator {term!r} uses unknown base factors {missing}")
        col = np.ones(len(full), dtype=int)
        for ch in term:
            col = col * full[:, base.index(ch)]
        columns.append(col)
    return np.column_stack(columns)


def fractional_factorial(
    bounds: np.ndarray,
    generators: str,
    levels: tuple[float, float] = (0.2, 0.8),
    n_center: int = 3,
) -> np.ndarray:
    """Fractional factorial at `levels` (fractions of each range) plus centre points."""
    bounds = np.asarray(bounds, dtype=float)
    coded = fractional_factorial_coded(generators)
    if coded.shape[1] != bounds.shape[0]:
        raise ValueError(f"Design has {coded.shape[1]} factors but bounds have {bounds.shape[0]}")
    low, high = levels
    unit = np.where(coded > 0, high, low).astype(float)
    centers = np.full((n_center, bounds.shape[0]), 0.5)
    return from_unit(np.vstack([unit, centers]), bounds)


def full_factorial(
    bounds: np.ndarray, levels_per_factor: int, span: tuple[float, float] = (0.0, 1.0)
) -> np.ndarray:
    """Full factorial grid with `levels_per_factor` evenly spaced levels inside `span`."""
    bounds = np.asarray(bounds, dtype=float)
    levels = np.linspace(span[0], span[1], levels_per_factor)
    grid = np.array(list(itertools.product(levels, repeat=bounds.shape[0])))
    return from_unit(grid, bounds)


def sample_known_feasible(
    problem: Problem, n: int, rng: np.random.Generator, max_tries: int = 50
) -> np.ndarray:
    """Draw n points that pass the problem's known constraints.

    Problems with a narrow feasible region (concrete, because of the applicability domain)
    provide their own `sample_candidates`; otherwise this does rejection sampling from the box.
    """
    sampler = getattr(problem, "sample_candidates", None)
    if callable(sampler):
        return sampler(n, rng)
    kept: list[np.ndarray] = []
    have, rate = 0, 1.0
    for _ in range(max_tries):
        need = n - have
        batch = int(np.ceil(need / max(rate, 0.01) * 1.2)) + 8
        X = random_design(problem.bounds, batch, rng)
        ok = problem.known_feasible(X)
        rate = max(float(ok.mean()), 1e-3)
        kept.append(X[ok])
        have += int(ok.sum())
        if have >= n:
            return np.vstack(kept)[:n]
    raise RuntimeError(f"Could only draw {have} of {n} known-feasible points")


def repair_to_feasible(
    X: np.ndarray, pool: np.ndarray, feasible: np.ndarray, bounds: np.ndarray
) -> np.ndarray:
    """Replace each infeasible row of X by the nearest unused point of a feasible `pool`.

    Distance is measured in unit-cube coordinates so every variable counts equally.
    """
    X = np.array(X, dtype=float, copy=True)
    feasible = np.asarray(feasible, dtype=bool)
    pool_u = to_unit(pool, bounds)
    used = np.zeros(len(pool), dtype=bool)
    for i in np.flatnonzero(~feasible):
        dist = np.sum((pool_u - to_unit(X[i], bounds)) ** 2, axis=1)
        dist[used] = np.inf
        j = int(np.argmin(dist))
        used[j] = True
        X[i] = pool[j]
    return X
