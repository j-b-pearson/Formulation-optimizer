"""Designs respect bounds; LHS is stratified; factorials are balanced; repair works."""

from __future__ import annotations

import numpy as np
import pytest

from formopt.config import load_config
from formopt.design import (
    fractional_factorial,
    fractional_factorial_coded,
    from_unit,
    full_factorial,
    lhs_design,
    random_design,
    repair_to_feasible,
    sample_known_feasible,
    to_unit,
)
from formopt.problems.enamel import EnamelProblem

BOUNDS_3D = np.array([[0.0, 1.0], [10.0, 20.0], [-5.0, 5.0]])
BOUNDS_5D = np.array([[0, 1], [25, 45], [0, 1], [420, 580], [10, 40]], dtype=float)


def _inside(X: np.ndarray, bounds: np.ndarray) -> bool:
    return bool(np.all((X >= bounds[:, 0]) & (X <= bounds[:, 1])))


@pytest.mark.parametrize("bounds", [BOUNDS_3D, BOUNDS_5D])
def test_designs_respect_bounds(bounds: np.ndarray) -> None:
    rng = np.random.default_rng(0)
    assert _inside(random_design(bounds, 50, rng), bounds)
    assert _inside(lhs_design(bounds, 50, rng), bounds)
    assert _inside(full_factorial(bounds, 3), bounds)


def test_unit_scaling_round_trip() -> None:
    X = random_design(BOUNDS_5D, 20, np.random.default_rng(1))
    np.testing.assert_allclose(from_unit(to_unit(X, BOUNDS_5D), BOUNDS_5D), X)


@pytest.mark.parametrize("n", [6, 25])
def test_lhs_one_point_per_stratum(n: int) -> None:
    X = lhs_design(BOUNDS_5D, n, np.random.default_rng(2))
    strata = np.floor(to_unit(X, BOUNDS_5D) * n).astype(int)
    for col in strata.T:
        assert sorted(col.tolist()) == list(range(n))


def test_fractional_factorial_2_5_1() -> None:
    coded = fractional_factorial_coded("a b c d abcd")
    assert coded.shape == (16, 5)
    assert np.all(coded.sum(axis=0) == 0)  # balanced: 8 low, 8 high per factor
    np.testing.assert_array_equal(coded.T @ coded, 16 * np.eye(5))  # orthogonal columns
    np.testing.assert_array_equal(coded[:, 4], np.prod(coded[:, :4], axis=1))  # E = ABCD
    X = fractional_factorial(BOUNDS_5D, "a b c d abcd", levels=(0.2, 0.8), n_center=3)
    assert X.shape == (19, 5)
    U = to_unit(X, BOUNDS_5D)
    np.testing.assert_allclose(np.unique(U[:16].round(10)), [0.2, 0.8])
    np.testing.assert_allclose(U[16:], 0.5)


def test_fractional_factorial_2_7_3() -> None:
    coded = fractional_factorial_coded("a b c d abc bcd acd")
    assert coded.shape == (16, 7)
    assert np.all(coded.sum(axis=0) == 0)
    np.testing.assert_array_equal(coded.T @ coded, 16 * np.eye(7))


def test_full_factorial_count() -> None:
    assert full_factorial(BOUNDS_3D, 3).shape == (27, 3)


def test_bad_generator_rejected() -> None:
    with pytest.raises(ValueError):
        fractional_factorial_coded("a b ae")


def test_repair_moves_only_infeasible_rows() -> None:
    rng = np.random.default_rng(3)
    X = random_design(BOUNDS_3D, 10, rng)
    feasible = np.array([True] * 7 + [False] * 3)
    pool = random_design(BOUNDS_3D, 100, rng)
    fixed = repair_to_feasible(X, pool, feasible, BOUNDS_3D)
    np.testing.assert_array_equal(fixed[:7], X[:7])
    for row in fixed[7:]:
        assert any(np.allclose(row, p) for p in pool)
    assert len({tuple(r) for r in fixed[7:]}) == 3  # no pool point reused


def test_sample_known_feasible_enamel() -> None:
    problem = EnamelProblem(load_config().enamel)
    X = sample_known_feasible(problem, 500, np.random.default_rng(4))
    assert X.shape == (500, 5)
    assert problem.known_feasible(X).all()
    assert X[:, 3].max() <= problem.cfg.max_cure_temp_c
