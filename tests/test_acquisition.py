"""Acquisition function properties and the spend estimate formula."""

from __future__ import annotations

import numpy as np
import pytest

from formopt.acquisition import (
    constrained_cost_acquisition,
    cooled_alpha,
    cost_aware,
    expected_improvement,
    known_cost_improvement,
    probability_of_feasibility,
    probability_of_feasibility_per_spec,
    probability_of_improvement,
    upper_confidence_bound,
)
from formopt.estimate import expected_spend_formula
from formopt.types import Spec

MU = np.array([-2.0, -0.5, 0.0, 0.3, 1.0, 4.0])


def test_ei_equals_plain_improvement_when_sigma_tiny() -> None:
    ei = expected_improvement(MU, np.full_like(MU, 1e-15), best=0.2)
    np.testing.assert_allclose(ei, np.maximum(MU - 0.2, 0.0), atol=1e-12)


def test_ei_non_negative() -> None:
    rng = np.random.default_rng(0)
    ei = expected_improvement(rng.normal(size=500), rng.uniform(0.0, 3.0, 500), best=0.5)
    assert np.all(ei >= 0)


def test_ei_increases_with_sigma_at_fixed_mu() -> None:
    sigmas = np.linspace(0.01, 3.0, 50)
    ei = expected_improvement(np.zeros(50), sigmas, best=0.5)
    assert np.all(np.diff(ei) > 0)


def test_ei_worked_example() -> None:
    ei = expected_improvement(np.array([5.0, 4.5]), np.array([2.0, 0.2]), best=4.0)
    np.testing.assert_allclose(ei, [1.40, 0.50], atol=0.01)


def test_pi_in_unit_interval_and_ucb() -> None:
    pi = probability_of_improvement(MU, np.full_like(MU, 0.7), best=0.0)
    assert np.all((pi >= 0) & (pi <= 1))
    assert np.all(np.diff(pi) >= 0)
    np.testing.assert_allclose(upper_confidence_bound(MU, np.ones_like(MU), beta=0.0), MU)


def test_pof_far_inside_limits_is_one_and_window_works() -> None:
    specs = [Spec("a", lower=30.0), Spec("v", lower=2000.0, upper=15000.0, transform="log")]
    preds = {
        "a": (np.array([60.0, 30.0]), np.array([1.0, 1.0])),
        "v": (np.log(np.array([5500.0, 5500.0])), np.array([0.01, 0.01])),
    }
    pof = probability_of_feasibility(preds, specs)
    assert pof[0] == pytest.approx(1.0, abs=1e-9)
    assert pof[1] == pytest.approx(0.5, abs=1e-6)  # mean exactly on the lower limit
    per_spec = probability_of_feasibility_per_spec(preds, specs)
    np.testing.assert_allclose(per_spec["a"] * per_spec["v"], pof)


def test_pof_worked_example() -> None:
    specs = [Spec("bd", lower=30.0), Spec("flex", lower=0.8)]
    preds = {"bd": (np.array([32.0]), np.array([2.0])), "flex": (np.array([0.85]), np.array([0.1]))}
    assert probability_of_feasibility(preds, specs)[0] == pytest.approx(0.841 * 0.691, abs=2e-3)


def test_known_cost_improvement_and_constrained_acquisition() -> None:
    cost = np.array([3.0, 2.5, 2.0])
    pof = np.array([0.9, 0.5, 0.1])
    np.testing.assert_allclose(known_cost_improvement(cost, None), 1.0)
    np.testing.assert_allclose(known_cost_improvement(cost, 2.8), [0.0, 0.3, 0.8])
    np.testing.assert_allclose(constrained_cost_acquisition(cost, pof, None), pof)
    np.testing.assert_allclose(constrained_cost_acquisition(cost, pof, 2.8), [0.0, 0.15, 0.08])


def test_cost_aware_alpha_zero_is_base() -> None:
    acq = np.array([0.1, 0.5, 2.0])
    run = np.array([300.0, 1200.0, 50.0])
    np.testing.assert_allclose(cost_aware(acq, run, 0.0), acq)
    np.testing.assert_allclose(cost_aware(acq, run, 1.0), acq / run)


def test_cost_aware_worked_example() -> None:
    acq, run = np.array([1.40, 0.50]), np.array([1200.0, 300.0])
    assert np.argmax(cost_aware(acq, run, 1.0)) == 1  # per dollar, the cheap run wins
    assert np.argmax(cost_aware(acq, run, 0.1)) == 0  # late in a cooled campaign, A wins


def test_cooled_alpha_schedule() -> None:
    assert cooled_alpha(0.0, 15000.0) == 1.0
    assert cooled_alpha(15000.0, 15000.0) == 0.0
    assert cooled_alpha(7500.0, 15000.0) == pytest.approx(0.5)
    assert cooled_alpha(20000.0, 15000.0) == 0.0


def test_expected_spend_formula_by_hand() -> None:
    # 100 + (1 - 0.5) * 200, plus tail 0.25 * mean cost 150 / mean p 0.5 = 75.
    assert expected_spend_formula(np.array([0.5, 0.5]), np.array([100.0, 200.0])) == 275.0
    assert expected_spend_formula(np.array([1.0, 0.3]), np.array([100.0, 200.0])) == 100.0


def test_expected_spend_finite_iff_some_pof_positive() -> None:
    assert np.isfinite(expected_spend_formula(np.array([0.0, 0.0]), np.ones(2), tail_pof=0.01))
    assert expected_spend_formula(np.array([0.0, 0.0]), np.ones(2)) == float("inf")
