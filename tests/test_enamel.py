"""Synthetic enamel simulator: behaviour a domain expert would check."""

from __future__ import annotations

import numpy as np
import pytest

from formopt.config import load_config
from formopt.problems.base import calibration_report, passes_specs, reference_optimum
from formopt.problems.enamel import EnamelProblem


@pytest.fixture(scope="module")
def problem() -> EnamelProblem:
    return EnamelProblem(load_config().enamel)


def _point(pai=0.3, solids=38.0, cat=0.3, temp=480.0, speed=25.0) -> np.ndarray:
    return np.array([pai, solids, cat, temp, speed], dtype=float)


def test_output_keys_and_shapes(problem: EnamelProblem) -> None:
    X = np.vstack([_point(), _point(temp=520.0), _point(speed=12.0)])
    out = problem.true_outputs(X)
    assert set(out) == {"breakdown_kv", "flex_pass", "viscosity_cp"}
    assert all(v.shape == (3,) for v in out.values())
    noisy = problem.observe(_point(), np.random.default_rng(0))
    assert set(noisy) == set(out) and all(isinstance(v, float) for v in noisy.values())
    assert problem.material_cost(X).shape == (3,) and problem.run_cost(X).shape == (3,)


def test_breakdown_rises_with_cure_when_undercured(problem: EnamelProblem) -> None:
    temps = np.linspace(420.0, 470.0, 6)
    X = np.array([_point(temp=t, speed=40.0) for t in temps])
    dose = problem.cure_dose(X)
    assert np.all(dose < problem.cfg.model.dose_max)
    assert np.all(np.diff(problem.true_outputs(X)["breakdown_kv"]) > 0)


def test_flexibility_drops_in_strong_overcure(problem: EnamelProblem) -> None:
    mild = _point(temp=470.0, speed=30.0, cat=0.0)
    strong = _point(temp=560.0, speed=10.0, cat=1.0)
    dose = problem.cure_dose(np.vstack([mild, strong]))
    assert dose[0] < problem.cfg.model.dose_max < dose[1]
    flex = problem.true_outputs(np.vstack([mild, strong]))["flex_pass"]
    assert flex[1] < 0.2 < 0.8 < flex[0]


def test_temperature_and_line_speed_interact(problem: EnamelProblem) -> None:
    """A hot oven with a fast line can reach the same cure dose as a cool oven, slow line."""
    cool_slow = _point(temp=460.0, speed=10.0)
    dose_cool = problem.cure_dose(cool_slow[None, :])[0]
    hot = _point(temp=540.0, speed=10.0)
    speed_needed = problem.cure_dose(hot[None, :])[0] / dose_cool * 10.0
    hot_fast = _point(temp=540.0, speed=speed_needed)
    assert problem.cure_dose(hot_fast[None, :])[0] == pytest.approx(dose_cool, rel=1e-9)
    assert speed_needed > 20.0


def test_viscosity_increases_with_solids(problem: EnamelProblem) -> None:
    X = np.array([_point(solids=s) for s in np.linspace(25.0, 45.0, 21)])
    assert np.all(np.diff(problem.true_outputs(X)["viscosity_cp"]) > 0)


def test_noise_matches_config(problem: EnamelProblem) -> None:
    x = _point()
    m = problem.cfg.model
    truth = {k: float(v[0]) for k, v in problem.true_outputs(x[None, :]).items()}
    draws = [problem.observe(x, np.random.default_rng([7, i])) for i in range(2000)]
    bd = np.array([d["breakdown_kv"] for d in draws])
    logv = np.log([d["viscosity_cp"] for d in draws])
    flex = np.array([d["flex_pass"] for d in draws])
    assert bd.std(ddof=1) == pytest.approx(m.sd_bd, rel=0.2)
    assert logv.std(ddof=1) == pytest.approx(m.sd_logv, rel=0.2)
    p = truth["flex_pass"]
    assert flex.std(ddof=1) == pytest.approx(np.sqrt(p * (1 - p) / m.n_flex_specimens), rel=0.2)
    assert set(np.unique(flex)) <= {k / m.n_flex_specimens for k in range(m.n_flex_specimens + 1)}


def test_known_constraint_rejects_hot_oven(problem: EnamelProblem) -> None:
    limit = problem.cfg.max_cure_temp_c
    X = np.vstack([_point(temp=limit - 1), _point(temp=limit), _point(temp=limit + 1)])
    assert problem.known_feasible(X).tolist() == [True, True, False]
    outside = _point(solids=50.0)
    assert not problem.known_feasible(outside[None, :])[0]


def test_cost_formulas_by_hand(problem: EnamelProblem) -> None:
    x = _point(pai=0.5, solids=40.0, cat=0.5, temp=500.0, speed=20.0)
    p = problem.prices
    resin = 0.5 * p["pai_resin"] + 0.5 * p["pei_resin"]
    material = 0.40 * resin + 0.60 * p["solvent"] + 0.005 * p["catalyst"]
    assert problem.material_cost(x[None, :])[0] == pytest.approx(material)
    rc = problem.cfg.run_cost
    energy = rc.energy_rate * (500.0 - rc.energy_temp_base) * (rc.energy_speed_ref / 20.0)
    run = rc.fixed + rc.batch_kg * material + energy
    assert problem.run_cost(x[None, :])[0] == pytest.approx(run)


def test_price_change_moves_cost_not_chemistry(problem: EnamelProblem) -> None:
    x = _point()[None, :]
    shocked = problem.with_prices({"pai_resin": problem.prices["pai_resin"] * 1.25})
    assert shocked.material_cost(x)[0] > problem.material_cost(x)[0]
    for key, value in problem.true_outputs(x).items():
        np.testing.assert_allclose(shocked.true_outputs(x)[key], value)
    with pytest.raises(KeyError):
        problem.with_prices({"unobtainium": 1.0})


def test_calibration_rules_hold(problem: EnamelProblem) -> None:
    cfg = load_config().benchmark.reference.model_copy(update={"n_samples": 20000})
    ref = reference_optimum(problem, cfg)
    report = calibration_report(problem, ref)
    assert report["feasible_fraction_ok"]
    assert report["run_cost_ratio_ok"]
    assert report["active_spec_ok"]
    assert not report["optimum_at_corner_on_every_variable"]
    truth = problem.true_outputs(np.asarray(ref["x"])[None, :])
    assert passes_specs(truth, problem.specs)[0]
