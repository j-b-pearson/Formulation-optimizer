"""The enamel run-cost spread family (offline, no benchmark runs)."""

from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np
import pytest
from pydantic import ValidationError

from formopt.config import load_config
from formopt.heterogeneity import with_run_cost_spread
from formopt.problems.enamel import EnamelProblem


def enamel(s: float | None) -> EnamelProblem:
    return EnamelProblem(with_run_cost_spread(load_config(), s).enamel)


def uniform_known_feasible(problem: EnamelProblem, n: int, seed: int) -> np.ndarray:
    box = problem.known_feasible_box()
    rng = np.random.default_rng(seed)
    return box[:, 0] + rng.random((n, box.shape[0])) * (box[:, 1] - box[:, 0])


def test_spread_benchmark_never_writes_into_reports_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    """`benchmark --run-cost-spread` without --out-dir must default to reports/v2, not to the
    frozen v1 folder, and must not log into the v1 MLflow experiment."""
    from formopt import benchmark, cli

    seen: dict = {}

    def fake_run_benchmark(cfg, problems, methods, seeds, **kwargs):
        seen.update(kwargs, s=cfg.enamel.run_cost_spread.s)
        return {}

    monkeypatch.setattr(benchmark, "run_benchmark", fake_run_benchmark)
    cfg = load_config()
    args = cli.build_parser().parse_args(["benchmark", "--run-cost-spread", "10"])
    cli.cmd_benchmark(cfg, args)
    assert seen["s"] == 10.0
    assert seen["out_dir"] == cfg.abs_paths.v2_dir / "s_10"
    assert seen["out_dir"] != cfg.abs_paths.reports_dir
    assert seen["experiment_suffix"] == "-v2-s_10"


def test_default_is_the_v1_formula_and_v1_fingerprint() -> None:
    p = enamel(None)
    assert p.run_cost_scale is None
    X = uniform_known_feasible(p, 500, 0)
    rc = p.cfg.run_cost
    energy = rc.energy_rate * (X[:, 3] - rc.energy_temp_base) * (rc.energy_speed_ref / X[:, 4])
    by_hand = rc.fixed + rc.batch_kg * p.material_cost(X) + energy
    assert np.array_equal(p.run_cost(X), by_hand)  # bit for bit, same operation order
    dump = p.cfg.model_dump()
    dump.pop("run_cost_spread")
    v1_payload = json.dumps({"cfg": dump, "prices": p.prices}, sort_keys=True).encode()
    assert p.fingerprint() == hashlib.sha1(v1_payload).hexdigest()
    assert p.reference_cache_suffix == ""


@pytest.mark.parametrize("s", [1.0, 3.0, 10.0])
def test_max_over_min_is_s_and_mean_is_v1(s: float) -> None:
    p = enamel(s)
    sc = p.run_cost_scale
    corners = np.array(list(itertools.product(*p.known_feasible_box())))
    rc_corners = p.run_cost(corners)
    assert rc_corners.max() / rc_corners.min() == pytest.approx(s, rel=1e-12)
    # Dense independent samples never fall outside the corner extremes (monotone driver).
    rc = p.run_cost(uniform_known_feasible(p, 200_000, 99))
    assert rc.min() >= sc.run_cost_min * (1 - 1e-12)
    assert rc.max() <= sc.run_cost_max * (1 + 1e-12)
    # Mean run cost stays at the v1 value (independent sample: Monte Carlo error only).
    v1 = enamel(None).run_cost(uniform_known_feasible(p, 200_000, 99))
    assert rc.mean() == pytest.approx(v1.mean(), rel=0.01)


def test_spread_one_is_constant_and_v1_is_a_member_of_the_family() -> None:
    X = uniform_known_feasible(enamel(None), 1000, 1)
    flat = enamel(1.0).run_cost(X)
    assert np.allclose(flat, flat[0])
    s_v1 = enamel(1.0).run_cost_scale.v1_effective_s
    assert s_v1 == pytest.approx(1897.6 / 299.0)  # (150 + 1747.6) / (150 + 149), by hand
    assert np.allclose(enamel(s_v1).run_cost(X), enamel(None).run_cost(X), rtol=1e-9)


def test_drivers_move_run_cost_in_the_documented_direction() -> None:
    p = enamel(10.0)
    x = np.array([[0.3, 35.0, 0.5, 480.0, 25.0]])
    base = p.run_cost(x)[0]
    for j, step, direction in [(0, 0.2, 1), (1, 5.0, 1), (2, 0.2, 1), (3, 40.0, 1), (4, 5.0, -1)]:
        y = x.copy()
        y[0, j] += step
        assert np.sign(p.run_cost(y)[0] - base) == direction


def test_spread_below_one_is_rejected_and_levels_get_their_own_cache() -> None:
    with pytest.raises(ValidationError):
        with_run_cost_spread(load_config(), 0.5)
    assert enamel(3.0).reference_cache_suffix == "_s3"
    assert enamel(3.0).fingerprint() != enamel(None).fingerprint()


def test_at_spread_one_cost_aware_and_plain_bo_choose_the_same_runs() -> None:
    from formopt.benchmark import make_strategy
    from formopt.loop import run_campaign

    cfg = with_run_cost_spread(load_config(), 1.0)
    problem = EnamelProblem(cfg.enamel)
    xs = {}
    for method in ("bo_plain", "bo_cost", "bo_cost_cooled"):
        strategy = make_strategy(method, cfg, "enamel", quick=True)
        strategy.n_candidates = 300
        result = run_campaign(problem, strategy, 6000.0, seed=1001)
        xs[method] = result.X()
    assert xs["bo_plain"].shape[0] > cfg.bo.n_init  # at least one BO step happened
    assert np.array_equal(xs["bo_plain"], xs["bo_cost"])
    assert np.array_equal(xs["bo_plain"], xs["bo_cost_cooled"])
