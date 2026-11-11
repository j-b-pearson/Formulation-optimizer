"""v2 H1 / H2 analysis on hand-made campaign metrics (offline, no benchmark runs)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from formopt.config import load_config
from formopt.heterogeneity import (
    analyze,
    growth_test,
    h1_verdict,
    h2_test,
    per_seed_advantage,
    starts_to_pay,
)

BUDGET = 1000.0


def metrics_frame(rows: dict[str, list[tuple[float, bool, float]]]) -> pd.DataFrame:
    """{method: [(spend_to_target, success, final_regret) per seed]} -> campaign metrics."""
    out = []
    for method, values in rows.items():
        for seed, (spend, success, regret) in enumerate(values, start=31):
            out.append(
                {
                    "method": method,
                    "seed": seed,
                    "spend_to_target": spend,
                    "success": success,
                    "final_regret": regret,
                    "n_experiments": 10,
                    "total_spend": 950.0,
                }
            )
    return pd.DataFrame(out)


def test_advantage_is_positive_when_cost_aware_is_better_and_censors_at_budget() -> None:
    m = metrics_frame(
        {
            "bo_plain": [(np.inf, False, 0.3), (800.0, True, 0.1), (500.0, True, 0.1)],
            "bo_cost_cooled": [(600.0, True, 0.1), (np.inf, False, 0.2), (500.0, True, 0.1)],
        }
    )
    adv = per_seed_advantage(m, "bo_cost_cooled", "bo_plain", BUDGET)
    assert adv["spend_saved"].tolist() == [400.0, -200.0, 0.0]  # 1000-600, 800-1000, 0
    assert adv["success_gain"].tolist() == [1, -1, 0]
    assert adv["regret_reduction"].round(6).tolist() == [0.2, -0.1, 0.0]


def test_growth_test_pairs_seeds_across_levels() -> None:
    low = pd.DataFrame({"seed": [1, 2, 3, 4], "spend_saved": [0.0, 0.0, 0.0, 0.0]})
    high = pd.DataFrame({"seed": [1, 2, 3, 4], "spend_saved": [100.0, 120.0, 80.0, 100.0]})
    res = growth_test(low, high, "spend_saved", n_boot=500, seed=0)
    assert res["growth"] == pytest.approx(100.0)
    assert 80.0 <= res["growth_ci_lo"] <= 100.0 <= res["growth_ci_hi"] <= 120.0
    with pytest.raises(ValueError):
        growth_test(low, high.assign(seed=[1, 2, 3, 5]), "spend_saved", 100, 0)


def growth_rows(spend_met: bool, success_met: bool) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"method": "m", "metric": "spend_saved", "criterion_met": spend_met},
            {"method": "m", "metric": "success_gain", "criterion_met": success_met},
            {"method": "m", "metric": "regret_reduction", "criterion_met": True},
        ]
    )


def test_h1_verdict_needs_both_co_primary_metrics() -> None:
    assert h1_verdict(growth_rows(True, True), "m") == "supported"
    assert h1_verdict(growth_rows(True, False), "m") == "partially supported"
    assert h1_verdict(growth_rows(False, False), "m") == "not supported"  # secondary ignored


def level_rows(spec: dict[float, tuple[float, float, float]]) -> pd.DataFrame:
    """{s: (spend ci_lo, spend mean, success mean)} with success CI never above zero."""
    rows = []
    for s, (spend_lo, spend_mean, succ_mean) in spec.items():
        rows.append(
            {
                "s": s,
                "method": "m",
                "metric": "spend_saved",
                "mean": spend_mean,
                "ci_above_zero": spend_lo > 0,
            }
        )
        rows.append(
            {
                "s": s,
                "method": "m",
                "metric": "success_gain",
                "mean": succ_mean,
                "ci_above_zero": False,
            }
        )
    return pd.DataFrame(rows)


def test_starts_to_pay_rule() -> None:
    assert (
        starts_to_pay(level_rows({1: (-1, 0, 0), 3: (10, 50, 0.01), 10: (20, 90, 0.1)}), "m") == 3
    )
    # A CI above zero on spend does not count if the success estimate is negative.
    assert (
        starts_to_pay(level_rows({1: (-1, 0, 0), 3: (10, 50, -0.02), 10: (20, 90, 0.0)}), "m") == 10
    )
    assert (
        starts_to_pay(level_rows({1: (-1, 0, 0), 3: (-5, 5, 0.0), 10: (-1, 9, 0.1)}), "m") is None
    )


def test_h2_counts_discordant_seeds() -> None:
    m = metrics_frame(
        {
            "bo_plain": [(np.inf, False, 0.3)] * 16 + [(500.0, True, 0.1)] * 4,
            "bo_cost": [(500.0, True, 0.1)] * 16 + [(np.inf, False, 0.3)] * 4,
        }
    )
    res = h2_test(m, "bo_cost", "bo_plain", BUDGET, n_boot=500, seed=0)
    assert res["paired_success_difference"] == pytest.approx(0.6)
    assert (res["seeds_only_method_succeeded"], res["seeds_only_baseline_succeeded"]) == (16, 4)
    assert res["success_rate_method"] == pytest.approx(0.8)
    assert 0.0 < res["mcnemar_exact_p_two_sided"] < 0.05
    assert res["replicated"] is True


def write_level(folder: Path, gain: float, n: int = 12) -> None:
    """A fake benchmark folder: cost-aware saves `gain` dollars on every seed."""
    rng = np.random.default_rng(0)
    rows = []
    for seed in range(31, 31 + n):
        base = float(rng.uniform(300, 900))
        for method, spend in (
            ("bo_plain", base),
            ("bo_cost", base - gain / 2),
            ("bo_cost_cooled", base - gain),
        ):
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "spend_to_target": spend,
                    "success": True,
                    "final_regret": 0.1,
                    "n_experiments": 10,
                    "total_spend": 950.0,
                }
            )
    folder.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(folder / "campaign_metrics_enamel.csv", index=False)
    info = {"budget": BUDGET, "seeds": list(range(31, 31 + n))}
    (folder / "benchmark_info_enamel.json").write_text(json.dumps(info), encoding="utf-8")


def test_analyze_end_to_end_on_fake_folders(tmp_path: Path) -> None:
    cfg = load_config()
    for s, gain in ((1, 0.0), (3, 40.0), (10, 120.0)):
        write_level(tmp_path / "sweep" / f"s_{s}", gain)
    write_level(tmp_path / "v1", 60.0)
    res = analyze(cfg, tmp_path / "sweep", tmp_path / "v1", tmp_path / "out", n_boot=300)
    growth = pd.read_csv(tmp_path / "out" / "h1_growth.csv")
    g = growth[(growth.method == "bo_cost_cooled") & (growth.metric == "spend_saved")].iloc[0]
    assert g["growth"] == pytest.approx(120.0) and bool(g["criterion_met"])
    # Success never differs here, so success growth is zero: H1 only partially supported.
    assert res["h1"]["bo_cost_cooled"]["verdict"] == "partially supported"
    assert res["h1"]["bo_cost_cooled"]["starts_to_pay_at_s"] == 3.0
    assert res["h2"]["paired_success_difference"] == 0.0 and res["h2"]["replicated"] is False
    for name in ("h1_per_level.csv", "h1_descriptive_by_level.csv", "h2.csv", "h1_h2_results.json"):
        assert (tmp_path / "out" / name).exists()
