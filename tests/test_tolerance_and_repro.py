"""Success-vs-tolerance curves and the v1 reproducibility comparison (offline)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from formopt.repro import compare_dirs, compare_frames
from formopt.tolerance import gap_summary, success_curves

REF = 100.0


def long_frame() -> pd.DataFrame:
    """Two seeds of one method. Seed 0 finds 104 at step 1 and 101 at step 3; seed 1 only ever
    finds 112 (at step 2) and stops after 3 experiments (budget gone)."""
    rows = []
    for step, (cost, ok) in enumerate([(130, False), (104, True), (120, True), (101, True)]):
        rows.append(
            {"method": "bo", "seed": 0, "step": step, "material_cost": cost, "truly_feasible": ok}
        )
    for step, (cost, ok) in enumerate([(90, False), (95, False), (112, True)]):
        rows.append(
            {"method": "bo", "seed": 1, "step": step, "material_cost": cost, "truly_feasible": ok}
        )
    return pd.DataFrame(rows)


def test_success_curves_by_hand() -> None:
    c = success_curves(long_frame(), REF, [0.02, 0.05, 0.15]).set_index(
        ["tolerance", "n_experiments"]
    )
    assert c.loc[(0.05, 1), "n_success"] == 0  # infeasible first run
    assert c.loc[(0.05, 2), "n_success"] == 1  # seed 0: 104 <= 105
    assert c.loc[(0.02, 3), "n_success"] == 0  # 104 > 102 until step 4
    assert c.loc[(0.02, 4), "n_success"] == 1  # 101 <= 102
    assert c.loc[(0.15, 4), "n_success"] == 2  # seed 1 keeps its 112 after it stopped
    assert c.loc[(0.15, 4), "success_rate"] == pytest.approx(1.0)
    assert c["ci_lo"].between(0, 1).all() and c["ci_hi"].between(0, 1).all()


def test_gap_summary_by_hand() -> None:
    g = gap_summary(long_frame(), REF).iloc[0]
    assert g["min_gap_pct"] == pytest.approx(1.0) and g["max_gap_pct"] == pytest.approx(12.0)
    assert g["share_found_any_in_spec"] == 1.0


def test_compare_frames_ignores_only_wall_time() -> None:
    a = pd.DataFrame({"x": [1.0, 2.0], "wall_time_s": [0.1, 0.2]})
    same = pd.DataFrame({"x": [1.0, 2.0], "wall_time_s": [9.0, 9.0]})
    assert compare_frames(a, same)["identical"] is True
    diff = pd.DataFrame({"x": [1.0, 2.0 + 1e-12], "wall_time_s": [0.1, 0.2]})
    res = compare_frames(a, diff)
    assert res["identical"] is False and res["differing_columns"] == ["x"]
    assert res["max_abs_diff"]["x"] == pytest.approx(1e-12, rel=0.01)


def test_compare_dirs(tmp_path: Path) -> None:
    for name in ("ref", "new"):
        d = tmp_path / name
        d.mkdir()
        frame = pd.DataFrame({"method": ["a"], "v": [np.pi], "wall_time_s": [len(name)]})
        for stem in ("summary", "paired", "campaign_metrics", "benchmark"):
            frame.to_csv(d / f"{stem}_toy.csv", index=False)
        info = {"seeds": [0, 1], "wall_time_s": len(name), "n_jobs": len(name)}
        (d / "benchmark_info_toy.json").write_text(json.dumps(info), encoding="utf-8")
    assert compare_dirs(tmp_path / "ref", tmp_path / "new", ["toy"])["all_identical"] is True
    (tmp_path / "new" / "benchmark_info_toy.json").write_text('{"seeds": [0, 2]}', "utf-8")
    report = compare_dirs(tmp_path / "ref", tmp_path / "new", ["toy"])
    assert report["all_identical"] is False
    assert report["problems"]["toy"]["benchmark_info"]["differing_keys"] == ["seeds"]


def test_compare_dirs_fails_when_a_file_is_missing(tmp_path: Path) -> None:
    """A table missing on one side (e.g. the git-ignored benchmark CSV in a fresh clone) was
    not compared, so the check must not report the folders as identical."""
    for name in ("ref", "new"):
        d = tmp_path / name
        d.mkdir()
        frame = pd.DataFrame({"method": ["a"], "v": [np.pi]})
        for stem in ("summary", "paired", "campaign_metrics", "benchmark"):
            frame.to_csv(d / f"{stem}_toy.csv", index=False)
        (d / "benchmark_info_toy.json").write_text('{"seeds": [0, 1]}', encoding="utf-8")
    (tmp_path / "ref" / "benchmark_toy.csv").unlink()
    report = compare_dirs(tmp_path / "ref", tmp_path / "new", ["toy"])
    assert report["all_identical"] is False
    assert report["problems"]["toy"]["benchmark"]["identical"] is None
