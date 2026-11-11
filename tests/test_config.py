"""The config file loads, validates and keeps its key business numbers."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from formopt.config import AppConfig, load_config


def test_config_loads_and_validates() -> None:
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert set(cfg.enamel.variables) == {
        "pai_fraction",
        "solids_pct",
        "catalyst_phr",
        "oven_temp_c",
        "line_speed_m_min",
    }
    assert cfg.enamel.specs["viscosity_cp"].transform == "log"
    assert cfg.enamel.budget > 0 and cfg.concrete.budget > 0
    assert cfg.benchmark.methods[0] == "random"


def test_paths_resolve_relative_to_config_file() -> None:
    cfg = load_config()
    assert cfg.abs_paths.reports_dir.is_absolute()
    # v2 moved the v1 outputs to reports/v1; every relative path is anchored at the config root.
    assert cfg.abs_paths.reports_dir == cfg.root / cfg.paths.reports_dir
    assert cfg.abs_paths.v2_dir == cfg.root / cfg.paths.v2_dir


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    text = load_config().source_path.read_text(encoding="utf-8")
    bad = tmp_path / "config.yaml"
    bad.write_text(text.replace("max_cure_temp_c:", "max_cure_tmp_c:"), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(bad)


def test_v2_settings_match_the_preregistration() -> None:
    """The pre-registered v2 choices (docs/PREREGISTRATION_v2.md) are what the config says."""
    cfg = load_config()
    v2 = cfg.v2
    seeds = range(v2.reporting_seed_start, v2.reporting_seed_start + v2.n_reporting_seeds)
    assert (seeds[0], seeds[-1]) == (31, 130)
    assert (v2.dev_seed_start, v2.dev_seed_start + v2.n_dev_seeds - 1) == (1001, 1010)
    assert v2.spread_levels == [1.0, 3.0, 10.0]
    assert v2.h1_headline_method == "bo_cost_cooled" and v2.h2_method == "bo_cost"
    assert cfg.enamel.run_cost_spread.s is None  # the default is the v1 run-cost formula
    th = cfg.lab_note.risk_thresholds
    assert (th.high_below, th.low_at_or_above) == (0.5, 0.8)


def test_risk_thresholds_must_be_ordered(tmp_path: Path) -> None:
    text = load_config().source_path.read_text(encoding="utf-8")
    bad = tmp_path / "config.yaml"
    bad.write_text(text.replace("high_below: 0.5", "high_below: 0.9"), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(bad)
