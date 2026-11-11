"""Typed configuration for the project.

`config.yaml` holds every number that a user might want to change (prices, specs, budgets,
simulator coefficients, seeds). This module validates it with Pydantic so a typo fails loudly
at load time instead of silently changing a benchmark.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class _Strict(BaseModel):
    """Base model that rejects unknown keys, so config typos are caught."""

    model_config = ConfigDict(extra="forbid")


class VariableCfg(_Strict):
    low: float
    high: float
    unit: str


class SpecCfg(_Strict):
    lower: float | None = None
    upper: float | None = None
    transform: Literal["none", "log"] = "none"


class EnamelModelCfg(_Strict):
    A: float
    Ea_over_R: float
    c_cat: float
    L_ref: float
    dose_max: float
    s_base: float
    b0: float
    b1: float
    k_bd: float
    k_pai: float
    k_over: float
    sd_bd: float
    f0: float
    f_over: float
    f_sol: float
    f_sol_scale: float
    s_ref: float
    f_pai: float
    n_flex_specimens: int
    v0: float
    v_sol: float
    v_pai: float
    v_cat: float
    sd_logv: float


class EnamelRunCostCfg(_Strict):
    fixed: float
    batch_kg: float
    energy_rate: float
    energy_temp_base: float
    energy_speed_ref: float


class RunCostSpreadCfg(_Strict):
    """Run-cost spread (v2 study). `s = None` keeps the v1 run-cost formula.

    With a number, run cost becomes k_s * (1 + (s - 1) * u(x)), where u(x) in [0, 1] is the v1
    variable cost (batch material plus oven energy and time) rescaled between its minimum and
    maximum over the known-feasible box, so the most expensive run costs s times the cheapest.
    k_s keeps the mean run cost over the known-feasible box at its v1 value.
    """

    s: float | None = Field(default=None, ge=1.0)
    mean_samples: int = Field(gt=0)  # uniform known-feasible samples for the v1 mean run cost
    mean_seed: int


class EnamelConfig(_Strict):
    variables: dict[str, VariableCfg]
    max_cure_temp_c: float
    specs: dict[str, SpecCfg]
    model: EnamelModelCfg
    prices: dict[str, float]
    run_cost: EnamelRunCostCfg
    run_cost_spread: RunCostSpreadCfg
    budget: float


class ConcreteRunCostCfg(_Strict):
    fixed: float
    batch_m3: float


class DomainCfg(_Strict):
    k: int = 5
    quantile: float = 0.95


class EmulatorCfg(_Strict):
    candidates: list[Literal["extra_trees", "hist_gb"]]
    n_estimators: int
    min_samples_leaf: int
    cv_folds: int
    seed: int


class ConcreteConfig(_Strict):
    age_days: float
    target_strength_mpa: float
    prices_per_kg: dict[str, float]
    run_cost: ConcreteRunCostCfg
    noise_sd: float | None
    noise_sd_cap: float
    domain: DomainCfg
    emulator: EmulatorCfg
    mix_id_round: float
    budget: float


class BOConfig(_Strict):
    n_init: int
    n_candidates: int
    quick_n_candidates: int
    gp_restarts: int
    gp_noise_bounds: tuple[float, float]
    gp_length_scale_bounds: tuple[float, float]
    gp_constant_bounds: tuple[float, float]


class FactorialCfg(_Strict):
    enamel_generators: str
    concrete_generators: str
    levels: tuple[float, float]
    n_center: int


class ReferenceCfg(_Strict):
    n_samples: int
    seed: int
    polish_starts: int
    polish_batch: int
    polish_radii: list[float]
    polish_max_moves: int


class BenchmarkConfig(_Strict):
    methods: list[str]
    seed_start: int
    n_seeds: int
    dev_seed_start: int
    n_dev_seeds: int
    quick_seeds: int
    target_tol: float
    grid_points: int
    n_boot: int
    bootstrap_seed: int
    headline_cost_method: str
    n_jobs: int
    random_batch: int
    factorial: FactorialCfg
    reference: ReferenceCfg


class PriceShockCfg(_Strict):
    min_pof: float
    n_candidates: int
    seed: int
    enamel_changes: dict[str, float]
    concrete_changes: dict[str, float]


class RiskThresholdsCfg(_Strict):
    """Risk label from the probability of passing all specs (PoF): below `high_below` is High,
    at or above `low_at_or_above` is Low, anything in between is Medium."""

    high_below: float = Field(ge=0.0, le=1.0)
    low_at_or_above: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> RiskThresholdsCfg:
        if self.high_below > self.low_at_or_above:
            raise ValueError("lab_note.risk_thresholds: high_below must be <= low_at_or_above")
        return self


class LabNoteCfg(_Strict):
    risk_thresholds: RiskThresholdsCfg
    max_retries: int = Field(ge=0)


class BatchCfg(_Strict):
    """Batch mode (exploratory): q runs per oven cycle with the kriging-believer heuristic."""

    q: int = Field(ge=1)
    methods: list[str]


class V2Cfg(_Strict):
    """v2 pre-registered settings (docs/PREREGISTRATION_v2.md). Change only with a logged
    deviation."""

    reporting_seed_start: int
    n_reporting_seeds: int
    fallback_n_reporting_seeds: int  # used only if the runtime estimate exceeds the limit
    dev_seed_start: int
    n_dev_seeds: int
    runtime_limit_hours: float
    timing_n_jobs: int
    spread_levels: list[float]
    sweep_methods: list[str]
    plain_method: str
    h1_headline_method: str
    h1_secondary_method: str
    h2_method: str
    n_boot: int
    bootstrap_seed: int
    robustness_n_boot: int
    robustness_seed: int
    tolerances: list[float]
    batch: BatchCfg


class PathsCfg(_Strict):
    raw_dir: Path
    processed_dir: Path
    reports_dir: Path
    figures_dir: Path
    v2_dir: Path

    def resolved(self, root: Path) -> PathsCfg:
        """Return a copy with every relative path anchored at `root`."""
        values = {
            name: (p if p.is_absolute() else root / p) for name, p in self.model_dump().items()
        }
        return PathsCfg(**values)


class AppConfig(_Strict):
    enamel: EnamelConfig
    concrete: ConcreteConfig
    bo: BOConfig
    benchmark: BenchmarkConfig
    price_shock: PriceShockCfg
    lab_note: LabNoteCfg
    v2: V2Cfg
    paths: PathsCfg
    source_path: Path | None = Field(default=None, exclude=True)

    @property
    def root(self) -> Path:
        """Folder that relative paths in the config are anchored to."""
        return self.source_path.parent if self.source_path else PROJECT_ROOT

    @property
    def abs_paths(self) -> PathsCfg:
        return self.paths.resolved(self.root)


class Settings(BaseSettings):
    """Environment variables (optionally from `.env`)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    formopt_config: str | None = None


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load and validate `config.yaml`.

    Order of precedence: explicit `path`, then the `FORMOPT_CONFIG` env var, then the
    `config.yaml` at the project root.
    """
    if path is None:
        env_path = Settings().formopt_config
        path = Path(env_path) if env_path else DEFAULT_CONFIG_PATH
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = AppConfig(**raw)
    cfg.source_path = path
    return cfg
