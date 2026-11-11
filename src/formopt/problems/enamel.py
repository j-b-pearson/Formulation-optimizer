"""SYNTHETIC magnet wire enamel simulator (an illustrative model, not a real enamel).

Five inputs (PAI fraction of the resin blend, solids, catalyst, oven temperature, line speed),
three measured outputs (breakdown proxy, flexibility pass rate, viscosity), a known oven
limit, a known material-cost formula and a known run-cost formula. All coefficients come from
`config.yaml` (DOMAIN_PRIMER section 3). Directions and orders of magnitude are plausible;
the numbers are illustrative.

Cure is modelled as an Arrhenius rate times residence time (proportional to 1 / line speed),
so oven temperature and line speed interact: a hot oven with a fast line can match a cooler
oven with a slow line.

v2 adds an optional run-cost spread `s` (config `enamel.run_cost_spread.s`). With `s = None`
(the default) the v1 run-cost formula is used unchanged.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass

import numpy as np

from formopt.config import EnamelConfig
from formopt.design import from_unit
from formopt.problems.base import BaseProblem
from formopt.types import Spec, Variable

KELVIN = 273.15
VARIABLE_ORDER = ["pai_fraction", "solids_pct", "catalyst_phr", "oven_temp_c", "line_speed_m_min"]


@dataclass(frozen=True)
class RunCostScale:
    """Constants of the v2 run-cost family: run_cost = k * (1 + (s - 1) * u(x)).

    u(x) = (driver(x) - driver_min) / (driver_max - driver_min) is the v1 variable run cost
    rescaled to [0, 1] over the known-feasible box, so the cheapest possible run costs k and the
    most expensive k * s. k is chosen so the mean run cost over the box equals the v1 mean.
    """

    s: float
    driver_min: float  # exact: the driver is monotone in every input, so extremes are corners
    driver_max: float
    mean_u: float  # Monte Carlo mean of u over the known-feasible box
    v1_mean_run_cost: float  # Monte Carlo mean of the v1 run cost over the same samples
    v1_fixed: float
    k: float

    @property
    def run_cost_min(self) -> float:
        return self.k

    @property
    def run_cost_max(self) -> float:
        return self.k * self.s

    @property
    def v1_effective_s(self) -> float:
        """Max / min of the v1 run cost over the known-feasible box (exact, at corners)."""
        return (self.v1_fixed + self.driver_max) / (self.v1_fixed + self.driver_min)


class EnamelProblem(BaseProblem):
    """The synthetic enamel virtual lab."""

    name = "enamel"

    def __init__(self, cfg: EnamelConfig, prices: dict[str, float] | None = None) -> None:
        if list(cfg.variables) != VARIABLE_ORDER:
            raise ValueError(f"Enamel variables must be {VARIABLE_ORDER} in that order")
        self.cfg = cfg
        self.prices = {**cfg.prices, **(prices or {})}
        self.variables = [Variable(n, v.low, v.high, v.unit) for n, v in cfg.variables.items()]
        self.specs = [Spec(o, s.lower, s.upper, s.transform) for o, s in cfg.specs.items()]
        s = cfg.run_cost_spread.s
        self.run_cost_scale = None if s is None else self.compute_run_cost_scale(float(s))

    # ----- structure ---------------------------------------------------------------------
    def fingerprint(self) -> str:
        dump = self.cfg.model_dump()
        if self.cfg.run_cost_spread.s is None:
            # v1 configs had no spread block: dropping it keeps v1 reference-cache keys valid.
            dump.pop("run_cost_spread")
        payload = {"cfg": dump, "prices": self.prices}
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    @property
    def reference_cache_suffix(self) -> str:
        """Separate reference-cache file per spread level, so levels never overwrite v1's."""
        s = self.cfg.run_cost_spread.s
        return "" if s is None else f"_s{s:g}"

    def with_prices(self, prices: dict[str, float]) -> EnamelProblem:
        """Copy with some prices replaced (chemistry unchanged): used for price shocks."""
        unknown = set(prices) - set(self.prices)
        if unknown:
            raise KeyError(f"Unknown enamel price keys {sorted(unknown)}")
        return EnamelProblem(self.cfg, {**self.prices, **prices})

    def known_feasible(self, X: np.ndarray) -> np.ndarray:
        """Inside the box and oven at or below the equipment limit. Never learned."""
        X = np.atleast_2d(X)
        return self.in_bounds(X) & (X[:, 3] <= self.cfg.max_cure_temp_c + 1e-9)

    def known_feasible_box(self) -> np.ndarray:
        """The known-feasible region as a box: the variable box with the oven capped at the
        equipment limit. (Here the known constraint is a bound, so the region is a box.)"""
        box = self.bounds.copy()
        box[3, 1] = min(box[3, 1], self.cfg.max_cure_temp_c)
        return box

    def noise_summary(self) -> dict[str, float]:
        """Measurement noise settings, reported in the calibration JSON."""
        m = self.cfg.model
        return {
            "breakdown_kv_sd": m.sd_bd,
            "breakdown_sd_over_spec_limit": m.sd_bd / float(self.cfg.specs["breakdown_kv"].lower),
            "viscosity_log_sd": m.sd_logv,
            "flex_specimens": float(m.n_flex_specimens),
        }

    # ----- physics (noise-free) ----------------------------------------------------------
    def cure_dose(self, X: np.ndarray) -> np.ndarray:
        """Cure dose = Arrhenius rate x residence time. Higher dose means more cure."""
        X = np.atleast_2d(X)
        m = self.cfg.model
        cat, temp, speed = X[:, 2], X[:, 3], X[:, 4]
        rate = m.A * np.exp(-m.Ea_over_R / (temp + KELVIN)) * (1.0 + m.c_cat * cat)
        return rate * (m.L_ref / speed)

    def true_outputs(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Noise-free outputs. `flex_pass` is the per-specimen pass probability."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        m = self.cfg.model
        pai, solids, cat = X[:, 0], X[:, 1], X[:, 2]
        dose = self.cure_dose(X)
        cure = 1.0 - np.exp(-dose)
        over = np.maximum(dose - m.dose_max, 0.0)
        build = m.b0 + m.b1 * (solids - m.s_base)
        breakdown = m.k_bd * cure * build * (1.0 + m.k_pai * pai) - m.k_over * over
        logit = m.f0 - m.f_over * over - m.f_sol * (solids - m.s_ref) / m.f_sol_scale
        logit = logit + m.f_pai * pai
        flex = 1.0 / (1.0 + np.exp(-logit))
        log_visc = np.log(m.v0) + m.v_sol * (solids - m.s_base) + m.v_pai * pai + m.v_cat * cat
        return {"breakdown_kv": breakdown, "flex_pass": flex, "viscosity_cp": np.exp(log_visc)}

    # ----- the "lab" ---------------------------------------------------------------------
    def observe(self, x: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
        """One noisy lab measurement. Draw order is fixed so common random numbers line up."""
        m = self.cfg.model
        truth = {k: float(v[0]) for k, v in self.true_outputs(np.asarray(x)[None, :]).items()}
        breakdown = truth["breakdown_kv"] + rng.normal(0.0, m.sd_bd)
        flex = rng.binomial(m.n_flex_specimens, truth["flex_pass"]) / m.n_flex_specimens
        visc = float(np.exp(np.log(truth["viscosity_cp"]) + rng.normal(0.0, m.sd_logv)))
        return {"breakdown_kv": float(breakdown), "flex_pass": float(flex), "viscosity_cp": visc}

    # ----- costs (known formulas, never modelled) ----------------------------------------
    def material_cost(self, X: np.ndarray) -> np.ndarray:
        """$/kg of liquid enamel as purchased: resin blend + solvent + catalyst."""
        X = np.atleast_2d(X)
        p = self.prices
        pai, solids, cat = X[:, 0], X[:, 1] / 100.0, X[:, 2]
        resin = pai * p["pai_resin"] + (1.0 - pai) * p["pei_resin"]
        return solids * resin + (1.0 - solids) * p["solvent"] + cat / 100.0 * p["catalyst"]

    def run_cost(self, X: np.ndarray) -> np.ndarray:
        """$ per line trial. v1 formula by default; the v2 spread family when `s` is set."""
        if self.run_cost_scale is None:
            return self.v1_run_cost(X)
        sc = self.run_cost_scale
        u = (self.run_cost_driver(X) - sc.driver_min) / (sc.driver_max - sc.driver_min)
        return sc.k * (1.0 + (sc.s - 1.0) * u)

    def v1_run_cost(self, X: np.ndarray) -> np.ndarray:
        """$ per line trial: fixed + batch material + energy/time (hotter, slower costs more).

        This is the v1 formula, kept with the same operation order so v1 reproduces bit for bit.
        """
        X = np.atleast_2d(X)
        rc = self.cfg.run_cost
        temp, speed = X[:, 3], X[:, 4]
        energy = rc.energy_rate * (temp - rc.energy_temp_base) * (rc.energy_speed_ref / speed)
        return rc.fixed + rc.batch_kg * self.material_cost(X) + energy

    def run_cost_driver(self, X: np.ndarray) -> np.ndarray:
        """The part of the v1 run cost that varies between runs (no fixed fee).

        Batch material (a pricier resin blend, more solids or more catalyst cost more per trial)
        plus oven energy and time: (temperature above base) x (dwell time, proportional to
        1 / line speed). Higher cure temperature and longer dwell cost more.
        """
        X = np.atleast_2d(X)
        rc = self.cfg.run_cost
        temp, speed = X[:, 3], X[:, 4]
        energy = rc.energy_rate * (temp - rc.energy_temp_base) * (rc.energy_speed_ref / speed)
        return rc.batch_kg * self.material_cost(X) + energy

    def compute_run_cost_scale(self, s: float) -> RunCostScale:
        """Constants that give max/min run cost = s with the v1 mean run cost.

        Extremes: the driver is monotone in each input over the known-feasible box (material
        cost rises with PAI, solids and catalyst at these prices; energy rises with temperature
        and falls with line speed), so its minimum and maximum sit at corners of the box. All 32
        corners are evaluated. The mean uses `mean_samples` uniform points of the box drawn
        with `mean_seed` (a private random stream, never shared with a campaign).
        """
        if s < 1.0:
            raise ValueError(f"Run-cost spread s must be >= 1, got {s}")
        box = self.known_feasible_box()
        corners = np.array(list(itertools.product(*box)))
        drv = self.run_cost_driver(corners)
        d_min, d_max = float(drv.min()), float(drv.max())
        if not d_max > d_min:
            raise ValueError("Run-cost driver is constant over the box; spread is undefined")
        spread = self.cfg.run_cost_spread
        rng = np.random.default_rng(spread.mean_seed)
        X = from_unit(rng.random((spread.mean_samples, box.shape[0])), box)
        mean_u = float(np.mean((self.run_cost_driver(X) - d_min) / (d_max - d_min)))
        v1_mean = float(np.mean(self.v1_run_cost(X)))
        k = v1_mean / (1.0 + (s - 1.0) * mean_u)
        return RunCostScale(s, d_min, d_max, mean_u, v1_mean, float(self.cfg.run_cost.fixed), k)
