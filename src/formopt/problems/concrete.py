"""Concrete virtual lab backed by the emulator trained on the real UCI data.

Inputs are the seven ingredients in kg per m3; age is fixed at 28 days (the specification
age). Known constraints keep the optimizer on physically meaningful mixes: every ingredient
inside its observed range, total mass and water-to-binder ratio inside their observed ranges,
and the applicability domain (close to real training mixes). The only unknown constraint is
28-day strength.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from formopt.config import AppConfig, ConcreteConfig
from formopt.data import INGREDIENTS
from formopt.emulator import ConcreteEmulator, water_binder
from formopt.problems.base import BaseProblem
from formopt.types import Spec, Variable

EMULATOR_FILE = "concrete_emulator.joblib"


class ConcreteProblem(BaseProblem):
    """Concrete mix design: cheapest $/m3 recipe whose 28-day strength meets the target."""

    name = "concrete"

    def __init__(
        self,
        emulator: ConcreteEmulator,
        cfg: ConcreteConfig,
        prices: dict[str, float] | None = None,
    ) -> None:
        self.emulator = emulator
        self.cfg = cfg
        self.prices = {**cfg.prices_per_kg, **(prices or {})}
        missing = set(INGREDIENTS) - set(self.prices)
        if missing:
            raise KeyError(f"Missing concrete prices for {sorted(missing)}")
        self.variables = [
            Variable(n, float(lo), float(hi), "kg/m3")
            for n, (lo, hi) in zip(INGREDIENTS, emulator.ingredient_bounds, strict=True)
        ]
        self.specs = [Spec("strength_mpa", lower=cfg.target_strength_mpa)]
        cap = cfg.noise_sd_cap
        self.noise_sd = (
            float(cfg.noise_sd)
            if cfg.noise_sd is not None
            else float(min(emulator.cv_report["residual_sd"], cap))
        )
        self._price_vec = np.array([self.prices[n] for n in INGREDIENTS])

    def fingerprint(self) -> str:
        emu = self.emulator
        payload = {
            "cfg": self.cfg.model_dump(),
            "prices": self.prices,
            "emulator": [emu.kind, emu.cv_report["rmse"], emu.domain.threshold],
            "bounds": emu.ingredient_bounds.tolist(),
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def with_prices(self, prices: dict[str, float]) -> ConcreteProblem:
        unknown = set(prices) - set(self.prices)
        if unknown:
            raise KeyError(f"Unknown concrete price keys {sorted(unknown)}")
        return ConcreteProblem(self.emulator, self.cfg, {**self.prices, **prices})

    def known_feasible(self, X: np.ndarray) -> np.ndarray:
        """Bounds, total mass, w/b and applicability domain. Cheap checks run first."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        emu = self.emulator
        total = X.sum(axis=1)
        wb = water_binder(X)
        ok = self.in_bounds(X)
        ok &= (total >= emu.total_mass_range[0]) & (total <= emu.total_mass_range[1])
        ok &= (wb >= emu.wb_range[0]) & (wb <= emu.wb_range[1])
        if ok.any():
            idx = np.flatnonzero(ok)
            ok[idx] = emu.domain.inside(X[idx])
        return ok

    def true_outputs(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Emulator mean at the specification age (the benchmark's 'truth')."""
        return {"strength_mpa": self.emulator.predict(X, self.cfg.age_days)}

    def observe(self, x: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
        """Emulator mean plus Gaussian noise (sd = grouped-CV residual sd, capped)."""
        mean = float(self.true_outputs(np.asarray(x)[None, :])["strength_mpa"][0])
        return {"strength_mpa": mean + float(rng.normal(0.0, self.noise_sd))}

    def material_cost(self, X: np.ndarray) -> np.ndarray:
        """$/m3 = sum of kg/m3 times $/kg. A known formula, never modelled."""
        return np.atleast_2d(X) @ self._price_vec

    def run_cost(self, X: np.ndarray) -> np.ndarray:
        """$ per trial batch: fixed testing cost plus the materials of a small batch."""
        rc = self.cfg.run_cost
        return rc.fixed + rc.batch_m3 * self.material_cost(X)

    def noise_summary(self) -> dict[str, float]:
        return {
            "strength_noise_sd_mpa": self.noise_sd,
            "grouped_cv_residual_sd_mpa": float(self.emulator.cv_report["residual_sd"]),
        }

    def target_sweep(
        self, targets: tuple[float, ...] = (40.0, 45.0, 50.0, 55.0, 60.0), n: int = 50_000
    ) -> dict[str, float]:
        """Share of uniform known-feasible mixes whose 28-day strength meets each target."""
        from formopt.design import sample_known_feasible

        X = sample_known_feasible(self, n, np.random.default_rng(2024))
        strength = self.true_outputs(X)["strength_mpa"]
        return {f"{t:g}": float(np.mean(strength >= t)) for t in targets}

    def calibration_extras(self) -> dict:
        return {
            "feasible_fraction_by_target_mpa": self.target_sweep(),
            "target_strength_mpa": self.cfg.target_strength_mpa,
            "age_days": self.cfg.age_days,
            "emulator_kind": self.emulator.kind,
            "domain_threshold": self.emulator.domain.threshold,
            "total_mass_range": list(self.emulator.total_mass_range),
            "wb_range": list(self.emulator.wb_range),
        }


def emulator_path(cfg: AppConfig) -> Path:
    return cfg.abs_paths.processed_dir / EMULATOR_FILE


def load_concrete_problem(cfg: AppConfig) -> ConcreteProblem:
    """Load the saved emulator and build the problem. Train it first if missing."""
    path = emulator_path(cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"No concrete emulator at {path}. Run: uv run python -m formopt train-emulator"
        )
    return ConcreteProblem(ConcreteEmulator.load(path), cfg.concrete)
