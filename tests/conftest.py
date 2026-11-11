"""Shared offline fixtures: a 40-row fake concrete table and a 2D toy problem.

Nothing here touches the network or downloaded data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

UCI_HEADERS = [
    "Cement (component 1)(kg in a m^3 mixture)",
    "Blast Furnace Slag (component 2)(kg in a m^3 mixture)",
    "Fly Ash (component 3)(kg in a m^3 mixture)",
    "Water  (component 4)(kg in a m^3 mixture)",
    "Superplasticizer (component 5)(kg in a m^3 mixture)",
    "Coarse Aggregate  (component 6)(kg in a m^3 mixture)",
    "Fine Aggregate (component 7)(kg in a m^3 mixture)",
    "Age (day)",
    "Concrete compressive strength(MPa, megapascals) ",
]

N_FAKE_MIXES = 18
N_FAKE_DUPLICATES = 4


def make_fake_concrete_raw(seed: int = 0) -> pd.DataFrame:
    """18 mixes, each tested at 7 and 28 days (36 rows), plus 4 exact duplicate rows."""
    rng = np.random.default_rng(seed)
    n = N_FAKE_MIXES
    cement = rng.uniform(150, 500, n).round(1)
    slag = rng.choice([0.0, 50.0, 100.0, 150.0], n)
    fly_ash = rng.choice([0.0, 60.0, 120.0], n)
    water = rng.uniform(140, 220, n).round(1)
    sp = rng.choice([0.0, 4.0, 8.0, 12.0], n)
    coarse = rng.uniform(850, 1100, n).round(1)
    fine = rng.uniform(650, 900, n).round(1)
    rows = []
    for i in range(n):
        wb = water[i] / (cement[i] + slag[i] + fly_ash[i])
        for age in (7.0, 28.0):
            strength = 95.0 * np.exp(-2.2 * wb) * (np.log(age) / np.log(28.0)) ** 0.5
            strength += rng.normal(0, 1.0)
            rows.append(
                [cement[i], slag[i], fly_ash[i], water[i], sp[i], coarse[i], fine[i], age, strength]
            )
    frame = pd.DataFrame(rows, columns=UCI_HEADERS)
    dupes = frame.iloc[[0, 5, 10, 21]]
    return pd.concat([frame, dupes], ignore_index=True)


@pytest.fixture
def fake_concrete_raw() -> pd.DataFrame:
    return make_fake_concrete_raw()


@pytest.fixture(scope="session")
def tiny_concrete_problem():
    """A ConcreteProblem backed by a tiny ExtraTrees emulator fitted on the fake 40-row frame."""
    from formopt.config import load_config
    from formopt.data import clean_concrete
    from formopt.emulator import train_concrete_emulator
    from formopt.problems.concrete import ConcreteProblem

    cfg = load_config().concrete
    emu_cfg = cfg.emulator.model_copy(
        update={"n_estimators": 25, "cv_folds": 3, "candidates": ["extra_trees"]}
    )
    small = cfg.model_copy(update={"emulator": emu_cfg, "target_strength_mpa": 35.0})
    df, _ = clean_concrete(make_fake_concrete_raw())
    emulator = train_concrete_emulator(df, small)
    return ConcreteProblem(emulator, small)


class ToyProblem:
    """2D toy lab: known cost x0 + x1, one unknown sin-based spec, run cost 1x to 5x.

    Known constraint: the corner x0 > 0.75 and x1 > 0.75 is forbidden (so factorial points at
    the 80 % level must be repaired). True optimum: x0 = x1 ~ 0.2145, cost ~ 0.429.
    """

    name = "toy"

    def __init__(self, noise_sd: float = 0.05) -> None:
        from formopt.types import Spec, Variable

        self.noise_sd = noise_sd
        self.variables = [Variable("x0", 0.0, 1.0, "-"), Variable("x1", 0.0, 1.0, "-")]
        self.specs = [Spec("g", lower=1.2)]
        self.prices = {"unit": 1.0}

    # BaseProblem-style helpers, written out so the fixture does not depend on them.
    @property
    def output_names(self) -> list[str]:
        return ["g"]

    @property
    def var_names(self) -> list[str]:
        return ["x0", "x1"]

    @property
    def bounds(self) -> np.ndarray:
        return np.array([[0.0, 1.0], [0.0, 1.0]])

    def as_dict(self, x: np.ndarray) -> dict[str, float]:
        return {"x0": float(x[0]), "x1": float(x[1])}

    def cost_cap(self) -> float:
        return 2.0 * self.prices["unit"]

    def fingerprint(self) -> str:
        return f"toy-{self.noise_sd}-{self.prices['unit']}"

    def in_bounds(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return np.all((X >= 0.0) & (X <= 1.0), axis=1)

    def known_feasible(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return self.in_bounds(X) & ~((X[:, 0] > 0.75) & (X[:, 1] > 0.75))

    def material_cost(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return self.prices["unit"] * (X[:, 0] + X[:, 1])

    def run_cost(self, X: np.ndarray) -> np.ndarray:
        return 1.0 + 4.0 * np.atleast_2d(X)[:, 0]

    def true_outputs(self, X: np.ndarray) -> dict[str, np.ndarray]:
        X = np.atleast_2d(X)
        return {"g": np.sin(3.0 * X[:, 0]) + np.sin(3.0 * X[:, 1])}

    def observe(self, x: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
        g = float(self.true_outputs(np.asarray(x)[None, :])["g"][0])
        return {"g": g + float(rng.normal(0.0, self.noise_sd))}

    def with_prices(self, prices: dict[str, float]) -> ToyProblem:
        new = ToyProblem(self.noise_sd)
        new.prices = {**self.prices, **prices}
        return new


TOY_OPT_COST = 2.0 * np.arcsin(0.6) / 3.0  # x0 = x1 = arcsin(0.6) / 3


@pytest.fixture
def toy_problem() -> ToyProblem:
    return ToyProblem()
