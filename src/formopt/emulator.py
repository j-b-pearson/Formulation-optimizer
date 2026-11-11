"""Concrete emulator: a tree ensemble trained on the real UCI data, used as a virtual lab.

It is deliberately NOT a Gaussian process (design decision D4): if the virtual lab and the
optimizer's surrogate were the same model class, Bayesian optimization would look better than
it really is. Validation is grouped by `mix_id` so the same recipe at different ages never sits
in both train and test. The applicability domain stops the optimizer from exploiting regions
where the emulator is extrapolating.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors

from formopt.config import ConcreteConfig, EmulatorCfg
from formopt.data import AGE, INGREDIENTS, TARGET

FEATURES = [*INGREDIENTS, "log_age"]
BINDER = ["cement", "slag", "fly_ash"]


def make_features(df: pd.DataFrame) -> np.ndarray:
    """Emulator inputs: the 7 ingredients (kg/m3) plus log(age in days)."""
    return np.column_stack([df[INGREDIENTS].to_numpy(float), np.log(df[AGE].to_numpy(float))])


def water_binder(X: np.ndarray) -> np.ndarray:
    """Water / (cement + slag + fly ash) for rows of the 7-ingredient matrix."""
    X = np.atleast_2d(X)
    return X[:, 3] / (X[:, 0] + X[:, 1] + X[:, 2])


def make_model(kind: str, cfg: EmulatorCfg) -> Any:
    """Build an unfitted emulator model of the given kind."""
    if kind == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=cfg.n_estimators,
            min_samples_leaf=cfg.min_samples_leaf,
            random_state=cfg.seed,
            n_jobs=1,
        )
    if kind == "hist_gb":
        return HistGradientBoostingRegressor(random_state=cfg.seed)
    raise ValueError(f"Unknown emulator kind {kind!r}")


def grouped_cv(
    model_factory: Callable[[], Any],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Grouped k-fold CV: R squared, RMSE and MAE overall and per fold, plus OOF predictions."""
    oof = np.empty_like(y, dtype=float)
    folds = []
    for train, test in GroupKFold(n_splits=n_splits).split(X, y, groups):
        model = model_factory().fit(X[train], y[train])
        pred = model.predict(X[test])
        oof[test] = pred
        folds.append(
            {
                "r2": float(r2_score(y[test], pred)),
                "rmse": float(np.sqrt(mean_squared_error(y[test], pred))),
                "mae": float(mean_absolute_error(y[test], pred)),
                "n_test": int(len(test)),
            }
        )
    resid = y - oof
    return {
        "r2": float(r2_score(y, oof)),
        "rmse": float(np.sqrt(mean_squared_error(y, oof))),
        "mae": float(mean_absolute_error(y, oof)),
        "residual_sd": float(np.std(resid, ddof=1)),
        "folds": folds,
        "oof_pred": oof,
    }


class ApplicabilityDomain:
    """kNN distance check: is a recipe close enough to real training mixes to trust?

    Ingredients are standardized; a point's score is its mean distance to the k nearest
    training mixes. The threshold is the `quantile` of the training mixes' own
    leave-one-out scores, so about 95 percent of real mixes count as "inside".
    """

    def __init__(self, X_train: np.ndarray, k: int = 5, quantile: float = 0.95) -> None:
        X_train = np.asarray(X_train, dtype=float)
        self.k = k
        self.mean_ = X_train.mean(axis=0)
        self.scale_ = X_train.std(axis=0, ddof=0)
        self.scale_[self.scale_ == 0] = 1.0
        self.X_train_std = self._standardize(X_train)
        # Brute force is fastest here (a few hundred mixes in 7D) and exact.
        self._nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute").fit(self.X_train_std)
        dist, _ = self._nn.kneighbors(self.X_train_std)
        self.train_scores_ = dist[:, 1:].mean(axis=1)  # drop self (distance 0)
        self.threshold = float(np.quantile(self.train_scores_, quantile))

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(X) - self.mean_) / self.scale_

    def score(self, X: np.ndarray) -> np.ndarray:
        """Mean distance (standardized units) to the k nearest training mixes."""
        dist, _ = self._nn.kneighbors(self._standardize(X), n_neighbors=self.k)
        return dist.mean(axis=1)

    def inside(self, X: np.ndarray) -> np.ndarray:
        """Boolean mask: True where the emulator is interpolating, not extrapolating."""
        return self.score(X) <= self.threshold


@dataclass
class ConcreteEmulator:
    """Fitted emulator plus everything the concrete problem needs to know about the data."""

    model: Any
    kind: str
    feature_names: list[str]
    domain: ApplicabilityDomain
    cv_report: dict[str, Any]
    ingredient_bounds: np.ndarray  # (7, 2) data min and max
    total_mass_range: tuple[float, float]
    wb_range: tuple[float, float]
    unique_mixes: np.ndarray  # (n_mix, 7) used for the domain and candidate sampling
    model_comparison: dict[str, dict[str, float]] = field(default_factory=dict)

    def predict(self, X_ingredients: np.ndarray, age_days: float) -> np.ndarray:
        """Predicted compressive strength (MPa) for recipes at a given age."""
        X = np.atleast_2d(np.asarray(X_ingredients, dtype=float))
        features = np.column_stack([X, np.full(len(X), np.log(age_days))])
        return self.model.predict(features)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path) -> ConcreteEmulator:
        return joblib.load(path)


def train_concrete_emulator(df: pd.DataFrame, cfg: ConcreteConfig) -> ConcreteEmulator:
    """Compare candidate model classes by grouped-CV RMSE, then fit the winner on all rows."""
    X = make_features(df)
    y = df[TARGET].to_numpy(float)
    groups = df["mix_id"].to_numpy()
    reports: dict[str, dict[str, Any]] = {}
    for kind in cfg.emulator.candidates:
        reports[kind] = grouped_cv(
            lambda kind=kind: make_model(kind, cfg.emulator), X, y, groups, cfg.emulator.cv_folds
        )
    best = min(reports, key=lambda k: reports[k]["rmse"])
    model = make_model(best, cfg.emulator).fit(X, y)
    mixes = df.drop_duplicates("mix_id")[INGREDIENTS].to_numpy(float)
    totals = df[INGREDIENTS].sum(axis=1).to_numpy(float)
    wb = water_binder(df[INGREDIENTS].to_numpy(float))
    comparison = {
        k: {m: float(v[m]) for m in ("r2", "rmse", "mae", "residual_sd")}
        for k, v in reports.items()
    }
    return ConcreteEmulator(
        model=model,
        kind=best,
        feature_names=FEATURES,
        domain=ApplicabilityDomain(mixes, k=cfg.domain.k, quantile=cfg.domain.quantile),
        cv_report=reports[best],
        ingredient_bounds=np.column_stack(
            [df[INGREDIENTS].min().to_numpy(float), df[INGREDIENTS].max().to_numpy(float)]
        ),
        total_mass_range=(float(totals.min()), float(totals.max())),
        wb_range=(float(wb.min()), float(wb.max())),
        unique_mixes=mixes,
        model_comparison=comparison,
    )


def sanity_checks(emulator: ConcreteEmulator, n_mixes: int = 10, seed: int = 0) -> dict:
    """Domain-expert checks: strength grows with age, and falls as water/binder rises."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(emulator.unique_mixes), size=n_mixes, replace=False)
    mixes = emulator.unique_mixes[idx]
    by_age = np.column_stack([emulator.predict(mixes, a) for a in (7.0, 28.0, 90.0)])
    nondecreasing = np.all(np.diff(by_age, axis=1) >= -1e-9, axis=1)

    # Partial dependence on w/b at 28 days: rescale water in every real mix to hit each w/b.
    grid = np.linspace(0.30, 0.70, 9)
    base = emulator.unique_mixes
    binder = base[:, 0] + base[:, 1] + base[:, 2]
    pd_values = []
    for wb in grid:
        X = base.copy()
        X[:, 3] = wb * binder
        pd_values.append(float(emulator.predict(X, 28.0).mean()))
    steps = np.diff(pd_values)
    return {
        "age_mixes_checked": int(n_mixes),
        "age_mixes_nondecreasing_7_28_90": int(nondecreasing.sum()),
        "age_predictions_7_28_90": by_age.round(2).tolist(),
        "pd_wb_grid": grid.round(3).tolist(),
        "pd_wb_strength_28d": [round(v, 3) for v in pd_values],
        "pd_wb_decreasing_steps": int((steps < 0).sum()),
        "pd_wb_total_steps": int(len(steps)),
    }
