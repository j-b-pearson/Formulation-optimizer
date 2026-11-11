"""Gaussian process surrogate: one GP per unknown spec output.

Kernel: Constant x Matern(nu = 2.5) with one length scale per input (ARD), plus a White noise
term so the GP learns how noisy the lab is. Inputs are scaled to the unit cube with the
problem bounds; outputs are standardized by sklearn (`normalize_y=True`). Viscosity is fit on
log(viscosity) because it spans an order of magnitude and its noise is multiplicative.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Kernel, Matern, WhiteKernel

from formopt.design import to_unit

MIN_VARIANCE = 1e-12


def make_kernel(
    d: int,
    length_scale_bounds: tuple[float, float] = (1e-2, 10.0),
    constant_bounds: tuple[float, float] = (1e-3, 1e3),
    noise_bounds: tuple[float, float] = (1e-6, 1.0),
) -> Kernel:
    """Constant * Matern 5/2 with ARD length scales + White noise (bounds in unit/std space)."""
    return ConstantKernel(1.0, constant_bounds) * Matern(
        length_scale=np.ones(d), length_scale_bounds=length_scale_bounds, nu=2.5
    ) + WhiteKernel(1e-3, noise_bounds)


@dataclass
class FittedGP:
    """A fitted GP plus what is needed to use it on natural-unit inputs."""

    model: GaussianProcessRegressor
    bounds: np.ndarray
    transform: str  # "none" or "log": the space the GP was fit in
    n_warnings: int = 0  # ConvergenceWarnings raised while fitting

    @property
    def noise_var(self) -> float:
        """Learned observation-noise variance, in the GP's (transformed) output units."""
        white = self.model.kernel_.k2.noise_level
        y_std = float(np.ravel(getattr(self.model, "_y_train_std", 1.0))[0])
        return float(white) * y_std**2

    def predict(self, X: np.ndarray, include_noise: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Mean and std in the transformed space.

        `include_noise=False` gives the uncertainty about the true response (used for
        probability of feasibility: does the recipe truly pass?). `include_noise=True` adds the
        learned lab noise (used to check calibration against noisy measurements).
        """
        U = to_unit(np.atleast_2d(X), self.bounds)
        mean, std = self.model.predict(U, return_std=True)
        var = std**2
        if not include_noise:
            var = var - self.noise_var
        return mean, np.sqrt(np.maximum(var, MIN_VARIANCE))


def fit_gp(
    X: np.ndarray,
    y: np.ndarray,
    bounds: np.ndarray,
    transform: str = "none",
    n_restarts: int = 2,
    seed: int = 0,
    length_scale_bounds: tuple[float, float] = (1e-2, 10.0),
    constant_bounds: tuple[float, float] = (1e-3, 1e3),
    noise_bounds: tuple[float, float] = (1e-6, 1.0),
) -> FittedGP:
    """Fit a GP by maximum marginal likelihood with restarts.

    `y` is given in raw units; with `transform="log"` the GP is fit on log(y). sklearn
    ConvergenceWarnings (hyperparameters hitting their bounds) are captured and counted rather
    than silenced globally; any other warning is passed through.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    y_fit = np.log(y) if transform == "log" else y
    kernel = make_kernel(X.shape[1], length_scale_bounds, constant_bounds, noise_bounds)
    model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=n_restarts,
        random_state=seed,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(to_unit(X, bounds), y_fit)
    n_conv = 0
    for w in caught:
        if issubclass(w.category, ConvergenceWarning):
            n_conv += 1
        else:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    return FittedGP(
        model=model, bounds=np.asarray(bounds, float), transform=transform, n_warnings=n_conv
    )
