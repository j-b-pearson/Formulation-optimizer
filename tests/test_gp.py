"""GP surrogate: shapes, interpolation, uncertainty behaviour, log transform."""

from __future__ import annotations

import numpy as np

from formopt.gp import fit_gp, make_kernel

BOUNDS_1D = np.array([[0.0, 1.0]])


def _sine_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.linspace(0.05, 0.95, 6)[:, None]
    return X, np.sin(6.0 * X[:, 0])


def test_kernel_has_one_length_scale_per_input() -> None:
    kernel = make_kernel(5)
    assert kernel.k1.k2.length_scale.shape == (5,)


def test_fit_predict_shapes() -> None:
    X, y = _sine_data()
    gp = fit_gp(X, y, BOUNDS_1D, seed=0)
    Xs = np.linspace(0, 1, 37)[:, None]
    mean, std = gp.predict(Xs)
    assert mean.shape == (37,) and std.shape == (37,)
    assert np.all(std > 0)
    assert isinstance(gp.n_warnings, int)


def test_interpolates_noise_free_data() -> None:
    X, y = _sine_data()
    gp = fit_gp(X, y, BOUNDS_1D, seed=0)
    mean, _ = gp.predict(X)
    np.testing.assert_allclose(mean, y, atol=0.05)


def test_std_small_near_data_large_far_away() -> None:
    X = np.array([[0.1], [0.15], [0.2], [0.25]])
    y = np.sin(6.0 * X[:, 0])
    gp = fit_gp(X, y, BOUNDS_1D, seed=0)
    _, std_near = gp.predict(np.array([[0.175]]))
    _, std_far = gp.predict(np.array([[0.9]]))
    assert std_near[0] < std_far[0]


def test_noise_term_is_learned_and_added_on_request() -> None:
    rng = np.random.default_rng(0)
    X = rng.random((40, 1))
    y = np.sin(6.0 * X[:, 0]) + rng.normal(0, 0.2, 40)
    gp = fit_gp(X, y, BOUNDS_1D, seed=0)
    assert 0.01 < gp.noise_var < 0.2
    _, latent = gp.predict(X[:5])
    _, total = gp.predict(X[:5], include_noise=True)
    np.testing.assert_allclose(total**2, latent**2 + gp.noise_var, rtol=1e-6)


def test_log_transform_round_trip() -> None:
    X = np.linspace(0.0, 1.0, 8)[:, None]
    y = 2000.0 * np.exp(1.5 * X[:, 0])  # viscosity-like, positive
    gp = fit_gp(X, y, BOUNDS_1D, transform="log", seed=0)
    assert gp.transform == "log"
    mean, _ = gp.predict(X)
    np.testing.assert_allclose(np.exp(mean), y, rtol=0.02)
