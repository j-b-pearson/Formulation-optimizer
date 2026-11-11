"""Core data types shared by problems, strategies, the loop and the benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Variable:
    """One input dimension (ingredient amount or process setting) with its allowed range."""

    name: str
    low: float
    high: float
    unit: str


@dataclass(frozen=True)
class Spec:
    """A pass/fail limit on one measured output.

    `transform="log"` means the surrogate GP is fit on log(output) and the limits are logged
    too (used for viscosity, which spans an order of magnitude).
    """

    output: str
    lower: float | None = None
    upper: float | None = None
    transform: str = "none"

    def transform_values(self, y: np.ndarray | float) -> np.ndarray:
        """Map raw output values into the space the GP is fit in."""
        y = np.asarray(y, dtype=float)
        return np.log(y) if self.transform == "log" else y

    def inverse_transform(self, z: np.ndarray | float) -> np.ndarray:
        """Map GP-space values back to raw output units."""
        z = np.asarray(z, dtype=float)
        return np.exp(z) if self.transform == "log" else z

    def transformed_limits(self) -> tuple[float | None, float | None]:
        """Spec limits in GP space (logged when `transform == "log"`)."""
        lo = None if self.lower is None else float(self.transform_values(self.lower))
        hi = None if self.upper is None else float(self.transform_values(self.upper))
        return lo, hi

    def passes(self, y: np.ndarray | float) -> np.ndarray:
        """Element-wise pass/fail for raw output values."""
        y = np.asarray(y, dtype=float)
        ok = np.ones(y.shape, dtype=bool)
        if self.lower is not None:
            ok &= y >= self.lower
        if self.upper is not None:
            ok &= y <= self.upper
        return ok


@dataclass
class Observation:
    """One experiment: what was run, what the lab measured, and what it cost."""

    x: np.ndarray  # shape (d,), natural units
    outputs: dict[str, float]  # measured (noisy) outputs
    run_cost: float  # $ for this experiment
    material_cost: float  # known formula ($/kg or $/m3), not measured
    passed_spec: bool  # judged on the noisy measurement (what the lab sees)
    truly_feasible: bool  # judged on noise-free truth (benchmark only)
    true_outputs: dict[str, float] = field(default_factory=dict)  # benchmark only
    # Surrogate mean and std (GP space, including noise) for each output at suggestion time.
    # Only BO strategies fill this; used for the calibration analysis.
    predicted: dict[str, tuple[float, float]] | None = None


@dataclass
class CampaignResult:
    """Full history of one optimization campaign (one problem, method and seed)."""

    problem: str
    method: str
    seed: int
    budget: float
    observations: list[Observation] = field(default_factory=list)
    wall_time_s: float = 0.0
    n_gp_warnings: int = 0

    @property
    def n_experiments(self) -> int:
        return len(self.observations)

    def X(self) -> np.ndarray:  # noqa: N802 (matrix name)
        """All tested inputs as an (n, d) array."""
        if not self.observations:
            return np.empty((0, 0))
        return np.vstack([o.x for o in self.observations])

    def cumulative_spend(self) -> np.ndarray:
        """Cumulative lab spend after each experiment."""
        return np.cumsum([o.run_cost for o in self.observations], dtype=float)

    def best_feasible_cost_trace(
        self, use_truth: bool = True, cost_if_none: float = np.inf
    ) -> np.ndarray:
        """Lowest material cost among feasible experiments so far, after each experiment.

        `use_truth=True` judges feasibility on the noise-free truth (the benchmark metric);
        `False` uses the lab's noisy pass/fail. Before any feasible run the value is
        `cost_if_none`.
        """
        best = cost_if_none
        trace = np.empty(len(self.observations))
        for i, obs in enumerate(self.observations):
            ok = obs.truly_feasible if use_truth else obs.passed_spec
            if ok and obs.material_cost < best:
                best = obs.material_cost
            trace[i] = best
        return trace
