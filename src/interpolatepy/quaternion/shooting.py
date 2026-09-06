"""Natural Riemannian cubic quaternion interpolation by multiple shooting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._shooting_solver import integrate
from ._shooting_solver import left_matrix
from ._shooting_solver import matching_system
from ._shooting_solver import newton_solve
from ._shooting_solver import rk4_step
from ._shooting_solver import rotation_residual
from .core import Quaternion


_EPSILON = 1e-12
_MIN_KEYFRAMES = 2


@dataclass(frozen=True, slots=True)
class ShootingConfig:
    """Settings for :class:`ShootingQuaternionInterpolation`.

    ``tolerance`` bounds the dimensionless orientation and scaled derivative
    matching residuals. RK4 integration starts with ``integration_steps`` per
    segment and is independently checked on a finer grid, up to
    ``max_integration_steps``. ``max_iterations`` is the total accepted Newton
    step budget across all integration grids. Output sampling does not affect
    the number of unknowns or the integration accuracy.
    """

    tolerance: float = 1e-8
    max_iterations: int = 30
    integration_steps: int = 16
    max_integration_steps: int = 1024

    def __post_init__(self) -> None:
        if not np.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive and finite")
        for name in ("max_iterations", "integration_steps", "max_integration_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_integration_steps <= self.integration_steps:
            raise ValueError("max_integration_steps must exceed integration_steps for independent verification")


class ShootingQuaternionInterpolation:
    """Interpolate rotations with natural Riemannian cubics and multiple shooting.

    Each keyframe interval has nine unknowns: the initial half angular velocity
    ``v``, its derivative ``a``, and an integration constant ``c``. In local
    segment time, the ODE is ``q' = q v``, ``v' = a``, ``a' = c - 2 cross(v, a)``.
    Damped Newton solves orientation interpolation, body velocity/acceleration
    continuity, and zero endpoint acceleration simultaneously. Analytic ODE
    sensitivities provide the sparse matching Jacobian.

    Parameters
    ----------
    time_points:
        Finite, strictly increasing physical keyframe times.
    quaternions:
        At least two finite, nonzero quaternions. Inputs are normalized and
        consecutive signs are aligned before initializing with segment SLERP.
    config:
        Numerical tolerances and iteration/integration limits.

    Notes
    -----
    The solution is a stationary curve of the continuous squared covariant
    acceleration functional, with natural boundary conditions. It is a local
    solution; global minimality is not guaranteed for widely separated data.
    It differs from SPRING's sampled, chord-parameterized curvature objective.
    A failed solve or integration accuracy check raises ``RuntimeError``.
    Angular velocity and acceleration are obtained from the integrated state,
    not by differencing sampled orientations.
    """

    def __init__(
        self,
        time_points: list[float] | np.ndarray,
        quaternions: list[Quaternion],
        config: ShootingConfig | None = None,
    ) -> None:
        self.config = config if config is not None else ShootingConfig()
        self.time_points = np.array(time_points, dtype=np.float64, copy=True)
        self.quaternions = self._prepare_keyframes(quaternions)
        self._durations = np.diff(self.time_points)
        self.t_min, self.t_max = float(self.time_points[0]), float(self.time_points[-1])
        self.num_variables = 9 * len(self._durations)
        values = np.array([[quaternion.w, quaternion.x, quaternion.y, quaternion.z] for quaternion in self.quaternions])
        conjugates = values[:-1].copy()
        conjugates[:, 1:] *= -1.0
        relative = np.einsum("mij,mj->mi", left_matrix(conjugates), values[1:])
        increments, _ = rotation_residual(relative)
        parameters = np.column_stack((increments, np.zeros((len(increments), 6))))
        self.iterations_run = 0
        steps = self.config.integration_steps
        while steps < self.config.max_integration_steps:
            parameters, iterations = newton_solve(
                parameters,
                values,
                self._durations,
                steps,
                self.config.tolerance * 0.1,
                self.config.max_iterations - self.iterations_run,
            )
            self.iterations_run += iterations
            finer_steps = min(2 * steps, self.config.max_integration_steps)
            residual, _ = matching_system(parameters, values, self._durations, finer_steps)
            self.residual_norm = float(np.max(np.abs(residual)))
            if self.residual_norm <= self.config.tolerance:
                self.integration_steps = finer_steps
                break
            steps = finer_steps
        else:
            raise RuntimeError("Multiple shooting integration accuracy exceeds tolerance at max_integration_steps")
        self._parameters = parameters
        _, _, self._nodes, energy = integrate(parameters, values[:-1], self.integration_steps, store_nodes=True)
        # Successive divisions retain zero energy on geodesics even when h**3
        # underflows, and avoid overflowing powers for long intervals.
        with np.errstate(over="ignore", under="ignore"):
            self.acceleration_energy = float(np.sum(4.0 * energy / self._durations / self._durations / self._durations))

    def _prepare_keyframes(self, quaternions: list[Quaternion]) -> list[Quaternion]:
        if self.time_points.ndim != 1:
            raise ValueError("time_points must be one-dimensional")
        if len(self.time_points) != len(quaternions):
            raise ValueError("Number of time points must match number of quaternions")
        if len(quaternions) < _MIN_KEYFRAMES:
            raise ValueError("At least 2 quaternions are required")
        if not np.all(np.isfinite(self.time_points)):
            raise ValueError("time_points must contain only finite values")
        durations = np.diff(self.time_points)
        if not np.all(np.isfinite(durations)) or not np.all(durations > 0.0):
            raise ValueError("Time points must have finite, strictly positive spacing")
        prepared: list[Quaternion] = []
        for index, quaternion in enumerate(quaternions):
            if not isinstance(quaternion, Quaternion):
                raise TypeError(f"Element {index} is not a Quaternion instance")
            values = np.array([quaternion.w, quaternion.x, quaternion.y, quaternion.z])
            norm = np.linalg.norm(values)
            if not np.isfinite(norm) or norm <= _EPSILON:
                raise ValueError(f"Quaternion {index} must be finite and non-zero")
            current = Quaternion(*map(float, values / norm))
            if prepared and prepared[-1].dot_product(current) < 0.0:
                current = -current
            prepared.append(current)
        return prepared

    def _check_time(self, time: float) -> float:
        if not np.isfinite(time):
            raise ValueError("Time must be finite")
        if time < self.t_min - _EPSILON or time > self.t_max + _EPSILON:
            raise ValueError(f"Time {time} outside valid range [{self.t_min}, {self.t_max}]")
        return float(np.clip(time, self.t_min, self.t_max))

    def _evaluate_state(self, time: float) -> tuple[np.ndarray, float]:
        time = self._check_time(time)
        segment = min(int(np.searchsorted(self.time_points, time, side="right")) - 1, len(self._durations) - 1)
        duration = float(self._durations[segment])
        fraction = float(np.clip((time - self.time_points[segment]) / duration, 0.0, 1.0))
        node = min(int(fraction * self.integration_steps), self.integration_steps)
        state = self._nodes[node, segment : segment + 1]
        remainder = fraction - node / self.integration_steps
        if remainder > 0.0:
            state, _, _ = rk4_step(state, self._parameters[segment : segment + 1, 6:9], remainder)
        return state[0], duration

    def evaluate(self, t: float) -> Quaternion:
        """Evaluate orientation; return original normalized keyframes exactly."""
        time = self._check_time(t)
        index = int(np.searchsorted(self.time_points, time))
        if index < len(self.time_points) and self.time_points[index] == time:
            return self.quaternions[index].copy()
        state, _ = self._evaluate_state(time)
        return Quaternion(*map(float, state[:4]))

    def evaluate_velocity(self, t: float) -> np.ndarray:
        """Return body angular velocity in radians per second."""
        state, duration = self._evaluate_state(t)
        return 2.0 * state[4:7] / duration

    def evaluate_acceleration(self, t: float) -> np.ndarray:
        """Return body angular acceleration in radians per second squared."""
        state, duration = self._evaluate_state(t)
        return 2.0 * state[7:10] / duration / duration

    def generate_trajectory(self, num_points: int = 100) -> tuple[np.ndarray, list[Quaternion]]:
        """Sample the solved continuous curve without rerunning optimization."""
        if isinstance(num_points, bool) or not isinstance(num_points, (int, np.integer)) or num_points < _MIN_KEYFRAMES:
            raise ValueError("num_points must be an integer of at least 2")
        times = np.linspace(self.t_min, self.t_max, num_points)
        return times, [self.evaluate(float(time)) for time in times]


__all__ = ["ShootingConfig", "ShootingQuaternionInterpolation"]
