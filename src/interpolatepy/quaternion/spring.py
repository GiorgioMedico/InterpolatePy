"""Curvature-minimizing quaternion interpolation with SPRING.

The implementation follows the discrete algorithm in Dam, Koch, and
Lillholm, *Quaternions, Interpolation and Animation* (DIKU-TR-98/5, 1998),
section 6.3.7.  A SLERP trajectory is relaxed with gradient descent to
minimize tangential curvature on the unit-quaternion sphere while every
keyframe remains fixed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._spring_slerp import slerp_segments
from ._spring_solver import curvature_energy
from ._spring_solver import curvature_energy_gradient
from ._spring_solver import iteration_budgets
from ._spring_solver import minimize
from ._spring_solver import nested_level_indices
from ._spring_solver import refinement_sample_counts
from .core import Quaternion


_EPSILON = 1e-12
_MIN_SAMPLES = 2


@dataclass(frozen=True, slots=True)
class SpringConfig:
    """Numerical settings for :class:`SpringQuaternionInterpolation`.

    Parameters
    ----------
    num_samples:
        Number of discrete frames, including the first and last keyframes.
        Frames are distributed among keyframe intervals in proportion to
        quaternion chord length, as proposed in the report.
    iterations:
        Total iteration budget across all refinement levels, unless the
        final-grid budget is explicitly overridden by ``final_iterations``.
    refinement_levels:
        Requested coarse-to-fine levels. Three levels use approximately 1/25,
        1/5, and all final samples, following the report's examples. Counts
        that would be too small or duplicate an earlier level are omitted.
    step_size:
        Initial length of each normalized-gradient step, including all coarse
        levels. Gauss-Newton uses a full model step and backtracking instead.
    norm_penalty:
        Weight of ``(||q||^2 - 1)^2`` in the discrete energy (equation 6.28).
    keyframe_curvature_weight:
        Extra weight for curvature centered on a keyframe. The report suggests
        approximately ``1.2`` to propagate curvature through sharp keyframes.
    tolerance:
        Stop when the movable part of the gradient has this Euclidean norm.
    solver:
        ``"gradient_descent"`` (default) retains the original algorithm.
        ``"gauss_newton"`` accelerates only the final grid with a banded
        least-squares model of exactly the same energy. Coarse solves stay
        identical, preserving the fixed samples and final starting state.
    final_iterations:
        Optional final-grid iteration budget, independent of the coarse
        solves. ``None`` retains the original shared budget. This lets both
        solvers be compared to convergence without changing coarse anchors.
    """

    num_samples: int = 101
    iterations: int = 300
    refinement_levels: int = 3
    step_size: float = 0.05
    norm_penalty: float = 100.0
    keyframe_curvature_weight: float = 1.2
    tolerance: float = 1e-9
    solver: str = "gradient_descent"
    final_iterations: int | None = None

    def __post_init__(self) -> None:
        if self.solver not in {"gradient_descent", "gauss_newton"}:
            raise ValueError("solver must be 'gradient_descent' or 'gauss_newton'")
        if self.final_iterations is not None and (
            isinstance(self.final_iterations, bool)
            or not isinstance(self.final_iterations, (int, np.integer))
            or self.final_iterations < 0
        ):
            raise ValueError("final_iterations must be a non-negative integer or None")
        if self.num_samples < _MIN_SAMPLES:
            raise ValueError("num_samples must be at least 2")
        if self.iterations < 0:
            raise ValueError("iterations must be non-negative")
        if self.refinement_levels < 1:
            raise ValueError("refinement_levels must be at least 1")
        if not np.isfinite(self.step_size) or self.step_size <= 0.0:
            raise ValueError("step_size must be positive and finite")
        if not np.isfinite(self.norm_penalty) or self.norm_penalty <= 0.0:
            raise ValueError("norm_penalty must be positive and finite")
        if not np.isfinite(self.keyframe_curvature_weight) or self.keyframe_curvature_weight <= 0.0:
            raise ValueError("keyframe_curvature_weight must be positive and finite")
        if not np.isfinite(self.tolerance) or self.tolerance < 0.0:
            raise ValueError("tolerance must be non-negative and finite")


def _quaternion_to_array(quaternion: Quaternion) -> np.ndarray:
    return np.array([quaternion.s_, *quaternion.v_], dtype=np.float64)


def _array_to_quaternion(values: np.ndarray) -> Quaternion:
    return Quaternion(float(values[0]), float(values[1]), float(values[2]), float(values[3]))


class SpringQuaternionInterpolation:
    """Numerical minimum-curvature interpolation of quaternion keyframes.

    ``SPRING`` stands for *Spherical Interpolation using Numerical Gradient
    descent*. The algorithm discretizes a quaternion curve, approximates its
    second derivative with centered differences, removes the radial component
    to obtain local curvature, and minimizes squared curvature while holding
    keyframes fixed.

    Parameters
    ----------
    time_points:
        Strictly increasing times for the quaternion keyframes.
    quaternions:
        Two or more non-zero quaternion keyframes. Inputs are normalized and
        signs are made consecutive so interpolation takes the short branch.
    config:
        Optional numerical settings. The default is :class:`SpringConfig`.

    Notes
    -----
    This is a discrete numerical trajectory, not an analytical spline. Values
    between optimized frames are evaluated with SLERP. Angular velocity and
    acceleration are centered finite-difference estimates. By default the
    solver performs the report's coarse-to-fine minimization: optimized frames
    from one level become fixed frames in the next level. If refinement
    increases curvature on the final grid, the final solve restarts from the
    original SLERP curve with only the keyframes fixed.

    ``stage_gradient_norms`` reports free-gradient norms before final
    normalization; ``converged`` reports final-grid stationarity (or an
    initially stationary target curve). A false value means the budget or
    line search ended without reaching ``tolerance``. No global-minimum or
    bitwise equality guarantee is made for this nonconvex problem. Different
    solvers need not agree after the same number of unconverged iterations.
    """

    def __init__(
        self,
        time_points: list[float] | np.ndarray,
        quaternions: list[Quaternion],
        config: SpringConfig | None = None,
    ) -> None:
        self.config = config if config is not None else SpringConfig()
        self.time_points = np.asarray(time_points, dtype=np.float64)
        self.quaternions = self._validate_and_prepare_keyframes(quaternions)

        if self.config.num_samples < len(self.quaternions):
            raise ValueError("num_samples must be at least the number of keyframes")

        intervals = self._allocate_intervals(self.config.num_samples)
        self.sample_times, initial_frames, self.keyframe_indices = self._initial_curve(intervals)
        self.refinement_sample_counts = refinement_sample_counts(
            len(self.quaternions), self.config.num_samples, self.config.refinement_levels
        )
        level_indices = nested_level_indices(len(initial_frames), self.keyframe_indices, self.refinement_sample_counts)
        budgets = iteration_budgets(self.config.iterations, len(level_indices))
        if self.config.final_iterations is not None:
            budgets = (*budgets[:-1], self.config.final_iterations)
        optimized_frames, stage_histories = self._optimize_levels(initial_frames, level_indices, budgets)
        final_curvature_weights = self._curvature_weights(level_indices[-1])

        # Equation 6.28 keeps frames approximately on H1. Normalize the final
        # finite-iteration result so every value exposed as a rotation is exact.
        optimized_frames /= np.linalg.norm(optimized_frames, axis=1)[:, None]
        optimized_frames[self.keyframe_indices] = np.array([_quaternion_to_array(q) for q in self.quaternions])

        self.samples = [_array_to_quaternion(frame) for frame in optimized_frames]
        self.initial_energy = curvature_energy(initial_frames, final_curvature_weights)
        self.final_energy = curvature_energy(optimized_frames, final_curvature_weights)
        self.stage_energy_history = stage_histories
        self.energy_history = stage_histories[-1]
        self.iterations_run = sum(len(history) - 1 for history in stage_histories)
        self.t_min = float(self.time_points[0])
        self.t_max = float(self.time_points[-1])
        self._derivative_step = float(np.min(np.diff(self.sample_times))) * 0.5

    def _validate_and_prepare_keyframes(self, quaternions: list[Quaternion]) -> list[Quaternion]:
        if self.time_points.ndim != 1:
            raise ValueError("time_points must be one-dimensional")
        if len(self.time_points) != len(quaternions):
            raise ValueError("Number of time points must match number of quaternions")
        if len(quaternions) < _MIN_SAMPLES:
            raise ValueError("At least 2 quaternions are required for interpolation")
        if not np.all(np.isfinite(self.time_points)):
            raise ValueError("time_points must contain only finite values")
        if not np.all(np.diff(self.time_points) > 0.0):
            raise ValueError("Time points must be strictly increasing")

        prepared: list[Quaternion] = []
        for index, quaternion in enumerate(quaternions):
            if not isinstance(quaternion, Quaternion):
                raise TypeError(f"Element {index} is not a Quaternion instance")
            norm = quaternion.norm()
            if not np.isfinite(norm) or norm <= _EPSILON:
                raise ValueError(f"Quaternion {index} must be finite and non-zero")
            current = _array_to_quaternion(_quaternion_to_array(quaternion) / norm)
            if prepared and prepared[-1].dot_product(current) < 0.0:
                current = -current
            prepared.append(current)
        return prepared

    def _allocate_intervals(self, num_samples: int) -> np.ndarray:
        """Distribute frames by quaternion chord length (report, page 55)."""
        segment_count = len(self.quaternions) - 1
        intervals = np.ones(segment_count, dtype=np.int64)
        remaining = num_samples - 1 - segment_count
        if remaining == 0:
            return intervals

        keyframe_values = np.array([_quaternion_to_array(q) for q in self.quaternions])
        chord_lengths = np.linalg.norm(np.diff(keyframe_values, axis=0), axis=1)
        if float(np.sum(chord_lengths)) <= _EPSILON:
            chord_lengths = np.diff(self.time_points)

        exact = remaining * chord_lengths / np.sum(chord_lengths)
        extra = np.floor(exact).astype(np.int64)
        intervals += extra
        leftover = int(remaining - np.sum(extra))
        if leftover:
            order = np.argsort(-(exact - extra), kind="stable")
            intervals[order[:leftover]] += 1
        return intervals

    def _initial_curve(self, intervals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        keyframe_indices = np.r_[0, np.cumsum(intervals)]
        fractions = np.empty(int(keyframe_indices[-1]) + 1)
        fractions[0] = 0.0
        segments = np.r_[0, np.repeat(np.arange(len(intervals)), intervals)]
        for index, count in enumerate(intervals):
            fractions[keyframe_indices[index] + 1 : keyframe_indices[index + 1] + 1] = np.arange(1, count + 1) / count
        sample_times = self.time_points[segments] + fractions * np.diff(self.time_points)[segments]
        sample_times[keyframe_indices] = self.time_points
        frames = slerp_segments(self.quaternions, segments, fractions)
        return sample_times, frames, keyframe_indices

    def _curvature_weights(self, level_indices: np.ndarray) -> np.ndarray:
        """Weight only curvature centered on an original internal keyframe."""
        weights = np.ones(len(level_indices) - 2, dtype=np.float64)
        keyframe_positions = np.searchsorted(level_indices, self.keyframe_indices[1:-1])
        weights[keyframe_positions - 1] = self.config.keyframe_curvature_weight
        return weights

    @staticmethod
    def _refine_initial_curve(
        source_indices: np.ndarray,
        source_frames: np.ndarray,
        target_indices: np.ndarray,
    ) -> np.ndarray:
        """SLERP new frames between the preceding level's fixed samples."""
        upper = np.searchsorted(source_indices, target_indices, side="left")
        existing = source_indices[upper] == target_indices
        refined = source_frames[upper].copy()
        # Existing coarse anchors retain their raw (possibly nonunit) values.
        # Only endpoints used to interpolate newly inserted samples are normalized.
        if np.any(~existing):
            lower = upper[~existing] - 1
            fractions = (target_indices[~existing] - source_indices[lower]) / (
                source_indices[lower + 1] - source_indices[lower]
            )
            endpoints = [_array_to_quaternion(frame).unit() for frame in source_frames]
            refined[~existing] = slerp_segments(endpoints, lower, fractions)
        return refined

    def _optimize_levels(
        self,
        final_initial_frames: np.ndarray,
        level_indices: tuple[np.ndarray, ...],
        budgets: tuple[int, ...],
    ) -> tuple[np.ndarray, tuple[tuple[float, ...], ...]]:
        # Check convergence on the target grid first: a coarse divided
        # difference can have truncation error even for a stationary geodesic.
        target_weights = self._curvature_weights(level_indices[-1])
        initial_curvature = curvature_energy(final_initial_frames, target_weights)
        _, initial_gradient = curvature_energy_gradient(final_initial_frames, target_weights, self.config.norm_penalty)
        initial_gradient[self.keyframe_indices] = 0.0
        converged = np.linalg.norm(initial_gradient) <= self.config.tolerance
        previous_indices: np.ndarray | None = None
        previous_frames: np.ndarray | None = None
        histories: list[tuple[float, ...]] = []
        gradient_norms: list[float] = []

        for current_indices, budget in zip(level_indices, budgets):
            if previous_indices is None or previous_frames is None or converged:
                stage_initial = final_initial_frames[current_indices].copy()
                fixed_final_indices = self.keyframe_indices
            else:
                stage_initial = self._refine_initial_curve(previous_indices, previous_frames, current_indices)
                fixed_final_indices = previous_indices

            if previous_indices is not None and len(current_indices) == len(final_initial_frames):
                # Coarse-grid truncation error can spoil an almost geodesic
                # curve. Restart the final solve if prolongation made it worse.
                normalized_initial = stage_initial / np.linalg.norm(stage_initial, axis=1)[:, None]
                if curvature_energy(normalized_initial, target_weights) > initial_curvature:
                    stage_initial = final_initial_frames.copy()
                    fixed_final_indices = self.keyframe_indices

            fixed_mask = np.isin(current_indices, fixed_final_indices)
            stage_frames, history = minimize(
                stage_initial,
                fixed_mask,
                self._curvature_weights(current_indices),
                self.config,
                iterations=0 if converged else budget,
                sample_indices=current_indices,
                solver=self.config.solver if len(current_indices) == len(final_initial_frames) else "gradient_descent",
            )
            _, gradient = curvature_energy_gradient(
                stage_frames, self._curvature_weights(current_indices), self.config.norm_penalty, current_indices
            )
            gradient[fixed_mask] = 0.0
            gradient_norms.append(float(np.linalg.norm(gradient)))
            histories.append(history)
            previous_indices = current_indices
            previous_frames = stage_frames

        if previous_frames is None:
            raise RuntimeError("SPRING refinement produced no stages")
        self.stage_gradient_norms = tuple(gradient_norms)
        self.converged = bool(converged or gradient_norms[-1] <= self.config.tolerance)
        return previous_frames, tuple(histories)

    def _check_time(self, t: float) -> float:
        if not np.isfinite(t):
            raise ValueError("Time must be finite")
        if t < self.t_min - _EPSILON or t > self.t_max + _EPSILON:
            raise ValueError(f"Time {t} outside valid range [{self.t_min}, {self.t_max}]")
        return float(np.clip(t, self.t_min, self.t_max))

    def evaluate(self, t: float) -> Quaternion:
        """Evaluate the optimized trajectory at time ``t``."""
        t = self._check_time(t)
        if t <= self.t_min:
            return self.samples[0].copy()
        if t >= self.t_max:
            return self.samples[-1].copy()

        upper = int(np.searchsorted(self.sample_times, t, side="right"))
        if upper >= len(self.samples):
            return self.samples[-1].copy()
        lower = upper - 1
        fraction = (t - self.sample_times[lower]) / (self.sample_times[upper] - self.sample_times[lower])
        return self.samples[lower].slerp(self.samples[upper], float(fraction)).unit()

    def evaluate_velocity(self, t: float) -> np.ndarray:
        """Estimate body angular velocity at time ``t`` in radians per second."""
        t = self._check_time(t)
        left = max(self.t_min, t - self._derivative_step)
        right = min(self.t_max, t + self._derivative_step)
        if right - left <= _EPSILON:
            return np.zeros(3)
        q_left = self.evaluate(left)
        q_right = self.evaluate(right)
        relative = q_left.inverse() * q_right
        if relative.s_ < 0.0:
            relative = -relative
        return 2.0 * relative.Log().v_ / (right - left)

    def evaluate_acceleration(self, t: float) -> np.ndarray:
        """Estimate body angular acceleration at time ``t`` in radians per second squared."""
        t = self._check_time(t)
        left = max(self.t_min, t - self._derivative_step)
        right = min(self.t_max, t + self._derivative_step)
        if right - left <= _EPSILON:
            return np.zeros(3)
        return (self.evaluate_velocity(right) - self.evaluate_velocity(left)) / (right - left)

    def generate_trajectory(self, num_points: int = 100) -> tuple[np.ndarray, list[Quaternion]]:
        """Evaluate ``num_points`` evenly spaced samples across the time range."""
        if num_points < _MIN_SAMPLES:
            raise ValueError("num_points must be at least 2")
        times = np.linspace(self.t_min, self.t_max, num_points)
        return times, [self.evaluate(float(t)) for t in times]


__all__ = ["SpringConfig", "SpringQuaternionInterpolation"]
