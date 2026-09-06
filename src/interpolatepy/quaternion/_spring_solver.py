"""Private numerical machinery for SPRING quaternion interpolation."""

from __future__ import annotations

from heapq import heappop
from heapq import heappush
from itertools import pairwise
from typing import Protocol

import numpy as np

from ._spring_energy import SpringEnergy
from ._spring_gauss_newton import descent_direction
from ._spring_gauss_newton import normal_matrix


_MIN_BACKTRACK_STEP = 1e-12
_MAX_BACKTRACKS = 30
_REFINEMENT_FACTOR = 5


class MinimizationSettings(Protocol):
    """Structural subset of settings required by :func:`minimize`."""

    @property
    def step_size(self) -> float: ...

    @property
    def norm_penalty(self) -> float: ...

    @property
    def tolerance(self) -> float: ...


def curvature_energy_gradient(
    frames: np.ndarray,
    curvature_weights: np.ndarray,
    norm_penalty: float,
    sample_indices: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Return the discrete curvature energy and its gradient.

    Refinement indices are measured in units of their mean spacing, keeping
    the original unit-spacing stencil on uniform grids. Uneven grids use
    second divided differences before removing the radial component.
    """
    trial = SpringEnergy(len(frames), curvature_weights, norm_penalty, sample_indices).evaluate(frames)
    return trial.energy, trial.gradient()


def curvature_energy(frames: np.ndarray, curvature_weights: np.ndarray) -> float:
    """Return the weighted discrete tangential-curvature energy."""
    return SpringEnergy(len(frames), curvature_weights, 0.0).evaluate(frames).energy


def refinement_sample_counts(
    keyframe_count: int,
    final_sample_count: int,
    requested_levels: int,
) -> tuple[int, ...]:
    """Create the report's approximately fivefold coarse-to-fine schedule."""
    minimum_coarse_count = min(final_sample_count, 2 * keyframe_count - 1)
    descending_counts = [final_sample_count]
    for _ in range(requested_levels - 1):
        next_count = max(
            minimum_coarse_count,
            round(descending_counts[-1] / _REFINEMENT_FACTOR),
        )
        if next_count >= descending_counts[-1]:
            break
        descending_counts.append(next_count)
    return tuple(reversed(descending_counts))


def iteration_budgets(total_iterations: int, stage_count: int) -> tuple[int, ...]:
    """Split a total budget with more work on earlier refinement levels.

    Three stages receive weights 4:3:2, matching the report's illustrative
    200/150/100 iteration schedule.
    """
    weights = np.arange(stage_count + 1, 1, -1, dtype=np.float64)
    exact = total_iterations * weights / np.sum(weights)
    budgets = np.floor(exact).astype(np.int64)
    remaining = total_iterations - int(np.sum(budgets))
    if remaining:
        order = np.argsort(-(exact - budgets), kind="stable")
        budgets[order[:remaining]] += 1
    return tuple(int(value) for value in budgets)


def nested_level_indices(
    final_sample_count: int,
    keyframe_indices: np.ndarray,
    sample_counts: tuple[int, ...],
) -> tuple[np.ndarray, ...]:
    """Select nested maximin subsets of the final discrete sample lattice."""
    selected = {int(index) for index in keyframe_indices}
    if not selected or min(selected) != 0 or max(selected) != final_sample_count - 1:
        raise ValueError("keyframe indices must include both ends of the final sample lattice")
    if any(count < len(selected) or count > final_sample_count for count in sample_counts):
        raise ValueError("refinement sample counts must fit the final sample lattice")
    levels: list[np.ndarray] = []
    intervals: list[tuple[int, int, int, int]] = []

    def add_interval(left: int, right: int) -> None:
        if right - left <= 1:
            return
        candidate = (left + right) // 2
        distance = min(candidate - left, right - candidate)
        heappush(intervals, (-distance, candidate, left, right))

    ordered_keyframes = sorted(selected)
    for left, right in pairwise(ordered_keyframes):
        add_interval(left, right)

    for sample_count in sample_counts:
        while len(selected) < sample_count:
            if not intervals:
                raise ValueError("sample count exceeds the final sample lattice")
            _, candidate, left, right = heappop(intervals)
            selected.add(candidate)
            add_interval(left, candidate)
            add_interval(candidate, right)
        levels.append(np.array(sorted(selected), dtype=np.int64))

    return tuple(levels)


def minimize(  # noqa: PLR0913
    initial_frames: np.ndarray,
    fixed_mask: np.ndarray,
    curvature_weights: np.ndarray,
    settings: MinimizationSettings,
    iterations: int,
    sample_indices: np.ndarray | None = None,
    solver: str = "gradient_descent",
) -> tuple[np.ndarray, tuple[float, ...]]:
    """Minimize the same SPRING energy with either descent or banded Gauss-Newton."""
    frames = initial_frames.copy()
    model = SpringEnergy(len(frames), curvature_weights, settings.norm_penalty, sample_indices)
    trial = model.evaluate(frames)
    energy, gradient = trial.energy, trial.gradient()
    fixed_frames = initial_frames[fixed_mask]
    free_mask = ~fixed_mask
    history = [energy]

    for _ in range(iterations):
        gradient[fixed_mask] = 0.0
        gradient_norm = float(np.linalg.norm(gradient))
        if gradient_norm <= settings.tolerance:
            break

        if solver == "gauss_newton":
            band = normal_matrix(
                frames, curvature_weights, settings.norm_penalty, fixed_mask, sample_indices, model.coefficients
            )
            direction = -descent_direction(band, gradient)
            lengths = np.linalg.norm(frames, axis=1)
            radial = np.sum(frames * direction, axis=1) / lengths
            tangent = direction - (radial / lengths)[:, None] * frames
            step = 1.0
        else:
            direction = gradient / gradient_norm
            step = settings.step_size
        accepted = False
        for _ in range(_MAX_BACKTRACKS):
            if solver == "gauss_newton":
                # Follow the sphere in tangential directions, but keep the
                # radial increment free. This is only a reparameterized line
                # search: the original off-sphere energy is still minimized.
                target_lengths = lengths - step * radial
                if np.any(target_lengths <= 0.0):
                    step *= 0.5
                    continue
                candidate = frames - step * tangent
                candidate *= (target_lengths / np.linalg.norm(candidate, axis=1))[:, None]
            else:
                candidate = frames - step * direction
            candidate[fixed_mask] = fixed_frames
            trial = model.evaluate(candidate)
            near_roundoff = solver == "gauss_newton" and abs(trial.energy - energy) <= 64.0 * np.finfo(float).eps * max(
                abs(energy), 1e-30
            )
            if trial.energy < energy or near_roundoff:
                candidate_gradient = trial.gradient()
                if trial.energy < energy or np.linalg.norm(candidate_gradient[free_mask]) < 0.5 * gradient_norm:
                    frames = candidate
                    energy = trial.energy
                    gradient = candidate_gradient
                    history.append(energy)
                    accepted = True
                    break
            step *= 0.5
            if step < _MIN_BACKTRACK_STEP:
                break
        if not accepted:
            break

    return frames, tuple(history)
