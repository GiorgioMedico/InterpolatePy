"""Private numerical machinery for SPRING quaternion interpolation."""

from __future__ import annotations

from heapq import heappop
from heapq import heappush
from itertools import pairwise
from typing import Protocol

import numpy as np


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
    gradient = np.zeros_like(frames)
    if sample_indices is None:
        spacing = np.ones(len(frames) - 1)
    else:
        spacing = np.diff(sample_indices).astype(np.float64)
        spacing /= np.mean(spacing)
    left_coefficient = 2.0 / (spacing[:-1] * (spacing[:-1] + spacing[1:]))
    right_coefficient = 2.0 / (spacing[1:] * (spacing[:-1] + spacing[1:]))
    center_coefficient = -(left_coefficient + right_coefficient)
    second_difference = (
        left_coefficient[:, None] * frames[:-2]
        + center_coefficient[:, None] * frames[1:-1]
        + right_coefficient[:, None] * frames[2:]
    )
    centers = frames[1:-1]
    norm_squared = np.einsum("ij,ij->i", centers, centers)
    projection_scale = np.einsum("ij,ij->i", second_difference, centers) / norm_squared
    curvature = second_difference - projection_scale[:, None] * centers
    weighted_curvature = curvature_weights[:, None] * curvature

    # Reverse accumulation through the spacing-aware second difference and
    # kappa = q'' - ((q'' . q) / (q . q)) q.
    gradient[:-2] += 2.0 * left_coefficient[:, None] * weighted_curvature
    gradient[1:-1] += 2.0 * center_coefficient[:, None] * weighted_curvature
    gradient[1:-1] += -2.0 * projection_scale[:, None] * weighted_curvature
    gradient[2:] += 2.0 * right_coefficient[:, None] * weighted_curvature

    residual = np.einsum("ij,ij->i", frames, frames) - 1.0
    gradient += 4.0 * norm_penalty * residual[:, None] * frames

    curvature_energy = float(np.sum(curvature_weights * np.einsum("ij,ij->i", curvature, curvature)))
    penalty_energy = float(norm_penalty * np.dot(residual, residual))
    return curvature_energy + penalty_energy, gradient


def curvature_energy(frames: np.ndarray, curvature_weights: np.ndarray) -> float:
    """Return the weighted discrete tangential-curvature energy."""
    energy, _ = curvature_energy_gradient(frames, curvature_weights, 0.0)
    return energy


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
) -> tuple[np.ndarray, tuple[float, ...]]:
    """Minimize the SPRING energy with normalized-gradient backtracking."""
    frames = initial_frames.copy()
    energy, gradient = curvature_energy_gradient(frames, curvature_weights, settings.norm_penalty, sample_indices)
    history = [energy]

    for _ in range(iterations):
        gradient[fixed_mask] = 0.0
        gradient_norm = float(np.linalg.norm(gradient))
        if gradient_norm <= settings.tolerance:
            break

        direction = gradient / gradient_norm
        step = settings.step_size
        accepted = False
        for _ in range(_MAX_BACKTRACKS):
            candidate = frames - step * direction
            candidate[fixed_mask] = initial_frames[fixed_mask]
            candidate_energy, candidate_gradient = curvature_energy_gradient(
                candidate, curvature_weights, settings.norm_penalty, sample_indices
            )
            if candidate_energy < energy:
                frames = candidate
                energy = candidate_energy
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
