"""Internal phase planner for Double-S trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .double_s import DoubleSTrajectory

EPSILON = 1e-6
MIN_GAMMA = 0.01
MAX_ITERATIONS = 50


@dataclass(frozen=True)
class _PlanCandidate:
    acceleration_max: float
    acceleration_min: float
    jerk_time_1: float
    jerk_time_2: float
    acceleration_time: float
    deceleration_time: float


def plan_trajectory(trajectory: DoubleSTrajectory) -> None:
    """Populate all phase parameters on a Double-S trajectory."""
    initial_position = trajectory.state.q_0
    final_position = trajectory.state.q_1
    initial_velocity = trajectory.state.v_0
    final_velocity = trajectory.state.v_1

    if np.isclose(final_position, initial_position):
        if np.isclose(final_velocity, initial_velocity):
            trajectory.T = 0.0
            return
        minimum_time = abs(final_velocity - initial_velocity) / trajectory.bounds.a_bound
        trajectory.T = max(minimum_time * 1.5, 0.1)
        return

    _transform_parameters(trajectory)
    _compute_initial_phase_times(trajectory)
    _compute_constant_velocity_time(trajectory)
    if trajectory.Tv < 0:
        trajectory.Tv = 0
        _fit_reduced_acceleration(trajectory)
    _finalize_plan(trajectory)


def _transform_parameters(trajectory: DoubleSTrajectory) -> None:
    """Transform state and limits so planning proceeds in one direction."""
    trajectory.sigma = np.sign(trajectory.state.q_1 - trajectory.state.q_0)
    sigma = trajectory.sigma
    bounds = trajectory.bounds
    trajectory.q_0_transformed = sigma * trajectory.state.q_0
    trajectory.q_1_transformed = sigma * trajectory.state.q_1
    trajectory.v_0_transformed = sigma * trajectory.state.v_0
    trajectory.v_1_transformed = sigma * trajectory.state.v_1
    trajectory.v_max = (sigma + 1) / 2 * bounds.v_bound + (sigma - 1) / 2 * (-bounds.v_bound)
    trajectory.v_min = (sigma + 1) / 2 * (-bounds.v_bound) + (sigma - 1) / 2 * bounds.v_bound
    trajectory.a_max = (sigma + 1) / 2 * bounds.a_bound + (sigma - 1) / 2 * (-bounds.a_bound)
    trajectory.a_min = (sigma + 1) / 2 * (-bounds.a_bound) + (sigma - 1) / 2 * bounds.a_bound
    trajectory.j_max = (sigma + 1) / 2 * bounds.j_bound + (sigma - 1) / 2 * (-bounds.j_bound)
    trajectory.j_min = (sigma + 1) / 2 * (-bounds.j_bound) + (sigma - 1) / 2 * bounds.j_bound


def _compute_initial_phase_times(trajectory: DoubleSTrajectory) -> None:
    """Compute acceleration and deceleration times at the requested limits."""
    if ((trajectory.v_max - trajectory.v_0_transformed) * trajectory.j_max) < trajectory.a_max**2:
        trajectory.Tj_1 = np.sqrt(max((trajectory.v_max - trajectory.v_0_transformed) / trajectory.j_max, 0))
        trajectory.Ta = 2 * trajectory.Tj_1
    else:
        trajectory.Tj_1 = trajectory.a_max / trajectory.j_max
        trajectory.Ta = trajectory.Tj_1 + (trajectory.v_max - trajectory.v_0_transformed) / trajectory.a_max

    if ((trajectory.v_max - trajectory.v_1_transformed) * trajectory.j_max) < trajectory.a_max**2:
        trajectory.Tj_2 = np.sqrt(max((trajectory.v_max - trajectory.v_1_transformed) / trajectory.j_max, 0))
        trajectory.Td = 2 * trajectory.Tj_2
    else:
        trajectory.Tj_2 = trajectory.a_max / trajectory.j_max
        trajectory.Td = trajectory.Tj_2 + (trajectory.v_max - trajectory.v_1_transformed) / trajectory.a_max


def _compute_constant_velocity_time(trajectory: DoubleSTrajectory) -> None:
    """Compute the duration of the constant-velocity phase."""
    if abs(trajectory.v_max) < EPSILON:
        trajectory.Tv = 0
        return
    trajectory.Tv = (
        (trajectory.q_1_transformed - trajectory.q_0_transformed) / trajectory.v_max
        - trajectory.Ta / 2 * (1 + trajectory.v_0_transformed / trajectory.v_max)
        - trajectory.Td / 2 * (1 + trajectory.v_1_transformed / trajectory.v_max)
    )


def _fit_reduced_acceleration(trajectory: DoubleSTrajectory) -> None:
    """Find feasible acceleration limits when maximum velocity is unreachable."""
    gamma_high = 1.0
    gamma_low = MIN_GAMMA
    gamma_mid = 0.5
    iteration = 0

    while iteration < MAX_ITERATIONS:
        iteration += 1
        gamma_mid = (gamma_high + gamma_low) / 2
        acceleration_max = gamma_mid * trajectory.bounds.a_bound
        acceleration_min = -gamma_mid * trajectory.bounds.a_bound
        jerk_time = acceleration_max / trajectory.j_max
        delta = (
            acceleration_max**4 / trajectory.j_max**2
            + 2 * (trajectory.v_0_transformed**2 + trajectory.v_1_transformed**2)
            + acceleration_max
            * (
                4 * (trajectory.q_1_transformed - trajectory.q_0_transformed)
                - 2 * acceleration_max / trajectory.j_max * (trajectory.v_0_transformed + trajectory.v_1_transformed)
            )
        )
        if delta < 0:
            gamma_high = gamma_mid
            continue

        acceleration_time = (
            acceleration_max**2 / trajectory.j_max - 2 * trajectory.v_0_transformed + np.sqrt(delta)
        ) / (2 * acceleration_max)
        deceleration_time = (
            acceleration_max**2 / trajectory.j_max - 2 * trajectory.v_1_transformed + np.sqrt(delta)
        ) / (2 * acceleration_max)

        if acceleration_time < 0:
            candidate = _zero_acceleration_phase(trajectory, acceleration_time, deceleration_time)
        elif deceleration_time < 0:
            candidate = _zero_deceleration_phase(trajectory, acceleration_time, deceleration_time)
        elif acceleration_time > 2 * jerk_time and deceleration_time > 2 * jerk_time:
            _apply_candidate(
                trajectory,
                _PlanCandidate(
                    acceleration_max,
                    acceleration_min,
                    jerk_time,
                    jerk_time,
                    acceleration_time,
                    deceleration_time,
                ),
            )
            break
        else:
            gamma_high = gamma_mid
            continue

        jerk_time_1, jerk_time_2, acceleration_time, deceleration_time = candidate
        if jerk_time_1 >= 0 and jerk_time_2 >= 0 and acceleration_time >= 0 and deceleration_time >= 0:
            _apply_candidate(
                trajectory,
                _PlanCandidate(
                    acceleration_max,
                    acceleration_min,
                    jerk_time_1,
                    jerk_time_2,
                    acceleration_time,
                    deceleration_time,
                ),
            )
            break
        gamma_high = gamma_mid


def _zero_acceleration_phase(
    trajectory: DoubleSTrajectory,
    acceleration_time: float,
    deceleration_time: float,
) -> tuple[float, float, float, float]:
    """Construct a candidate with no acceleration phase."""
    velocity_sum = trajectory.v_1_transformed + trajectory.v_0_transformed
    if abs(velocity_sum) < EPSILON:
        return 0.0, 0.0, 0.0, 0.0
    acceleration_time = 0
    deceleration_time = 2 * (trajectory.q_1_transformed - trajectory.q_0_transformed) / velocity_sum
    argument = trajectory.j_max * (trajectory.q_1_transformed - trajectory.q_0_transformed) - np.sqrt(
        trajectory.j_max
        * (
            trajectory.j_max * (trajectory.q_1_transformed - trajectory.q_0_transformed) ** 2
            + velocity_sum**2 * (trajectory.v_1_transformed - trajectory.v_0_transformed)
        )
    )
    jerk_time_2 = argument / (trajectory.j_max * velocity_sum) if abs(argument) > EPSILON else 0
    return 0.0, jerk_time_2, acceleration_time, deceleration_time


def _zero_deceleration_phase(
    trajectory: DoubleSTrajectory,
    acceleration_time: float,
    deceleration_time: float,
) -> tuple[float, float, float, float]:
    """Construct a candidate with no deceleration phase."""
    velocity_sum = trajectory.v_1_transformed + trajectory.v_0_transformed
    if abs(velocity_sum) < EPSILON:
        return 0.0, 0.0, 0.0, 0.0
    deceleration_time = 0
    acceleration_time = 2 * (trajectory.q_1_transformed - trajectory.q_0_transformed) / velocity_sum
    argument = trajectory.j_max * (trajectory.q_1_transformed - trajectory.q_0_transformed) - np.sqrt(
        trajectory.j_max
        * (
            trajectory.j_max * (trajectory.q_1_transformed - trajectory.q_0_transformed) ** 2
            - velocity_sum**2 * (trajectory.v_1_transformed - trajectory.v_0_transformed)
        )
    )
    jerk_time_1 = argument / (trajectory.j_max * velocity_sum) if abs(argument) > EPSILON else 0
    return jerk_time_1, 0.0, acceleration_time, deceleration_time


def _apply_candidate(
    trajectory: DoubleSTrajectory,
    candidate: _PlanCandidate,
) -> None:
    """Apply a feasible reduced-acceleration candidate."""
    trajectory.a_max = candidate.acceleration_max
    trajectory.a_min = candidate.acceleration_min
    trajectory.Tj_1 = candidate.jerk_time_1
    trajectory.Tj_2 = candidate.jerk_time_2
    trajectory.Ta = candidate.acceleration_time
    trajectory.Td = candidate.deceleration_time


def _finalize_plan(trajectory: DoubleSTrajectory) -> None:
    """Normalize phase durations and calculate derived limits."""
    trajectory.a_lim_a = trajectory.j_max * trajectory.Tj_1
    trajectory.a_lim_d = -trajectory.j_max * trajectory.Tj_2
    trajectory.Ta = max(trajectory.Ta, 0)
    trajectory.Td = max(trajectory.Td, 0)
    trajectory.Tv = max(trajectory.Tv, 0)
    trajectory.Tj_1 = max(trajectory.Tj_1, 0)
    trajectory.Tj_2 = max(trajectory.Tj_2, 0)

    if trajectory.Ta <= trajectory.Tj_1:
        trajectory.v_lim = trajectory.v_0_transformed + trajectory.j_max * trajectory.Ta**2 / 2
    else:
        trajectory.v_lim = trajectory.v_0_transformed + (trajectory.Ta - trajectory.Tj_1) * trajectory.a_lim_a
    trajectory.T = round((trajectory.Ta + trajectory.Tv + trajectory.Td) * 1000) / 1000


__all__ = ["EPSILON", "MAX_ITERATIONS", "MIN_GAMMA", "plan_trajectory"]
