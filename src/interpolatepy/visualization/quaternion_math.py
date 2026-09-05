"""Projection and metric helpers for quaternion visualizations."""

from __future__ import annotations

import numpy as np

from interpolatepy.quaternion.core import Quaternion

_MIN_QUATERNIONS = 2


def stereographic_projection(q: Quaternion, singularity_threshold: float) -> np.ndarray:
    """Project a unit quaternion to Modified Rodrigues Parameters."""
    q_unit = q.unit()
    if abs(q_unit.w + 1.0) < singularity_threshold:
        q_unit = -q_unit
    if abs(q_unit.w + 1.0) < singularity_threshold:
        raise ValueError("Quaternion is too close to singularity at w = -1. Cannot perform stereographic projection.")

    denominator = 1.0 + q_unit.w
    return np.array([q_unit.x / denominator, q_unit.y / denominator, q_unit.z / denominator])


def inverse_stereographic_projection(mrp: np.ndarray) -> Quaternion:
    """Convert Modified Rodrigues Parameters back to a unit quaternion."""
    mrp_norm_squared = np.dot(mrp, mrp)
    scalar = (1 - mrp_norm_squared) / (1 + mrp_norm_squared)
    vector_scale = 2 / (1 + mrp_norm_squared)
    x, y, z = vector_scale * mrp
    return Quaternion(scalar, x, y, z)


def project_trajectory(quaternions: list[Quaternion], singularity_threshold: float) -> np.ndarray:
    """Project all nonsingular quaternions in a trajectory to three dimensions."""
    projected_points = []
    for quaternion in quaternions:
        try:
            projected_points.append(stereographic_projection(quaternion, singularity_threshold))
        except ValueError as error:
            print(f"Warning: Skipping quaternion due to singularity: {error}")
    return np.array(projected_points) if projected_points else np.empty((0, 3))


def quaternion_distance(first: Quaternion, second: Quaternion) -> float:
    """Return the norm of the difference between two quaternions."""
    return (first - second).norm()


def velocity_magnitudes(
    quaternions: list[Quaternion], time_points: list[float] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Compute neighboring quaternion-distance averages along a trajectory."""
    if len(quaternions) < _MIN_QUATERNIONS:
        raise ValueError("Need at least 2 quaternions to compute velocity")

    count = len(quaternions)
    velocities = np.zeros(count)
    velocities[0] = quaternion_distance(quaternions[1], quaternions[0])
    for index in range(1, count - 1):
        distance_backward = quaternion_distance(quaternions[index], quaternions[index - 1])
        distance_forward = quaternion_distance(quaternions[index + 1], quaternions[index])
        velocities[index] = (distance_backward + distance_forward) / 2.0
    velocities[-1] = quaternion_distance(quaternions[-1], quaternions[-2])

    if time_points is None:
        times = np.arange(count, dtype=float)
    else:
        if len(time_points) != count:
            raise ValueError("Time points length must match quaternions length")
        times = np.array(time_points)
    return times, velocities


__all__ = [
    "inverse_stereographic_projection",
    "project_trajectory",
    "quaternion_distance",
    "stereographic_projection",
    "velocity_magnitudes",
]
