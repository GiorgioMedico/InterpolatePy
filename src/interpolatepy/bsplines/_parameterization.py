"""Shared parameterization strategies for B-spline data points."""

from __future__ import annotations

import numpy as np

_CENTRIPETAL_POWER = 0.5


def parameterize_points(points: np.ndarray, method: str) -> np.ndarray:
    """Return normalized parameters using spacing, chord, or centripetal length."""
    point_count = len(points)
    last_index = point_count - 1
    parameters = np.zeros(point_count, dtype=np.float64)
    parameters[-1] = 1.0

    if method == "equally_spaced":
        for index in range(1, last_index):
            parameters[index] = index / last_index
        return parameters

    if method == "chord_length":
        power = 1.0
    elif method == "centripetal":
        power = _CENTRIPETAL_POWER
    else:
        raise ValueError(f"Unknown method: {method}. Options are 'equally_spaced', 'chord_length', or 'centripetal'.")

    lengths = [float(np.linalg.norm(points[index] - points[index - 1])) ** power for index in range(1, point_count)]
    total_length = sum(lengths)
    for index in range(1, last_index):
        parameters[index] = parameters[index - 1] + lengths[index - 1] / total_length
    return parameters


__all__ = ["parameterize_points"]
