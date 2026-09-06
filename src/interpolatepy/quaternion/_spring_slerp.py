"""Batch the existing log/exp SLERP without changing its small-angle rules."""

from __future__ import annotations

import numpy as np

from .core import Quaternion


def slerp_segments(quaternions: list[Quaternion], segments: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    """Interpolate many samples, computing each adjacent pair's log only once.

    Fractions lie in [0, 1]. Endpoints are copied verbatim, including their
    signs; interior points use the same shortest-arc choice and first-order
    small-angle exponential as ``Quaternion.slerp``.
    """
    values = np.array([[q.s_, *q.v_] for q in quaternions])
    result = values[segments].copy()
    at_end = fractions == 1.0  # noqa: RUF069 -- preserve exact SLERP endpoint semantics
    result[at_end] = values[segments[at_end] + 1]
    interior = (fractions > 0.0) & ~at_end
    if not np.any(interior):
        return result

    logs = np.empty((len(quaternions) - 1, 3))
    for segment in np.unique(segments[interior]):
        start, end = quaternions[segment], quaternions[segment + 1]
        relative = start.i() * (end if start.dot_prod(end) >= 0.0 else -end)
        logs[segment] = relative.Log().v_
    vectors = logs[segments[interior]] * fractions[interior, None]
    # Batched three-vector dot products preserve the scalar norm's ordering.
    theta = np.sqrt((vectors[:, None, :] @ vectors[:, :, None])[:, 0, 0])
    scalar = np.ones(len(theta))
    regular = theta >= Quaternion.EPSILON
    scalar[regular] = np.cos(theta[regular])
    vectors[regular] = vectors[regular] * np.sin(theta[regular, None]) / theta[regular, None]
    starts = values[segments[interior]]
    result[interior, 0] = starts[:, 0] * scalar - (starts[:, None, 1:] @ vectors[:, :, None])[:, 0, 0]
    result[interior, 1:] = starts[:, :1] * vectors + scalar[:, None] * starts[:, 1:] + np.cross(starts[:, 1:], vectors)
    return result
