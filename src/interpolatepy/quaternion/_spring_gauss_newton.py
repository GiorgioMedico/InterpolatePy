"""Banded Gauss-Newton models for the unchanged discrete SPRING energy."""

from __future__ import annotations

import numpy as np
from scipy.linalg import LinAlgError
from scipy.linalg import solveh_banded

from ._spring_energy import SpringEnergy

_COMPONENTS = 4
_BAND_ROWS = 12


def normal_matrix(  # noqa: PLR0913
    frames: np.ndarray,
    weights: np.ndarray,
    norm_penalty: float,
    fixed_mask: np.ndarray,
    sample_indices: np.ndarray | None,
    coefficients: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Return ``2 J.T J`` in lower-banded storage, with fixed increments zero.

    Residuals are weighted tangential second differences and
    ``sqrt(norm_penalty) * (q.q - 1)``: their squared norm is exactly the
    existing SPRING energy, including its off-sphere norm penalty.
    Each curvature residual touches three consecutive four-component frames,
    so the normal matrix has only eleven lower diagonals.
    """
    if coefficients is None:
        coefficients = SpringEnergy(len(frames), weights, norm_penalty, sample_indices).coefficients
    left, center, right = coefficients
    q = frames[1:-1]
    squared_norm = np.sum(q * q, axis=1)
    difference = left[:, None] * frames[:-2] + center[:, None] * q + right[:, None] * frames[2:]
    radial = np.sum(q * difference, axis=1) / squared_norm
    curvature = difference - radial[:, None] * q
    projection = np.eye(_COMPONENTS) - q[:, :, None] * q[:, None, :] / squared_norm[:, None, None]
    jacobians = (
        left[:, None, None] * projection,
        (center - radial)[:, None, None] * projection
        - q[:, :, None] * curvature[:, None, :] / squared_norm[:, None, None],
        right[:, None, None] * projection,
    )
    diagonal = 8.0 * norm_penalty * frames[:, :, None] * frames[:, None, :]
    first = np.zeros((len(frames) - 1, _COMPONENTS, _COMPONENTS))
    second = np.zeros((len(frames) - 2, _COMPONENTS, _COMPONENTS))
    for index, jacobian in enumerate(jacobians):
        diagonal[index : index + len(q)] += 2.0 * weights[:, None, None] * (jacobian.transpose(0, 2, 1) @ jacobian)
    first[:-1] += 2.0 * weights[:, None, None] * (jacobians[1].transpose(0, 2, 1) @ jacobians[0])
    first[1:] += 2.0 * weights[:, None, None] * (jacobians[2].transpose(0, 2, 1) @ jacobians[1])
    second += 2.0 * weights[:, None, None] * (jacobians[2].transpose(0, 2, 1) @ jacobians[0])
    diagonal[fixed_mask] = np.eye(_COMPONENTS)
    first[fixed_mask[:-1] | fixed_mask[1:]] = 0.0
    second[fixed_mask[:-2] | fixed_mask[2:]] = 0.0
    band = np.zeros((_BAND_ROWS, _COMPONENTS * len(frames)))
    for offset, blocks in enumerate((diagonal, first, second)):
        columns = _COMPONENTS * np.arange(len(blocks))
        for row in range(_COMPONENTS):
            for column in range(_COMPONENTS):
                distance = _COMPONENTS * offset + row - column
                if distance >= 0:
                    band[distance, columns + column] = blocks[:, row, column]
    return band


def descent_direction(band: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    """Solve the banded model, adding damping only if factorization fails."""
    scale = max(1.0, float(np.max(band[0])))
    damping = 0.0
    for attempt in range(8):
        candidate = band.copy(order="F")
        candidate[0] += damping
        try:
            direction = solveh_banded(
                candidate, -gradient.ravel(), lower=True, overwrite_ab=True, overwrite_b=True
            ).reshape(gradient.shape)
        except LinAlgError:
            damping = scale * 1e-12 if attempt == 0 else damping * 10.0
            continue
        if np.all(np.isfinite(direction)) and np.sum(direction * gradient) < 0.0:
            return direction
        damping = scale * 1e-12 if attempt == 0 else damping * 10.0
    return -gradient / np.linalg.norm(gradient)
