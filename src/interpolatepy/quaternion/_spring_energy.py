"""Reusable SPRING stencils and lazy gradients for backtracking trials."""

from __future__ import annotations

import numpy as np


class SpringEnergy:
    """Cache the fixed grid coefficients for one minimization stage."""

    def __init__(
        self,
        sample_count: int,
        weights: np.ndarray,
        norm_penalty: float,
        sample_indices: np.ndarray | None = None,
    ) -> None:
        self.weights = weights
        self.norm_penalty = norm_penalty
        spacing = np.ones(sample_count - 1) if sample_indices is None else np.diff(sample_indices).astype(np.float64)
        spacing /= np.mean(spacing)
        left = 2.0 / (spacing[:-1] * (spacing[:-1] + spacing[1:]))
        right = 2.0 / (spacing[1:] * (spacing[:-1] + spacing[1:]))
        self.coefficients = (left, -(left + right), right)

    def evaluate(self, frames: np.ndarray) -> EnergyEvaluation:
        """Evaluate energy, retaining intermediates for an optional gradient."""
        return EnergyEvaluation(self, frames)


class EnergyEvaluation:
    """A trial's energy; compute its gradient only if the trial needs it.

    The model and frames must remain unchanged until ``gradient`` is called.
    No buffers are shared between evaluations.
    """

    def __init__(self, model: SpringEnergy, frames: np.ndarray) -> None:
        self.model = model
        self.frames = frames
        left, center, right = model.coefficients
        second_difference = left[:, None] * frames[:-2] + center[:, None] * frames[1:-1] + right[:, None] * frames[2:]
        centers = frames[1:-1]
        norm_squared = np.einsum("ij,ij->i", centers, centers)
        self.projection_scale = np.einsum("ij,ij->i", second_difference, centers) / norm_squared
        self.curvature = second_difference - self.projection_scale[:, None] * centers
        self.residual = np.einsum("ij,ij->i", frames, frames) - 1.0
        curvature_energy = float(np.sum(model.weights * np.einsum("ij,ij->i", self.curvature, self.curvature)))
        penalty_energy = float(model.norm_penalty * np.dot(self.residual, self.residual))
        self.energy = curvature_energy + penalty_energy

    def gradient(self) -> np.ndarray:
        """Reverse accumulation through the unchanged curvature and norm residuals."""
        left, center, right = self.model.coefficients
        weighted_curvature = self.model.weights[:, None] * self.curvature
        gradient = np.zeros_like(self.frames)
        gradient[:-2] += 2.0 * left[:, None] * weighted_curvature
        gradient[1:-1] += 2.0 * center[:, None] * weighted_curvature
        gradient[1:-1] += -2.0 * self.projection_scale[:, None] * weighted_curvature
        gradient[2:] += 2.0 * right[:, None] * weighted_curvature
        gradient += 4.0 * self.model.norm_penalty * self.residual[:, None] * self.frames
        return gradient
