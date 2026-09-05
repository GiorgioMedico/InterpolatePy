"""Adapters for geometric path classes.

The C++ constructors use different parameter names from Python:
- ``LinearPath``: C++ ``pi, pf`` vs Python ``pi, pf`` (same)
- ``CircularPath``: C++ ``axis, axis_point, circle_point`` vs Python ``r, d, pi``

Adds ``evaluate_at()`` and ``all_traj()`` convenience methods that the pure-Python
classes provide but the C++ classes do not.
"""

from __future__ import annotations

import numpy as np

from interpolatepy._backend import get_cpp_module

_cpp = get_cpp_module()

_CppLinearPath = _cpp.path.LinearPath
_CppCircularPath = _cpp.path.CircularPath


class LinearPath(_CppLinearPath):  # type: ignore[valid-type, misc]
    """C++-backed LinearPath with Python-compatible constructor."""

    def __init__(self, pi: np.ndarray, pf: np.ndarray) -> None:
        self._pi = np.asarray(pi, dtype=float)
        self._pf = np.asarray(pf, dtype=float)
        super().__init__(self._pi, self._pf)
        diff = self._pf - self._pi
        length = float(np.linalg.norm(diff))
        self._tangent = diff / length if length > 0 else np.zeros(3)

    @property
    def pi(self) -> np.ndarray:
        """Initial point."""
        return self._pi

    @property
    def pf(self) -> np.ndarray:
        """Final point."""
        return self._pf

    @property
    def tangent(self) -> np.ndarray:
        """Unit tangent vector."""
        return self._tangent

    @property
    def total_length(self) -> float:
        """Alias for ``length`` property."""
        return self.length

    def evaluate_at(
        self, s_values: float | list[float] | np.ndarray
    ) -> dict[str, np.ndarray]:
        """Evaluate position, velocity, acceleration at arc-length values."""
        s_arr = np.atleast_1d(np.asarray(s_values, dtype=float))
        s_clipped = np.clip(s_arr, 0, self.length)
        n = len(s_clipped)
        positions = np.zeros((n, 3))
        for i, s in enumerate(s_clipped):
            positions[i] = self.position(float(s))
        velocities = np.tile(self.velocity(0.0), (n, 1))
        accelerations = np.zeros((n, 3))
        return {
            "position": positions,
            "velocity": velocities,
            "acceleration": accelerations,
            "s": s_clipped,
        }

    def all_traj(self, num_points: int = 100) -> dict[str, np.ndarray]:
        """Generate complete trajectory along the path."""
        s_values = np.linspace(0, self.length, num_points)
        return self.evaluate_at(s_values)


class CircularPath(_CppCircularPath):  # type: ignore[valid-type, misc]
    """C++-backed CircularPath with Python-compatible parameter names."""

    def __init__(self, r: np.ndarray, d: np.ndarray, pi: np.ndarray) -> None:
        self._r = np.asarray(r, dtype=float)
        self._d = np.asarray(d, dtype=float)
        self._pi = np.asarray(pi, dtype=float)
        super().__init__(
            axis=self._r,
            axis_point=self._d,
            circle_point=self._pi,
        )

    @property
    def r(self) -> np.ndarray:
        """Axis vector."""
        return self._r

    @property
    def d(self) -> np.ndarray:
        """Point on the axis."""
        return self._d

    @property
    def pi(self) -> np.ndarray:
        """Point on the circle."""
        return self._pi

    def evaluate_at(
        self, s_values: float | list[float] | np.ndarray
    ) -> dict[str, np.ndarray]:
        """Evaluate position, velocity, acceleration at arc-length values."""
        s_arr = np.atleast_1d(np.asarray(s_values, dtype=float))
        n = len(s_arr)
        positions = np.zeros((n, 3))
        velocities = np.zeros((n, 3))
        accelerations = np.zeros((n, 3))
        for i, s in enumerate(s_arr):
            positions[i] = self.position(float(s))
            velocities[i] = self.velocity(float(s))
            accelerations[i] = self.acceleration(float(s))
        return {
            "position": positions,
            "velocity": velocities,
            "acceleration": accelerations,
            "s": s_arr,
        }

    def all_traj(self, num_points: int = 100) -> dict[str, np.ndarray]:
        """Generate complete trajectory around the circle."""
        s_values = np.linspace(0, 2 * np.pi * self.radius, num_points)
        return self.evaluate_at(s_values)
