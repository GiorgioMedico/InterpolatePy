"""Adapters for the cubic-spline family.

Each class subclasses the C++ pybind11 class and adds Python-only convenience
methods (``plot()``) and property aliases where attribute names differ between
the Python and C++ implementations.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from interpolatepy._backend import get_cpp_module
from interpolatepy.splines.cubic import CubicSpline as _PyCubicSpline

_cpp = get_cpp_module()

_CppCubicSpline = _cpp.CubicSpline
_CppCubicSmoothingSpline = _cpp.CubicSmoothingSpline
_CppCubicSplineWithAcc1 = _cpp.CubicSplineWithAcceleration1
_CppCubicSplineWithAcc2 = _cpp.CubicSplineWithAcceleration2


def _plot_spline(spline: Any, num_points: int = 1000) -> None:
    """Plot a native spline through its common evaluation interface."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    t_points = np.asarray(spline.t_points)
    q_points = np.asarray(spline.q_points)
    times = np.linspace(t_points[0], t_points[-1], num_points)

    _figure, (position_ax, velocity_ax, acceleration_ax) = plt.subplots(
        3, 1, figsize=(10, 8), sharex=True
    )
    position_ax.plot(times, spline.evaluate(times), "b-", linewidth=2)
    position_ax.plot(t_points, q_points, "ro", markersize=8)
    position_ax.set_ylabel("Position")
    position_ax.grid(True)

    velocity_ax.plot(times, spline.evaluate_velocity(times), "g-", linewidth=2)
    velocity_ax.set_ylabel("Velocity")
    velocity_ax.grid(True)

    acceleration_ax.plot(times, spline.evaluate_acceleration(times), "r-", linewidth=2)
    acceleration_ax.set_ylabel("Acceleration")
    acceleration_ax.set_xlabel("Time")
    acceleration_ax.grid(True)
    plt.tight_layout()


class CubicSpline(_CppCubicSpline):  # type: ignore[valid-type, misc]
    """C++-backed CubicSpline with Python ``plot()`` method."""

    plot = _PyCubicSpline.plot

    @property
    def n(self) -> int:
        """Number of polynomial segments (alias for ``n_segments``)."""
        return self.n_segments


class CubicSmoothingSpline(_CppCubicSmoothingSpline):  # type: ignore[valid-type, misc]
    """C++-backed CubicSmoothingSpline with Python property aliases."""

    plot = _plot_spline

    @property
    def t(self) -> Any:
        """Alias mapping Python ``t`` to C++ ``t_points``."""
        return self.t_points

    @property
    def q(self) -> Any:
        """Alias mapping Python ``q`` to C++ ``q_points``."""
        return self.q_points

    @property
    def s(self) -> Any:
        """Alias mapping Python ``s`` to C++ ``s_points``."""
        return self.s_points


class CubicSplineWithAcceleration1(_CppCubicSplineWithAcc1):  # type: ignore[valid-type, misc]
    """C++-backed CubicSplineWithAcceleration1 with property aliases."""

    plot = _plot_spline

    @property
    def t(self) -> Any:
        """Alias mapping Python ``t`` to C++ ``t_points``."""
        return self.t_points

    @property
    def q(self) -> Any:
        """Alias mapping Python ``q`` to C++ ``q_points``."""
        return self.q_points


class CubicSplineWithAcceleration2(_CppCubicSplineWithAcc2):  # type: ignore[valid-type, misc]
    """C++-backed CubicSplineWithAcceleration2 with property aliases."""

    plot = _plot_spline
