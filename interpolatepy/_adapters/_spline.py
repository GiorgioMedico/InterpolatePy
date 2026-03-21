"""Adapters for the cubic-spline family.

Each class subclasses the C++ pybind11 class and adds Python-only convenience
methods (``plot()``) and property aliases where attribute names differ between
the Python and C++ implementations.
"""

from __future__ import annotations

from typing import Any

from interpolatepy._backend import get_cpp_module
from interpolatepy.cubic_spline import CubicSpline as _PyCubicSpline

_cpp = get_cpp_module()

_CppCubicSpline = _cpp.CubicSpline
_CppCubicSmoothingSpline = _cpp.CubicSmoothingSpline
_CppCubicSplineWithAcc1 = _cpp.CubicSplineWithAcceleration1
_CppCubicSplineWithAcc2 = _cpp.CubicSplineWithAcceleration2


class CubicSpline(_CppCubicSpline):  # type: ignore[valid-type, misc]
    """C++-backed CubicSpline with Python ``plot()`` method."""

    plot = _PyCubicSpline.plot

    @property
    def n(self) -> int:
        """Number of polynomial segments (alias for ``n_segments``)."""
        return self.n_segments


class CubicSmoothingSpline(_CppCubicSmoothingSpline):  # type: ignore[valid-type, misc]
    """C++-backed CubicSmoothingSpline with Python property aliases."""

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
