"""Adapters for the B-spline family.

Adds ``plot_2d()``, ``plot_3d()``, ``__repr__`` and plotting helpers that
exist only in the pure-Python implementations.
"""

from __future__ import annotations

from interpolatepy._backend import get_cpp_module
from interpolatepy.b_spline import BSpline as _PyBSpline
from interpolatepy.b_spline_interpolate import BSplineInterpolator as _PyBSplineInterpolator

_cpp = get_cpp_module()

_CppBSpline = _cpp.bspline.BSpline
_CppCubicBSplineInterpolation = _cpp.bspline.CubicBSplineInterpolation
_CppBSplineInterpolator = _cpp.bspline.BSplineInterpolator
_CppApproximationBSpline = _cpp.bspline.ApproximationBSpline
_CppSmoothingCubicBSpline = _cpp.bspline.SmoothingCubicBSpline


class BSpline(_CppBSpline):  # type: ignore[valid-type, misc]
    """C++-backed BSpline with Python plotting and repr."""

    # Class constants used by plot methods
    DIM_2 = 2
    DIM_3 = 3

    plot_2d = _PyBSpline.plot_2d
    plot_3d = _PyBSpline.plot_3d
    __repr__ = _PyBSpline.__repr__


class CubicBSplineInterpolation(_CppCubicBSplineInterpolation):  # type: ignore[valid-type, misc]
    """C++-backed CubicBSplineInterpolation."""

    DIM_2 = 2
    DIM_3 = 3

    plot_2d = _PyBSpline.plot_2d
    plot_3d = _PyBSpline.plot_3d
    __repr__ = _PyBSpline.__repr__


class BSplineInterpolator(_CppBSplineInterpolator):  # type: ignore[valid-type, misc]
    """C++-backed BSplineInterpolator with plotting helpers."""

    DIM_2 = 2
    DIM_3 = 3

    plot_2d = _PyBSpline.plot_2d
    plot_3d = _PyBSpline.plot_3d
    plot_with_points = _PyBSplineInterpolator.plot_with_points
    plot_with_points_3d = _PyBSplineInterpolator.plot_with_points_3d
    __repr__ = _PyBSpline.__repr__


class ApproximationBSpline(_CppApproximationBSpline):  # type: ignore[valid-type, misc]
    """C++-backed ApproximationBSpline."""

    DIM_2 = 2
    DIM_3 = 3

    plot_2d = _PyBSpline.plot_2d
    plot_3d = _PyBSpline.plot_3d
    __repr__ = _PyBSpline.__repr__


class SmoothingCubicBSpline(_CppSmoothingCubicBSpline):  # type: ignore[valid-type, misc]
    """C++-backed SmoothingCubicBSpline."""

    DIM_2 = 2
    DIM_3 = 3

    plot_2d = _PyBSpline.plot_2d
    plot_3d = _PyBSpline.plot_3d
    __repr__ = _PyBSpline.__repr__
