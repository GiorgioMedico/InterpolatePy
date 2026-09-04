"""Adapters for the B-spline family.

Adds ``plot_2d()``, ``plot_3d()``, ``__repr__`` and plotting helpers that
exist only in the pure-Python implementations.
"""

from __future__ import annotations

import numpy as np

from interpolatepy._backend import get_cpp_module
from interpolatepy.b_spline import BSpline as _PyBSpline
from interpolatepy.b_spline_interpolate import BSplineInterpolator as _PyBSplineInterpolator
from interpolatepy.b_spline_smooth import BSplineParams as _PyBSplineParams

_cpp = get_cpp_module()

_CppBSpline = _cpp.bspline.BSpline
_CppCubicBSplineInterpolation = _cpp.bspline.CubicBSplineInterpolation
_CppBSplineInterpolator = _cpp.bspline.BSplineInterpolator
_CppApproximationBSpline = _cpp.bspline.ApproximationBSpline
_CppSmoothingCubicBSpline = _cpp.bspline.SmoothingCubicBSpline
_CppBSplineParams = _cpp.bspline.BSplineParams


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

    def __init__(  # noqa: PLR0913
        self,
        degree: int,
        points: list | np.ndarray,
        times: list | np.ndarray | None = None,
        initial_velocity: float | list | np.ndarray | None = None,
        final_velocity: float | list | np.ndarray | None = None,
        initial_acceleration: float | list | np.ndarray | None = None,
        final_acceleration: float | list | np.ndarray | None = None,
        cyclic: bool = False,
    ) -> None:
        """Normalize Python scalar and vector inputs for the native constructor."""
        point_array = np.asarray(points, dtype=np.float64)
        if point_array.ndim == 1:
            point_array = point_array.reshape(-1, 1)

        time_array = None if times is None else np.asarray(times, dtype=np.float64)

        def as_vector(
            value: float | list | np.ndarray | None,
        ) -> np.ndarray | None:
            if value is None:
                return None
            return np.atleast_1d(np.asarray(value, dtype=np.float64))

        super().__init__(
            degree,
            point_array,
            time_array,
            as_vector(initial_velocity),
            as_vector(final_velocity),
            as_vector(initial_acceleration),
            as_vector(final_acceleration),
            cyclic,
        )

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

    def __init__(
        self,
        points: list | np.ndarray,
        params: _PyBSplineParams | None = None,
    ) -> None:
        if params is None:
            params = _PyBSplineParams()

        method_map = {
            "equally_spaced": _cpp.bspline.Parameterization.EquallySpaced,
            "chord_length": _cpp.bspline.Parameterization.ChordLength,
            "centripetal": _cpp.bspline.Parameterization.Centripetal,
        }
        if params.method not in method_map:
            raise ValueError(f"Unknown parameterization method: {params.method}")

        cpp_params = _CppBSplineParams()
        cpp_params.mu = params.mu
        cpp_params.weights = (
            None
            if params.weights is None
            else np.asarray(params.weights, dtype=np.float64)
        )
        cpp_params.v0 = (
            None if params.v0 is None else np.asarray(params.v0, dtype=np.float64)
        )
        cpp_params.vn = (
            None if params.vn is None else np.asarray(params.vn, dtype=np.float64)
        )
        cpp_params.method = method_map[params.method]
        cpp_params.enforce_endpoints = params.enforce_endpoints
        cpp_params.auto_derivatives = params.auto_derivatives

        point_array = np.asarray(points, dtype=np.float64)
        if point_array.ndim == 1:
            point_array = point_array.reshape(-1, 1)
        super().__init__(point_array, cpp_params)

    plot_2d = _PyBSpline.plot_2d
    plot_3d = _PyBSpline.plot_3d
    __repr__ = _PyBSpline.__repr__
