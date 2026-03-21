"""Resolve the active backend and export the public API symbols.

This module contains the conditional import logic that selects between
C++-backed adapters and pure-Python implementations based on the
``HAS_CPP`` flag from ``_backend``.
"""

from __future__ import annotations

from ._backend import HAS_CPP

if HAS_CPP:
    # ── C++-backed classes and functions ─────────────────────────────
    from ._adapters import (
        # Spline
        CubicSpline,
        CubicSmoothingSpline,
        CubicSplineWithAcceleration1,
        CubicSplineWithAcceleration2,
        SplineParameters,
        SplineConfig,
        # B-spline
        BSpline,
        ApproximationBSpline,
        CubicBSplineInterpolation,
        BSplineInterpolator,
        BSplineParams,
        SmoothingCubicBSpline,
        # Motion profiles
        DoubleSTrajectory,
        StateParams,
        TrajectoryBounds,
        BoundaryCondition,
        PolynomialTrajectory,
        TimeInterval,
        TrapezoidalTrajectory,
        ParabolicBlendTrajectory,
        # Paths
        CircularPath,
        LinearPath,
        # Quaternion
        QuaternionSpline,
        SquadC2,
        LogQuaternionInterpolation,
        ModifiedLogQuaternionInterpolation,
        # Free functions
        solve_tridiagonal,
        smoothing_spline_with_tolerance,
        linear_traj,
        compute_trajectory_frames,
        circular_trajectory_with_derivatives,
        helicoidal_trajectory_with_derivatives,
    )
else:
    # ── Pure-Python fallback ─────────────────────────────────────────
    # Core spline algorithms
    from .cubic_spline import CubicSpline  # type: ignore[assignment]
    from .c_s_smoothing import CubicSmoothingSpline  # type: ignore[assignment]
    from .c_s_smoot_search import SplineConfig
    from .c_s_smoot_search import smoothing_spline_with_tolerance
    from .c_s_with_acc1 import CubicSplineWithAcceleration1  # type: ignore[assignment]
    from .c_s_with_acc2 import CubicSplineWithAcceleration2  # type: ignore[assignment]
    from .c_s_with_acc2 import SplineParameters

    # B-spline family
    from .b_spline import BSpline  # type: ignore[assignment]
    from .b_spline_approx import ApproximationBSpline  # type: ignore[assignment]
    from .b_spline_cubic import CubicBSplineInterpolation  # type: ignore[assignment]
    from .b_spline_interpolate import BSplineInterpolator  # type: ignore[assignment]
    from .b_spline_smooth import BSplineParams
    from .b_spline_smooth import SmoothingCubicBSpline  # type: ignore[assignment]

    # Motion profiles
    from .double_s import DoubleSTrajectory  # type: ignore[assignment]
    from .double_s import StateParams
    from .double_s import TrajectoryBounds
    from .polynomials import BoundaryCondition
    from .polynomials import PolynomialTrajectory  # type: ignore[assignment]
    from .polynomials import TimeInterval
    from .trapezoidal import TrapezoidalTrajectory  # type: ignore[assignment]

    # Path planning
    from .simple_paths import CircularPath  # type: ignore[assignment]
    from .simple_paths import LinearPath  # type: ignore[assignment]
    from .lin_poly_parabolic import ParabolicBlendTrajectory  # type: ignore[assignment]

    # Quaternion interpolation
    from .quat_spline import QuaternionSpline  # type: ignore[assignment]
    from .squad_c2 import SquadC2  # type: ignore[assignment]
    from .log_quat import LogQuaternionInterpolation  # type: ignore[assignment]
    from .log_quat import ModifiedLogQuaternionInterpolation  # type: ignore[assignment]

    # Free functions
    from .tridiagonal_inv import solve_tridiagonal
    from .linear import linear_traj
    from .frenet_frame import compute_trajectory_frames
    from .frenet_frame import circular_trajectory_with_derivatives
    from .frenet_frame import helicoidal_trajectory_with_derivatives

__all__ = [
    "ApproximationBSpline",
    "BSpline",
    "BSplineInterpolator",
    "BSplineParams",
    "BoundaryCondition",
    "CircularPath",
    "CubicBSplineInterpolation",
    "CubicSmoothingSpline",
    "CubicSpline",
    "CubicSplineWithAcceleration1",
    "CubicSplineWithAcceleration2",
    "DoubleSTrajectory",
    "LinearPath",
    "LogQuaternionInterpolation",
    "ModifiedLogQuaternionInterpolation",
    "ParabolicBlendTrajectory",
    "PolynomialTrajectory",
    "QuaternionSpline",
    "SmoothingCubicBSpline",
    "SplineConfig",
    "SplineParameters",
    "SquadC2",
    "StateParams",
    "TimeInterval",
    "TrajectoryBounds",
    "TrapezoidalTrajectory",
    "circular_trajectory_with_derivatives",
    "compute_trajectory_frames",
    "helicoidal_trajectory_with_derivatives",
    "linear_traj",
    "smoothing_spline_with_tolerance",
    "solve_tridiagonal",
]
