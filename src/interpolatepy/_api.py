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
        ShootingQuaternionInterpolation,
        SpringQuaternionInterpolation,
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
    from .splines.acceleration1 import CubicSplineWithAcceleration1  # type: ignore[assignment]
    from .splines.acceleration2 import CubicSplineWithAcceleration2  # type: ignore[assignment]
    from .splines.acceleration2 import SplineParameters
    from .splines.cubic import CubicSpline  # type: ignore[assignment]
    from .splines.search import SplineConfig
    from .splines.search import smoothing_spline_with_tolerance
    from .splines.smoothing import CubicSmoothingSpline  # type: ignore[assignment]

    # B-spline family
    from .bsplines.approximation import ApproximationBSpline  # type: ignore[assignment]
    from .bsplines.core import BSpline  # type: ignore[assignment]
    from .bsplines.cubic import CubicBSplineInterpolation  # type: ignore[assignment]
    from .bsplines.interpolation import BSplineInterpolator  # type: ignore[assignment]
    from .bsplines.smoothing import BSplineParams
    from .bsplines.smoothing import SmoothingCubicBSpline  # type: ignore[assignment]

    # Motion profiles
    from .motion.double_s import DoubleSTrajectory  # type: ignore[assignment]
    from .motion.double_s import StateParams
    from .motion.double_s import TrajectoryBounds
    from .motion.parabolic import ParabolicBlendTrajectory  # type: ignore[assignment]
    from .motion.polynomial import BoundaryCondition
    from .motion.polynomial import PolynomialTrajectory  # type: ignore[assignment]
    from .motion.polynomial import TimeInterval
    from .motion.trapezoidal import TrapezoidalTrajectory  # type: ignore[assignment]

    # Path planning
    from .paths.geometric import CircularPath  # type: ignore[assignment]
    from .paths.geometric import LinearPath  # type: ignore[assignment]

    # Quaternion interpolation
    from .quaternion.logarithmic import LogQuaternionInterpolation  # type: ignore[assignment]
    from .quaternion.logarithmic import ModifiedLogQuaternionInterpolation  # type: ignore[assignment]
    from .quaternion.spline import QuaternionSpline  # type: ignore[assignment]
    from .quaternion.shooting import ShootingQuaternionInterpolation  # type: ignore[assignment]
    from .quaternion.spring import SpringQuaternionInterpolation  # type: ignore[assignment]
    from .quaternion.squad import SquadC2  # type: ignore[assignment]

    # Free functions
    from .linalg import solve_tridiagonal
    from .paths.frenet import circular_trajectory_with_derivatives
    from .paths.frenet import compute_trajectory_frames
    from .paths.frenet import helicoidal_trajectory_with_derivatives
    from .paths.linear import linear_traj

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
    "ShootingQuaternionInterpolation",
    "SmoothingCubicBSpline",
    "SplineConfig",
    "SplineParameters",
    "SpringQuaternionInterpolation",
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
