"""
InterpolatePy: A comprehensive Python library for trajectory planning and interpolation.

This package provides smooth trajectory generation with precise control over position,
velocity, acceleration, and jerk profiles for robotics, animation, and scientific computing.

When the compiled C++ extension is available, computation-heavy classes are
automatically backed by the native implementation.  Set the environment variable
``INTERPOLATEPY_NO_CPP=1`` to force pure-Python mode.
"""

from .version import __version__

from ._backend import HAS_CPP

# ── Backend-aware classes and functions ──────────────────────────────
from ._api import (
    ApproximationBSpline,
    BSpline,
    BSplineInterpolator,
    BSplineParams,
    BoundaryCondition,
    CircularPath,
    CubicBSplineInterpolation,
    CubicSmoothingSpline,
    CubicSpline,
    CubicSplineWithAcceleration1,
    CubicSplineWithAcceleration2,
    DoubleSTrajectory,
    LinearPath,
    LogQuaternionInterpolation,
    ModifiedLogQuaternionInterpolation,
    ParabolicBlendTrajectory,
    PolynomialTrajectory,
    QuaternionSpline,
    SmoothingCubicBSpline,
    SplineConfig,
    SplineParameters,
    SquadC2,
    StateParams,
    TimeInterval,
    TrajectoryBounds,
    TrapezoidalTrajectory,
    circular_trajectory_with_derivatives,
    compute_trajectory_frames,
    helicoidal_trajectory_with_derivatives,
    linear_traj,
    smoothing_spline_with_tolerance,
    solve_tridiagonal,
)

# ── Always pure-Python (no C++ equivalent) ───────────────────────────
from .motion import CalculationParams
from .motion import InterpolationParams
from .motion import PolynomialTrajectoryParams
from .motion import PolynomialTrajectoryParams as TrajectoryParams
from .motion import TrapezoidalTrajectoryParams
from .paths import plot_frames
from .quaternion import Quaternion

# Protocols
from .protocols import CurveEvaluator
from .protocols import GeometricPath
from .protocols import QuaternionTrajectory
from .protocols import ScalarTrajectory
from .protocols import TrajectoryFunction

__all__ = [
    "HAS_CPP",
    "ApproximationBSpline",
    "BSpline",
    "BSplineInterpolator",
    "BSplineParams",
    "BoundaryCondition",
    "CalculationParams",
    "CircularPath",
    "CubicBSplineInterpolation",
    "CubicSmoothingSpline",
    "CubicSpline",
    "CubicSplineWithAcceleration1",
    "CubicSplineWithAcceleration2",
    "CurveEvaluator",
    "DoubleSTrajectory",
    "GeometricPath",
    "InterpolationParams",
    "LinearPath",
    "LogQuaternionInterpolation",
    "ModifiedLogQuaternionInterpolation",
    "ParabolicBlendTrajectory",
    "PolynomialTrajectory",
    "PolynomialTrajectoryParams",
    "Quaternion",
    "QuaternionSpline",
    "QuaternionTrajectory",
    "ScalarTrajectory",
    "SmoothingCubicBSpline",
    "SplineConfig",
    "SplineParameters",
    "SquadC2",
    "StateParams",
    "TimeInterval",
    "TrajectoryBounds",
    "TrajectoryFunction",
    "TrajectoryParams",
    "TrapezoidalTrajectory",
    "TrapezoidalTrajectoryParams",
    "__version__",
    "circular_trajectory_with_derivatives",
    "compute_trajectory_frames",
    "helicoidal_trajectory_with_derivatives",
    "linear_traj",
    "plot_frames",
    "smoothing_spline_with_tolerance",
    "solve_tridiagonal",
]
