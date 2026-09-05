"""C++-backed classes with Python-facing compatibility helpers.

This package is only imported when ``_backend.HAS_CPP`` is ``True``.  Each
submodule normalizes the native constructors, results, and common convenience
methods used by the public API. Implementation-level diagnostics can still be
backend-specific.
"""

from __future__ import annotations

# ── Spline family ────────────────────────────────────────────────────
from ._spline import (
    CubicSmoothingSpline,
    CubicSpline,
    CubicSplineWithAcceleration1,
    CubicSplineWithAcceleration2,
)

# ── B-spline family ──────────────────────────────────────────────────
from ._bspline import (
    ApproximationBSpline,
    BSpline,
    BSplineInterpolator,
    CubicBSplineInterpolation,
    SmoothingCubicBSpline,
)

# ── Motion profiles ──────────────────────────────────────────────────
from ._motion import (
    DoubleSTrajectory,
    ParabolicBlendTrajectory,
    PolynomialTrajectory,
    TrapezoidalTrajectory,
)

# ── Quaternion interpolation ─────────────────────────────────────────
from ._quaternion import (
    LogQuaternionInterpolation,
    ModifiedLogQuaternionInterpolation,
    QuaternionSpline,
    SquadC2,
)

# ── Path adapters ────────────────────────────────────────────────────
from ._paths import (
    CircularPath,
    LinearPath,
)

# ── Direct re-exports (no adapter needed) ────────────────────────────
from ._direct import (
    BoundaryCondition,
    BSplineParams,
    circular_trajectory_with_derivatives,
    compute_trajectory_frames,
    helicoidal_trajectory_with_derivatives,
    linear_traj,
    smoothing_spline_with_tolerance,
    solve_tridiagonal,
    SplineConfig,
    SplineParameters,
    StateParams,
    TimeInterval,
    TrajectoryBounds,
)

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
