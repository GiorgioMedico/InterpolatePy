"""
InterpolatePy: A comprehensive Python library for trajectory planning and interpolation.

This package provides smooth trajectory generation with precise control over position,
velocity, acceleration, and jerk profiles for robotics, animation, and scientific computing.
"""

from .version import __version__

# Core spline algorithms
from .cubic_spline import CubicSpline
from .c_s_smoothing import CubicSmoothingSpline
from .c_s_smoot_search import SplineConfig
from .c_s_with_acc1 import CubicSplineWithAcceleration1
from .c_s_with_acc2 import CubicSplineWithAcceleration2
from .c_s_with_acc2 import SplineParameters

# B-spline family
from .b_spline import BSpline
from .b_spline_approx import ApproximationBSpline
from .b_spline_cubic import CubicBSplineInterpolation
from .b_spline_interpolate import BSplineInterpolator
from .b_spline_smooth import BSplineParams
from .b_spline_smooth import SmoothingCubicBSpline

# Motion profiles
from .double_s import DoubleSTrajectory
from .double_s import StateParams
from .double_s import TrajectoryBounds
from .polynomials import BoundaryCondition
from .polynomials import PolynomialTrajectory
from .polynomials import TimeInterval
from .polynomials import TrajectoryParams
from .trapezoidal import CalculationParams
from .trapezoidal import InterpolationParams
from .trapezoidal import TrapezoidalTrajectory

# Path planning
from .simple_paths import CircularPath
from .simple_paths import LinearPath
from .lin_poly_parabolic import ParabolicBlendTrajectory

# Quaternion interpolation
from .quat_core import Quaternion
from .quat_spline import QuaternionSpline
from .squad_c2 import SquadC2

# Linear interpolation
from .linear import linear_traj

# Frenet frame utilities
from .frenet_frame import compute_trajectory_frames

# Utility functions
from .tridiagonal_inv import solve_tridiagonal

__all__ = [
    # Core spline algorithms
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
    "DoubleSTrajectory",
    "InterpolationParams",
    "LinearPath",
    "ParabolicBlendTrajectory",
    "PolynomialTrajectory",
    "Quaternion",
    "QuaternionSpline",
    "SmoothingCubicBSpline",
    "SplineConfig",
    "SplineParameters",
    "SquadC2",
    "StateParams",
    "TimeInterval",
    "TrajectoryBounds",
    "TrajectoryParams",
    "TrapezoidalTrajectory",
    # Version and functions
    "__version__",
    "compute_trajectory_frames",
    "linear_traj",
    "solve_tridiagonal",
]
