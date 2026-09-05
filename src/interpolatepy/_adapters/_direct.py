"""Backend exports that need no class adapter.

Some helpers intentionally retain their Python implementation because the
native function has a different public signature or return type.
"""

from interpolatepy._backend import get_cpp_module
from interpolatepy.bsplines.smoothing import BSplineParams
from interpolatepy.motion.polynomial import BoundaryCondition
from interpolatepy.motion.polynomial import TimeInterval
from interpolatepy.paths.frenet import compute_trajectory_frames
from interpolatepy.paths.linear import linear_traj
from interpolatepy.splines.search import SplineConfig
from interpolatepy.splines.search import smoothing_spline_with_tolerance

_cpp = get_cpp_module()

# Data / config classes
SplineParameters = _cpp.SplineParameters

StateParams = _cpp.motion.StateParams
TrajectoryBounds = _cpp.motion.TrajectoryBounds

# Free functions
solve_tridiagonal = _cpp.solve_tridiagonal
circular_trajectory_with_derivatives = _cpp.path.circular_trajectory_with_derivatives
helicoidal_trajectory_with_derivatives = _cpp.path.helicoidal_trajectory_with_derivatives

__all__ = [
    "BSplineParams",
    "BoundaryCondition",
    "SplineConfig",
    "SplineParameters",
    "StateParams",
    "TimeInterval",
    "TrajectoryBounds",
    "circular_trajectory_with_derivatives",
    "compute_trajectory_frames",
    "helicoidal_trajectory_with_derivatives",
    "linear_traj",
    "smoothing_spline_with_tolerance",
    "solve_tridiagonal",
]
