"""Backend exports that need no class adapter.

Some helpers intentionally retain their Python implementation because the
native function has a different public signature or return type.
"""

from interpolatepy._backend import get_cpp_module
from interpolatepy.b_spline_smooth import BSplineParams
from interpolatepy.c_s_smoot_search import SplineConfig
from interpolatepy.c_s_smoot_search import smoothing_spline_with_tolerance
from interpolatepy.frenet_frame import compute_trajectory_frames
from interpolatepy.linear import linear_traj
from interpolatepy.polynomials import BoundaryCondition
from interpolatepy.polynomials import TimeInterval

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
