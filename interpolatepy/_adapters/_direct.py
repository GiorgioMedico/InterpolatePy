"""Direct re-exports from the C++ backend (no adapter wrapper needed)."""

from interpolatepy._backend import get_cpp_module

_cpp = get_cpp_module()

# Data / config classes
SplineParameters = _cpp.SplineParameters
SplineConfig = _cpp.SplineConfig
BSplineParams = _cpp.bspline.BSplineParams

StateParams = _cpp.motion.StateParams
TrajectoryBounds = _cpp.motion.TrajectoryBounds
BoundaryCondition = _cpp.motion.BoundaryCondition
TimeInterval = _cpp.motion.TimeInterval

# Free functions
solve_tridiagonal = _cpp.solve_tridiagonal
smoothing_spline_with_tolerance = _cpp.smoothing_spline_with_tolerance
linear_traj = _cpp.path.linear_traj
compute_trajectory_frames = _cpp.path.compute_frenet_frames
circular_trajectory_with_derivatives = _cpp.path.circular_trajectory_with_derivatives
helicoidal_trajectory_with_derivatives = _cpp.path.helicoidal_trajectory_with_derivatives
