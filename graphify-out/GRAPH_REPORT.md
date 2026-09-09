# Graph Report - InterpolatePy  (2026-09-09)

## Corpus Check
- 238 files · ~203,234 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 4245 nodes · 7202 edges · 213 communities (205 shown, 5 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 388 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `a75d7be5`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- CubicSpline
- BSpline
- CubicSmoothingSpline
- BoundaryCondition
- SquadC2
- _api.py
- SmoothingCubicBSpline
- shooting_solver.cpp
- Quaternion
- test_spring_quaternion.cpp
- TrajectoryParams
- CubicSplineWithAcceleration1
- .setup_test_data
- ApproximationBSpline
- SplineConfig
- ModifiedLogQuaternionInterpolation
- CubicBSplineInterpolation
- .identity
- CircularPath
- .from_euler_angles
- QuaternionTrajectoryVisualizer
- config.hpp
- vector
- SquadC2
- LinearPath
- TestQuaternionBasicOperations
- test_bspline_variants.cpp
- API reference
- test_b_spline_variants.py
- _adapters/__init__.py
- TrajectoryParams
- quaternion.cpp
- quaternion_example.cpp
- SpringQuaternionInterpolation
- Troubleshooting
- DoubleSTrajectory
- TestQuaternionDynamics
- .slerp
- LogQuaternionInterpolation
- TrajectoryBounds
- QuaternionSpline
- _bspline.py
- _spline.py
- TestTridiagonalSolver
- FullTrajectoryResult
- PYBIND11_MODULE
- motion_types.hpp
- shooting.py
- PolynomialTrajectory
- SpringQuaternionInterpolation
- TestQuaternionConversions
- TestQuaternionSpline
- test_quaternion_spline.cpp
- LinearPath
- QuaternionSpline
- StateParams
- compute_trajectory_frames
- ndarray
- ndarray
- _spring_solver.py
- test_spring_quaternion.py
- trapezoidal_ex.py
- InterpolationParams
- BSplineInterpolator
- compute_frenet_frames
- TrajectoryResult
- paths/__init__.py
- spring_quaternion_ex.py
- quaternion/core.py
- TestQuaternionMathematics
- coefficients_
- Changelog
- ndarray
- .inverse_stereographic_projection
- TestPlottingFunctionality
- cubic_spline_with_acc1.cpp
- example_utils.hpp
- quat_visualization_ex.py
- ParabolicBlendTrajectory
- spring_quaternion_interpolation.cpp
- test_paths.cpp
- interpolatepy/__init__.py
- protocols_ex.py
- spring.py
- ndarray
- DoubleSTrajectory
- TestQuaternionTrajectoryProtocol
- TestBSplineEvaluation
- _interpolation_system.py
- linear_traj
- SpringConfig
- concepts_example.cpp
- .stereographic_projection
- TestBSplineKnotHandling
- TestBSplineInterpolatorAdvanced
- TestBackendDetection
- TestParabolicBlendTrajectoryConstruction
- TestScalarTrajectoryProtocol
- smoothing_cubic_bspline.cpp
- evaluate
- ndarray
- TestFrenetFrames
- test_protocols.py
- test_spring_performance.py
- TestGeometricPathProtocol
- parabolic_linear_example.cpp
- SplineConfig
- cubic_smoothing_spline.cpp
- quaternion_spline.cpp
- c_s_with_acc1_ex.py
- User guide
- b_spline_approx_ex.py
- b_spline_ex.py
- log_quat_new_ex.py
- _approximation_system.py
- TestLinearTrajectoryScalar
- TestPathPlanningPerformance
- TestPolynomialTrajectoryHeuristicVelocities
- SpringConfig
- modified_log_quaternion_interpolation.cpp
- Quaternion interpolation
- polynomials_ex.py
- .evaluate_full
- TestEdgeCasesAndErrorHandling
- test_polynomial_trajectory.cpp
- ParabolicBlendTrajectory
- Algorithms
- simple_paths_ex.py
- .create_uniform_knots
- test_lin_poly_parabolic.py
- test_path_planning.py
- TestLinearTrajectoryVector
- TestMotionProfilePerformance
- TestDoubleSTrajectoryEdgeCases
- cubic_smoothing_example.cpp
- bspline_approx_smooth_example.cpp
- test_bspline.cpp
- Contributing
- Spline interpolation
- main
- TestParabolicBlendTrajectoryPerformance
- test_native_adapter_api.py
- bspline_example.cpp
- approximation_bspline.cpp
- log_quaternion_interpolation.cpp
- test_shooting_quaternion.cpp
- Run Python examples
- TimingResult
- plot_individual_methods
- main
- ._setup_spline
- PlotStyle
- TestBSplineCurveGeneration
- TestCubicSplinePerformance
- TestParabolicBlendTrajectoryMathematicalProperties
- TestParabolicBlendTrajectoryEdgeCases
- TestLinearTrajectoryPerformance
- TestLinearTrajectoryMathematicalProperties
- TestDoubleSTrajectoryConstruction
- test_package_structure.py
- TestTimeInterval
- TestBoundaryCondition
- bspline_interpolator_example.cpp
- cubic_bspline_interpolation.cpp
- SpringEnergy
- test_data.hpp
- Installation
- b_spline_interpolate_ex.py
- TestNumericalEquivalence
- TestParabolicBlendTrajectoryPlotting
- InterpolatePy algorithm guide
- cubic_spline_acc1_example.cpp
- circular_path.cpp
- linear_path.cpp
- test_double_s_trajectory.cpp
- test_parabolic_blend_trajectory.cpp
- test_quaternion.cpp
- test_smoothing_search.cpp
- Architecture
- Motion profiles
- Path planning
- .__init__
- TestLinearTrajectoryInputValidation
- TestCurveEvaluatorProtocol
- .create_test_spline_trajectory
- trapezoidal_example.cpp
- ShootingConfig
- bspline_interpolator.cpp
- test_cubic_smoothing_spline.cpp
- InterpolatePy
- linear_ex.py
- QuaternionTrajectory
- InterpolatePy
- TestBSplineVariantsPerformance
- polynomial_example.cpp
- Quick start
- .create_periodic_knots
- .plot
- MinimizationSettings
- test_motion_profiles.py
- TestTrajectoryFunctionProtocol
- main
- main
- modified_log_quat_interp
- test_linear.py
- _solve_error_message
- .plot_with_points
- .plot_with_points_3d
- scripts/__init__.py
- InterpolatePy

## God Nodes (most connected - your core abstractions)
1. `Quaternion` - 269 edges
2. `BSpline` - 94 edges
3. `CubicSpline` - 54 edges
4. `BoundaryCondition` - 50 edges
5. `BSplineInterpolator` - 49 edges
6. `DoubleSTrajectory` - 48 edges
7. `TimeInterval` - 48 edges
8. `QuaternionSpline` - 46 edges
9. `TrajectoryBounds` - 45 edges
10. `ParabolicBlendTrajectory` - 45 edges

## Surprising Connections (you probably didn't know these)
- `test_shooting_native_adapter_matches_python_reference()` --uses--> `ShootingQuaternionInterpolation`  [INFERRED]
  tests/test_native_adapter_api.py → src/interpolatepy/quaternion/shooting.py
- `test_spring_native_adapter_matches_python_reference()` --uses--> `SpringQuaternionInterpolation`  [INFERRED]
  tests/test_native_adapter_api.py → src/interpolatepy/quaternion/spring.py
- `Quaternion::identity()` --calls--> `Quaternion`  [EXTRACTED]
  cpp/src/quaternion.cpp → src/interpolatepy/quaternion/core.py
- `Quaternion::from_euler_angles()` --calls--> `Quaternion`  [EXTRACTED]
  cpp/src/quaternion.cpp → src/interpolatepy/quaternion/core.py
- `Quaternion::operator*()` --calls--> `Quaternion`  [EXTRACTED]
  cpp/src/quaternion.cpp → src/interpolatepy/quaternion/core.py

## Import Cycles
- None detected.

## Communities (213 total, 5 thin omitted)

### Community 0 - "CubicSpline"
Cohesion: 0.03
Nodes (51): CubicSpline, ndarray, Compute the interior velocities from the C2-continuity equations. The method…, Compute the coefficients for each cubic polynomial segment. For each segment k,…, Evaluate the spline at time t. Parameters ---------- t : float or numpy.ndarray…, Evaluate the velocity at time t. Parameters ---------- t : float or…, Evaluate the acceleration at time t. Parameters ---------- t : float or…, Plot the spline trajectory along with its velocity and acceleration profiles.… (+43 more)

### Community 1 - "BSpline"
Cohesion: 0.04
Nodes (47): BSpline, Axes3D, A class for representing and evaluating B-spline curves. Parameters ----------…, Plot a 3D B-spline curve. Parameters ---------- num_points : int, optional…, Return a string representation of the B-spline. Returns ------- str String…, Comprehensive tests for the B-spline curve implementation. This module contains…, Test that non-decreasing knots raise ValueError., Test that invalid knot-control point relationship raises ValueError. (+39 more)

### Community 2 - "CubicSmoothingSpline"
Cohesion: 0.04
Nodes (42): CubicSmoothingSpline, Cubic smoothing spline trajectory planning with control over the smoothness…, CaptureFixture, FixtureFunction, parametrize, Comprehensive tests for smoothing spline implementations. This module contains…, Test suite comparing different smoothing approaches., Test consistency across different smoothing algorithms. (+34 more)

### Community 3 - "BoundaryCondition"
Cohesion: 0.05
Nodes (44): BoundaryCondition, PolynomialTrajectory, Polynomial trajectory generation for smooth motion profiles. This module…, Generate smooth polynomial trajectories with specified boundary conditions.…, Generate a 3rd order polynomial trajectory with specified boundary conditions.…, Generate a 5th order polynomial trajectory with specified boundary conditions.…, Boundary conditions for polynomial trajectory generation. Parameters ----------…, Generate a 7th order polynomial trajectory with specified boundary conditions.… (+36 more)

### Community 4 - "SquadC2"
Cohesion: 0.04
Nodes (43): SquadC2, Comprehensive tests for SQUAD C2 quaternion interpolation implementation. This…, Test extended sequence with multiple waypoints., Test that virtual waypoints have appropriate time spacing., Test the corrected intermediate quaternion formula (Equation 5)., Test the corrected intermediate quaternion formula from Equation 5., Test that all intermediate quaternions are unit quaternions., Equation 5 must use the durations `evaluate` actually traverses. Equation 5… (+35 more)

### Community 5 - "_api.py"
Cohesion: 0.05
Nodes (40): Resolve the active backend and export the public API symbols. This module…, ndarray, Efficient tridiagonal matrix solver using the Thomas algorithm. This module…, Solve a tridiagonal system using the Thomas algorithm. This function solves the…, solve_tridiagonal(), CubicSplineWithAcceleration2, ndarray, Container for spline initialization parameters. Parameters ---------- v0 :… (+32 more)

### Community 6 - "SmoothingCubicBSpline"
Cohesion: 0.06
Nodes (36): example_8_12(), Compare 3D cubic B-spline smoothing for several lambda values., BSplineParams, ndarray, Calculate normalized parameters for the approximation points., Calculate the cubic knot vector from the parameters ūₖ. The construction is: u0…, Construct the B matrix for the smoothing functional. B contains the basis…, Parameters for initializing a SmoothingCubicBSpline. Attributes ---------- mu :… (+28 more)

### Community 7 - "shooting_solver.cpp"
Cohesion: 0.06
Nodes (54): pair, ShootingConfig, State, vector, Vector3d, ShootingQuaternionInterpolation::evaluate(), ShootingQuaternionInterpolation::evaluate_acceleration(), ShootingQuaternionInterpolation::evaluate_state() (+46 more)

### Community 8 - "Quaternion"
Cohesion: 0.04
Nodes (28): _py_to_cpp(), Convert a Python Quaternion to a C++ Quaternion., Quaternion, Quaternion addition: q1 + q2 = [s1+s2, v1+v2], Quaternion subtraction: q1 - q2 = [s1-s2, v1-v2], Quaternion multiplication or scalar multiplication., Right scalar multiplication: c * q, Quaternion division or scalar division. (+20 more)

### Community 9 - "test_spring_quaternion.cpp"
Cohesion: 0.05
Nodes (51): Band, cholesky_solve(), Frames, Indices, pair, vector, Vector3d, descent_direction() (+43 more)

### Community 10 - "TrajectoryParams"
Cohesion: 0.05
Nodes (36): __dir__(), __getattr__(), Any, Backend-neutral motion-profile API., Resolve public names lazily and through the active backend when applicable., Return the public domain API for interactive discovery., CalculationParams, Module for generating and managing trapezoidal velocity profiles for trajectory… (+28 more)

### Community 11 - "CubicSplineWithAcceleration1"
Cohesion: 0.06
Nodes (29): CubicSplineWithAcceleration1, ndarray, Initialize the cubic spline with velocity and acceleration constraints.…, Cubic spline trajectory planning with both velocity and acceleration…, Add two extra points at t₁ and tₙ₋₁ to satisfy acceleration constraints. The…, Solve for the accelerations by setting up and solving the linear system A ω =…, Compute the polynomial coefficients for each segment using equation (4.25). For…, Get the indices in the expanded array that correspond to original points.… (+21 more)

### Community 12 - ".setup_test_data"
Cohesion: 0.05
Nodes (30): parametrize, Comprehensive tests for the logarithmic quaternion interpolation…, Test evaluation between control points., Test evaluation at boundaries and outside range., Test angular velocity evaluation., Test angular acceleration evaluation., Test trajectory generation., Test suite for ModifiedLogQuaternionInterpolation (mLQI) class. (+22 more)

### Community 13 - "ApproximationBSpline"
Cohesion: 0.06
Nodes (31): ApproximationBSpline, ndarray, Calculate normalized parameter values for the approximation points., Compute knot vector following the algorithm in Section 8.5.1. Parameters…, A class for B-spline curve approximation of a set of points. Inherits from…, Compute endpoint-constrained weighted least-squares control points., Calculate the approximation error as the sum of squared distances. Computes sum…, Refine the approximation by adding more control points. Adds control points… (+23 more)

### Community 14 - "SplineConfig"
Cohesion: 0.07
Nodes (31): example_prescribed_tolerance(), Example of finding a smoothing spline with prescribed tolerance., CubicSmoothingSpline, ndarray, Configuration parameters for smoothing spline calculation., Find a cubic smoothing spline with a maximum approximation error smaller than a…, smoothing_spline_with_tolerance(), SplineConfig (+23 more)

### Community 15 - "ModifiedLogQuaternionInterpolation"
Cohesion: 0.07
Nodes (31): _PyLogQuaternionInterpolation, __dir__(), __getattr__(), Any, Backend-neutral quaternion interpolation API., Resolve public names lazily and through the active backend when applicable., Return the public domain API for interactive discovery., _canonicalize_double_cover() (+23 more)

### Community 16 - "CubicBSplineInterpolation"
Cohesion: 0.06
Nodes (31): CubicBSplineInterpolation, BSpline, ndarray, Calculate normalized parameters for the interpolation points., Calculate the knot vector based on the parameters ūₖ. Returns ------- ndarray…, Calculate the control points by solving a system of equations. Parameters…, Initialize a cubic B-spline interpolation of a set of points. Parameters…, A class for cubic B-spline interpolation of a set of points. This class… (+23 more)

### Community 17 - ".identity"
Cohesion: 0.06
Nodes (27): LogQuaternionInterpolation::evaluate(), Create identity quaternion [1, 0, 0, 0], Create quaternion from rotation angle and axis. Args: angle: Rotation angle in…, Evaluate the interpolated quaternion at time ``t``., Test the continuous axis-angle recovery algorithm., Test handling of small angles where axis is indeterminate., Test rotation matrix conversion round trip., Test round-trip projection consistency. (+19 more)

### Community 18 - "CircularPath"
Cohesion: 0.07
Nodes (23): CircularPath, A circular path in 3D space defined by an axis and a point on the circle. This…, Test suite for CircularPath class., Test basic CircularPath construction., Test CircularPath construction with list inputs., Test position evaluation on unit circle., Test position evaluation with non-zero center., Test velocity for circular path. (+15 more)

### Community 19 - ".from_euler_angles"
Cohesion: 0.11
Nodes (32): Create quaternion from Euler angles (roll, pitch, yaw) in radians, ndarray, Evaluate orientation; return original normalized keyframes exactly., Return body angular velocity in radians per second., Return body angular acceleration in radians per second squared., Sample the solved continuous curve without rerunning optimization., Interpolate rotations with natural Riemannian cubics and multiple shooting.…, ShootingQuaternionInterpolation (+24 more)

### Community 20 - "QuaternionTrajectoryVisualizer"
Cohesion: 0.07
Nodes (24): Any, Figure, ndarray, QuaternionTrajectoryVisualizer, Create a simple 3D plot of quaternion trajectory using stereographic…, Create a 3D plot showing only waypoints (no trajectory lines). Args: waypoints:…, Calculate the distance between two quaternions using quaternion norm. Args: q1:…, Compute velocity magnitudes using the formula: V(qi) = [||qi - qi-1|| + ||qi -… (+16 more)

### Community 21 - "config.hpp"
Cohesion: 0.07
Nodes (19): bind_smoothing_spline(), module_, BSplineParams, auto_derivatives, enforce_endpoints, method, mu, v0 (+11 more)

### Community 22 - "vector"
Cohesion: 0.05
Nodes (22): main(), vector, SquadC2Config, normalize_quaternions, quaternions, time_points, validate_continuity, "CubicSpline C2 continuity" (+14 more)

### Community 23 - "SquadC2"
Cohesion: 0.08
Nodes (20): ndarray, Validate input data for SQUAD_C2 construction., Add duplicate virtual endpoint waypoints: Q = [q₁, q₁ᵛⁱʳᵗ, q₂, ..., qₙ₋₁ᵛⁱʳᵗ,…, Compute the segment durations hᵢ that the trajectory actually traverses.…, Compute intermediate quaternion using the corrected formula from Equation (5)…, Setup intermediate quaternions and polynomial parameterizations for all…, C²-Continuous, Zero-Clamped Quaternion Interpolation using SQUAD with Quintic…, Spherical linear interpolation between two quaternions. (+12 more)

### Community 24 - "LinearPath"
Cohesion: 0.08
Nodes (21): LinearPath, A linear path between two points in 3D space. This class represents a straight-…, Test that position evaluation clamps arc length to valid range., Test that velocity is constant for linear path., Test that acceleration is zero for linear path., Test LinearPath in 3D space., Test LinearPath with 2D points (should still work)., Advanced test suite for LinearPath functionality. (+13 more)

### Community 25 - "TestQuaternionBasicOperations"
Cohesion: 0.06
Nodes (19): Create quaternion from 3x3 or 4x4 rotation matrix., Test rotation matrix conversion edge cases., Test edge cases for angle-axis constructor., Test quaternion creation from 3x3 rotation matrix., Test quaternion creation from 4x4 transformation matrix., Test error handling for invalid rotation matrix sizes., Test basic arithmetic operations., Test quaternion multiplication. (+11 more)

### Community 26 - "test_bspline_variants.cpp"
Cohesion: 0.06
Nodes (34): "ApproximationBSpline construction", "ApproximationBSpline different degrees", "ApproximationBSpline error calculation", "ApproximationBSpline knot vector properties", "ApproximationBSpline original data storage", "ApproximationBSpline parameterization methods", "ApproximationBSpline validation", "BSpline variant inheritance" (+26 more)

### Community 27 - "API reference"
Cohesion: 0.06
Nodes (35): API reference, ApproximationBSpline, B-splines, BSpline, BSplineInterpolator, CircularPath, CubicBSplineInterpolation, CubicSmoothingSpline (+27 more)

### Community 28 - "test_b_spline_variants.py"
Cohesion: 0.09
Nodes (24): B-spline curve approximation with least squares fitting. This module provides…, Axes, Plot a 2D B-spline curve with customizable styling. Parameters ----------…, __dir__(), __getattr__(), Any, Backend-neutral B-spline API., Resolve public names through the active backend. (+16 more)

### Community 29 - "_adapters/__init__.py"
Cohesion: 0.09
Nodes (22): _CppShooting, ModuleType, Backend exports that need no class adapter. Some helpers intentionally retain…, C++-backed classes with Python-facing compatibility helpers. This package is…, Native adapter for the Double-S motion profile., Native adapter for parabolic-blend trajectories., Native adapter for trapezoidal trajectories., Adapters for geometric path classes. The C++ constructors use different… (+14 more)

### Community 30 - "TrajectoryParams"
Cohesion: 0.08
Nodes (22): Parameters for multipoint polynomial trajectory generation. Parameters…, TrajectoryParams, FixtureFunction, parametrize, Test suite for TrajectoryParams dataclass., Test TrajectoryParams creation with minimal parameters., Test TrajectoryParams creation with all parameters., Test that different orders can be specified. (+14 more)

### Community 31 - "quaternion.cpp"
Cohesion: 0.07
Nodes (24): Matrix3d, Matrix4d, pair, tuple, Vector3d, Quaternion::conjugate(), Quaternion::dot(), Quaternion::E() (+16 more)

### Community 32 - "quaternion_example.cpp"
Cohesion: 0.10
Nodes (24): pair, string, vector, log_quaternion_example(), main(), make_waypoints(), method_comparison(), modified_log_quaternion_example() (+16 more)

### Community 33 - "SpringQuaternionInterpolation"
Cohesion: 0.09
Nodes (20): _CppSpringQuaternionInterpolation, _PySpringConfig, _checked_py_to_cpp(), _cpp_to_py(), Any, ndarray, _PyQuaternion, _py_to_cpp() (+12 more)

### Community 34 - "Troubleshooting"
Cohesion: 0.06
Nodes (32): A constrained interpolation system is ill-conditioned, A documented top-level name cannot be imported, A helper exists only on the Python backend, Axis and angle appear swapped, B-spline questions, Backend issues, Bounds are invalid or the move is infeasible, Derivatives do not match expected angular velocity (+24 more)

### Community 35 - "DoubleSTrajectory"
Cohesion: 0.11
Nodes (26): DoubleSTrajectory, _apply_candidate(), _compute_constant_velocity_time(), _compute_initial_phase_times(), _finalize_plan(), _fit_reduced_acceleration(), plan_trajectory(), _PlanCandidate (+18 more)

### Community 36 - "TestQuaternionDynamics"
Cohesion: 0.07
Nodes (18): Trapezoidal quaternion integration. Returns: (updated_quat,…, Trapezoidal quaternion scalar part integration., Trapezoidal quaternion vector part integration., Test suite for edge cases and error handling., Test numerical stability with near-zero values., Test numerical stability with large rotation angles., Test interpolation between quaternions that are nearly opposite., Test interpolation between identical quaternions. (+10 more)

### Community 37 - ".slerp"
Cohesion: 0.08
Nodes (17): SquadC2::evaluate(), Quaternion dot product: q1·q2 = s1*s2 + v1·v2, Spherical Linear Interpolation (Slerp)., Spherical Linear Interpolation derivative., Static version of slerp_prime, Spherical Cubic Interpolation (Squad)., Spherical Cubic Interpolation derivative., Force SLERP interpolation at given time, regardless of current method setting.… (+9 more)

### Community 38 - "LogQuaternionInterpolation"
Cohesion: 0.11
Nodes (17): _CppLogQuaternionInterpolation, _CppModifiedLogQuaternionInterpolation, _PyModifiedLogQuaternionInterpolation, _cpp_to_py(), LogQuaternionInterpolation, ModifiedLogQuaternionInterpolation, Any, ndarray (+9 more)

### Community 39 - "TrajectoryBounds"
Cohesion: 0.09
Nodes (18): Bounds for trajectory planning. Parameters ---------- v_bound : float Velocity…, TrajectoryBounds, Test suite for DoubleSTrajectory evaluation methods., Test basic trajectory evaluation., Test trajectory evaluation with array inputs., Test that boundary conditions are satisfied., Test that velocity bounds are not exceeded., Test that acceleration bounds are not exceeded. (+10 more)

### Community 40 - "QuaternionSpline"
Cohesion: 0.08
Nodes (17): ndarray, QuaternionSpline, Quaternion interpolation at given time. Returns: (interpolated_quaternion,…, Quaternion spline interpolator for smooth trajectory planning. Supports both…, Quaternion interpolation with angular velocity. Returns:…, Get the time range of the spline, Check if this spline has no data, Get the current interpolation method (+9 more)

### Community 41 - "_bspline.py"
Cohesion: 0.08
Nodes (23): _CppApproximationBSpline, _CppBSpline, _CppBSplineInterpolator, _CppCubicBSplineInterpolation, _CppSmoothingCubicBSpline, _PyBSplineParams, ApproximationBSpline, BSpline (+15 more)

### Community 42 - "_spline.py"
Cohesion: 0.08
Nodes (22): _CppCubicSmoothingSpline, _CppCubicSpline, _CppCubicSplineWithAcc1, _CppCubicSplineWithAcc2, CubicSmoothingSpline, CubicSpline, CubicSplineWithAcceleration1, CubicSplineWithAcceleration2 (+14 more)

### Community 43 - "TestTridiagonalSolver"
Cohesion: 0.10
Nodes (17): FixtureFunction, NDArray, parametrize, Tests for the tridiagonal matrix solver implementation. This module contains…, Create a tridiagonal system with a known analytical solution. This creates a…, Test that the tridiagonal solver produces correct results. Parameters…, Test the solver against a system with known analytical solution. Parameters…, Test the solver's stability with ill-conditioned matrices. (+9 more)

### Community 44 - "FullTrajectoryResult"
Cohesion: 0.10
Nodes (22): FullTrajectoryResult, acceleration, jerk, position, velocity, PolynomialTrajectory(), map, string (+14 more)

### Community 45 - "PYBIND11_MODULE"
Cohesion: 0.08
Nodes (17): bind_acc_splines(), module_, bind_bspline(), module_, bind_cubic_spline(), module_, bind_motion(), module_ (+9 more)

### Community 46 - "motion_types.hpp"
Cohesion: 0.11
Nodes (18): map, string, example_asymmetric_velocities(), example_negative_displacement(), example_phase_durations(), example_standard(), example_velocity_matching(), main() (+10 more)

### Community 47 - "shooting.py"
Cohesion: 0.16
Nodes (22): csc_matrix, Natural Riemannian cubic quaternion interpolation by multiple shooting., _advance_sensitivity(), integrate(), left_matrix(), matching_system(), newton_solve(), ndarray (+14 more)

### Community 48 - "PolynomialTrajectory"
Cohesion: 0.10
Nodes (18): _multipoint_evaluator(), PolynomialTrajectory, BoundaryCondition, TimeInterval, Native adapter for polynomial trajectories., Compute heuristic intermediate velocities via C++., Generate a multi-segment polynomial trajectory., Build the native callable for each polynomial segment. (+10 more)

### Community 49 - "SpringQuaternionInterpolation"
Cohesion: 0.16
Nodes (14): _array_to_quaternion(), ndarray, _quaternion_to_array(), Numerical minimum-curvature interpolation of quaternion keyframes. ``SPRING``…, Distribute frames by quaternion chord length (report, page 55)., Weight only curvature centered on an original internal keyframe., SLERP new frames between the preceding level's fixed samples., Evaluate the optimized trajectory at time ``t``. (+6 more)

### Community 50 - "TestQuaternionConversions"
Cohesion: 0.10
Nodes (16): FixtureFunction, parametrize, Test suite for performance benchmarks., Benchmark basic quaternion operations., Benchmark SLERP interpolation performance., Benchmark spline interpolation performance., Benchmark conversion operations performance., Test quaternion creation from Euler angles. (+8 more)

### Community 51 - "TestQuaternionSpline"
Cohesion: 0.10
Nodes (14): Test changing interpolation method on existing spline., Test spline method validation., Test forced interpolation methods regardless of spline setting., Test SQUAD interpolation with insufficient points., Test spline interpolation with velocity computation., Test behavior with empty spline., Test suite for quaternion spline functionality., Create test data for spline testing. (+6 more)

### Community 52 - "test_quaternion_spline.cpp"
Cohesion: 0.08
Nodes (24): vector, "LogQuaternionInterpolation construction", "LogQuaternionInterpolation endpoints", "LogQuaternionInterpolation supports two quaternions for every degree", "LogQuaternionInterpolation validation", "LogQuaternionInterpolation velocity", make_test_quats(), make_test_times() (+16 more)

### Community 53 - "LinearPath"
Cohesion: 0.12
Nodes (12): _CppCircularPath, _CppLinearPath, CircularPath, LinearPath, ndarray, Evaluate position, velocity, acceleration at arc-length values., Generate complete trajectory around the circle., C++-backed LinearPath with Python-compatible constructor. (+4 more)

### Community 54 - "QuaternionSpline"
Cohesion: 0.09
Nodes (13): _CppQuaternionSpline, _CppSquadC2, setter, QuaternionSpline, Change the interpolation method for subsequent evaluations., Return the active interpolation method name., Return (t_min, t_max)., C++-backed SquadC2 returning Python Quaternions. (+5 more)

### Community 55 - "StateParams"
Cohesion: 0.12
Nodes (22): example_asymmetric_velocities(), example_factory_method(), example_negative_displacement(), example_standard_trajectory(), example_velocity_matching(), main(), Example demonstrating the usage of the DoubleSTrajectory class for motion…, Demonstrate matching velocities when positions are the same. (+14 more)

### Community 56 - "compute_trajectory_frames"
Cohesion: 0.12
Nodes (23): example_8_5(), example_8_6(), example_rot(), Examples demonstrating the computation and visualization of Frenet frames. This…, Recreate Example 8.5 using the general approach., Recreate Example 8.6., Recreate Example Rotations., angular_error_deg() (+15 more)

### Community 57 - "ndarray"
Cohesion: 0.10
Nodes (13): ndarray, Calculate second derivative with respect to arc length. For linear path, this…, Evaluate position, velocity, and acceleration at specific arc length values.…, Generate a complete trajectory along the entire linear path. Parameters…, Initialize a circular path. Parameters ---------- r : array_like Unit vector of…, Calculate position at arc length s. Parameters ---------- s : float or…, Calculate first derivative with respect to arc length. Parameters ---------- s…, Calculate second derivative with respect to arc length. Parameters ---------- s… (+5 more)

### Community 58 - "ndarray"
Cohesion: 0.10
Nodes (11): ndarray, Quaternion time derivative. The quaternion time derivative (quaternion…, Matrix E for quaternion dynamics. E = sI - S(v) for BASE_FRAME (sign=0) E = sI…, Create skew-symmetric matrix from vector. S(v) = [[ 0, -v3, v2], [ v3, 0, -v1],…, Return angular velocity from quaternion and its time derivative. Solves: q̇ =…, Rotation matrix from unit quaternion. R = (s² - v·v)I + 2vv^T + 2s*S(v), Transformation matrix from quaternion., Convert quaternion to axis-angle representation. Returns the canonical form… (+3 more)

### Community 59 - "_spring_solver.py"
Cohesion: 0.13
Nodes (18): EnergyEvaluation, ndarray, Reusable SPRING stencils and lazy gradients for backtracking trials., Evaluate energy, retaining intermediates for an optional gradient., A trial's energy; compute its gradient only if the trial needs it. The model…, Reverse accumulation through the unchanged curvature and norm residuals., Cache the fixed grid coefficients for one minimization stage., SpringEnergy (+10 more)

### Community 60 - "test_spring_quaternion.py"
Cohesion: 0.18
Nodes (23): _curved_keyframes(), Any, ndarray, parametrize, SpringConfig, Tests for the SPRING quaternion interpolation algorithm., _same_orientation(), test_spring_analytic_gradient_matches_centered_difference() (+15 more)

### Community 61 - "trapezoidal_ex.py"
Cohesion: 0.12
Nodes (23): example_10_complex_velocity_profile(), example_1_basic_trajectory(), example_2_nonzero_velocities(), example_3_negative_displacement(), example_4_duration_constrained(), example_5_triangular_profile(), example_6_asymmetric_profile(), example_7_multi_point_custom_velocities() (+15 more)

### Community 62 - "InterpolationParams"
Cohesion: 0.11
Nodes (15): C++-backed trajectory matching the pure-Python class API., Generate a single-segment trapezoidal trajectory., Calculate intermediate velocities heuristically., Generate a multi-segment trajectory through waypoints., TrapezoidalTrajectory, InterpolationParams, Parameters for multi-point interpolation. Parameters ---------- points :…, Test suite for trapezoidal waypoint interpolation. (+7 more)

### Community 63 - "BSplineInterpolator"
Cohesion: 0.12
Nodes (14): BSplineInterpolator, A B-spline that interpolates a set of points with specified degrees of…, Test suite for BSplineInterpolator class., Test basic BSplineInterpolator construction., Test interpolation with different degrees., Test interpolation accuracy for known functions., Test interpolation end conditions., Test input validation and error cases. (+6 more)

### Community 64 - "compute_frenet_frames"
Cohesion: 0.13
Nodes (20): circular_path_example(), frenet_frame_circular(), frenet_frame_helicoidal(), linear_path_example(), main(), FrenetFrame, binormal, curvature (+12 more)

### Community 65 - "TrajectoryResult"
Cohesion: 0.10
Nodes (15): TrajectoryResult, acceleration, position, velocity, TrapezoidalTrajectory(), vector, ParabolicBlendTrajectory::evaluate(), ParabolicBlendTrajectory::ParabolicBlendTrajectory() (+7 more)

### Community 66 - "paths/__init__.py"
Cohesion: 0.11
Nodes (20): angular_error_deg(), main(), ndarray, Modified Logarithmic Quaternion Interpolation (mLQI) along a cylindrical helix…, Geodesic angle between two rotation matrices, in degrees., circular_trajectory_with_derivatives(), helicoidal_trajectory_with_derivatives(), plot_frames() (+12 more)

### Community 67 - "spring_quaternion_ex.py"
Cohesion: 0.16
Nodes (22): angular_speed(), benchmark_methods(), create_waypoints(), discrete_tangential_curvature(), _evaluate_batch(), main(), _median_runtime(), plot_comparison() (+14 more)

### Community 68 - "quaternion/core.py"
Cohesion: 0.15
Nodes (16): Adapters for the quaternion interpolation family. The C++ ``evaluate()``…, inverse_stereographic_projection(), project_trajectory(), ndarray, quaternion_distance(), Projection and metric helpers for quaternion visualizations., Project a unit quaternion to Modified Rodrigues Parameters., Convert Modified Rodrigues Parameters back to a unit quaternion. (+8 more)

### Community 69 - "TestQuaternionMathematics"
Cohesion: 0.09
Nodes (12): Test suite for core quaternion mathematical operations., Test quaternion conjugate operation., Test quaternion norm calculation., Test quaternion normalization to unit length., Test unit normalization edge cases., Test quaternion inverse operation., Test inverse operation edge cases., Test quaternion exponential function. (+4 more)

### Community 70 - "coefficients_"
Cohesion: 0.15
Nodes (19): SegmentInfo, span, VectorXd, CubicSpline::compute_coefficients(), CubicSpline::CubicSpline(), CubicSpline::evaluate(), CubicSpline::evaluate_acceleration(), CubicSpline::evaluate_velocity() (+11 more)

### Community 71 - "Changelog"
Cohesion: 0.10
Nodes (20): 1.0.1 — 2025-03-26, 1.1.0 — 2025-05-17, 2.0.0 — 2025-08-06, 3.0.0 — 2026-03-21, 3.0.1 — 2026-05-14, 3.1.0 — 2026-05-21, 3.2.0 — 2026-09-04, 3.2.1 — 2026-09-04 (+12 more)

### Community 72 - "ndarray"
Cohesion: 0.14
Nodes (11): CurveEvaluator, GeometricPath, ndarray, Protocol, Protocol definitions for InterpolatePy trajectory and curve interfaces. Defines…, Protocol for callable trajectory functions returning (pos, vel, acc).…, Protocol for scalar (1D) trajectory evaluation. Conforming classes provide…, Protocol for parametric curve evaluation with derivative support. Conforming… (+3 more)

### Community 73 - ".inverse_stereographic_projection"
Cohesion: 0.12
Nodes (13): Convert Modified Rodrigues Parameters back to quaternion. Args: mrp: 3D point…, FixtureFunction, parametrize, Test suite for inverse stereographic projection., Test inverse projection of origin., Test inverse projection with various MRP points., Test inverse projection with large MRP values., Test suite for performance benchmarks. (+5 more)

### Community 74 - "TestPlottingFunctionality"
Cohesion: 0.12
Nodes (12): patch, Test suite for plotting methods with matplotlib mocking., Test basic 3D trajectory plotting., Test 3D trajectory plotting with custom options., Test 3D trajectory plotting with empty quaternion list., Test basic angular velocity plotting., Test angular velocity plotting with custom time points., Test angular velocity plotting with empty quaternion list. (+4 more)

### Community 75 - "cubic_spline_with_acc1.cpp"
Cohesion: 0.12
Nodes (14): bind_tridiagonal(), module_, VectorXd, solve_tridiagonal(), CubicSpline::compute_velocities(), SegmentInfo, span, VectorXd (+6 more)

### Community 76 - "example_utils.hpp"
Cohesion: 0.16
Nodes (18): example_8_8(), example_with_derivatives(), main(), function, MatrixXd, string, tuple, Vector3d (+10 more)

### Community 77 - "quat_visualization_ex.py"
Cohesion: 0.16
Nodes (19): create_waypoint_trajectory(), demo_combined_plot(), demo_simple_3d_plot(), demo_velocity_plot(), generate_interpolated_trajectories(), main(), plot_3d_trajectory_comparison(), plot_quaternion_components_comparison() (+11 more)

### Community 78 - "ParabolicBlendTrajectory"
Cohesion: 0.14
Nodes (12): ParabolicBlendTrajectory, Class to generate trajectories composed of linear segments with parabolic…, Test suite for trajectory generation functionality., Test trajectory generation with two waypoints., Test trajectory generation with three waypoints., Test that initial and final velocities are zero., Test position continuity throughout trajectory., Test trajectory evaluation outside valid time range. (+4 more)

### Community 79 - "spring_quaternion_interpolation.cpp"
Cohesion: 0.15
Nodes (17): Frames, pair, SpringConfig, vector, Vector3d, SpringQuaternionInterpolation::allocate_intervals(), SpringQuaternionInterpolation::create_initial_curve(), SpringQuaternionInterpolation::evaluate() (+9 more)

### Community 80 - "test_paths.cpp"
Cohesion: 0.11
Nodes (18): "circular_trajectory_with_derivatives", "CircularPath 3D", "CircularPath acceleration", "CircularPath point on axis", "CircularPath vector evaluation", "CircularPath velocity", "CircularPath XY plane", "Frenet frame circular path" (+10 more)

### Community 81 - "interpolatepy/__init__.py"
Cohesion: 0.11
Nodes (11): example_8_8(), Interpolate a ten-point 3D path with a cubic B-spline., CubicSmoothingSpline, Compare smoothing splines with different μ values., textbook_example(), plot_trajectory_with_waypoints(), Plot the trajectory along with waypoints highlighted. The function generates…, This module contains examples of Python code. (+3 more)

### Community 82 - "protocols_ex.py"
Cohesion: 0.16
Nodes (18): evaluate_curve(), example_curve_evaluator(), example_geometric_path(), example_quaternion_trajectory(), main(), ndarray, Example demonstrating protocol-based generic functions in InterpolatePy.…, Sample a parametric curve and its first derivative. Parameters ---------- curve… (+10 more)

### Community 83 - "spring.py"
Cohesion: 0.12
Nodes (17): Curvature-minimizing quaternion interpolation with SPRING. The implementation…, ndarray, Batch the existing log/exp SLERP without changing its small-angle rules., Interpolate many samples, computing each adjacent pair's log only once.…, slerp_segments(), curvature_energy_gradient(), iteration_budgets(), nested_level_indices() (+9 more)

### Community 84 - "ndarray"
Cohesion: 0.13
Nodes (10): ndarray, Construct matrices A and C for the linear system. Builds the tridiagonal matrix…, Solve the linear system to find the accelerations. Solves either the pure…, Compute the approximated positions using equation (4.36). For pure…, Compute polynomial coefficients for each segment. For each segment k from 0 to…, Evaluate the spline at time t. Parameters ---------- t : float or list[float]…, Evaluate the velocity at time t. Parameters ---------- t : float or list[float]…, Evaluate the acceleration at time t. Parameters ---------- t : float or… (+2 more)

### Community 85 - "DoubleSTrajectory"
Cohesion: 0.17
Nodes (9): _CppDoubleSTrajectory, DoubleSTrajectory, ndarray, C++-backed trajectory matching the Python scalar-trajectory protocol., Evaluate position, velocity, acceleration, and jerk at time *t*., Return the total trajectory duration., Return the duration of each trajectory phase., Create a trajectory callable and return it with its duration. (+1 more)

### Community 86 - "TestQuaternionTrajectoryProtocol"
Cohesion: 0.14
Nodes (10): QuaternionSpline, LogQuaternionInterpolation, SquadC2, Tests for QuaternionTrajectory protocol conformance., SquadC2 should work through QuaternionTrajectory interface., QuaternionSpline.evaluate should return a Quaternion., QuaternionSpline.evaluate_velocity should return an ndarray., QuaternionSpline.evaluate_acceleration should return an ndarray. (+2 more)

### Community 87 - "TestBSplineEvaluation"
Cohesion: 0.11
Nodes (10): Test suite for B-spline curve evaluation., Test evaluation of 1D linear B-spline., Test evaluation of 2D B-spline curve., Test evaluation of 3D B-spline curve., Test that endpoints are handled correctly., Test that zero-order derivative equals evaluation., Test basic derivative evaluation., Test that derivative order validation works. (+2 more)

### Community 88 - "_interpolation_system.py"
Cohesion: 0.24
Nodes (15): _add_cyclic_rows(), _add_derivative_row(), _add_endpoint_rows(), _add_interpolation_rows(), compute_control_points(), _InterpolationSystem, ndarray, Linear-system construction for exact B-spline interpolation. (+7 more)

### Community 89 - "linear_traj"
Cohesion: 0.15
Nodes (11): linear_traj(), ndarray, Generate points along a linear trajectory using NumPy vectorization. This…, Test suite for edge cases., Test trajectory with zero time duration., Test trajectory with very small time duration., Test trajectory with very large time duration., Test trajectory with negative time values. (+3 more)

### Community 90 - "SpringConfig"
Cohesion: 0.24
Nodes (15): Numerical settings for :class:`SpringQuaternionInterpolation`. Parameters…, SpringConfig, keyframes(), orientation_error(), ndarray, parametrize, Same-objective and convergence checks for the two discrete SPRING solvers., test_banded_model_matches_original_energy_and_numerical_jacobian() (+7 more)

### Community 91 - "concepts_example.cpp"
Cohesion: 0.25
Nodes (13): string, example_curve_evaluator(), example_geometric_path(), example_quaternion_trajectory(), example_scalar_trajectory(), main(), print_concept_conformance(), sample_curve() (+5 more)

### Community 92 - ".stereographic_projection"
Cohesion: 0.17
Nodes (9): Project a unit quaternion to 3D space using stereographic projection. Uses…, Test suite for stereographic projection methods., Test stereographic projection of identity quaternion., Test stereographic projection with various quaternions., Test mathematical correctness of stereographic projection., Test singularity handling near w = -1., Test exact singularity at w = -1., Test projection normalizes non-unit quaternions. (+1 more)

### Community 93 - "TestBSplineKnotHandling"
Cohesion: 0.12
Nodes (9): Test suite for knot vector handling and span finding., Test basic knot span finding., Test that knot span finding uses caching correctly., Test knot span finding at boundary conditions., Test that out-of-range parameters raise ValueError., Test uniform knot vector creation., Test uniform knot vector creation with custom domain., Test uniform knot creation input validation. (+1 more)

### Community 94 - "TestBSplineInterpolatorAdvanced"
Cohesion: 0.12
Nodes (9): Advanced test suite for BSplineInterpolator functionality., Test interpolation with various degrees comprehensively., Test interpolation of complex curves., Test accuracy of cubic degree interpolation., Test smoothness properties of cubic interpolation., Test interpolation with sufficient number of points., Test derivative evaluation if available., Test behavior with closed curve data. (+1 more)

### Community 95 - "TestBackendDetection"
Cohesion: 0.12
Nodes (8): Tests for C++ backend detection and switching., Tests that the backend detection mechanism works correctly., INTERPOLATEPY_NO_CPP=1 should force pure-Python mode., Without env var, C++ backend should be active if .so is present., Verify C++ backend classes are used when HAS_CPP is True., Quaternion should always be the pure-Python class., TestBackendDetection, TestCppClassTypes

### Community 96 - "TestParabolicBlendTrajectoryConstruction"
Cohesion: 0.12
Nodes (9): Test construction with single waypoint., Test suite for ParabolicBlendTrajectory construction and validation., Test basic construction with valid parameters., Test construction with custom sampling interval., Test construction with Python lists., Test construction with numpy arrays., Test that mismatched array lengths raise ValueError., Test construction with empty arrays. (+1 more)

### Community 97 - "TestScalarTrajectoryProtocol"
Cohesion: 0.16
Nodes (9): CubicSpline, DoubleSTrajectory, Tests for ScalarTrajectory protocol conformance., DoubleSTrajectory.evaluate should return position only (not a tuple)., evaluate_full should still return the 4-tuple., Individual methods should match evaluate_full output., CubicSpline methods should be callable through protocol., CubicSpline should NOT satisfy GeometricPath. (+1 more)

### Community 98 - "smoothing_cubic_bspline.cpp"
Cohesion: 0.21
Nodes (14): BSplineParams, MatrixXd, Parameterization, VectorXd, SmoothingCubicBSpline::calculate_approximation_error(), SmoothingCubicBSpline::calculate_control_points_impl(), SmoothingCubicBSpline::calculate_control_points_with_endpoints(), SmoothingCubicBSpline::calculate_knot_vector() (+6 more)

### Community 99 - "evaluate"
Cohesion: 0.21
Nodes (13): BSpline::basis_function_derivatives(), BSpline::basis_functions(), BSpline::BSpline(), BSpline::create_periodic_knots(), BSpline::create_uniform_knots(), BSpline::evaluate(), BSpline::evaluate_derivative(), BSpline::generate_curve_points() (+5 more)

### Community 100 - "ndarray"
Cohesion: 0.18
Nodes (8): ndarray, Find the knot span index for a given parameter value u. Parameters ---------- u…, Calculate all non-zero basis functions at parameter value u. Parameters…, Evaluate the B-spline curve at parameter value u. Parameters ---------- u :…, Calculate derivatives of basis functions up to the specified order. Parameters…, Evaluate the derivative of the B-spline curve at parameter value u. Parameters…, Generate points along the B-spline curve for visualization. Parameters…, Initialize a B-spline curve. Parameters ---------- degree : int The degree of…

### Community 101 - "TestFrenetFrames"
Cohesion: 0.17
Nodes (9): ndarray, Test suite for Frenet frame computation., Create a trajectory function for linear path., Create a trajectory function for circular path., Test Frenet frames for linear trajectory., Test Frenet frames with tool orientation., Test Frenet frames with roll-pitch-yaw orientation., Test Frenet frames with edge cases. (+1 more)

### Community 102 - "test_protocols.py"
Cohesion: 0.18
Nodes (14): circular_path(), cubic_spline(), linear_path(), log_quat_interp(), fixture, quaternion_spline(), Tests for protocol conformance and functional behavior. Verifies that all…, LogQuaternionInterpolation instance. (+6 more)

### Community 103 - "test_spring_performance.py"
Cohesion: 0.25
Nodes (14): eager_minimize(), ndarray, parametrize, SpringConfig, Regression checks for result-preserving SPRING implementation optimizations., Frozen pre-optimization line search, computing every trial gradient., test_batched_slerp_matches_scalar_operations(), test_batched_slerp_retains_small_angle_and_antipodal_rules() (+6 more)

### Community 104 - "TestGeometricPathProtocol"
Cohesion: 0.16
Nodes (8): CircularPath, LinearPath, LinearPath should NOT satisfy ScalarTrajectory., Tests for GeometricPath protocol conformance., LinearPath should work through GeometricPath interface., CircularPath should work through GeometricPath interface., LinearPath should NOT satisfy QuaternionTrajectory (no evaluate method)., TestGeometricPathProtocol

### Community 105 - "parabolic_linear_example.cpp"
Cohesion: 0.19
Nodes (11): example_linear_scalar(), example_linear_vector(), example_parabolic_blend(), main(), MatrixXd, LinearTrajResult, accelerations, positions (+3 more)

### Community 106 - "SplineConfig"
Cohesion: 0.14
Nodes (14): optional, VectorXd, SplineConfig, debug, max_iterations, v0, vn, weights (+6 more)

### Community 107 - "cubic_smoothing_spline.cpp"
Cohesion: 0.16
Nodes (9): optional, SegmentInfo, span, VectorXd, CubicSmoothingSpline::CubicSmoothingSpline(), CubicSmoothingSpline::evaluate(), CubicSmoothingSpline::evaluate_acceleration(), CubicSmoothingSpline::evaluate_velocity() (+1 more)

### Community 108 - "quaternion_spline.cpp"
Cohesion: 0.15
Nodes (10): vector, Vector3d, QuaternionSpline::compute_intermediates(), QuaternionSpline::evaluate(), QuaternionSpline::evaluate_acceleration(), QuaternionSpline::evaluate_velocity(), QuaternionSpline::QuaternionSpline(), Method (+2 more)

### Community 109 - "c_s_with_acc1_ex.py"
Cohesion: 0.20
Nodes (13): CubicSplineWithAcceleration1, camera_pan_example(), compare_boundary_conditions(), drone_height_example(), multi_dimensional_example(), Example for planning a drone's height trajectory with acceleration limits and…, Example for creating a 3D trajectory using three independent splines for x, y,…, Example comparing different boundary conditions for the same waypoints.… (+5 more)

### Community 110 - "User guide"
Cohesion: 0.14
Nodes (14): B-spline curves, Callable trajectory generators, Double-S trajectories, Evaluation conventions, Geometric paths, Imports and backend routing, Input rules, Plotting (+6 more)

### Community 111 - "b_spline_approx_ex.py"
Cohesion: 0.18
Nodes (13): create_test_shapes(), example_approximation(), example_degree_comparison(), example_different_shapes(), example_method_comparison(), example_noise_sensitivity(), ndarray, Demonstrate B-spline approximation with the example from Section 8.5. Args:… (+5 more)

### Community 112 - "b_spline_ex.py"
Cohesion: 0.21
Nodes (13): create_simple_3d_bspline(), demonstrate_3d_bspline(), demonstration(), example_b6(), example_bspline(), plot_basis_functions(), BSpline, Plot the basis functions and mark the evaluation point. Args: bspline: The… (+5 more)

### Community 113 - "log_quat_new_ex.py"
Cohesion: 0.21
Nodes (13): create_basic_trajectory(), create_complex_trajectory(), demo_basic_interpolation(), demo_comparison_with_traditional_methods(), demo_lqi_vs_mlqi_comparison(), main(), Logarithmic Quaternion Interpolation (LQI) Examples This example demonstrates…, Detailed comparison between LQI and mLQI methods. (+5 more)

### Community 114 - "_approximation_system.py"
Cohesion: 0.23
Nodes (13): approximate_control_points(), ApproximationProblem, _basis_row(), _build_matrices(), BSpline, ndarray, Weighted least-squares system for B-spline approximation., Expand the locally nonzero basis functions into a full matrix row. (+5 more)

### Community 115 - "TestLinearTrajectoryScalar"
Cohesion: 0.14
Nodes (8): Test scalar trajectory extrapolation outside time range., Test scalar trajectory with single time point., Test suite for scalar linear trajectories., Test basic scalar linear trajectory., Test scalar trajectory with negative displacement., Test scalar trajectory with zero displacement., Test scalar trajectory with non-zero start time., TestLinearTrajectoryScalar

### Community 116 - "TestPathPlanningPerformance"
Cohesion: 0.19
Nodes (9): FixtureFunction, parametrize, Benchmark path evaluation performance., Benchmark Frenet frame computation performance., Helper method for creating circular trajectory function., Test Frenet frames for circular trajectory., Test suite for performance benchmarks., Benchmark path construction performance. (+1 more)

### Community 117 - "TestPolynomialTrajectoryHeuristicVelocities"
Cohesion: 0.14
Nodes (8): Test suite for heuristic velocity calculation., Test basic heuristic velocity calculation., Test heuristic velocities for linear trajectory., Test heuristic velocities for parabolic trajectory., Test heuristic velocities with non-uniform time spacing., Test heuristic velocities with minimum number of points., Test heuristic velocities with some identical points., TestPolynomialTrajectoryHeuristicVelocities

### Community 118 - "SpringConfig"
Cohesion: 0.15
Nodes (11): string, SpringConfig, final_iterations, iterations, keyframe_curvature_weight, norm_penalty, num_samples, refinement_levels (+3 more)

### Community 119 - "modified_log_quaternion_interpolation.cpp"
Cohesion: 0.18
Nodes (11): MatrixXd, optional, pair, vector, Vector4d, VectorXd, ModifiedLogQuaternionInterpolation::evaluate(), ModifiedLogQuaternionInterpolation::evaluate_acceleration() (+3 more)

### Community 120 - "Quaternion interpolation"
Cohesion: 0.15
Nodes (12): A faster solver for the same discrete SPRING problem, Choosing a method, Compare orientations correctly, Construct and convert orientations, Logarithmic quaternion interpolation, Modified logarithmic interpolation, Multiple shooting for natural Riemannian cubics, Quaternion interpolation (+4 more)

### Community 121 - "polynomials_ex.py"
Cohesion: 0.23
Nodes (12): main(), multipoint_interpolation_example(), multipoint_interpolation_no_vel_example(), plot_trajectory(), TimeInterval, Example of using polynomial trajectories for interpolation. This example…, Demonstrate multi-point interpolation with different polynomial orders., Plot the generated trajectory. Parameters ---------- trajectory_func :… (+4 more)

### Community 122 - ".evaluate_full"
Cohesion: 0.21
Nodes (7): ndarray, Evaluate all trajectory components at time t. Parameters ---------- t : float…, Evaluate position at time t. Parameters ---------- t : float or ndarray Time(s)…, Evaluate velocity at time t. Parameters ---------- t : float or ndarray Time(s)…, Evaluate acceleration at time t. Parameters ---------- t : float or ndarray…, Evaluate jerk at time t. Parameters ---------- t : float or ndarray Time(s) at…, Static factory method to create a trajectory function and return its duration.…

### Community 123 - "TestEdgeCasesAndErrorHandling"
Cohesion: 0.15
Nodes (7): Test suite for edge cases and error handling., Test numerical stability near singularity points., Test handling of large angle rotations., Test handling of near-zero quaternions., Test trajectory with nearly opposite quaternions., Test error handling for invalid MRP dimensions., TestEdgeCasesAndErrorHandling

### Community 125 - "test_polynomial_trajectory.cpp"
Cohesion: 0.17
Nodes (11): "Heuristic velocities", "Multipoint trajectory", "Order 3 basic trajectory", "Order 3 jerk constant", "Order 3 negative displacement", "Order 3 nonzero velocities", "Order 5 basic trajectory", "Order 5 nonzero accelerations" (+3 more)

### Community 126 - "ParabolicBlendTrajectory"
Cohesion: 0.32
Nodes (6): _CppParabolicBlendTrajectory, ParabolicBlendTrajectory, ndarray, C++-backed parabolic blend with Python-compatible evaluation methods., Return a Python-style trajectory callable and total duration., Plot position, velocity, and acceleration samples.

### Community 127 - "Algorithms"
Cohesion: 0.17
Nodes (12): Acceleration-constrained splines, Algorithms, B-splines, Cubic interpolation, Cubic smoothing, Double-S motion, Numerical practice, Paths and frames (+4 more)

### Community 128 - "simple_paths_ex.py"
Cohesion: 0.27
Nodes (11): circular_path_example(), linear_path_example(), main(), plot_3d_path(), plot_motion_profiles(), ndarray, Simple example demonstrating how to use geometric paths with polynomial motion…, Example with a linear path and polynomial motion. (+3 more)

### Community 129 - ".create_uniform_knots"
Cohesion: 0.24
Nodes (8): Create a uniform knot vector for a B-spline with appropriate multiplicity at…, FixtureFunction, parametrize, Test suite for performance benchmarks., Benchmark B-spline construction performance., Benchmark B-spline evaluation performance., Benchmark basis function calculation performance., TestBSplinePerformance

### Community 130 - "test_lin_poly_parabolic.py"
Cohesion: 0.17
Nodes (7): Linear trajectories with parabolic blending at via points. This module…, Comprehensive tests for linear-parabolic trajectory implementation. This module…, Integration tests combining multiple features., Test complete workflow similar to the example script., Test trajectory with different blend durations at each point., Compare trajectory with small blend durations (approximates linear)., TestParabolicBlendTrajectoryIntegration

### Community 131 - "test_path_planning.py"
Cohesion: 0.17
Nodes (7): Module for simple geometric path primitives. Provides basic geometric path…, Comprehensive tests for path planning implementations. This module contains…, Test suite for edge cases in path planning., Test linear path with very small length., Test circular path with very small radius., Test paths with large coordinate values., TestPathPlanningEdgeCases

### Community 132 - "TestLinearTrajectoryVector"
Cohesion: 0.17
Nodes (7): Test suite for vector linear trajectories., Test basic 2D vector trajectory., Test 3D vector trajectory., Test vector trajectory with numpy array inputs., Test vector trajectories with different dimensions., Test vector trajectory with mixed positive/negative directions., TestLinearTrajectoryVector

### Community 133 - "TestMotionProfilePerformance"
Cohesion: 0.23
Nodes (8): FixtureFunction, parametrize, Test suite for performance benchmarks., Benchmark DoubleSTrajectory construction performance., Benchmark DoubleSTrajectory evaluation performance., Benchmark trapezoidal trajectory generation performance., Benchmark waypoint interpolation performance., TestMotionProfilePerformance

### Community 134 - "TestDoubleSTrajectoryEdgeCases"
Cohesion: 0.17
Nodes (7): Test suite for DoubleSTrajectory edge cases., Test trajectory with zero displacement., Test trajectory with very small displacement., Test trajectory with large displacement., Test trajectory with negative displacement., Test trajectory with non-zero initial and final velocities., TestDoubleSTrajectoryEdgeCases

### Community 135 - "cubic_smoothing_example.cpp"
Cohesion: 0.22
Nodes (8): bind_smoothing_search(), module_, main(), smoothing_mu_example(), smoothing_tolerance_example(), span, smoothing_spline_with_tolerance(), SplineConfig

### Community 136 - "bspline_approx_smooth_example.cpp"
Cohesion: 0.33
Nodes (10): MatrixXd, example_8_12(), example_approximation(), example_cp_count_comparison(), example_degree_comparison(), example_smoothing_mu_comparison(), example_smoothing_with_derivatives(), example_weighted_approximation() (+2 more)

### Community 137 - "test_bspline.cpp"
Cohesion: 0.18
Nodes (10): "BSpline basis functions", "BSpline construction", "BSpline construction validation", "BSpline curve generation", "BSpline edge cases", "BSpline evaluation", "BSpline knot span", "BSpline numerical stability" (+2 more)

### Community 138 - "Contributing"
Cohesion: 0.18
Nodes (11): Backend parity, Before submitting a change, C++ build and tests, Contributing, Development setup, Documentation, Example programs, Pull requests (+3 more)

### Community 139 - "Spline interpolation"
Cohesion: 0.18
Nodes (10): Approximate or smooth vector data, Common mistakes, Constrain endpoint acceleration, Construct a B-spline from controls, Interpolate B-spline samples, Interpolate scalar waypoints, Plotting, Select smoothing by error tolerance (+2 more)

### Community 140 - "main"
Cohesion: 0.25
Nodes (10): close_figures(), documentation_files(), is_standalone_example(), main(), Path, Execute standalone Python examples embedded in the project documentation., Return Markdown sources that can contain public examples., Identify snippets that declare their own InterpolatePy imports. (+2 more)

### Community 141 - "TestParabolicBlendTrajectoryPerformance"
Cohesion: 0.24
Nodes (7): FixtureFunction, Test suite for performance benchmarks., Benchmark trajectory construction., Benchmark trajectory generation., Benchmark trajectory evaluation., Benchmark trajectory with many waypoints., TestParabolicBlendTrajectoryPerformance

### Community 142 - "test_native_adapter_api.py"
Cohesion: 0.18
Nodes (5): parametrize, Regression tests for the package-level API with the C++ backend active., test_shooting_native_adapter_matches_python_reference(), test_spring_gauss_newton_native_adapter_matches_python(), test_spring_native_adapter_matches_python_reference()

### Community 143 - "bspline_example.cpp"
Cohesion: 0.38
Nodes (8): create_example_bspline(), demonstrate_3d_bspline(), demonstrate_basic_bspline(), demonstrate_curve_generation(), demonstrate_periodic_knots(), example_b6(), main(), BSpline()

### Community 144 - "approximation_bspline.cpp"
Cohesion: 0.33
Nodes (9): ApproximationBSpline::approximate_control_points(), ApproximationBSpline::ApproximationBSpline(), ApproximationBSpline::calculate_approximation_error(), ApproximationBSpline::compute_knots(), ApproximationBSpline::compute_parameters(), MatrixXd, optional, Parameterization (+1 more)

### Community 145 - "log_quaternion_interpolation.cpp"
Cohesion: 0.22
Nodes (9): MatrixXd, optional, vector, Vector3d, VectorXd, LogQuaternionInterpolation::evaluate_acceleration(), LogQuaternionInterpolation::evaluate_velocity(), LogQuaternionInterpolation::LogQuaternionInterpolation() (+1 more)

### Community 146 - "test_shooting_quaternion.cpp"
Cohesion: 0.20
Nodes (9): vector, curved_keyframes(), "Shooting agrees with a known scalar natural cubic", "Shooting handles extreme finite time units without false convergence or NaN energy", "Shooting is invariant to quaternion scale signs and time units", "Shooting output resolution does not alter the solve", "Shooting preserves a rotated constant speed geodesic", "Shooting preserves keyframes and matches natural C2 boundaries" (+1 more)

### Community 147 - "Run Python examples"
Cohesion: 0.20
Nodes (10): B-splines, C++ examples, Example programs, Motion profiles, Paths and generic interfaces, Quaternion interpolation, Run every Python example headlessly, Run Python examples (+2 more)

### Community 148 - "TimingResult"
Cohesion: 0.20
Nodes (7): print_timing_table(), Print aligned median construction and evaluation timings., Median wall-clock timings for one interpolation method., Return median construction time in milliseconds., Return median time for the complete evaluation batch in milliseconds., Return median batch time normalized by its sample count., TimingResult

### Community 149 - "plot_individual_methods"
Cohesion: 0.24
Nodes (9): create_simple_trajectory(), main(), plot_individual_methods(), NDArray, Simple SQUAD vs SQUAD C2 Quaternion Interpolation Example This example…, Main demonstration function., Create a simple quaternion trajectory for demonstration., Plot individual method analysis with separate plots for each interpolation… (+1 more)

### Community 150 - "main"
Cohesion: 0.27
Nodes (9): Namespace, check_backend(), main(), parse_args(), Path, Run every Python example in a separate headless subprocess., Parse command-line options., Verify that native mode actually loaded the compiled extension. (+1 more)

### Community 151 - "._setup_spline"
Cohesion: 0.20
Nodes (5): Set the interpolation method for this spline. Args: method: "slerp", "squad",…, Initialize quaternion spline interpolator. Args: time_points: List of time…, Setup this quaternion as a spline interpolator. Args: time_points: List of time…, Validate input data for spline construction., Precompute intermediate quaternions for smooth Squad interpolation

### Community 152 - "PlotStyle"
Cohesion: 0.27
Nodes (7): PlotStyle, Configuration for plot styling., Test suite for PlotStyle dataclass., Test PlotStyle initialization with default values., Test PlotStyle initialization with custom values., Test PlotStyle initialization with some custom values., TestPlotStyle

### Community 153 - "TestBSplineCurveGeneration"
Cohesion: 0.20
Nodes (6): Test suite for curve point generation and sampling., Test basic curve point generation., Test curve point generation for 1D case., Test curve point generation for 3D case., Test curve point generation with different point counts., TestBSplineCurveGeneration

### Community 154 - "TestCubicSplinePerformance"
Cohesion: 0.27
Nodes (7): FixtureFunction, parametrize, Test suite for performance benchmarks., Benchmark spline construction performance., Benchmark spline evaluation performance., Benchmark derivative evaluation performance., TestCubicSplinePerformance

### Community 155 - "TestParabolicBlendTrajectoryMathematicalProperties"
Cohesion: 0.20
Nodes (6): Test suite for mathematical properties and accuracy., Test that trajectory passes through corrected via points., Test that velocity profile is reasonably smooth., Test that acceleration is non-zero in blends and zero in linear segments., Test energy-related properties of the trajectory., TestParabolicBlendTrajectoryMathematicalProperties

### Community 156 - "TestParabolicBlendTrajectoryEdgeCases"
Cohesion: 0.20
Nodes (6): Test suite for edge cases and error conditions., Test trajectory with identical consecutive waypoints., Test with very small time intervals between waypoints., Test trajectory with negative position values., Test trajectory with only one waypoint., TestParabolicBlendTrajectoryEdgeCases

### Community 157 - "TestLinearTrajectoryPerformance"
Cohesion: 0.27
Nodes (7): FixtureFunction, parametrize, Test suite for performance benchmarks., Benchmark scalar trajectory performance., Benchmark vector trajectory performance., Benchmark performance with large datasets., TestLinearTrajectoryPerformance

### Community 158 - "TestLinearTrajectoryMathematicalProperties"
Cohesion: 0.20
Nodes (6): Test suite for mathematical properties verification., Test that trajectory is truly linear., Test that trajectory interpolates correctly at boundary points., Test that midpoint has correct value., Test linear superposition for vector trajectories., TestLinearTrajectoryMathematicalProperties

### Community 159 - "TestDoubleSTrajectoryConstruction"
Cohesion: 0.20
Nodes (6): Test basic DoubleSTrajectory construction., Test that non-numeric state parameters raise TypeError., Test construction with various start/end states., A feasible zero-duration phase must not leave jerk times undefined., Test suite for DoubleSTrajectory construction and validation., TestDoubleSTrajectoryConstruction

### Community 160 - "test_package_structure.py"
Cohesion: 0.20
Nodes (9): Tests for the installed package layout and compatibility surface., New domain namespaces expose the backend-neutral public classes., Only the modern domain modules are shipped at the package root., PEP 561 marker is present in the installed package., Numerical use should not import the optional plotting stack., test_core_import_does_not_eagerly_load_matplotlib(), test_domain_namespaces_match_top_level_api(), test_flat_compatibility_modules_are_removed() (+1 more)

### Community 161 - "TestTimeInterval"
Cohesion: 0.20
Nodes (6): Test TimeInterval with negative start time., Test TimeInterval with zero duration., Test suite for TimeInterval dataclass., Test TimeInterval creation., Test that we can calculate duration from time interval., TestTimeInterval

### Community 162 - "TestBoundaryCondition"
Cohesion: 0.20
Nodes (6): Test suite for BoundaryCondition dataclass., Test BoundaryCondition creation with minimal parameters., Test BoundaryCondition creation with all parameters., Test that default values are correctly applied., Test BoundaryCondition with negative values., TestBoundaryCondition

### Community 163 - "bspline_interpolator_example.cpp"
Cohesion: 0.42
Nodes (8): example_3d(), example_cubic_bspline(), example_cyclic(), example_degree5(), example_jerk_continuous(), main(), print_scalar_trajectory(), BSplineInterpolator()

### Community 164 - "cubic_bspline_interpolation.cpp"
Cohesion: 0.39
Nodes (8): MatrixXd, optional, Parameterization, VectorXd, CubicBSplineInterpolation::calculate_control_points(), CubicBSplineInterpolation::calculate_knot_vector(), CubicBSplineInterpolation::calculate_parameters(), CubicBSplineInterpolation::CubicBSplineInterpolation()

### Community 165 - "SpringEnergy"
Cohesion: 0.28
Nodes (6): Frames, size_t, vector, Vector3d, SpringEnergy, norm_penalty_

### Community 166 - "test_data.hpp"
Cohesion: 0.22
Nodes (6): "TrapezoidalTrajectory duration-based", "TrapezoidalTrajectory edge cases", "TrapezoidalTrajectory heuristic velocities", "TrapezoidalTrajectory velocity-based", "TrapezoidalTrajectory waypoint interpolation", "Tridiagonal solver"

### Community 167 - "Installation"
Cohesion: 0.22
Nodes (9): Build and activate the Python extension, Build and test the C++ library, Development checkout, Force the Python backend, Install from PyPI, Installation, Installation troubleshooting, Optional C++ backend (+1 more)

### Community 168 - "b_spline_interpolate_ex.py"
Cohesion: 0.22
Nodes (8): example_cubic_bspline(), example_cyclic_bspline(), example_jerk_continuous_bspline(), Plot a cubic B-spline trajectory with endpoint velocity constraints., Plot a cyclic degree-four B-spline trajectory., Demonstrate a simple 3D B-spline interpolation with a basic curved path., Plot a degree-four B-spline trajectory with continuous jerk., simple_3d_example()

### Community 169 - "TestNumericalEquivalence"
Cohesion: 0.25
Nodes (5): fixture, C++ and Python DoubleSTrajectory should produce matching results., Compare Python and C++ results for key algorithms., C++ and Python CubicSpline should produce identical results., TestNumericalEquivalence

### Community 170 - "TestParabolicBlendTrajectoryPlotting"
Cohesion: 0.28
Nodes (6): patch, Test suite for plotting functionality., Test plot method without providing trajectory data., Test plot method with provided trajectory data., Test plot method with custom sampling interval., TestParabolicBlendTrajectoryPlotting

### Community 171 - "InterpolatePy algorithm guide"
Cohesion: 0.25
Nodes (8): B-spline curves, Backend and numerical notes, Choose by problem, InterpolatePy algorithm guide, Paths and Frenet frames, Quaternion interpolation, Scalar cubic splines, Scalar motion profiles

### Community 172 - "cubic_spline_acc1_example.cpp"
Cohesion: 0.46
Nodes (7): camera_pan_example(), compare_boundary_conditions(), drone_height_example(), main(), multi_dimensional_example(), robot_joint_example(), simple_example()

### Community 173 - "circular_path.cpp"
Cohesion: 0.36
Nodes (7): CircularPath::acceleration(), CircularPath::CircularPath(), CircularPath::position(), CircularPath::velocity(), MatrixXd, Vector3d, VectorXd

### Community 174 - "linear_path.cpp"
Cohesion: 0.36
Nodes (7): MatrixXd, Vector3d, VectorXd, LinearPath::acceleration(), LinearPath::LinearPath(), LinearPath::position(), LinearPath::velocity()

### Community 175 - "test_double_s_trajectory.cpp"
Cohesion: 0.25
Nodes (7): "DoubleSTrajectory boundary conditions", "DoubleSTrajectory bounds validation", "DoubleSTrajectory construction", "DoubleSTrajectory edge cases", "DoubleSTrajectory evaluation", "DoubleSTrajectory phase durations", "DoubleSTrajectory velocity bounds"

### Community 176 - "test_parabolic_blend_trajectory.cpp"
Cohesion: 0.25
Nodes (7): "ParabolicBlendTrajectory construction", "ParabolicBlendTrajectory identical waypoints", "ParabolicBlendTrajectory negative positions", "ParabolicBlendTrajectory position continuity", "ParabolicBlendTrajectory single waypoint", "ParabolicBlendTrajectory three-point", "ParabolicBlendTrajectory two-point"

### Community 177 - "test_quaternion.cpp"
Cohesion: 0.25
Nodes (7): "Quaternion arithmetic", "Quaternion construction", "Quaternion conversions", "Quaternion exp/log", "Quaternion intermediate", "Quaternion SLERP", "Quaternion SQUAD"

### Community 178 - "test_smoothing_search.cpp"
Cohesion: 0.25
Nodes (7): vector, make_noisy_sine(), "Smoothing spline with tolerance - basic search", "Smoothing spline with tolerance - convergence", "Smoothing spline with tolerance - result spline is usable", "Smoothing spline with tolerance - with boundary conditions", "Smoothing spline with tolerance - with weights"

### Community 179 - "Architecture"
Cohesion: 0.25
Nodes (8): Adapter layer, Architecture, Backend detection, C++ targets, Overview, Public import routing, Source layout, Testing the two implementations

### Community 180 - "Motion profiles"
Cohesion: 0.25
Nodes (7): Choosing a profile, Double-S: bound jerk, acceleration, and speed, Linear segments with parabolic blends, Motion profiles, Polynomial boundary interpolation, Trapezoidal interpolation through via points, Trapezoidal velocity profile

### Community 181 - "Path planning"
Cohesion: 0.25
Nodes (7): Add a time law, Circular path, Custom curve, Frenet frames, Linear path, Path planning, Tool orientation

### Community 182 - ".__init__"
Cohesion: 0.32
Nodes (5): BSpline, ndarray, Create the knot vector based on the degree. - For odd degrees (3, 5): knots at…, Build and solve the interpolation system for its control points., Initialize a B-spline interpolator. Parameters ---------- degree : int The…

### Community 183 - "TestLinearTrajectoryInputValidation"
Cohesion: 0.25
Nodes (5): Test suite for input validation and type handling., Test trajectory with list inputs., Test trajectory with mixed input types., Test trajectory with single-element position arrays., TestLinearTrajectoryInputValidation

### Community 184 - "TestCurveEvaluatorProtocol"
Cohesion: 0.25
Nodes (5): Tests for CurveEvaluator protocol conformance., BSplineInterpolator should satisfy CurveEvaluator., BSpline should satisfy CurveEvaluator., CubicSpline should NOT satisfy CurveEvaluator (no evaluate_derivative)., TestCurveEvaluatorProtocol

### Community 185 - ".create_test_spline_trajectory"
Cohesion: 0.25
Nodes (4): Create a test quaternion spline trajectory., Test visualization with quaternion spline trajectory., Test velocity analysis with quaternion spline trajectory., Test visualization consistency across interpolation methods.

### Community 186 - "trapezoidal_example.cpp"
Cohesion: 0.52
Nodes (6): example_basic_trajectory(), example_duration_based(), example_multipoint_heuristic(), example_nonzero_velocities(), example_time_constrained(), main()

### Community 187 - "ShootingConfig"
Cohesion: 0.29
Nodes (5): ShootingConfig, integration_steps, max_integration_steps, max_iterations, tolerance

### Community 188 - "bspline_interpolator.cpp"
Cohesion: 0.48
Nodes (6): BSplineInterpolator::BSplineInterpolator(), BSplineInterpolator::compute_control_points(), BSplineInterpolator::create_knot_vector(), MatrixXd, optional, VectorXd

### Community 189 - "test_cubic_smoothing_spline.cpp"
Cohesion: 0.29
Nodes (6): "CubicSmoothingSpline construction", "CubicSmoothingSpline edge cases", "CubicSmoothingSpline evaluation", "CubicSmoothingSpline exact interpolation (mu=1)", "CubicSmoothingSpline smoothing effect", "CubicSmoothingSpline with weights"

### Community 190 - "InterpolatePy"
Cohesion: 0.29
Nodes (7): Backend selection, Find the right tool, First trajectory, Install, InterpolatePy, Next steps, Version 3.2 highlights

### Community 191 - "linear_ex.py"
Cohesion: 0.38
Nodes (6): main(), Example file demonstrating the use of linear_traj function. This example shows:…, Demonstrate linear_traj with scalar positions., Demonstrate linear_traj with vector positions (2D points)., scalar_trajectory_example(), vector_trajectory_example()

### Community 192 - "QuaternionTrajectory"
Cohesion: 0.29
Nodes (4): create_factories(), Create fresh-construction callables for every compared method., QuaternionTrajectory, Protocol for quaternion-valued trajectory evaluation. Conforming classes…

### Community 193 - "InterpolatePy"
Cohesion: 0.29
Nodes (7): Development, Installation, InterpolatePy, License and citation, Optional C++ backend, Quick start, What is included

### Community 194 - "TestBSplineVariantsPerformance"
Cohesion: 0.33
Nodes (5): FixtureFunction, Test suite for performance benchmarks of B-spline variants., Benchmark construction performance for different variants., Benchmark evaluation performance across variants., TestBSplineVariantsPerformance

### Community 195 - "polynomial_example.cpp"
Cohesion: 0.60
Nodes (5): example_multipoint_heuristic(), example_order3_two_point(), example_order5_two_point(), example_order7_comparison(), main()

### Community 196 - "Quick start"
Cohesion: 0.33
Nodes (6): B-spline through two or more points, Geometric path plus a time law, Jerk-limited point-to-point motion, Quaternion keyframes, Quick start, Scalar spline

### Community 197 - ".create_periodic_knots"
Cohesion: 0.33
Nodes (3): Create a periodic (uniform) knot vector for a B-spline. Parameters ----------…, Test periodic knot vector creation., Test periodic knot creation input validation.

### Community 198 - ".plot"
Cohesion: 0.33
Nodes (3): ndarray, Plot the trajectory's position, velocity, and acceleration. If trajectory data…, Generate the parabolic blend trajectory function. Returns -------…

### Community 199 - "MinimizationSettings"
Cohesion: 0.33
Nodes (3): MinimizationSettings, Protocol, Structural subset of settings required by :func:`minimize`.

### Community 200 - "test_motion_profiles.py"
Cohesion: 0.33
Nodes (4): Comprehensive tests for motion profile implementations. This module contains…, Test suite for DoubleSTrajectory static methods., Test static trajectory creation method., TestDoubleSTrajectoryStaticMethods

### Community 201 - "TestTrajectoryFunctionProtocol"
Cohesion: 0.33
Nodes (4): Tests for TrajectoryFunction protocol conformance., A simple callable returning a 3-tuple should satisfy TrajectoryFunction., Lambda should also satisfy TrajectoryFunction., TestTrajectoryFunctionProtocol

### Community 202 - "main"
Cohesion: 0.50
Nodes (3): main(), Compare multiple-shooting setup/sampling with coarse and dense SPRING. Run with…, Print median warm construction and fixed-size output sampling timings.

### Community 203 - "main"
Cohesion: 0.50
Nodes (3): main(), Compare solvers of the SAME discrete SPRING problem at a common tolerance.…, Measure time, stationarity and angular agreement on identical inputs.

### Community 204 - "modified_log_quat_interp"
Cohesion: 0.50
Nodes (3): ModifiedLogQuaternionInterpolation, modified_log_quat_interp(), ModifiedLogQuaternionInterpolation instance.

### Community 206 - "_solve_error_message"
Cohesion: 0.67
Nodes (3): LinAlgError, Build an actionable message for an unsolvable interpolation system., _solve_error_message()

## Knowledge Gaps
- **420 isolated node(s):** `mu`, `weights`, `v0`, `vn`, `method` (+415 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 2049 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Quaternion` connect `Quaternion` to `BoundaryCondition`, `SquadC2`, `shooting_solver.cpp`, `.setup_test_data`, `ModifiedLogQuaternionInterpolation`, `.identity`, `.from_euler_angles`, `QuaternionTrajectoryVisualizer`, `plot_individual_methods`, `._setup_spline`, `SquadC2`, `TestQuaternionBasicOperations`, `_adapters/__init__.py`, `quaternion.cpp`, `quaternion_example.cpp`, `SpringQuaternionInterpolation`, `TestQuaternionDynamics`, `.slerp`, `LogQuaternionInterpolation`, `QuaternionSpline`, `shooting.py`, `SpringQuaternionInterpolation`, `TestQuaternionConversions`, `TestQuaternionSpline`, `QuaternionSpline`, `compute_trajectory_frames`, `.create_test_spline_trajectory`, `ndarray`, `test_spring_quaternion.py`, `QuaternionTrajectory`, `spring_quaternion_ex.py`, `quaternion/core.py`, `TestQuaternionMathematics`, `ndarray`, `.inverse_stereographic_projection`, `TestPlottingFunctionality`, `quat_visualization_ex.py`, `spring_quaternion_interpolation.cpp`, `interpolatepy/__init__.py`, `spring.py`, `SpringConfig`, `.stereographic_projection`, `test_spring_performance.py`, `quaternion_spline.cpp`, `log_quat_new_ex.py`, `modified_log_quaternion_interpolation.cpp`, `TestEdgeCasesAndErrorHandling`?**
  _High betweenness centrality (0.371) - this node is a cross-community bridge._
- **Why does `BSpline` connect `BSpline` to `.create_uniform_knots`, `ndarray`, `_api.py`, `.create_periodic_knots`, `SmoothingCubicBSpline`, `_bspline.py`, `ApproximationBSpline`, `CubicBSplineInterpolation`, `_approximation_system.py`, `TestBSplineEvaluation`, `TestBSplineCurveGeneration`, `test_b_spline_variants.py`, `TestBSplineKnotHandling`, `BSplineInterpolator`?**
  _High betweenness centrality (0.100) - this node is a cross-community bridge._
- **Why does `evaluate()` connect `evaluate` to `quaternion_example.cpp`, `smoothing_cubic_bspline.cpp`, `SpringEnergy`, `coefficients_`, `shooting_solver.cpp`, `test_spring_quaternion.cpp`, `cubic_spline_with_acc1.cpp`, `cubic_smoothing_spline.cpp`, `quaternion_spline.cpp`, `spring_quaternion_interpolation.cpp`, `approximation_bspline.cpp`?**
  _High betweenness centrality (0.063) - this node is a cross-community bridge._
- **Are the 56 inferred relationships involving `Quaternion` (e.g. with `LogQuaternionInterpolation` and `ModifiedLogQuaternionInterpolation`) actually correct?**
  _`Quaternion` has 56 INFERRED edges - model-reasoned connections that need verification._
- **Are the 19 inferred relationships involving `BSpline` (e.g. with `ApproximationBSpline` and `BSpline`) actually correct?**
  _`BSpline` has 19 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `CubicSpline` (e.g. with `CubicSpline` and `TestNumericalEquivalence`) actually correct?**
  _`CubicSpline` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `BoundaryCondition` (e.g. with `circular_path_example()` and `linear_path_example()`) actually correct?**
  _`BoundaryCondition` has 11 INFERRED edges - model-reasoned connections that need verification._