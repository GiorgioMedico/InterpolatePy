# InterpolatePy algorithm guide

This page is a compact map of the algorithms exposed by InterpolatePy 3.2.0.
For constructor signatures and complete method documentation, use the
[API reference](docs/api-reference.md). For worked code, use the
[tutorials](docs/user-guide.md) and [`examples/`](examples/).

## Choose by problem

| Problem | Start with | Why |
| --- | --- | --- |
| Interpolate scalar waypoints with specified endpoint velocities | `CubicSpline` | Piecewise cubic, C2 across interior knots, inexpensive evaluation |
| Fit noisy scalar samples | `CubicSmoothingSpline` | Trades waypoint error against curvature through `mu` and weights |
| Also constrain endpoint accelerations | `CubicSplineWithAcceleration1` or `CubicSplineWithAcceleration2` | Adds endpoint acceleration conditions with two different constructions |
| Evaluate a curve from known B-spline control points | `BSpline` | Direct control over degree, knots, and control points |
| Pass a B-spline through parametric samples | `BSplineInterpolator` | Degrees 3, 4, and 5; optional velocity/acceleration/cyclic constraints |
| Approximate many points with fewer controls | `ApproximationBSpline` | Weighted least-squares curve reduction |
| Smooth a vector-valued curve | `SmoothingCubicBSpline` | Cubic B-spline fit with a smoothness/fidelity trade-off |
| Limit velocity, acceleration, and jerk | `DoubleSTrajectory` | Seven-phase jerk-limited Double-S profile |
| Limit velocity and acceleration | `TrapezoidalTrajectory` | Trapezoidal or triangular velocity profile |
| Match endpoint derivatives exactly | `PolynomialTrajectory` | Cubic, quintic, or seventh-order polynomial boundary interpolation |
| Blend a scalar via-point sequence | `ParabolicBlendTrajectory` | Linear segments joined by parabolic blends |
| Interpolate two orientations | `Quaternion.slerp()` | Shortest-path spherical interpolation |
| Interpolate orientation keyframes | `QuaternionSpline` | Piecewise SLERP/SQUAD selected explicitly or automatically |
| Require a smooth SQUAD-style orientation trajectory | `SquadC2` | Virtual endpoints and smooth angular derivative evaluation |
| Interpolate rotations in logarithmic coordinates | `LogQuaternionInterpolation` | B-spline interpolation of continuous rotation vectors |
| Decouple rotation angle and axis | `ModifiedLogQuaternionInterpolation` | Separate angle and unit-axis spline state |
| Describe a 3D line or circle geometrically | `LinearPath` or `CircularPath` | Arc-length parameterized position and geometric derivatives |
| Attach a moving frame to a parametric curve | `compute_trajectory_frames()` | Frenet tangent, normal, and binormal, with optional tool rotation |

## Scalar cubic splines

For times `t_i`, positions `q_i`, and segment duration `h_i`, each
`CubicSpline` segment is a cubic polynomial. Coefficients are chosen so that it
passes through both segment endpoints, matches waypoint velocities, and has
continuous acceleration at interior waypoints. `v0` and `vn` are clamped
endpoint velocities; the defaults are zero velocity, not natural (zero
curvature) boundary conditions.

`CubicSmoothingSpline` minimizes a combination of weighted data error and curve
roughness. Its parameter follows these conventions:

- `mu=1` gives exact interpolation.
- Smaller positive `mu` values emphasize smoothness.
- An infinite point weight pins that sample exactly.
- `mu` must be in `(0, 1]`.

Use `smoothing_spline_with_tolerance()` when the input requirement is a maximum
waypoint error rather than a preselected `mu`. It returns the spline, selected
`mu`, achieved maximum error, and iteration count.

The acceleration variants have different APIs:

- `CubicSplineWithAcceleration1(..., v0, vn, a0, an)` adds virtual endpoint
  waypoints.
- `CubicSplineWithAcceleration2(..., params=SplineParameters(...))` replaces
  the first and last cubic pieces with quintic segments when acceleration
  constraints are supplied.

## B-spline curves

A degree-`p` B-spline is

\[
C(u)=\sum_i N_{i,p}(u)P_i,
\]

where `P_i` are control points and `N_{i,p}` are basis functions induced by the
knot vector. The valid domain of a `BSpline` instance is
`[spline.u_min, spline.u_max]`. Use `evaluate_derivative(u, order)` for curve
derivatives and `generate_curve_points()` for sampled parameters and points.

`BSplineInterpolator` accepts scalar or vector-valued samples and optional
times. Degrees 3, 4, and 5 are supported. Since version 3.2.0, all three degrees
work with two or more samples; missing endpoint constraints are completed with
generated natural higher-derivative rows. Explicit velocity and acceleration
constraints take precedence.

`CubicBSplineInterpolation` is a cubic curve interpolator with chord-length,
centripetal, or equally-spaced parameterization. `ApproximationBSpline` uses a
chosen number of control points, while `SmoothingCubicBSpline` uses
`BSplineParams` to configure smoothing, weighting, endpoint enforcement, and
parameterization.

## Scalar motion profiles

`DoubleSTrajectory` constructs a jerk-limited point-to-point motion from
`StateParams` and `TrajectoryBounds`. The useful sampling distinction is:

```python
position = trajectory.evaluate(t)
velocity = trajectory.evaluate_velocity(t)
acceleration = trajectory.evaluate_acceleration(t)
jerk = trajectory.evaluate_jerk(t)
position, velocity, acceleration, jerk = trajectory.evaluate_full(t)
```

Scalar and NumPy-array times are supported. Times are clipped to the planned
duration. Inspect `get_duration()` and `get_phase_durations()` for timing.

`TrapezoidalTrajectory.generate_trajectory()` returns a callable and duration.
Its `TrajectoryParams` lives in `interpolatepy.trapezoidal`; the top-level
`interpolatepy.TrajectoryParams` is the distinct multipoint polynomial
configuration class.

`PolynomialTrajectory` returns callables whose outputs are `(position,
velocity, acceleration, jerk)`. Cubic order matches position and velocity;
quintic also matches acceleration; seventh order also matches jerk.

`ParabolicBlendTrajectory.generate()` returns a `(position, velocity,
acceleration)` callable and total duration for a via-point path.

## Quaternion interpolation

Quaternions `q` and `-q` represent the same orientation. The interpolation
classes account for this double cover when constructing their paths. Compare
orientations using the absolute quaternion dot product rather than component
equality when sign is irrelevant.

`QuaternionSpline` supports `"slerp"`, `"squad"`, and `"auto"`. Its
`evaluate()`, `evaluate_velocity()`, and `evaluate_acceleration()` methods match
the quaternion trajectory protocol. `SquadC2` adds virtual endpoints and
derivative-aware interpolation.

The logarithmic interpolators support B-spline degrees 3, 4, and 5 and, since
3.2.0, accept as few as two quaternion waypoints. `evaluate_velocity()` and
`evaluate_acceleration()` return derivatives of the interpolated internal
coordinates. Use `get_physical_kinematics()` when physical angular velocity and
angular acceleration are required.

## Paths and Frenet frames

`LinearPath` is parameterized by arc length from zero to `path.length`.
`CircularPath` takes an axis `r`, a point `d` on that axis, and a point `pi` on
the circle; one turn spans `2*pi*path.radius`. Their `velocity()` and
`acceleration()` methods are derivatives with respect to path length, not time.

`compute_trajectory_frames()` accepts a callable returning `(position,
first_derivative, second_derivative)` for each parameter. Each returned
`frames[i]` is a 3-by-3 rotation matrix whose columns are tangent, normal, and
binormal. A scalar `tool_orientation` rotates about the local binormal; a
three-tuple applies roll, pitch, and yaw.

## Backend and numerical notes

- Import supported APIs from `interpolatepy`, not implementation modules, to
  allow backend selection.
- The Python implementation is always available. `interpolatepy.HAS_CPP`
  reports whether the optional native extension was loaded.
- Vectorized time evaluation is supported by scalar cubic splines and
  Double-S trajectories. General B-spline and quaternion evaluators take one
  parameter at a time; sample them in a comprehension.
- Validate strictly increasing time arrays and keep units consistent across
  positions, derivatives, and bounds.
- Do not extrapolate B-splines beyond `u_min` and `u_max`.
