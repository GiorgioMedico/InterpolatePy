# Algorithms

This guide summarizes the numerical families in InterpolatePy 3.2.0 and helps
select between them. The [API reference](api-reference.md) supplies signatures;
the tutorials supply complete code.

## Selection table

| Requirement | Algorithm | Output continuity or constraint |
| --- | --- | --- |
| Scalar waypoints plus endpoint velocities | `CubicSpline` | Piecewise cubic, C2 at interior knots |
| Scalar fitting with noise | `CubicSmoothingSpline` | Adjustable fidelity/roughness |
| Endpoint velocity and acceleration | Acceleration spline variants | Explicit endpoint derivatives |
| Known B-spline controls and knots | `BSpline` | Degree-dependent continuity |
| Pass a parametric curve through samples | `BSplineInterpolator` | Degrees 3, 4, or 5 |
| Reduce a dense point set | `ApproximationBSpline` | Weighted least-squares fit |
| Smooth vector samples | `SmoothingCubicBSpline` | Cubic B-spline smoothing |
| Bound jerk, acceleration, and speed | `DoubleSTrajectory` | Piecewise constant jerk |
| Bound acceleration and speed | `TrapezoidalTrajectory` | Trapezoidal or triangular speed |
| Match endpoint derivatives | `PolynomialTrajectory` | Order 3, 5, or 7 |
| Blend scalar via points | `ParabolicBlendTrajectory` | Linear pieces with quadratic blends |
| Two orientations | Quaternion SLERP | Spherical geodesic segment |
| Orientation keyframes | `QuaternionSpline` | Piecewise SLERP or SQUAD |
| Smooth orientation derivatives | `SquadC2` | SQUAD-style C2 construction |
| Rotation-vector interpolation | LQI / mLQI | Degree-3/4/5 B-spline state |
| 3D line or circle | Path primitives | Arc-length parameterized geometry |
| Frame along a curve | Frenet helpers | Orthonormal moving frame |

## Cubic interpolation

On segment `i`, InterpolatePy evaluates

\[
q_i(\tau)=a_{i0}+a_{i1}\tau+a_{i2}\tau^2+a_{i3}\tau^3,
\qquad \tau=t-t_i.
\]

The coefficients interpolate the segment endpoints and match velocities at
each knot. Interior waypoint velocities come from a tridiagonal system that
enforces acceleration continuity. The constructor's `v0` and `vn` fix endpoint
velocities; their zero defaults define a clamped, rest-to-rest spline.

Use independent scalar splines for each axis of a vector trajectory when all
axes share the same time knots. This does not enforce a bound on the combined
vector norm.

## Cubic smoothing

The smoothing family balances weighted sample error and an integrated curvature
penalty. InterpolatePy expresses the trade-off with `mu` and internally uses

\[
\lambda=\frac{1-\mu}{6\mu}.
\]

`mu=1` means exact interpolation; smaller positive values increase smoothing.
Per-point weights control relative fidelity, and infinite weight fixes a sample.
Use the tolerance search helper to choose `mu` by a maximum-error requirement.

## Acceleration-constrained splines

Both variants interpolate the supplied scalar samples and honor endpoint
velocities and accelerations, but they construct the boundary differently.

- Method 1 adds virtual samples near both endpoints before solving a cubic
  spline. It is configured directly with `v0`, `vn`, `a0`, and `an`.
- Method 2 starts with a cubic spline and replaces its boundary segments with
  quintics when `SplineParameters.a0` or `.an` is not `None`.

They need not trace identical curves between waypoints. Compare derivative
peaks as well as endpoint values when choosing one.

## B-splines

A degree-`p` B-spline curve is

\[
C(u)=\sum_i N_{i,p}(u)P_i.
\]

The basis functions have local support, so moving one control point changes
only part of the curve. Knot multiplicity reduces continuity at that knot. For
a valid `BSpline`, the usual size relation is
`len(knots) = len(control_points) + degree + 1`.

`BSplineInterpolator` solves for controls that pass through the data and honor
provided boundary derivatives. Valid degrees are 3, 4, and 5. Version 3.2.0
supports two or more samples at every valid degree by supplying nonconflicting
higher-derivative boundary rows where explicit constraints are absent.

Parameterization affects curve shape:

- `"equally_spaced"` ignores point spacing;
- `"chord_length"` weights intervals by Euclidean distance;
- `"centripetal"` uses the square root of chord length and can reduce looping
  near unevenly spaced points.

`CubicBSplineInterpolation` is a cubic convenience implementation.
`ApproximationBSpline` deliberately does not pass through every input point;
its number of controls governs compression. `SmoothingCubicBSpline` combines a
cubic B-spline model with error and smoothness terms configured by
`BSplineParams`.

## Double-S motion

A Double-S profile consists of acceleration, optional constant-velocity, and
deceleration stages, with jerk ramps at stage boundaries. The planner transforms
negative displacement into a positive planning coordinate and maps results back
to the original sign.

Use `TrajectoryBounds` for positive speed, acceleration, and jerk magnitudes.
The planner may shorten or remove plateau phases for short moves. Always use the
computed `get_duration()` instead of assuming a seven-phase profile.

Sampling methods are deliberately component-oriented:

```python
q = trajectory.evaluate(t)
qd = trajectory.evaluate_velocity(t)
qdd = trajectory.evaluate_acceleration(t)
qddd = trajectory.evaluate_jerk(t)
q, qd, qdd, qddd = trajectory.evaluate_full(t)
```

## Trapezoidal motion

A trapezoidal profile accelerates at a bounded rate, optionally cruises, then
decelerates. If the move is too short to reach `vmax`, the speed profile is
triangular. A `duration` can be requested instead of `vmax` when it is feasible.

`generate_trajectory()` returns a callable and duration for one segment.
`interpolate_waypoints()` joins multiple segments and can infer intermediate
velocities heuristically.

## Polynomial motion

Polynomial order determines which boundary derivatives can be matched:

| Order | Boundary values used |
| --- | --- |
| 3 | position and velocity |
| 5 | position, velocity, and acceleration |
| 7 | position, velocity, acceleration, and jerk |

Higher order is not automatically better: it can introduce larger interior
peaks. Sample and check every derivative that matters to the actuator.

## Quaternion methods

For unit quaternions, SLERP follows a constant-rate spherical path between two
orientations. Because `q` and `-q` encode the same rotation, interpolation code
chooses a consistent sign to avoid taking the long path.

`QuaternionSpline` applies SLERP or SQUAD segment-wise. `SquadC2` extends the
sequence with virtual boundary orientations and exposes numerical angular
velocity and acceleration.

LQI recovers a continuous rotation vector

\[
r(t)=\theta(t)\hat n(t)
\]

and interpolates it with `BSplineInterpolator`. mLQI splines the angle and axis
components separately, optionally normalizing the axis after evaluation. Both
accept degrees 3, 4, and 5 and at least two waypoints.

Coordinate-state derivatives returned by logarithmic interpolators are not
always physical angular velocity and acceleration. On the Python backend,
`get_physical_kinematics()` performs the required mapping.

## Paths and frames

`LinearPath` and `CircularPath` use arc length `s`, which makes `velocity(s)` a
unit tangent for regular paths. These geometric derivatives must be combined
with a time law using the chain rule to obtain physical derivatives.

For a parametric curve `p(u)`, the Frenet helper computes the normalized tangent,
normal from the tangent derivative, and binormal from their cross product. In
the returned matrix, these three vectors are columns. At zero speed or nearly
zero curvature, the implementation selects a stable fallback direction; a
Frenet normal is not geometrically unique there.

## Numerical practice

- Scale times and positions to avoid extreme coefficient magnitudes.
- Inspect derivative peaks between waypoints, not only at the samples.
- Use strictly increasing finite parameters.
- Evaluate B-splines only inside `[u_min, u_max]`.
- Compare quaternion orientations with `abs(q1.dot_prod(q2))` when sign is
  irrelevant.
- Use the [protocols](api-reference.md#runtime-checkable-protocols) for generic
  code, but remember that runtime protocol checks verify attribute presence,
  not units or numerical semantics.
