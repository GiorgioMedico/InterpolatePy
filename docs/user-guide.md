# User guide

InterpolatePy separates three ideas that are easy to conflate:

1. a scalar trajectory maps time to position and derivatives;
2. a parametric curve maps a parameter to a vector point and derivatives;
3. a geometric path maps arc length to a point, independent of timing.

Choosing the correct family makes units and return values much clearer.

## Imports and backend routing

Import supported classes and functions from the package root:

```python
from interpolatepy import CubicSpline
from interpolatepy import DoubleSTrajectory
from interpolatepy import QuaternionSpline
```

These names are resolved by `_api.py`. When the native extension is available,
they refer to adapter classes; otherwise they refer to Python implementations.
Direct imports such as `interpolatepy.splines.cubic.CubicSpline` bypass that
routing and always select the implementation module.

Use `interpolatepy.HAS_CPP` for diagnostics, not for normal application
branching. The common evaluation surface should be sufficient for typical code.

## Evaluation conventions

### Scalar splines

The cubic spline family uses:

```python
position = spline.evaluate(t)
velocity = spline.evaluate_velocity(t)
acceleration = spline.evaluate_acceleration(t)
```

These methods accept a scalar or a NumPy array. Scalar input returns a scalar;
array input returns an array with the same time shape.

### Double-S trajectories

`DoubleSTrajectory` follows the same component methods and adds jerk:

```python
position = trajectory.evaluate(t)
jerk = trajectory.evaluate_jerk(t)
position, velocity, acceleration, jerk = trajectory.evaluate_full(t)
```

Do not unpack `evaluate()`; it returns position only.

### Callable trajectory generators

`TrapezoidalTrajectory`, `PolynomialTrajectory`, and
`ParabolicBlendTrajectory` create callables instead of exposing the scalar
evaluation interface directly:

- trapezoidal and parabolic-blend callables return `(q, qd, qdd)`;
- polynomial callables return `(q, qd, qdd, qddd)`.

The generator also returns the duration when it computes one.

### B-spline curves

General B-spline methods take one parameter at a time:

```python
point = curve.evaluate(u)
first_derivative = curve.evaluate_derivative(u, order=1)
parameters, points = curve.generate_curve_points(num_points=100)
```

The valid parameter interval is `[curve.u_min, curve.u_max]`. Derivatives are
with respect to `u`, not physical time unless `u` is itself time.

### Quaternion trajectories

Quaternion interpolation classes return a `Quaternion` from `evaluate()` and
NumPy coordinate vectors from derivative methods. Evaluate one time at a time.
For logarithmic interpolators, internal-coordinate derivatives and physical
angular kinematics are distinct; see the
[quaternion tutorial](tutorials/quaternion-interpolation.md).

### Geometric paths

`LinearPath` and `CircularPath` are parameterized by arc length `s`:

```python
point = path.position(s)
tangent = path.velocity(s)
curvature_vector = path.acceleration(s)
batch = path.evaluate_at([0.0, s])
```

The method names `velocity` and `acceleration` mean first and second geometric
derivatives with respect to arc length. Combine a path with a scalar time law to
obtain physical velocity and acceleration.

## Input rules

- Time points must be strictly increasing and match the number of samples.
- Keep position, time, velocity, acceleration, and jerk units consistent.
- Quaternion constructors use radians.
- Normalize arbitrary quaternions with `.unit()` before interpolation when the
  constructor does not do so for you.
- A `CircularPath` axis must be nonzero, and its circle point must not lie on
  the axis.
- `CubicSmoothingSpline.mu` must be in `(0, 1]`; `mu=1` interpolates exactly.
- Bounds for `DoubleSTrajectory` must be positive and endpoint speeds must be
  compatible with the velocity bound.

## Similar names

Two public data classes are both called `TrajectoryParams` in their defining
modules:

```python
from interpolatepy import TrajectoryParams  # polynomial multipoint configuration
from interpolatepy.motion.trapezoidal import TrajectoryParams as TrapezoidalParams
```

The top-level name is the polynomial class. Use the module-qualified alias for
trapezoidal generation.

The two acceleration spline implementations also differ at construction time:

```python
from interpolatepy import CubicSplineWithAcceleration1
from interpolatepy import CubicSplineWithAcceleration2
from interpolatepy import SplineParameters

method_1 = CubicSplineWithAcceleration1(
    [0.0, 1.0, 2.0], [0.0, 1.0, 0.0], a0=0.0, an=0.0
)
method_2 = CubicSplineWithAcceleration2(
    [0.0, 1.0, 2.0],
    [0.0, 1.0, 0.0],
    SplineParameters(a0=0.0, an=0.0),
)
```

## Selecting an algorithm

Use [Algorithms](algorithms.md) for a comparison table and mathematical
overview. In short:

- use `CubicSpline` for a small scalar waypoint sequence;
- use smoothing splines when exact passage through noisy data is undesirable;
- use B-splines for vector-valued parametric curves and local control;
- use Double-S when jerk limits matter, trapezoidal profiles when acceleration
  and velocity limits are sufficient, and polynomials for endpoint derivative
  matching;
- use quaternion algorithms for orientation—do not spline Euler angles through
  singularities;
- keep geometry (`LinearPath`, `CircularPath`) separate from the time law.

## Plotting

Plotting is available on scalar spline classes, B-spline curves, the parabolic
blend generator, the quaternion visualization helper module, and the Frenet
`plot_frames()` function. Motion-profile objects generally expose numerical
samples rather than a `plot()` method.

Matplotlib figures are not shown until application code calls `plt.show()` in
most workflows. For headless checks, set `MPLBACKEND=Agg`.

## Tutorials

- [Spline interpolation](tutorials/spline-interpolation.md)
- [Motion profiles](tutorials/motion-profiles.md)
- [Quaternion interpolation](tutorials/quaternion-interpolation.md)
- [Path planning](tutorials/path-planning.md)

The [Examples](examples.md) page maps these topics to runnable scripts, and the
[API reference](api-reference.md) is generated from current docstrings.
