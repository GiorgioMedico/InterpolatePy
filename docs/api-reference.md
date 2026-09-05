# API reference

This page documents the supported Python API for InterpolatePy 3.2.1. Import
these objects from `interpolatepy` unless a module-qualified exception is shown.

!!! note "Backend-dependent class identity"
    MkDocs renders the Python implementation docstrings. The package root may
    resolve computation-heavy names to native adapter classes when
    `HAS_CPP=True`. The adapters align the primary construction and evaluation
    workflows, but some diagnostic and convenience methods remain Python-only;
    see [Architecture](architecture.md#adapter-layer) for current limitations.

## Package information

`interpolatepy.__version__` contains the installed version and
`interpolatepy.HAS_CPP` reports backend selection.

## Organized imports

The package root remains the shortest stable API. Domain-oriented namespaces
provide the same backend selection when grouped imports make application code
clearer:

```python
from interpolatepy.bsplines import BSpline
from interpolatepy.motion import DoubleSTrajectory
from interpolatepy.paths import LinearPath
from interpolatepy.quaternion import QuaternionSpline
from interpolatepy.splines import CubicSpline
```

Concrete modules such as `interpolatepy.splines.cubic` contain the Python
implementations and intentionally bypass native-backend selection.

## Scalar splines

### CubicSpline

::: interpolatepy.CubicSpline
    options:
      members:
        - __init__
        - evaluate
        - evaluate_velocity
        - evaluate_acceleration
        - plot

### CubicSmoothingSpline

::: interpolatepy.CubicSmoothingSpline
    options:
      members:
        - __init__
        - evaluate
        - evaluate_velocity
        - evaluate_acceleration
        - plot

### Smoothing tolerance search

::: interpolatepy.SplineConfig

::: interpolatepy.smoothing_spline_with_tolerance

### CubicSplineWithAcceleration1

::: interpolatepy.CubicSplineWithAcceleration1
    options:
      members:
        - __init__
        - evaluate
        - evaluate_velocity
        - evaluate_acceleration
        - plot

### CubicSplineWithAcceleration2

::: interpolatepy.CubicSplineWithAcceleration2
    options:
      members:
        - __init__
        - evaluate
        - evaluate_velocity
        - evaluate_acceleration
        - plot

::: interpolatepy.SplineParameters

## B-splines

### BSpline

::: interpolatepy.BSpline

### BSplineInterpolator

::: interpolatepy.BSplineInterpolator

Degrees 3, 4, and 5 accept two or more points. `times` defaults to generated
parameters. Provide endpoint velocity or acceleration vectors when those
derivatives need to be constrained.

### CubicBSplineInterpolation

::: interpolatepy.CubicBSplineInterpolation

### ApproximationBSpline

::: interpolatepy.ApproximationBSpline

### SmoothingCubicBSpline

::: interpolatepy.SmoothingCubicBSpline

::: interpolatepy.BSplineParams

## Motion profiles

### DoubleSTrajectory

::: interpolatepy.DoubleSTrajectory

::: interpolatepy.StateParams

::: interpolatepy.TrajectoryBounds

`DoubleSTrajectory.evaluate(t)` returns only position.
`DoubleSTrajectory.evaluate_full(t)` returns `(position, velocity,
acceleration, jerk)`.

### TrapezoidalTrajectory

::: interpolatepy.TrapezoidalTrajectory

The trapezoidal configuration type is module-qualified because the package-root
`TrajectoryParams` name belongs to polynomial trajectories:

::: interpolatepy.motion.trapezoidal.TrajectoryParams

::: interpolatepy.CalculationParams

::: interpolatepy.InterpolationParams

### PolynomialTrajectory

::: interpolatepy.PolynomialTrajectory

::: interpolatepy.BoundaryCondition

::: interpolatepy.TimeInterval

::: interpolatepy.TrajectoryParams

### ParabolicBlendTrajectory

::: interpolatepy.ParabolicBlendTrajectory

## Quaternion operations and interpolation

### Quaternion

::: interpolatepy.Quaternion

`Quaternion.to_axis_angle()` returns `(axis, angle)`. Angles passed to
`from_angle_axis()` and `from_euler_angles()` are radians.

### QuaternionSpline

::: interpolatepy.QuaternionSpline

### SquadC2

::: interpolatepy.SquadC2

### LogQuaternionInterpolation

::: interpolatepy.LogQuaternionInterpolation

### ModifiedLogQuaternionInterpolation

::: interpolatepy.ModifiedLogQuaternionInterpolation

The logarithmic interpolators support degree 3, 4, or 5 with at least two
quaternion waypoints.

## Geometric paths and frames

### LinearPath

::: interpolatepy.LinearPath

### CircularPath

::: interpolatepy.CircularPath

### Frenet-frame functions

::: interpolatepy.compute_trajectory_frames

::: interpolatepy.helicoidal_trajectory_with_derivatives

::: interpolatepy.circular_trajectory_with_derivatives

::: interpolatepy.plot_frames

## Numerical utilities

::: interpolatepy.linear_traj

::: interpolatepy.solve_tridiagonal

## Runtime-checkable protocols

The protocols use structural typing: a class conforms when it implements the
required methods, without inheriting from the protocol.

::: interpolatepy.ScalarTrajectory

::: interpolatepy.CurveEvaluator

::: interpolatepy.GeometricPath

::: interpolatepy.QuaternionTrajectory

::: interpolatepy.TrajectoryFunction

## Public export list

The authoritative export list is `interpolatepy.__all__`. Internal module
names, implementation attributes, C++ submodules, and visualization helpers not
listed there may change without a compatibility guarantee.
