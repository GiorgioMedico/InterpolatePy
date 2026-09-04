# Spline interpolation

This tutorial covers scalar cubic splines and vector-valued B-spline curves.
The distinction matters: scalar splines accept NumPy arrays of times directly,
while general B-spline curves evaluate one parameter at a time.

## Interpolate scalar waypoints

```python
import numpy as np

from interpolatepy import CubicSpline

t_points = [0.0, 1.0, 2.5, 4.0]
q_points = [0.0, 1.0, -0.5, 2.0]
spline = CubicSpline(t_points, q_points, v0=0.0, vn=0.0)

t = np.linspace(t_points[0], t_points[-1], 250)
q = spline.evaluate(t)
qd = spline.evaluate_velocity(t)
qdd = spline.evaluate_acceleration(t)

assert np.allclose(spline.evaluate(np.asarray(t_points)), q_points)
assert np.isclose(spline.evaluate_velocity(t_points[0]), 0.0)
assert np.isclose(spline.evaluate_velocity(t_points[-1]), 0.0)
```

`v0` and `vn` are endpoint velocities. Their default value is zero, which
creates a clamped rest-to-rest spline. A natural spline instead constrains
endpoint acceleration; `CubicSpline` does not use that convention.

At an interior knot, position, velocity, and acceleration are continuous. Jerk
is piecewise constant and may jump.

## Smooth noisy scalar samples

`mu` controls the fit/roughness trade-off. One gives exact interpolation;
smaller positive values smooth more strongly.

```python
import numpy as np

from interpolatepy import CubicSmoothingSpline

rng = np.random.default_rng(42)
t_points = np.linspace(0.0, 6.0, 31)
q_points = np.sin(t_points) + 0.12 * rng.standard_normal(t_points.size)

spline = CubicSmoothingSpline(
    t_points.tolist(),
    q_points.tolist(),
    mu=0.2,
    v0=0.0,
    vn=0.0,
)

t = np.linspace(0.0, 6.0, 200)
smoothed = spline.evaluate(t)
assert smoothed.shape == t.shape
```

Weights are relative penalties on sample error. Use `np.inf` to pin a sample:

```python
import numpy as np

from interpolatepy import CubicSmoothingSpline

t_points = [0.0, 1.0, 2.0, 3.0]
q_points = [0.0, 1.2, 0.8, 2.0]
weights = [np.inf, 1.0, 1.0, np.inf]
spline = CubicSmoothingSpline(t_points, q_points, mu=0.1, weights=weights)

assert np.isclose(spline.evaluate(0.0), q_points[0])
assert np.isclose(spline.evaluate(3.0), q_points[-1])
```

## Select smoothing by error tolerance

When the requirement is a maximum deviation rather than a chosen `mu`, use the
binary-search helper:

```python
import numpy as np

from interpolatepy import SplineConfig
from interpolatepy import smoothing_spline_with_tolerance

t_points = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
q_points = np.array([0.0, 1.1, 0.1, 1.9, 1.0])
config = SplineConfig(max_iterations=40)

spline, mu, max_error, iterations = smoothing_spline_with_tolerance(
    t_points,
    q_points,
    tolerance=0.25,
    config=config,
)

assert 0.0 < mu <= 1.0
assert iterations <= config.max_iterations
assert max_error <= 0.25 + 1e-6
```

The returned error is measured at the supplied samples, not over every point of
the continuous curve.

## Constrain endpoint acceleration

Method 1 takes derivative values directly:

```python
import numpy as np

from interpolatepy import CubicSplineWithAcceleration1

spline = CubicSplineWithAcceleration1(
    [0.0, 1.0, 2.0, 3.0],
    [0.0, 1.0, -0.5, 0.0],
    v0=0.0,
    vn=0.0,
    a0=0.5,
    an=-0.5,
)

assert np.isclose(spline.evaluate_acceleration(0.0), 0.5)
assert np.isclose(spline.evaluate_acceleration(3.0), -0.5)
```

Method 2 uses `SplineParameters`:

```python
import numpy as np

from interpolatepy import CubicSplineWithAcceleration2
from interpolatepy import SplineParameters

params = SplineParameters(v0=0.0, vn=0.0, a0=0.5, an=-0.5)
spline = CubicSplineWithAcceleration2(
    [0.0, 1.0, 2.0, 3.0],
    [0.0, 1.0, -0.5, 0.0],
    params,
)

assert np.isclose(spline.evaluate_acceleration(0.0), params.a0)
assert np.isclose(spline.evaluate_acceleration(3.0), params.an)
```

The constructions differ between samples. Sample velocity and acceleration
densely before choosing one for a constrained mechanism.

## Construct a B-spline from controls

For degree `p`, the knot vector has
`len(control_points) + p + 1` entries:

```python
import numpy as np

from interpolatepy import BSpline

degree = 3
control_points = np.array(
    [[0.0, 0.0], [1.0, 2.0], [2.0, -1.0], [3.0, 1.0], [4.0, 0.0]]
)
knots = BSpline.create_uniform_knots(degree, len(control_points))
curve = BSpline(degree, knots, control_points)

u = np.linspace(curve.u_min, curve.u_max, 150)
points = np.array([curve.evaluate(value) for value in u])
tangents = np.array([curve.evaluate_derivative(value, 1) for value in u])

assert points.shape == (150, 2)
assert tangents.shape == (150, 2)
```

## Interpolate B-spline samples

`BSplineInterpolator` supports degree 3, 4, or 5, optional times, endpoint
derivatives, and cyclic constraints. As of 3.2.0, two samples are sufficient at
all valid degrees.

```python
import numpy as np

from interpolatepy import BSplineInterpolator

points = np.array([[0.0, 0.0], [2.0, 1.0]])
times = np.array([0.0, 2.0])
curve = BSplineInterpolator(
    degree=5,
    points=points,
    times=times,
    initial_velocity=np.array([1.0, 0.0]),
)

assert np.allclose(curve.evaluate(0.0), points[0])
assert np.allclose(curve.evaluate(2.0), points[1])
assert np.allclose(curve.evaluate_derivative(0.0, 1), [1.0, 0.0])
```

Without explicit `times`, the class generates a parameter sequence. Use
`curve.u_min` and `curve.u_max` instead of assuming `[0, 1]`.

## Approximate or smooth vector data

Approximation reduces a point set to a selected number of controls:

```python
import numpy as np

from interpolatepy import ApproximationBSpline

x = np.linspace(0.0, 2.0 * np.pi, 40)
points = np.column_stack((x, np.sin(x)))
curve = ApproximationBSpline(points, num_control_points=10, degree=3)
parameters, fitted = curve.generate_curve_points(120)

assert fitted.shape == (120, 2)
assert parameters.shape == (120,)
```

B-spline smoothing uses a configuration object:

```python
import numpy as np

from interpolatepy import BSplineParams
from interpolatepy import SmoothingCubicBSpline

points = np.array(
    [[0.0, 0.0], [1.0, 1.2], [2.0, 0.7], [3.0, 2.1], [4.0, 2.0]]
)
params = BSplineParams(mu=0.8, enforce_endpoints=True, auto_derivatives=True)
curve = SmoothingCubicBSpline(points, params)

errors = curve.calculate_approximation_error()
assert errors.shape == (len(points),)
```

## Plotting

Scalar spline classes provide `plot()`. B-spline classes provide `plot_2d()`
and `plot_3d()` according to control-point dimension. These functions create or
populate Matplotlib axes; application code controls when to call `plt.show()`.

## Common mistakes

- Calling a zero-velocity cubic spline “natural.”
- Passing `a0` and `an` directly to method 2 instead of using
  `SplineParameters`.
- Sending an array to `BSpline.evaluate()`; sample scalar parameters in a loop.
- Evaluating outside `[u_min, u_max]`.
- Expecting approximation or smoothing curves to pass through every sample.
- Treating a curve derivative with respect to `u` as a physical time
  derivative without applying the chain rule.
