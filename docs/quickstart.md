# Quick start

All examples on this page use the public `interpolatepy` namespace. That keeps
the code independent of whether the Python or optional C++ backend is active.

## Scalar spline

`CubicSpline` interpolates scalar waypoints and accepts endpoint velocities:

```python
import numpy as np

from interpolatepy import CubicSpline

t_points = [0.0, 1.0, 2.0, 3.0]
q_points = [0.0, 1.0, -0.5, 2.0]
spline = CubicSpline(t_points, q_points, v0=0.0, vn=0.0)

t = np.linspace(t_points[0], t_points[-1], 200)
position = spline.evaluate(t)
velocity = spline.evaluate_velocity(t)
acceleration = spline.evaluate_acceleration(t)

assert position.shape == t.shape
assert np.isclose(position[0], q_points[0])
assert np.isclose(position[-1], q_points[-1])
```

The zero defaults for `v0` and `vn` clamp endpoint velocity. They are not
natural (zero-acceleration) boundary conditions.

## Jerk-limited point-to-point motion

```python
import numpy as np

from interpolatepy import DoubleSTrajectory
from interpolatepy import StateParams
from interpolatepy import TrajectoryBounds

state = StateParams(q_0=0.0, q_1=10.0, v_0=0.0, v_1=0.0)
bounds = TrajectoryBounds(v_bound=4.0, a_bound=3.0, j_bound=8.0)
trajectory = DoubleSTrajectory(state, bounds)

t = np.linspace(0.0, trajectory.get_duration(), 200)
position, velocity, acceleration, jerk = trajectory.evaluate_full(t)

assert np.isclose(position[0], state.q_0)
assert np.isclose(position[-1], state.q_1)
```

`evaluate()` returns only position. Use `evaluate_full()` when you need all four
signals.

## B-spline through two or more points

Degrees 3, 4, and 5 can interpolate as few as two samples in version 3.2.0:

```python
import numpy as np

from interpolatepy import BSplineInterpolator

points = np.array([[0.0, 0.0], [2.0, 1.0]])
times = np.array([0.0, 2.0])
curve = BSplineInterpolator(degree=5, points=points, times=times)

midpoint = curve.evaluate(1.0)
tangent = curve.evaluate_derivative(1.0, order=1)

assert midpoint.shape == (2,)
assert tangent.shape == (2,)
assert np.allclose(curve.evaluate(times[0]), points[0])
assert np.allclose(curve.evaluate(times[-1]), points[-1])
```

B-spline evaluators accept one parameter value at a time. Use a comprehension
to sample a curve:

```python
import numpy as np

from interpolatepy import BSplineInterpolator

points = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0]])
curve = BSplineInterpolator(degree=3, points=points)
u = np.linspace(curve.u_min, curve.u_max, 100)
samples = np.array([curve.evaluate(value) for value in u])
```

## Quaternion keyframes

```python
import numpy as np

from interpolatepy import Quaternion
from interpolatepy import QuaternionSpline

times = [0.0, 1.0, 2.0]
orientations = [
    Quaternion.identity(),
    Quaternion.from_angle_axis(np.pi / 3.0, np.array([0.0, 0.0, 1.0])),
    Quaternion.from_euler_angles(0.2, 0.4, 0.8),
]
rotation = QuaternionSpline(times, orientations, interpolation_method="auto")

q = rotation.evaluate(0.5)
angular_velocity = rotation.evaluate_velocity(0.5)
matrix = q.to_rotation_matrix()

assert matrix.shape == (3, 3)
assert angular_velocity.shape == (3,)
```

## Geometric path plus a time law

A path uses arc length `s`; a motion law maps time to `s`:

```python
import numpy as np

from interpolatepy import DoubleSTrajectory
from interpolatepy import LinearPath
from interpolatepy import StateParams
from interpolatepy import TrajectoryBounds

path = LinearPath(np.array([0.0, 0.0, 0.0]), np.array([3.0, 4.0, 0.0]))
law = DoubleSTrajectory(
    StateParams(q_0=0.0, q_1=path.length, v_0=0.0, v_1=0.0),
    TrajectoryBounds(v_bound=2.0, a_bound=2.0, j_bound=4.0),
)

t = np.linspace(0.0, law.get_duration(), 100)
s = law.evaluate(t)
positions = np.array([path.position(value) for value in s])

assert np.allclose(positions[0], path.pi)
assert np.allclose(positions[-1], path.pf)
```

Continue with the [User guide](user-guide.md), choose a focused
[tutorial](user-guide.md#tutorials), or browse the checked
[example programs](examples.md).
