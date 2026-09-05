# Quaternion interpolation

Quaternions avoid Euler-angle singularities, but introduce a double cover: `q`
and `-q` represent the same orientation. InterpolatePy's trajectory classes
choose consistent signs across keyframes.

## Construct and convert orientations

```python
import numpy as np

from interpolatepy import Quaternion

identity = Quaternion.identity()
about_z = Quaternion.from_angle_axis(
    np.pi / 2.0, np.array([0.0, 0.0, 1.0])
)
from_rpy = Quaternion.from_euler_angles(roll=0.2, pitch=0.3, yaw=0.4)

matrix = from_rpy.to_rotation_matrix()
roll, pitch, yaw = from_rpy.to_euler_angles()
axis, angle = about_z.to_axis_angle()

assert matrix.shape == (3, 3)
assert np.isclose(angle, np.pi / 2.0)
assert np.allclose(axis, [0.0, 0.0, 1.0])
```

Notice the asymmetric naming: `from_angle_axis(angle, axis)` accepts angle
first, while `to_axis_angle()` returns axis first.

Normalize arbitrary components with `.unit()` before treating them as a
rotation.

## SLERP between two orientations

```python
import numpy as np

from interpolatepy import Quaternion

start = Quaternion.identity()
end = Quaternion.from_angle_axis(np.pi, np.array([0.0, 0.0, 1.0]))
halfway = start.slerp(end, 0.5)
axis, angle = halfway.to_axis_angle()

assert np.isclose(angle, np.pi / 2.0)
```

The interpolation parameter is dimensionless and lies between zero and one.

## QuaternionSpline keyframes

```python
from interpolatepy import Quaternion
from interpolatepy import QuaternionSpline

times = [0.0, 1.0, 2.0, 3.0, 4.0]
orientations = [
    Quaternion.identity(),
    Quaternion.from_euler_angles(0.1, 0.2, 0.1),
    Quaternion.from_euler_angles(0.3, 0.5, 0.4),
    Quaternion.from_euler_angles(0.2, 0.7, 0.8),
    Quaternion.from_euler_angles(0.0, 0.4, 1.0),
]
spline = QuaternionSpline(times, orientations, interpolation_method="auto")

q = spline.evaluate(2.5)
omega = spline.evaluate_velocity(2.5)
alpha = spline.evaluate_acceleration(2.5)
```

The available method strings are `"slerp"`, `"squad"`, and `"auto"`. In
SQUAD or automatic mode, boundary segments use SLERP and eligible interior
segments use SQUAD. With too few keyframes for SQUAD, evaluation falls back to
SLERP.

`interpolate_at_time()` also returns a status code for legacy callers;
`evaluate()` is the protocol-compatible interface.

## SquadC2

Use `SquadC2` when zero-clamped boundaries and smoother angular derivatives are
important:

```python
import numpy as np

from interpolatepy import Quaternion
from interpolatepy import SquadC2

times = [0.0, 1.0, 2.0, 3.0]
orientations = [
    Quaternion.identity(),
    Quaternion.from_angle_axis(0.4, np.array([1.0, 0.0, 0.0])),
    Quaternion.from_angle_axis(0.8, np.array([0.0, 1.0, 0.0])),
    Quaternion.from_angle_axis(1.0, np.array([0.0, 0.0, 1.0])),
]
spline = SquadC2(times, orientations)

t = 1.5
q = spline.evaluate(t)
omega = spline.evaluate_velocity(t)
alpha = spline.evaluate_acceleration(t)
assert omega.shape == (3,)
assert alpha.shape == (3,)
```

Input quaternions are normalized by default. `validate_continuity` controls the
implementation's construction-time continuity diagnostics.

## Logarithmic quaternion interpolation

LQI unwraps keyframes into continuous rotation vectors and interpolates those
vectors with a B-spline:

```python
import numpy as np

from interpolatepy import LogQuaternionInterpolation
from interpolatepy import Quaternion

times = [0.0, 2.0]
orientations = [
    Quaternion.identity(),
    Quaternion.from_angle_axis(1.2, np.array([0.0, 0.0, 1.0])),
]
interpolator = LogQuaternionInterpolation(times, orientations, degree=5)

q = interpolator.evaluate(1.0)
rotation_vector_rate = interpolator.evaluate_velocity(1.0)
assert rotation_vector_rate.shape == (3,)
```

Degrees 3, 4, and 5 all accept two or more keyframes in 3.2.0.

On the Python backend, distinguish rotation-vector derivatives from physical
angular kinematics:

```python
import numpy as np

from interpolatepy import Quaternion
from interpolatepy.quaternion.logarithmic import LogQuaternionInterpolation

times = [0.0, 1.0, 2.0]
orientations = [
    Quaternion.identity(),
    Quaternion.from_euler_angles(0.2, 0.3, 0.1),
    Quaternion.from_euler_angles(0.5, 0.4, 0.7),
]
interpolator = LogQuaternionInterpolation(times, orientations)
omega, alpha = interpolator.get_physical_kinematics(1.0)
assert omega.shape == (3,)
assert alpha.shape == (3,)
```

`get_physical_kinematics()` and acceleration boundary arguments are currently
Python-backend helpers. Force the fallback for code that relies on them.

## Modified logarithmic interpolation

mLQI splines the angle and axis separately. Its internal derivative vector has
four components: angle plus three axis components.

```python
import numpy as np

from interpolatepy import ModifiedLogQuaternionInterpolation
from interpolatepy import Quaternion

times = [0.0, 1.0, 2.0]
orientations = [
    Quaternion.identity(),
    Quaternion.from_angle_axis(0.5, np.array([1.0, 0.0, 0.0])),
    Quaternion.from_angle_axis(1.0, np.array([0.0, 1.0, 0.0])),
]
interpolator = ModifiedLogQuaternionInterpolation(
    times, orientations, degree=3, normalize_axis=True
)

q = interpolator.evaluate(0.75)
coordinate_rate = interpolator.evaluate_velocity(0.75)
assert coordinate_rate.shape == (4,)
```

Keep `normalize_axis=True` unless the interpolated axis is already known to
stay unit length.

## SPRING minimum-curvature interpolation

`SpringQuaternionInterpolation` implements SPRING (*Spherical Interpolation
using Numerical Gradient descent*) from Dam, Koch, and Lillholm's 1998 report
*Quaternions, Interpolation and Animation*. It starts with piecewise SLERP and
numerically relaxes the in-between frames to minimize tangential curvature on
the unit-quaternion sphere. Keyframes remain fixed.

```python
import numpy as np

from interpolatepy import Quaternion
from interpolatepy import SpringConfig
from interpolatepy import SpringQuaternionInterpolation

times = [0.0, 1.0, 2.0, 3.0]
orientations = [
    Quaternion.identity(),
    Quaternion.from_angle_axis(0.8, np.array([1.0, 0.0, 0.0])),
    Quaternion.from_angle_axis(1.0, np.array([0.0, 1.0, 0.0])),
    Quaternion.from_angle_axis(1.2, np.array([0.0, 0.0, 1.0])),
]
spring = SpringQuaternionInterpolation(
    times,
    orientations,
    SpringConfig(num_samples=81, iterations=300),
)

q = spring.evaluate(1.5)
omega = spring.evaluate_velocity(1.5)
alpha = spring.evaluate_acceleration(1.5)
assert np.isclose(q.norm(), 1.0)
assert spring.final_energy <= spring.initial_energy
```

SPRING is a global, iterative method: changing one keyframe can affect the
whole curve, and increasing `num_samples` or `iterations` increases setup
cost. Intermediate values are SLERP evaluations of the optimized discrete
frames; angular derivatives are numerical estimates. The default three-level
solver refines a coarse curve approximately fivefold at each level. Every
optimized coarse frame is held fixed at the next level, following the report's
multi-step procedure. Set `refinement_levels=1` for a one-stage solve.

`refinement_sample_counts` and `stage_energy_history` are immutable tuples for
inspecting convergence. `energy_history` is the final-stage history, while
`initial_energy` and `final_energy` compare the original piecewise-SLERP curve
and the final optimized curve on the same final grid.

## Compare orientations correctly

Component equality rejects the equivalent pair `q` and `-q`. For unit
quaternions, use the absolute dot product:

```python
from interpolatepy import Quaternion

q1 = Quaternion.identity()
q2 = -q1
same_orientation = abs(q1.dot_prod(q2)) > 1.0 - 1e-9
assert same_orientation
```

## Choosing a method

| Need | Method |
| --- | --- |
| One segment | `Quaternion.slerp()` |
| Simple keyframe API | `QuaternionSpline` |
| Zero-clamped smooth SQUAD construction | `SquadC2` |
| Global numerical minimum-curvature curve | `SpringQuaternionInterpolation` |
| Continuous rotation-vector representation | LQI |
| Separate angle/axis state | mLQI |

Rotations close to 180 degrees are intrinsically branch-sensitive. Inspect the
actual orientation path, angular velocity, and angular acceleration for the
chosen keyframes rather than selecting solely by method name.
