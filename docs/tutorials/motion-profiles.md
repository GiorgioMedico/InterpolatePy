# Motion profiles

Motion profiles map time to scalar position and derivatives. This tutorial
compares jerk-limited Double-S motion, trapezoidal velocity profiles, polynomial
boundary interpolation, and parabolic via-point blends.

## Double-S: bound jerk, acceleration, and speed

```python
import numpy as np

from interpolatepy import DoubleSTrajectory
from interpolatepy import StateParams
from interpolatepy import TrajectoryBounds

state = StateParams(q_0=0.0, q_1=12.0, v_0=0.0, v_1=0.0)
bounds = TrajectoryBounds(v_bound=4.0, a_bound=3.0, j_bound=8.0)
trajectory = DoubleSTrajectory(state, bounds)

t = np.linspace(0.0, trajectory.get_duration(), 300)
q, qd, qdd, qddd = trajectory.evaluate_full(t)

assert np.isclose(q[0], state.q_0)
assert np.isclose(q[-1], state.q_1)
assert np.max(np.abs(qd)) <= bounds.v_bound + 1e-6
assert np.max(np.abs(qdd)) <= bounds.a_bound + 1e-6
```

The class also offers component methods:

```python
from interpolatepy import DoubleSTrajectory
from interpolatepy import StateParams
from interpolatepy import TrajectoryBounds

trajectory = DoubleSTrajectory(
    StateParams(q_0=0.0, q_1=1.0, v_0=0.0, v_1=0.0),
    TrajectoryBounds(v_bound=1.0, a_bound=2.0, j_bound=4.0),
)
t = trajectory.get_duration() / 2.0
q = trajectory.evaluate(t)
qd = trajectory.evaluate_velocity(t)
qdd = trajectory.evaluate_acceleration(t)
qddd = trajectory.evaluate_jerk(t)
```

`evaluate()` returns position only. The trajectory duration and individual
phase durations are computed during construction:

```python
from interpolatepy import DoubleSTrajectory
from interpolatepy import StateParams
from interpolatepy import TrajectoryBounds

trajectory = DoubleSTrajectory(
    StateParams(q_0=5.0, q_1=-2.0, v_0=0.0, v_1=0.0),
    TrajectoryBounds(v_bound=3.0, a_bound=2.0, j_bound=5.0),
)
phases = trajectory.get_phase_durations()
assert phases["total"] == trajectory.get_duration()
```

Short moves may have no constant-speed or constant-acceleration plateau. Do not
assume all seven phases have positive duration.

## Trapezoidal velocity profile

The trapezoidal `TrajectoryParams` is defined in its module; it is not the
top-level polynomial configuration with the same name.

```python
import numpy as np

from interpolatepy import TrapezoidalTrajectory
from interpolatepy.trapezoidal import TrajectoryParams

params = TrajectoryParams(
    q0=0.0,
    q1=10.0,
    v0=0.0,
    v1=0.0,
    amax=3.0,
    vmax=4.0,
)
evaluate, duration = TrapezoidalTrajectory.generate_trajectory(params)

q0, v0, _ = evaluate(0.0)
q1, v1, _ = evaluate(duration)
assert np.isclose(q0, params.q0)
assert np.isclose(q1, params.q1)
assert np.isclose(v0, params.v0)
assert np.isclose(v1, params.v1)
```

For a fixed feasible duration, provide `duration` and `amax` instead of `vmax`:

```python
from interpolatepy import TrapezoidalTrajectory
from interpolatepy.trapezoidal import TrajectoryParams

params = TrajectoryParams(
    q0=0.0,
    q1=5.0,
    v0=0.0,
    v1=0.0,
    amax=3.0,
    duration=4.0,
)
evaluate, duration = TrapezoidalTrajectory.generate_trajectory(params)
assert duration == 4.0
```

If a move is too short to reach the requested `vmax`, the planner returns a
triangular velocity profile.

## Trapezoidal interpolation through via points

```python
import numpy as np

from interpolatepy import InterpolationParams
from interpolatepy import TrapezoidalTrajectory

params = InterpolationParams(
    points=[0.0, 4.0, 1.0, 6.0],
    v0=0.0,
    vn=0.0,
    amax=4.0,
    vmax=5.0,
)
evaluate, duration = TrapezoidalTrajectory.interpolate_waypoints(params)

q_start, _, _ = evaluate(0.0)
q_end, _, _ = evaluate(duration)
assert np.isclose(q_start, params.points[0])
assert np.isclose(q_end, params.points[-1])
```

Supply `times` to prescribe the waypoint schedule or `inter_velocities` to
prescribe internal waypoint speeds. Otherwise the helper computes values
heuristically.

## Polynomial boundary interpolation

```python
import numpy as np

from interpolatepy import BoundaryCondition
from interpolatepy import PolynomialTrajectory
from interpolatepy import TimeInterval

initial = BoundaryCondition(position=0.0, velocity=0.0, acceleration=0.0)
final = BoundaryCondition(position=2.0, velocity=0.0, acceleration=0.0)
interval = TimeInterval(start=0.0, end=3.0)

evaluate = PolynomialTrajectory.order_5_trajectory(initial, final, interval)
q0, v0, a0, _ = evaluate(interval.start)
q1, v1, a1, _ = evaluate(interval.end)

assert np.allclose([q0, v0, a0], [0.0, 0.0, 0.0])
assert np.allclose([q1, v1, a1], [2.0, 0.0, 0.0])
```

Choose the lowest order that represents the boundary conditions:

- order 3: position and velocity;
- order 5: also acceleration;
- order 7: also jerk.

All returned polynomial callables produce `(position, velocity, acceleration,
jerk)`.

For multiple segments, use the top-level polynomial `TrajectoryParams`:

```python
from interpolatepy import PolynomialTrajectory
from interpolatepy import TrajectoryParams

params = TrajectoryParams(
    points=[0.0, 1.0, -0.5, 2.0],
    times=[0.0, 1.0, 2.0, 4.0],
    velocities=[0.0, 0.5, 0.5, 0.0],
    order=3,
)
evaluate = PolynomialTrajectory.multipoint_trajectory(params)
q, qd, qdd, qddd = evaluate(1.5)
```

## Linear segments with parabolic blends

`dt_blend` has one entry per waypoint and controls the local blend duration:

```python
import numpy as np

from interpolatepy import ParabolicBlendTrajectory

planner = ParabolicBlendTrajectory(
    q=[0.0, 2.0, 1.0, 4.0],
    t=[0.0, 1.0, 2.5, 4.0],
    dt_blend=[0.2, 0.3, 0.3, 0.2],
)
evaluate, duration = planner.generate()

samples = np.array([evaluate(value) for value in np.linspace(0.0, duration, 100)])
assert samples.shape == (100, 3)
```

Use positive blend durations that fit inside neighboring time intervals. The
class does not enforce every scheduling condition on your behalf.

## Choosing a profile

| Requirement | Prefer |
| --- | --- |
| Hard jerk, acceleration, and speed magnitudes | Double-S |
| Hard acceleration and speed magnitudes | Trapezoidal |
| Exact endpoint derivatives and fixed duration | Polynomial |
| Simple via-point blending | Parabolic blend |

Regardless of the choice, sample the final trajectory densely and verify every
physical limit. Per-axis scalar bounds do not imply a bound on the norm of a
multi-axis vector.
