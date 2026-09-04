# Path planning

`LinearPath` and `CircularPath` describe geometry independently of time. Their
parameter `s` is arc length. A scalar motion profile supplies `s(t)` when the
path must be traversed dynamically.

## Linear path

```python
import numpy as np

from interpolatepy import LinearPath

start = np.array([0.0, 0.0, 0.0])
end = np.array([3.0, 4.0, 0.0])
path = LinearPath(start, end)

assert np.isclose(path.length, 5.0)
assert np.allclose(path.position(0.0), start)
assert np.allclose(path.position(path.length), end)
assert np.allclose(path.velocity(2.0), [0.6, 0.8, 0.0])
assert np.allclose(path.acceleration(2.0), np.zeros(3))
```

`position()` clamps values before the start or after the end. `velocity()` is
the unit tangent with respect to arc length, not a velocity in units per second.

Batch helpers return a dictionary with `s`, `position`, `velocity`, and
`acceleration` arrays:

```python
import numpy as np

from interpolatepy import LinearPath

path = LinearPath(np.zeros(3), np.array([1.0, 2.0, 2.0]))
samples = path.all_traj(num_points=25)
assert samples["position"].shape == (25, 3)
```

## Circular path

A circle is defined by an axis, a point on the axis, and a point on the circle:

```python
import numpy as np

from interpolatepy import CircularPath

axis = np.array([0.0, 0.0, 1.0])
axis_point = np.array([0.0, 0.0, 0.0])
circle_point = np.array([2.0, 0.0, 0.0])
path = CircularPath(r=axis, d=axis_point, pi=circle_point)

quarter_turn = 0.5 * np.pi * path.radius
assert np.allclose(path.position(0.0), circle_point)
assert np.allclose(path.position(quarter_turn), [0.0, 2.0, 0.0], atol=1e-12)
```

One full turn spans `2 * np.pi * path.radius`. Unlike the finite line,
`CircularPath.position()` does not clamp its arc length.

## Add a time law

For a path `p(s)` and scalar law `s(t)`, the chain rule gives

\[
\dot p = p'(s)\dot s,
\qquad
\ddot p = p''(s)\dot s^2 + p'(s)\ddot s.
\]

```python
import numpy as np

from interpolatepy import DoubleSTrajectory
from interpolatepy import LinearPath
from interpolatepy import StateParams
from interpolatepy import TrajectoryBounds

path = LinearPath(np.zeros(3), np.array([3.0, 4.0, 0.0]))
law = DoubleSTrajectory(
    StateParams(q_0=0.0, q_1=path.length, v_0=0.0, v_1=0.0),
    TrajectoryBounds(v_bound=2.0, a_bound=2.0, j_bound=5.0),
)

t = np.linspace(0.0, law.get_duration(), 120)
s, sd, sdd, _ = law.evaluate_full(t)
position = np.array([path.position(value) for value in s])
velocity = np.array([path.velocity(value) for value in s]) * sd[:, None]
acceleration = np.array(
    [
        path.acceleration(value) * speed**2
        + path.velocity(value) * tangential_acceleration
        for value, speed, tangential_acceleration in zip(s, sd, sdd, strict=True)
    ]
)

assert np.allclose(position[0], path.pi)
assert np.allclose(position[-1], path.pf)
assert np.allclose(velocity[[0, -1]], 0.0, atol=1e-9)
```

The same composition applies to a circle; the curvature term then contributes
normal acceleration.

## Frenet frames

`compute_trajectory_frames()` accepts a parametric curve callable that returns
position, first derivative, and second derivative:

```python
from functools import partial

import numpy as np

from interpolatepy import compute_trajectory_frames
from interpolatepy import helicoidal_trajectory_with_derivatives

helix = partial(helicoidal_trajectory_with_derivatives, r=2.0, d=0.4)
u = np.linspace(0.0, 4.0 * np.pi, 100)
points, frames = compute_trajectory_frames(helix, u)

assert points.shape == (100, 3)
assert frames.shape == (100, 3, 3)
assert np.allclose(frames[0].T @ frames[0], np.eye(3), atol=1e-12)
```

Each `frames[i]` stores tangent, normal, and binormal as its columns. A frame is
a rotation matrix; `frames[i][:, 0]` is the tangent.

At a straight segment or stationary point, the classical Frenet normal is not
unique. The implementation chooses a stable perpendicular fallback.

## Tool orientation

A scalar tool orientation rotates about the frame's local binormal. A tuple
applies roll, pitch, and yaw in radians:

```python
from functools import partial

import numpy as np

from interpolatepy import circular_trajectory_with_derivatives
from interpolatepy import compute_trajectory_frames

circle = partial(circular_trajectory_with_derivatives, r=1.5)
u = np.linspace(0.0, 2.0 * np.pi, 40)
points, tool_frames = compute_trajectory_frames(
    circle,
    u,
    tool_orientation=(np.deg2rad(10.0), np.deg2rad(20.0), 0.0),
)
assert tool_frames.shape == (40, 3, 3)
```

## Custom curve

Analytical derivatives give the best frame quality:

```python
import numpy as np

from interpolatepy import compute_trajectory_frames

def parabola(u: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position = np.array([u, u**2, 0.0])
    first = np.array([1.0, 2.0 * u, 0.0])
    second = np.array([0.0, 2.0, 0.0])
    return position, first, second

u = np.linspace(-1.0, 1.0, 21)
points, frames = compute_trajectory_frames(parabola, u)
assert points.shape == (21, 3)
```

Use `plot_frames()` to visualize a subset of frames on a 3D Matplotlib axis.
The [`frenet_frame_ex.py`](https://github.com/GiorgioMedico/InterpolatePy/blob/main/examples/frenet_frame_ex.py) program contains a
complete plotting workflow.
