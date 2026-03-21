# Path Planning Tutorial

This tutorial covers InterpolatePy's geometric path primitives and Frenet frame computation for spatial trajectory planning.

## Geometric Path Primitives

InterpolatePy provides two basic geometric paths parameterized by arc length.

### Linear Path

A straight line between two 3D points:

```python
from interpolatepy import LinearPath
import numpy as np

# Define endpoints
pi = np.array([0, 0, 0])   # Start
pf = np.array([5, 3, 2])   # End

path = LinearPath(pi, pf)

# Path properties
print(f"Path length: {path.length:.2f}")
print(f"Tangent (constant): {path.tangent}")

# Evaluate by arc length parameter s
s = 2.0   # 2 units along the path
position = path.position(s)
velocity = path.velocity(s)       # Constant tangent vector
acceleration = path.acceleration(s)  # Zero for straight line

print(f"Position at s={s}: {position}")
```

### Circular Path

A circular arc in 3D space defined by an axis of rotation, a point on the axis, and a starting point:

```python
from interpolatepy import CircularPath
import numpy as np

# Define circular arc
r = np.array([0, 0, 1])    # Rotation axis (Z-axis)
d = np.array([0, 0, 0])    # Point on axis (origin)
pi = np.array([1, 0, 0])   # Starting point on circle

path = CircularPath(r, d, pi)

# The radius is computed from pi and the axis
print(f"Radius: {path.radius:.2f}")

# Evaluate at arc length
s = np.pi   # Half circle
position = path.position(s)
velocity = path.velocity(s)          # Tangent to circle
acceleration = path.acceleration(s)  # Centripetal

print(f"Position at s=pi: {position}")
print(f"Acceleration (centripetal): {acceleration}")
```

## Combining Paths with Motion Laws

Geometric paths define *where* to move. Motion laws define *how fast*. Combine them for complete trajectories:

```python
from interpolatepy import LinearPath, PolynomialTrajectory, BoundaryCondition, TimeInterval
import numpy as np
import matplotlib.pyplot as plt

# 1. Define the geometric path
pi = np.array([0, 0, 0])
pf = np.array([10, 5, 3])
path = LinearPath(pi, pf)

# 2. Define the motion law (how arc length varies with time)
# Start and end at rest, traverse the full path length
initial = BoundaryCondition(position=0.0, velocity=0.0)
final = BoundaryCondition(position=path.length, velocity=0.0)
interval = TimeInterval(start=0.0, end=5.0)

motion_law = PolynomialTrajectory.order_5_trajectory(initial, final, interval)

# 3. Combine: evaluate motion law to get arc length, then path to get position
t_eval = np.linspace(0, 5, 200)
positions = []
velocities = []

for t in t_eval:
    s, ds_dt, _, _ = motion_law(t)       # Arc length and its derivatives
    pos = path.position(s)                 # 3D position
    vel = path.velocity(s) * ds_dt         # Chain rule: dp/dt = dp/ds * ds/dt
    positions.append(pos)
    velocities.append(np.linalg.norm(vel))

positions = np.array(positions)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 3D path
ax3d = fig.add_subplot(121, projection="3d")
ax3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], "b-", linewidth=2)
ax3d.scatter(*pi, color="green", s=100, label="Start")
ax3d.scatter(*pf, color="red", s=100, label="End")
ax3d.set_xlabel("X")
ax3d.set_ylabel("Y")
ax3d.set_zlabel("Z")
ax3d.set_title("Linear Path with 5th-Order Motion Law")
ax3d.legend()

# Speed profile
axes[1].plot(t_eval, velocities, "r-", linewidth=2)
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Speed")
axes[1].set_title("Speed Profile (smooth start/stop)")
axes[1].grid(True)

plt.tight_layout()
plt.show()
```

## Frenet Frame Computation

The Frenet-Serret frame provides a local coordinate system (tangent, normal, binormal) at each point along a curve. This is essential for tool orientation in CNC machining and robotic path following.

### Basic Usage

```python
from interpolatepy import compute_trajectory_frames
from interpolatepy import helicoidal_trajectory_with_derivatives
import numpy as np

# Define a helicoidal (helical) trajectory
r = 2.0   # Radius
d = 0.5   # Pitch per radian

def helix_func(u):
    return helicoidal_trajectory_with_derivatives(u, r, d)

# Compute frames at sampled points
u_values = np.linspace(0, 4 * np.pi, 100)
points, frames = compute_trajectory_frames(helix_func, u_values)

# points: (100, 3) array of 3D positions
# frames: (100, 3, 3) array where frames[i] = [tangent, normal, binormal]

print(f"Tangent at u=0: {frames[0, 0]}")
print(f"Normal at u=0:  {frames[0, 1]}")
print(f"Binormal at u=0: {frames[0, 2]}")
```

### Visualizing Frames

```python
from interpolatepy import compute_trajectory_frames, plot_frames
from interpolatepy import helicoidal_trajectory_with_derivatives
import numpy as np
import matplotlib.pyplot as plt

r, d = 2.0, 0.5
u_values = np.linspace(0, 4 * np.pi, 100)

def helix_func(u):
    return helicoidal_trajectory_with_derivatives(u, r, d)

points, frames = compute_trajectory_frames(helix_func, u_values)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
plot_frames(ax, points, frames, scale=0.5, skip=10)
ax.set_title("Helicoidal Trajectory with Frenet Frames")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.tight_layout()
plt.show()
```

### Tool Orientation

For CNC machining or robotic applications, you can specify an additional tool orientation relative to the Frenet frame:

```python
from interpolatepy import compute_trajectory_frames
from interpolatepy import circular_trajectory_with_derivatives
import numpy as np

def circle_func(u):
    return circular_trajectory_with_derivatives(u, radius=3.0)

u_values = np.linspace(0, 2 * np.pi, 50)

# Add tool tilt: (roll, pitch, yaw) relative to Frenet frame
points, frames = compute_trajectory_frames(
    circle_func,
    u_values,
    tool_orientation=(0.1, -0.2, 0.0)  # Small roll and pitch tilt
)
```

### Custom Path Functions

Any function returning `(position, first_derivative, second_derivative)` works:

```python
from interpolatepy import compute_trajectory_frames
import numpy as np

def lissajous_3d(u):
    """3D Lissajous curve with derivatives."""
    position = np.array([
        np.sin(2 * u),
        np.sin(3 * u),
        np.sin(5 * u) * 0.5
    ])
    first_derivative = np.array([
        2 * np.cos(2 * u),
        3 * np.cos(3 * u),
        5 * np.cos(5 * u) * 0.5
    ])
    second_derivative = np.array([
        -4 * np.sin(2 * u),
        -9 * np.sin(3 * u),
        -25 * np.sin(5 * u) * 0.5
    ])
    return position, first_derivative, second_derivative

u_values = np.linspace(0, 2 * np.pi, 200)
points, frames = compute_trajectory_frames(lissajous_3d, u_values)
```

## Built-in Trajectory Functions

InterpolatePy provides ready-made trajectory functions for common curves:

| Function | Description |
|----------|-------------|
| `helicoidal_trajectory_with_derivatives(u, r, d)` | Helical curve with radius `r` and pitch `d` |
| `circular_trajectory_with_derivatives(u, radius)` | Circle in the XY plane |

## Next Steps

- **[API Reference](../api-reference.md#path-planning)** for complete method documentation
- **[Algorithms Guide](../algorithms.md)** for Frenet-Serret mathematical foundations
- **Example scripts**: `examples/simple_paths_ex.py`, `examples/frenet_frame_ex.py`
