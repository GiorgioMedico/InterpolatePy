# API Reference

Complete API documentation for all InterpolatePy classes, functions, and data structures.

## Overview

InterpolatePy is organized into the following modules:

- **[Spline Interpolation](#spline-interpolation)**: Cubic splines, B-splines, and smoothing variants
- **[Motion Profiles](#motion-profiles)**: S-curves, trapezoidal, and polynomial trajectories  
- **[Quaternion Interpolation](#quaternion-interpolation)**: 3D rotation interpolation methods
- **[Path Planning](#path-planning)**: Geometric paths and coordinate frames
- **[Utilities](#utilities)**: Helper functions and data structures

## Spline Interpolation

### CubicSpline {#cubic-spline}

::: interpolatepy.CubicSpline
    options:
      members:
        - __init__
        - evaluate
        - evaluate_velocity
        - evaluate_acceleration
        - plot
      show_source: false
      show_root_heading: true
      show_root_toc_entry: false

**Example:**
```python
from interpolatepy import CubicSpline
import numpy as np

# Create spline through waypoints
t_points = [0, 1, 2, 3, 4]
q_points = [0, 1, 4, 2, 0]
spline = CubicSpline(t_points, q_points, v0=0.0, vn=0.0)

# Evaluate trajectory
t = 1.5
position = spline.evaluate(t)
velocity = spline.evaluate_velocity(t)
acceleration = spline.evaluate_acceleration(t)

# Vectorized evaluation
t_array = np.linspace(0, 4, 100)
trajectory = spline.evaluate(t_array)
```

### CubicSmoothingSpline {#cubic-smoothing-spline}

::: interpolatepy.CubicSmoothingSpline
    options:
      members:
        - __init__
        - evaluate
        - evaluate_velocity
        - evaluate_acceleration
        - plot
      show_source: false

**Example:**
```python
from interpolatepy import CubicSmoothingSpline

# Noisy data points
t_points = [0, 0.5, 1.0, 1.5, 2.0]
q_noisy = [0.1, 1.05, -0.02, -0.98, 0.05]

# Create smoothing spline
spline = CubicSmoothingSpline(t_points, q_noisy, mu=0.1)
spline.plot()
```

<a id="cubic-spline-with-acceleration"></a>
### CubicSplineWithAcceleration1 {#cubic-spline-with-acceleration1}

::: interpolatepy.CubicSplineWithAcceleration1
    options:
      members:
        - __init__
        - evaluate
        - evaluate_velocity
        - evaluate_acceleration
        - plot
        - original_indices
      show_source: false

### CubicSplineWithAcceleration2 {#cubic-spline-with-acceleration2}

::: interpolatepy.CubicSplineWithAcceleration2
    options:
      members:
        - __init__
        - evaluate
        - evaluate_velocity
        - evaluate_acceleration
        - plot
      show_source: false

### B-Spline Family

#### BSpline {#b-spline}

::: interpolatepy.BSpline
    options:
      show_source: false

#### BSplineInterpolator {#b-spline-interpolator}

::: interpolatepy.BSplineInterpolator
    options:
      show_source: false

**Example:**
```python
from interpolatepy import BSplineInterpolator
import numpy as np

# 2D curve data
points = np.array([[0, 0], [1, 2], [3, 1], [4, 3], [5, 0]])
times = np.array([0, 1, 2, 3, 4])

# Create B-spline interpolator
bspline = BSplineInterpolator(
    degree=3, 
    points=points, 
    times=times,
    initial_velocity=np.array([1, 0]),
    final_velocity=np.array([0, -1])
)

# Evaluate curve
t = 2.5
position = bspline.evaluate(t)
velocity = bspline.evaluate_velocity(t)
```

#### ApproximationBSpline {#approximation-b-spline}

::: interpolatepy.ApproximationBSpline
    options:
      show_source: false

#### SmoothingCubicBSpline {#smoothing-cubic-b-spline}

::: interpolatepy.SmoothingCubicBSpline
    options:
      show_source: false

## Motion Profiles

### DoubleSTrajectory {#double-s-trajectory}

::: interpolatepy.DoubleSTrajectory
    options:
      show_source: false

#### StateParams

::: interpolatepy.StateParams
    options:
      show_source: false

#### TrajectoryBounds

::: interpolatepy.TrajectoryBounds
    options:
      show_source: false

**Example:**
```python
from interpolatepy import DoubleSTrajectory, StateParams, TrajectoryBounds

# Define trajectory parameters
state = StateParams(
    q_0=0.0,     # Initial position
    q_1=10.0,    # Final position
    v_0=0.0,     # Initial velocity
    v_1=0.0      # Final velocity
)

bounds = TrajectoryBounds(
    v_bound=5.0,   # Maximum velocity
    a_bound=10.0,  # Maximum acceleration
    j_bound=30.0   # Maximum jerk
)

# Create S-curve trajectory
trajectory = DoubleSTrajectory(state, bounds)

# Evaluate trajectory
t = trajectory.get_duration() / 2
position = trajectory.evaluate(t)
velocity = trajectory.evaluate_velocity(t)
acceleration = trajectory.evaluate_acceleration(t)
jerk = trajectory.evaluate_jerk(t)

print(f"Duration: {trajectory.get_duration():.2f}s")
```

### TrapezoidalTrajectory {#trapezoidal-trajectory}

::: interpolatepy.TrapezoidalTrajectory
    options:
      show_source: false

#### TrajectoryParams

::: interpolatepy.TrajectoryParams
    options:
      show_source: false

**Example:**
```python
from interpolatepy import TrapezoidalTrajectory, TrajectoryParams

# Define trajectory parameters
params = TrajectoryParams(
    q0=0.0,       # Initial position
    q1=5.0,       # Final position
    v0=0.0,       # Initial velocity
    v1=0.0,       # Final velocity
    amax=2.0,     # Maximum acceleration
    vmax=1.5      # Maximum velocity
)

# Generate trajectory
traj_func, duration = TrapezoidalTrajectory.generate_trajectory(params)

# Evaluate at specific time
t = 1.0
pos, vel, acc = traj_func(t)
print(f"At t={t}: pos={pos:.2f}, vel={vel:.2f}, acc={acc:.2f}")
```

### PolynomialTrajectory {#polynomial-trajectory}

::: interpolatepy.PolynomialTrajectory
    options:
      show_source: false

#### BoundaryCondition

::: interpolatepy.BoundaryCondition
    options:
      show_source: false

#### TimeInterval

::: interpolatepy.TimeInterval
    options:
      show_source: false

**Example:**
```python
from interpolatepy import PolynomialTrajectory, BoundaryCondition, TimeInterval

# Define boundary conditions
initial = BoundaryCondition(
    position=0.0,
    velocity=0.0,
    acceleration=0.0,
    jerk=0.0
)

final = BoundaryCondition(
    position=1.0,
    velocity=0.0,
    acceleration=0.0,
    jerk=0.0
)

interval = TimeInterval(t0=0.0, t1=2.0)

# Generate 7th-order polynomial
traj_func = PolynomialTrajectory.order_7_trajectory(initial, final, interval)

# Evaluate trajectory
t = 1.0
pos, vel, acc, jerk = traj_func(t)
```

### ParabolicBlendTrajectory {#parabolic-blend-trajectory}

::: interpolatepy.ParabolicBlendTrajectory
    options:
      show_source: false

## Quaternion Interpolation

### Quaternion

::: interpolatepy.Quaternion
    options:
      members:
        - __init__
        - norm
        - unit
        - inverse
        - conjugate
        - slerp
        - squad
        - Log
        - exp
        - to_axis_angle
        - from_angle_axis
        - from_euler_angles
        - to_euler_angles
        - identity
      show_source: false

**Example:**
```python
from interpolatepy import Quaternion
import numpy as np

# Create quaternions
q1 = Quaternion.identity()
q2 = Quaternion.from_angle_axis(np.pi/2, [0, 0, 1])  # 90° about Z

# SLERP interpolation
t = 0.5
q_interp = q1.slerp(q2, t)

# Convert to axis-angle
axis, angle = q_interp.to_axis_angle()
print(f"Interpolated rotation: {np.degrees(angle):.1f}° about {axis}")

# Convert to Euler angles
roll, pitch, yaw = q_interp.to_euler_angles()
print(f"Euler angles: roll={np.degrees(roll):.1f}°, "
      f"pitch={np.degrees(pitch):.1f}°, yaw={np.degrees(yaw):.1f}°")
```

### SquadC2 {#squad-c2}

::: interpolatepy.SquadC2
    options:
      show_source: false

**Example:**
```python
from interpolatepy import SquadC2, Quaternion
import numpy as np

# Define rotation waypoints
times = [0, 1, 2, 3]
orientations = [
    Quaternion.identity(),
    Quaternion.from_angle_axis(np.pi/2, [1, 0, 0]),
    Quaternion.from_angle_axis(np.pi, [0, 1, 0]),
    Quaternion.from_angle_axis(np.pi/4, [0, 0, 1])
]

# Create C² continuous quaternion spline
squad = SquadC2(times, orientations)

# Evaluate smooth rotation trajectory
t = 1.5
orientation = squad.evaluate(t)
angular_velocity = squad.evaluate_velocity(t)
angular_acceleration = squad.evaluate_acceleration(t)
```

### QuaternionSpline {#quaternion-spline}

::: interpolatepy.QuaternionSpline
    options:
      show_source: false

<a id="log-quaternion-interpolation"></a>
<a id="modified-log-quaternion-interpolation"></a>

## Path Planning

<a id="simple-linear-path"></a>
### LinearPath {#linear-path}

::: interpolatepy.LinearPath
    options:
      show_source: false

**Example:**
```python
from interpolatepy import LinearPath
import numpy as np

# Define line segment
pi = np.array([0, 0, 0])    # Start point
pf = np.array([5, 3, 2])    # End point

path = LinearPath(pi, pf)

# Evaluate along path (s = arc length parameter)
s = 2.0  # 2 units along path
position = path.position(s)
velocity = path.velocity(s)      # Unit tangent vector
acceleration = path.acceleration(s)  # Zero for straight line

# Evaluate multiple points
s_values = np.linspace(0, path.total_length, 50)
trajectory = path.evaluate_at(s_values)
```

<a id="simple-circular-path"></a>
### CircularPath {#circular-path}

::: interpolatepy.CircularPath
    options:
      show_source: false

**Example:**
```python
from interpolatepy import CircularPath
import numpy as np

# Define circular arc
r = np.array([0, 0, 1])     # Axis direction (Z-axis)
d = np.array([0, 0, 0])     # Point on axis (origin)
pi = np.array([1, 0, 0])    # Point on circle

path = CircularPath(r, d, pi)

# Evaluate along arc
s = np.pi  # π units of arc length (180°)
position = path.position(s)
velocity = path.velocity(s)      # Tangent to circle
acceleration = path.acceleration(s)  # Centripetal acceleration

# Evaluate complete circle
s_values = np.linspace(0, 2*np.pi*path.radius, 100)
trajectory = path.evaluate_at(s_values)
```

<a id="frenet-frame-computation"></a>
### Frenet Frame Computation {#frenet-frame}

::: interpolatepy.compute_trajectory_frames

**Example:**
```python
from interpolatepy import compute_trajectory_frames
import numpy as np

def helix_path(u):
    """Parametric helix with derivatives."""
    r, pitch = 2.0, 0.5
    
    position = np.array([
        r * np.cos(u),
        r * np.sin(u),
        pitch * u
    ])
    
    first_derivative = np.array([
        -r * np.sin(u),
        r * np.cos(u),
        pitch
    ])
    
    second_derivative = np.array([
        -r * np.cos(u),
        -r * np.sin(u),
        0
    ])
    
    return position, first_derivative, second_derivative

# Compute Frenet frames
u_values = np.linspace(0, 4*np.pi, 100)
points, frames = compute_trajectory_frames(
    helix_path, 
    u_values,
    tool_orientation=(0.1, 0.2, 0.0)  # Additional tool rotation
)

# frames[i] contains [tangent, normal, binormal] vectors at points[i]
```

## Utilities

### Linear Interpolation

::: interpolatepy.linear_traj

**Example:**
```python
from interpolatepy import linear_traj
import numpy as np

# Linear interpolation between points
p0 = [0, 0, 0]
p1 = [5, 3, 2]
t0, t1 = 0.0, 2.0

time_array = np.linspace(t0, t1, 50)
positions, velocities, accelerations = linear_traj(p0, p1, t0, t1, time_array)

# positions[i] = interpolated position at time_array[i]
# velocities[i] = constant velocity vector
# accelerations[i] = zero vector (constant velocity)
```

### Tridiagonal Solver

::: interpolatepy.solve_tridiagonal

**Example:**
```python
from interpolatepy import solve_tridiagonal
import numpy as np

# Solve tridiagonal system Ax = b
# where A has diagonals a (lower), b (main), c (upper)

n = 5
a = np.array([0, 1, 1, 1, 1])      # Lower diagonal (first element unused)
b = np.array([2, 2, 2, 2, 2])      # Main diagonal
c = np.array([1, 1, 1, 1, 0])      # Upper diagonal (last element unused)
d = np.array([1, 2, 3, 4, 5])      # Right-hand side

# Solve system
x = solve_tridiagonal(a, b, c, d)
print(f"Solution: {x}")
```

## Data Structures

### Configuration Classes

The following dataclasses are used for algorithm configuration:

#### SplineConfig

::: interpolatepy.SplineConfig
    options:
      show_source: false

#### SplineParameters

::: interpolatepy.SplineParameters
    options:
      show_source: false

#### BSplineParams

::: interpolatepy.BSplineParams
    options:
      show_source: false

## Performance Notes

### Evaluation Complexity

| Algorithm | Setup | Single Eval | Vectorized Eval | Memory |
|-----------|-------|-------------|-----------------|---------|
| `CubicSpline` | O(n) | O(log n) | O(k log n) | O(n) |
| `DoubleSTrajectory` | O(1) | O(1) | O(k) | O(1) |
| `BSplineInterpolator` | O(n²) | O(p) | O(kp) | O(n) |
| `SquadC2` | O(n) | O(1) | O(k) | O(n) |
| `TrapezoidalTrajectory` | O(1) | O(1) | O(k) | O(1) |

Where:
- n = number of waypoints
- k = number of evaluation points  
- p = B-spline degree

### Memory Usage

- **Splines**: Store coefficient arrays (4n floats for cubic splines)
- **Motion Profiles**: Store only parameters (typically <10 floats)
- **Quaternions**: Store waypoints and intermediate calculations (8n floats)

### Vectorization

All algorithms support vectorized evaluation:

```python
# Efficient vectorized evaluation
t_array = np.linspace(0, 10, 1000)
result = algorithm.evaluate(t_array)  # Single call

# Less efficient scalar evaluation
result = [algorithm.evaluate(t) for t in t_array]  # 1000 calls
```

## Error Handling

### Common Exceptions

#### ValueError
- Invalid input dimensions
- Non-monotonic time sequences
- Evaluation outside trajectory domain

#### RuntimeError
- Numerical instability in solver
- Failed convergence in iterative algorithms

#### TypeError
- Incorrect input types
- Missing required parameters

### Safe Evaluation Patterns

```python
# Bounds checking
try:
    result = spline.evaluate(t)
except ValueError as e:
    # Handle out-of-bounds evaluation
    if t < spline.t_points[0]:
        result = spline.evaluate(spline.t_points[0])
    elif t > spline.t_points[-1]:
        result = spline.evaluate(spline.t_points[-1])
    else:
        raise e

# Input validation
def validate_waypoints(t_points, q_points):
    if len(t_points) != len(q_points):
        raise ValueError("Mismatched array lengths")
    if not all(t_points[i] < t_points[i+1] for i in range(len(t_points)-1)):
        raise ValueError("Time points must be strictly increasing")
```

## Version Information

Current version: **2.0.0**

Access version information:
```python
import interpolatepy
print(interpolatepy.__version__)
```

For the complete changelog, see [Changelog](changelog.md).