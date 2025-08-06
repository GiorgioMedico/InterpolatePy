# Quick Start Guide

This guide will get you started with InterpolatePy in just a few minutes. We'll cover the most common use cases and show you how to create your first smooth trajectories.

## Prerequisites

Make sure you have InterpolatePy installed:

```bash
pip install InterpolatePy
```

## Your First Trajectory

Let's start with a simple cubic spline through waypoints:

```python
import numpy as np
import matplotlib.pyplot as plt
from interpolatepy import CubicSpline

# Define waypoints
t_points = [0.0, 2.0, 4.0, 6.0, 8.0]
q_points = [0.0, 3.0, -1.0, 2.0, 0.0]

# Create cubic spline with zero initial and final velocities
spline = CubicSpline(t_points, q_points, v0=0.0, vn=0.0)

# Evaluate at any time
position = spline.evaluate(3.0)
velocity = spline.evaluate_velocity(3.0)
acceleration = spline.evaluate_acceleration(3.0)

print(f"At t=3.0: pos={position:.2f}, vel={velocity:.2f}, acc={acceleration:.2f}")

# Plot the complete trajectory
spline.plot()
plt.show()
```

**What you get:**
- Smooth C² continuous trajectory through all waypoints
- Built-in visualization showing position, velocity, and acceleration
- Zero velocities at start and end points

## Common Use Cases

### 1. Jerk-Limited Motion (S-Curves)

Perfect for robotics and automation where smooth acceleration is critical:

```python
from interpolatepy import DoubleSTrajectory, StateParams, TrajectoryBounds

# Define motion parameters
state = StateParams(
    q_0=0.0,     # Start position
    q_1=10.0,    # End position  
    v_0=0.0,     # Start velocity
    v_1=0.0      # End velocity
)

bounds = TrajectoryBounds(
    v_bound=5.0,   # Max velocity
    a_bound=10.0,  # Max acceleration
    j_bound=30.0   # Max jerk
)

# Generate S-curve trajectory
trajectory = DoubleSTrajectory(state, bounds)

# Get total duration
print(f"Duration: {trajectory.get_duration():.2f}s")

# Evaluate at specific times
t_eval = np.linspace(0, trajectory.get_duration(), 100)
positions = [trajectory.evaluate(t) for t in t_eval]
velocities = [trajectory.evaluate_velocity(t) for t in t_eval]

# Visualize all profiles
trajectory.plot()
plt.show()
```

### 2. Smooth Rotation Interpolation

For 3D orientation control with quaternions:

```python
from interpolatepy import Quaternion, SquadC2
import numpy as np

# Define rotation waypoints
orientations = [
    Quaternion.identity(),                                    # No rotation
    Quaternion.from_angle_axis(np.pi/2, [1, 0, 0]),         # 90° about X
    Quaternion.from_angle_axis(np.pi, [0, 1, 0]),           # 180° about Y
    Quaternion.from_angle_axis(np.pi/4, [0, 0, 1])          # 45° about Z
]

times = [0.0, 1.0, 2.0, 3.0]

# Create smooth quaternion trajectory
quat_spline = SquadC2(times, orientations)

# Evaluate smooth rotation
t = 1.5
orientation = quat_spline.evaluate(t)
angular_velocity = quat_spline.evaluate_velocity(t)

print(f"Orientation at t={t}: {orientation}")
print(f"Angular velocity: {angular_velocity}")
```

### 3. Noise-Robust Curve Fitting

When your data has noise, use smoothing splines:

```python
from interpolatepy import smoothing_spline_with_tolerance, SplineConfig

# Noisy data points
t_noisy = np.linspace(0, 10, 20)
q_noisy = np.sin(t_noisy) + 0.1 * np.random.randn(20)

# Automatically find optimal smoothing
config = SplineConfig(max_iterations=50)
spline, mu, error, iterations = smoothing_spline_with_tolerance(
    np.array(t_noisy), 
    np.array(q_noisy), 
    tolerance=0.05,  # Maximum allowed deviation
    config=config
)

print(f"Optimal smoothing parameter: {mu:.6f}")

# Plot original data vs smoothed curve
spline.plot()
plt.scatter(t_noisy, q_noisy, color='red', alpha=0.6, label='Noisy data')
plt.legend()
plt.show()
```

### 4. Polynomial Trajectories

For precise boundary condition control:

```python
from interpolatepy import PolynomialTrajectory, BoundaryCondition, TimeInterval

# Define precise boundary conditions
initial = BoundaryCondition(
    position=0.0,
    velocity=0.0,
    acceleration=0.0,
    jerk=0.0
)

final = BoundaryCondition(
    position=5.0,
    velocity=0.0,
    acceleration=0.0,
    jerk=0.0
)

interval = TimeInterval(t0=0.0, t1=3.0)

# Generate 7th-order polynomial trajectory
traj_func = PolynomialTrajectory.order_7_trajectory(initial, final, interval)

# Evaluate complete trajectory
t_eval = np.linspace(0, 3, 100)
results = [traj_func(t) for t in t_eval]
positions = [r[0] for r in results]
velocities = [r[1] for r in results]
accelerations = [r[2] for r in results]
jerks = [r[3] for r in results]

# Plot all derivatives
fig, axes = plt.subplots(4, 1, figsize=(10, 8))
axes[0].plot(t_eval, positions, label='Position')
axes[1].plot(t_eval, velocities, label='Velocity')  
axes[2].plot(t_eval, accelerations, label='Acceleration')
axes[3].plot(t_eval, jerks, label='Jerk')

for ax in axes:
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
```

## Key Concepts

### 1. Continuity Classes

InterpolatePy algorithms provide different levels of smoothness:

- **C⁰**: Continuous position (basic interpolation)
- **C¹**: Continuous position and velocity  
- **C²**: Continuous position, velocity, and acceleration (cubic splines)

### 2. Boundary Conditions

Control trajectory behavior at endpoints:

```python
# Zero velocity at endpoints (natural spline)
spline = CubicSpline(t_points, q_points, v0=0.0, vn=0.0)

# Specified velocities
spline = CubicSpline(t_points, q_points, v0=2.0, vn=-1.5)

# With acceleration constraints
from interpolatepy import CubicSplineWithAcceleration1
spline = CubicSplineWithAcceleration1(
    t_points, q_points,
    v0=0.0, vn=0.0,
    a0=0.0, an=0.0
)
```

### 3. Evaluation Methods

All trajectory objects provide consistent evaluation methods:

```python
# Position
position = spline.evaluate(t)

# First derivative (velocity)
velocity = spline.evaluate_velocity(t)

# Second derivative (acceleration)  
acceleration = spline.evaluate_acceleration(t)

# Some algorithms also provide:
jerk = trajectory.evaluate_jerk(t)  # Third derivative
```

### 4. Vectorized Evaluation

Evaluate multiple time points efficiently:

```python
# Single time point
pos = spline.evaluate(1.5)

# Multiple time points (vectorized)
t_array = np.linspace(0, 10, 100)
positions = spline.evaluate(t_array)  # Returns numpy array
```

## Performance Tips

### 1. Use Vectorized Operations

```python
# Efficient: vectorized evaluation
t_eval = np.linspace(0, 10, 1000)
positions = spline.evaluate(t_eval)

# Inefficient: loop over single evaluations
positions = [spline.evaluate(t) for t in t_eval]
```

### 2. Reuse Trajectory Objects

```python
# Create once, evaluate many times
spline = CubicSpline(t_points, q_points)

# Efficient: reuse the same spline object
for t in time_sequence:
    pos = spline.evaluate(t)
    # Process position...
```

### 3. Choose Appropriate Algorithms

| Use Case | Recommended Algorithm | Complexity |
|----------|----------------------|------------|
| Smooth waypoint interpolation | `CubicSpline` | <span class="complexity">O(n)</span> |
| Jerk-limited motion | `DoubleSTrajectory` | <span class="complexity">O(1)</span> |
| Noisy data fitting | `CubicSmoothingSpline` | <span class="complexity">O(n)</span> |
| 3D rotation | `SquadC2` | <span class="complexity">O(1)</span> |
| High-degree control | `BSplineInterpolator` | <span class="complexity">O(n)</span> |

## Common Patterns

### Pattern 1: Trajectory with Via Points

```python
def create_trajectory_with_via_points(waypoints, durations):
    """Create smooth trajectory through multiple waypoints."""
    t_points = np.cumsum([0] + durations)
    return CubicSpline(t_points, waypoints, v0=0.0, vn=0.0)

# Usage
waypoints = [0, 5, -2, 8, 3]
durations = [2, 3, 2, 4]  # Time between waypoints
traj = create_trajectory_with_via_points(waypoints, durations)
```

### Pattern 2: Trajectory Synchronization

```python
def synchronize_trajectories(trajectories):
    """Find common duration for multiple trajectories."""
    max_duration = max(traj.get_duration() for traj in trajectories)
    
    # Evaluate all at synchronized time points
    t_sync = np.linspace(0, max_duration, 100)
    sync_data = {}
    
    for i, traj in enumerate(trajectories):
        sync_data[f'traj_{i}'] = [
            traj.evaluate(min(t, traj.get_duration())) for t in t_sync
        ]
    
    return t_sync, sync_data
```

### Pattern 3: Trajectory Blending

```python
def blend_trajectories(traj1, traj2, blend_time, total_time):
    """Smoothly blend between two trajectories."""
    def blended_traj(t):
        if t <= blend_time:
            return traj1.evaluate(t)
        elif t >= total_time - blend_time:
            return traj2.evaluate(t - (total_time - traj2.get_duration()))
        else:
            # Linear blend in middle region
            alpha = (t - blend_time) / (total_time - 2*blend_time)
            return (1-alpha) * traj1.evaluate(t) + alpha * traj2.evaluate(t)
    
    return blended_traj
```

## Next Steps

Now that you've mastered the basics:

1. **[User Guide](user-guide.md)** - Dive deeper into specific algorithms
2. **[Tutorials](tutorials/spline-interpolation.md)** - Step-by-step guides for each category
3. **[API Reference](api-reference.md)** - Complete function documentation
4. **[Examples](examples.md)** - Real-world applications and advanced usage
5. **[Algorithms](algorithms.md)** - Mathematical foundations and theory

## Troubleshooting

### Common Issues

**Q: My spline has unwanted oscillations**
A: Try using `CubicSmoothingSpline` with a small smoothing parameter, or reduce the number of waypoints.

**Q: Trajectory violates velocity/acceleration limits**  
A: Use motion profiles like `DoubleSTrajectory` or `TrapezoidalTrajectory` that enforce constraints.

**Q: Quaternion interpolation flips unexpectedly**
A: Use `SquadC2` which handles quaternion double-cover automatically.

**Q: Performance is slower than expected**
A: Ensure you're using vectorized evaluation and consider algorithm complexity for your use case.

Ready to create some smooth trajectories! 🚀