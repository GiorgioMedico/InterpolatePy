# Spline Interpolation Tutorial

This tutorial covers InterpolatePy's spline interpolation algorithms, from basic cubic splines to advanced B-spline methods. You'll learn when to use each algorithm and how to handle common challenges.

## What Are Splines?

Splines are piecewise polynomial curves that maintain smoothness across segment boundaries. They're ideal for:

- **Waypoint interpolation**: Creating smooth paths through specific points
- **Data fitting**: Approximating noisy or sparse datasets
- **Trajectory generation**: Producing C² continuous motion profiles

## Basic Cubic Splines

### Getting Started

The most common spline is the cubic spline, which provides C² continuity (continuous position, velocity, and acceleration).

```python
import numpy as np
import matplotlib.pyplot as plt
from interpolatepy import CubicSpline

# Define waypoints
t_points = [0, 1, 2, 3, 4, 5]
q_points = [0, 1, 3, 2, 4, 2]

# Create cubic spline
spline = CubicSpline(t_points, q_points, v0=0.0, vn=0.0)

# Visualize
spline.plot()
plt.show()
```

### Understanding Boundary Conditions

Boundary conditions control how the spline behaves at the endpoints:

```python
import matplotlib.pyplot as plt

# Same waypoints
t_points = [0, 1, 2, 3, 4]
q_points = [0, 2, 1, 3, 1]

# Different boundary conditions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Natural spline (zero curvature at ends)
spline_natural = CubicSpline(t_points, q_points)  # Default: v0=0, vn=0
t_eval = np.linspace(0, 4, 100)
axes[0, 0].plot(t_eval, spline_natural.evaluate(t_eval), 'b-', linewidth=2)
axes[0, 0].scatter(t_points, q_points, color='red', s=50, zorder=5)
axes[0, 0].set_title('Natural Spline (v₀=0, vₙ=0)')
axes[0, 0].grid(True)

# Specified initial velocity
spline_v0 = CubicSpline(t_points, q_points, v0=2.0, vn=0.0)
axes[0, 1].plot(t_eval, spline_v0.evaluate(t_eval), 'g-', linewidth=2)
axes[0, 1].scatter(t_points, q_points, color='red', s=50, zorder=5)
axes[0, 1].set_title('Initial Velocity v₀=2.0')
axes[0, 1].grid(True)

# Specified final velocity
spline_vn = CubicSpline(t_points, q_points, v0=0.0, vn=-1.5)
axes[1, 0].plot(t_eval, spline_vn.evaluate(t_eval), 'm-', linewidth=2)
axes[1, 0].scatter(t_points, q_points, color='red', s=50, zorder=5)
axes[1, 0].set_title('Final Velocity vₙ=-1.5')
axes[1, 0].grid(True)

# Both velocities specified
spline_both = CubicSpline(t_points, q_points, v0=1.0, vn=-1.0)
axes[1, 1].plot(t_eval, spline_both.evaluate(t_eval), 'orange', linewidth=2)
axes[1, 1].scatter(t_points, q_points, color='red', s=50, zorder=5)
axes[1, 1].set_title('Both Velocities (v₀=1.0, vₙ=-1.0)')
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()
```

### Analyzing Continuity Properties

Let's verify the C² continuity properties:

```python
import numpy as np
import matplotlib.pyplot as plt

# Create spline
t_points = [0, 1, 2, 3]
q_points = [0, 1, 0, 2]
spline = CubicSpline(t_points, q_points)

# Evaluate around waypoint (t=1)
t_test = 1.0
eps_values = np.logspace(-8, -1, 8)
continuity_errors = {'position': [], 'velocity': [], 'acceleration': []}

for eps in eps_values:
    # Left and right limits
    pos_left = spline.evaluate(t_test - eps)
    pos_right = spline.evaluate(t_test + eps)
    vel_left = spline.evaluate_velocity(t_test - eps)
    vel_right = spline.evaluate_velocity(t_test + eps)
    acc_left = spline.evaluate_acceleration(t_test - eps)
    acc_right = spline.evaluate_acceleration(t_test + eps)
    
    # Compute errors
    continuity_errors['position'].append(abs(pos_right - pos_left))
    continuity_errors['velocity'].append(abs(vel_right - vel_left))
    continuity_errors['acceleration'].append(abs(acc_right - acc_left))

# Plot continuity verification
fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(eps_values, continuity_errors['position'], 'o-', label='Position (C⁰)')
ax.loglog(eps_values, continuity_errors['velocity'], 's-', label='Velocity (C¹)')
ax.loglog(eps_values, continuity_errors['acceleration'], '^-', label='Acceleration (C²)')
ax.set_xlabel('ε (distance from waypoint)')
ax.set_ylabel('Continuity Error')
ax.set_title('C² Continuity Verification at Waypoint')
ax.legend()
ax.grid(True)
plt.show()

print("Cubic splines provide:")
print("✓ C⁰ continuity (continuous position)")
print("✓ C¹ continuity (continuous velocity)")  
print("✓ C² continuity (continuous acceleration)")
```

## Handling Noisy Data with Smoothing Splines

When your data contains noise, exact interpolation may not be desirable. Smoothing splines balance fidelity to the data with curve smoothness.

### Manual Smoothing Parameter

```python
import numpy as np
import matplotlib.pyplot as plt
from interpolatepy import CubicSmoothingSpline

# Generate noisy data
np.random.seed(42)
t_true = np.linspace(0, 10, 50)
q_true = np.sin(t_true) + 0.5 * np.sin(3 * t_true)
q_noisy = q_true + 0.2 * np.random.randn(len(t_true))

# Try different smoothing parameters
smoothing_params = [0.0, 0.01, 0.1, 1.0]
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, mu in enumerate(smoothing_params):
    # Create smoothing spline
    spline = CubicSmoothingSpline(t_true.tolist(), q_noisy.tolist(), mu=mu)
    
    # Evaluate
    t_eval = np.linspace(0, 10, 200)
    q_smooth = spline.evaluate(t_eval)
    
    # Plot
    axes[i].plot(t_true, q_true, 'g-', linewidth=2, label='True signal')
    axes[i].scatter(t_true, q_noisy, alpha=0.6, color='red', s=20, label='Noisy data')
    axes[i].plot(t_eval, q_smooth, 'b-', linewidth=2, label=f'Smoothed (μ={mu})')
    axes[i].set_title(f'Smoothing Parameter μ = {mu}')
    axes[i].legend()
    axes[i].grid(True)

plt.tight_layout()
plt.show()
```

### Automatic Smoothing Parameter Selection

```python
from interpolatepy import smoothing_spline_with_tolerance

# Automatically find optimal smoothing parameter
tolerance = 0.1  # Maximum allowed deviation from data points
spline_auto = smoothing_spline_with_tolerance(
    t_true.tolist(), 
    q_noisy.tolist(), 
    tolerance=tolerance,
    max_iterations=50
)

print(f"Optimal smoothing parameter: μ = {spline_auto.mu:.6f}")

# Compare with manual selection
fig, ax = plt.subplots(figsize=(12, 6))

# Plot results
t_eval = np.linspace(0, 10, 200)
q_auto = spline_auto.evaluate(t_eval)

ax.plot(t_true, q_true, 'g-', linewidth=2, label='True signal')
ax.scatter(t_true, q_noisy, alpha=0.6, color='red', s=20, label='Noisy data')
ax.plot(t_eval, q_auto, 'b-', linewidth=2, 
        label=f'Auto-smoothed (μ={spline_auto.mu:.4f})')

# Add tolerance bands
ax.fill_between(t_true, q_noisy - tolerance, q_noisy + tolerance, 
                alpha=0.2, color='orange', label=f'Tolerance ±{tolerance}')

ax.set_xlabel('Time')
ax.set_ylabel('Position')
ax.set_title('Automatic Smoothing Parameter Selection')
ax.legend()
ax.grid(True)
plt.show()

# Compute deviation from data points
deviations = [abs(spline_auto.evaluate(t) - q) for t, q in zip(t_true, q_noisy)]
max_deviation = max(deviations)
print(f"Maximum deviation from data: {max_deviation:.3f} (tolerance: {tolerance})")
```

## Advanced Splines with Acceleration Boundary Conditions

Sometimes you need to specify not just velocity but also acceleration at the endpoints.

### Method 1: Virtual Waypoint Insertion

```python
from interpolatepy import CubicSplineWithAcceleration1

# Define trajectory with acceleration constraints
t_points = [0, 2, 4, 6, 8]
q_points = [0, 3, -1, 2, 0]

# Create spline with acceleration boundary conditions
spline_acc = CubicSplineWithAcceleration1(
    t_points, q_points,
    v0=0.0, vn=0.0,    # Zero velocity at endpoints
    a0=2.0, an=-1.0    # Specified accelerations
)

# Compare with regular cubic spline
spline_regular = CubicSpline(t_points, q_points, v0=0.0, vn=0.0)

# Plot comparison
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
t_eval = np.linspace(0, 8, 200)

# Position
pos_acc = spline_acc.evaluate(t_eval)
pos_reg = spline_regular.evaluate(t_eval)
axes[0].plot(t_eval, pos_acc, 'b-', linewidth=2, label='With acceleration BC')
axes[0].plot(t_eval, pos_reg, 'r--', linewidth=2, label='Regular spline')
axes[0].scatter(t_points, q_points, color='red', s=50, zorder=5)
axes[0].set_ylabel('Position')
axes[0].set_title('Position Comparison')
axes[0].legend()
axes[0].grid(True)

# Velocity  
vel_acc = spline_acc.evaluate_velocity(t_eval)
vel_reg = spline_regular.evaluate_velocity(t_eval)
axes[1].plot(t_eval, vel_acc, 'b-', linewidth=2, label='With acceleration BC')
axes[1].plot(t_eval, vel_reg, 'r--', linewidth=2, label='Regular spline')
axes[1].set_ylabel('Velocity')
axes[1].set_title('Velocity Comparison')
axes[1].legend()
axes[1].grid(True)

# Acceleration
acc_acc = spline_acc.evaluate_acceleration(t_eval)
acc_reg = spline_regular.evaluate_acceleration(t_eval)
axes[2].plot(t_eval, acc_acc, 'b-', linewidth=2, label='With acceleration BC')
axes[2].plot(t_eval, acc_reg, 'r--', linewidth=2, label='Regular spline')
axes[2].axhline(y=2.0, color='blue', linestyle=':', alpha=0.7, label='Initial acc = 2.0')
axes[2].axhline(y=-1.0, color='blue', linestyle=':', alpha=0.7, label='Final acc = -1.0')
axes[2].set_xlabel('Time')
axes[2].set_ylabel('Acceleration')
axes[2].set_title('Acceleration Comparison')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()

# Verify boundary conditions
print("Boundary condition verification:")
print(f"Initial acceleration: {spline_acc.evaluate_acceleration(0):.3f} (target: 2.0)")
print(f"Final acceleration: {spline_acc.evaluate_acceleration(8):.3f} (target: -1.0)")
```

### Method 2: Quintic End Segments

```python
from interpolatepy import CubicSplineWithAcceleration2

# Same problem with Method 2
spline_acc2 = CubicSplineWithAcceleration2(
    t_points, q_points,
    v0=0.0, vn=0.0,
    a0=2.0, an=-1.0
)

# Compare both methods
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

pos_acc1 = spline_acc.evaluate(t_eval)
pos_acc2 = spline_acc2.evaluate(t_eval)
vel_acc1 = spline_acc.evaluate_velocity(t_eval)
vel_acc2 = spline_acc2.evaluate_velocity(t_eval)
acc_acc1 = spline_acc.evaluate_acceleration(t_eval)
acc_acc2 = spline_acc2.evaluate_acceleration(t_eval)

# Position
axes[0].plot(t_eval, pos_acc1, 'b-', linewidth=2, label='Method 1 (Virtual waypoints)')
axes[0].plot(t_eval, pos_acc2, 'g--', linewidth=2, label='Method 2 (Quintic ends)')
axes[0].scatter(t_points, q_points, color='red', s=50, zorder=5)
axes[0].set_ylabel('Position')
axes[0].legend()
axes[0].grid(True)

# Velocity
axes[1].plot(t_eval, vel_acc1, 'b-', linewidth=2, label='Method 1')
axes[1].plot(t_eval, vel_acc2, 'g--', linewidth=2, label='Method 2')
axes[1].set_ylabel('Velocity')
axes[1].legend()
axes[1].grid(True)

# Acceleration
axes[2].plot(t_eval, acc_acc1, 'b-', linewidth=2, label='Method 1')
axes[2].plot(t_eval, acc_acc2, 'g--', linewidth=2, label='Method 2')
axes[2].set_xlabel('Time')
axes[2].set_ylabel('Acceleration')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()

print("\\nMethod comparison:")
print(f"Method 1 - Initial acc: {spline_acc.evaluate_acceleration(0):.6f}")
print(f"Method 2 - Initial acc: {spline_acc2.evaluate_acceleration(0):.6f}")
print(f"Method 1 - Final acc: {spline_acc.evaluate_acceleration(8):.6f}")
print(f"Method 2 - Final acc: {spline_acc2.evaluate_acceleration(8):.6f}")
```

## B-Spline Interpolation

B-splines provide more flexibility than cubic splines, supporting different degrees and local control.

### Basic B-Spline Interpolation

```python
from interpolatepy import BSplineInterpolator
import numpy as np

# 2D curve example
t_points = np.array([0, 1, 2, 3, 4, 5])
waypoints_2d = np.array([
    [0, 0],    # Start point
    [1, 2],    # 
    [3, 1],    # Waypoints
    [4, 3],    #
    [5, 2],    #
    [6, 0]     # End point
])

# Create B-spline interpolators with different degrees
degrees = [3, 4, 5]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, degree in enumerate(degrees):
    # Create B-spline
    bspline = BSplineInterpolator(
        degree=degree,
        points=waypoints_2d,
        times=t_points,
        initial_velocity=np.array([1, 0]),  # Initial direction
        final_velocity=np.array([0, -1])    # Final direction
    )
    
    # Evaluate curve
    t_eval = np.linspace(0, 5, 200)
    curve_points = np.array([bspline.evaluate(t) for t in t_eval])
    
    # Plot
    axes[i].plot(curve_points[:, 0], curve_points[:, 1], 'b-', linewidth=2, 
                 label=f'Degree {degree} B-spline')
    axes[i].scatter(waypoints_2d[:, 0], waypoints_2d[:, 1], 
                    color='red', s=60, zorder=5, label='Waypoints')
    axes[i].set_xlabel('X')
    axes[i].set_ylabel('Y')
    axes[i].set_title(f'B-Spline Degree {degree}')
    axes[i].legend()
    axes[i].grid(True)
    axes[i].axis('equal')

plt.tight_layout()
plt.show()
```

### B-Spline vs Cubic Spline Comparison

```python
# 1D comparison
t_points_1d = [0, 1, 2, 3, 4]
q_points_1d = [0, 2, -1, 3, 1]

# Create both types
cubic_spline = CubicSpline(t_points_1d, q_points_1d, v0=0, vn=0)
bspline_interp = BSplineInterpolator(
    degree=3,
    points=np.array([[q] for q in q_points_1d]),  # Convert to 2D array
    times=np.array(t_points_1d),
    initial_velocity=np.array([0]),
    final_velocity=np.array([0])
)

# Evaluate and compare
t_eval = np.linspace(0, 4, 200)
cubic_values = cubic_spline.evaluate(t_eval)
bspline_values = np.array([bspline_interp.evaluate(t)[0] for t in t_eval])

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Position comparison
axes[0].plot(t_eval, cubic_values, 'b-', linewidth=2, label='Cubic Spline')
axes[0].plot(t_eval, bspline_values, 'r--', linewidth=2, label='B-Spline (degree 3)')
axes[0].scatter(t_points_1d, q_points_1d, color='black', s=60, zorder=5)
axes[0].set_ylabel('Position')
axes[0].set_title('Cubic Spline vs B-Spline Comparison')
axes[0].legend()
axes[0].grid(True)

# Difference
difference = cubic_values - bspline_values
axes[1].plot(t_eval, difference, 'g-', linewidth=2)
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Difference')
axes[1].set_title('Difference Between Methods')
axes[1].grid(True)

plt.tight_layout()
plt.show()

print(f"Maximum difference: {np.max(np.abs(difference)):.8f}")
print("Note: Small differences are due to different parameterizations")
```

## Practical Applications

### Multi-Axis Robot Trajectory

```python
# Simulate 6-DOF robot arm trajectory
import numpy as np
import matplotlib.pyplot as plt

# Joint limits and waypoints
joint_names = ['Base', 'Shoulder', 'Elbow', 'Wrist1', 'Wrist2', 'Wrist3']
joint_limits = [
    (-180, 180),  # Base
    (-90, 90),    # Shoulder  
    (-150, 150),  # Elbow
    (-180, 180),  # Wrist1
    (-90, 90),    # Wrist2
    (-180, 180)   # Wrist3
]

# Define trajectory waypoints (degrees)
time_points = [0, 2, 4, 6, 8, 10]
joint_waypoints = {
    'Base': [0, 45, 90, 45, -30, 0],
    'Shoulder': [0, 30, -45, 60, 15, 0],
    'Elbow': [0, -60, 90, -30, 45, 0],
    'Wrist1': [0, 90, -90, 0, 135, 0],
    'Wrist2': [0, -30, 45, -60, 30, 0],
    'Wrist3': [0, 180, -180, 90, -90, 0]
}

# Create splines for each joint  
joint_splines = {}
for joint, waypoints in joint_waypoints.items():
    joint_splines[joint] = CubicSpline(
        time_points,
        np.radians(waypoints),  # Convert to radians
        v0=0.0, vn=0.0         # Zero velocity at start/end
    )

# Evaluate trajectories
t_eval = np.linspace(0, 10, 300)
joint_trajectories = {}

for joint, spline in joint_splines.items():
    joint_trajectories[joint] = {
        'position': np.degrees(spline.evaluate(t_eval)),
        'velocity': np.degrees(spline.evaluate_velocity(t_eval)),
        'acceleration': np.degrees(spline.evaluate_acceleration(t_eval))
    }

# Plot results
fig, axes = plt.subplots(6, 3, figsize=(18, 20))

for i, joint in enumerate(joint_names):
    # Position
    axes[i, 0].plot(t_eval, joint_trajectories[joint]['position'], 'b-', linewidth=2)
    axes[i, 0].scatter(time_points, joint_waypoints[joint], color='red', s=40, zorder=5)
    axes[i, 0].axhline(y=joint_limits[i][0], color='r', linestyle='--', alpha=0.7)
    axes[i, 0].axhline(y=joint_limits[i][1], color='r', linestyle='--', alpha=0.7)
    axes[i, 0].set_ylabel(f'{joint}\\nPosition (°)')
    axes[i, 0].grid(True)
    
    # Velocity
    axes[i, 1].plot(t_eval, joint_trajectories[joint]['velocity'], 'g-', linewidth=2)
    axes[i, 1].set_ylabel(f'{joint}\\nVelocity (°/s)')
    axes[i, 1].grid(True)
    
    # Acceleration
    axes[i, 2].plot(t_eval, joint_trajectories[joint]['acceleration'], 'm-', linewidth=2)
    axes[i, 2].set_ylabel(f'{joint}\\nAcceleration (°/s²)')
    axes[i, 2].grid(True)

# Labels for bottom row
axes[5, 0].set_xlabel('Time (s)')
axes[5, 1].set_xlabel('Time (s)')
axes[5, 2].set_xlabel('Time (s)')

# Titles for top row
axes[0, 0].set_title('Joint Positions')
axes[0, 1].set_title('Joint Velocities')
axes[0, 2].set_title('Joint Accelerations')

plt.tight_layout()
plt.show()

# Check for joint limit violations
print("Joint Limit Analysis:")
for i, joint in enumerate(joint_names):
    positions = joint_trajectories[joint]['position']
    min_pos, max_pos = np.min(positions), np.max(positions)
    limit_min, limit_max = joint_limits[i]
    
    violation = min_pos < limit_min or max_pos > limit_max
    status = "❌ VIOLATION" if violation else "✅ OK"
    
    print(f"{joint:>10}: [{min_pos:6.1f}°, {max_pos:6.1f}°] "
          f"(limits: [{limit_min:4.0f}°, {limit_max:4.0f}°]) {status}")
```

### Smoothing Noisy Sensor Data

```python
# Simulate noisy sensor data from a robot trajectory
np.random.seed(123)

# Generate true trajectory
t_true = np.linspace(0, 20, 200)
q_true = 2 * np.sin(0.5 * t_true) + np.sin(2 * t_true) + 0.1 * t_true

# Add realistic sensor noise
measurement_times = np.linspace(0, 20, 80)  # Sparse measurements
sensor_noise = 0.15 * np.random.randn(len(measurement_times))
q_measured = np.interp(measurement_times, t_true, q_true) + sensor_noise

# Apply different smoothing approaches
smoothing_methods = {
    'No Smoothing': CubicSpline(measurement_times.tolist(), q_measured.tolist()),
    'Light Smoothing': CubicSmoothingSpline(measurement_times.tolist(), q_measured.tolist(), mu=0.01),
    'Medium Smoothing': CubicSmoothingSpline(measurement_times.tolist(), q_measured.tolist(), mu=0.1),
    'Auto Smoothing': smoothing_spline_with_tolerance(measurement_times.tolist(), q_measured.tolist(), tolerance=0.1)
}

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

t_eval = np.linspace(0, 20, 400)

for i, (method_name, spline) in enumerate(smoothing_methods.items()):
    q_smooth = spline.evaluate(t_eval)
    
    axes[i].plot(t_true, q_true, 'g-', linewidth=2, alpha=0.8, label='True trajectory')
    axes[i].scatter(measurement_times, q_measured, color='red', alpha=0.6, s=20, label='Noisy measurements')
    axes[i].plot(t_eval, q_smooth, 'b-', linewidth=2, label='Smoothed')
    
    # Calculate RMS error
    q_true_interp = np.interp(t_eval, t_true, q_true)
    rms_error = np.sqrt(np.mean((q_smooth - q_true_interp)**2))
    
    axes[i].set_title(f'{method_name}\\nRMS Error: {rms_error:.3f}')
    axes[i].legend()
    axes[i].grid(True)
    axes[i].set_xlabel('Time (s)')
    axes[i].set_ylabel('Position')

plt.tight_layout()
plt.show()

# Print smoothing parameters
print("Smoothing Parameters:")
for method_name, spline in smoothing_methods.items():
    if hasattr(spline, 'mu'):
        print(f"{method_name:>15}: μ = {spline.mu:.6f}")
    else:
        print(f"{method_name:>15}: No smoothing (exact interpolation)")
```

## Performance Considerations

### Evaluation Speed Comparison

```python
import time
import numpy as np

# Test different spline types with large datasets
n_waypoints = 1000
n_evaluations = 10000

# Generate test data
t_waypoints = np.linspace(0, 100, n_waypoints)
q_waypoints = np.sin(0.1 * t_waypoints) + 0.1 * np.random.randn(n_waypoints)
t_eval = np.linspace(0, 100, n_evaluations)

# Test algorithms
algorithms = {
    'CubicSpline': CubicSpline(t_waypoints.tolist(), q_waypoints.tolist()),
    'SmoothingSpline': CubicSmoothingSpline(t_waypoints.tolist(), q_waypoints.tolist(), mu=0.01)
}

print(f"Performance Test: {n_waypoints} waypoints, {n_evaluations} evaluations")
print("-" * 60)

for name, algorithm in algorithms.items():
    # Time setup (already done above)
    
    # Time vectorized evaluation
    start_time = time.time()
    result_vectorized = algorithm.evaluate(t_eval)
    vectorized_time = time.time() - start_time
    
    # Time scalar evaluation (smaller sample)
    t_eval_small = t_eval[::100]  # Every 100th point
    start_time = time.time()
    result_scalar = [algorithm.evaluate(t) for t in t_eval_small]
    scalar_time = time.time() - start_time
    scalar_time_scaled = scalar_time * 100  # Scale to full size
    
    print(f"{name:>15}: Vectorized: {vectorized_time:.3f}s, "
          f"Scalar: {scalar_time_scaled:.3f}s, "
          f"Speedup: {scalar_time_scaled/vectorized_time:.1f}x")

print("\\n💡 Always use vectorized evaluation for better performance!")
```

## Common Pitfalls and Solutions

### 1. Non-Monotonic Time Points

```python
# Common mistake: unsorted time points
try:
    t_bad = [0, 2, 1, 3, 4]  # Not monotonic!
    q_points = [0, 1, 2, 3, 4]
    spline = CubicSpline(t_bad, q_points)
except ValueError as e:
    print(f"❌ Error: {e}")

# Solution: Sort the data
t_unsorted = [0, 2, 1, 3, 4]
q_unsorted = [0, 1, 2, 3, 4]

# Sort by time
sorted_indices = np.argsort(t_unsorted)
t_sorted = [t_unsorted[i] for i in sorted_indices]
q_sorted = [q_unsorted[i] for i in sorted_indices]

spline = CubicSpline(t_sorted, q_sorted)
print("✅ Fixed: Data sorted by time")
```

### 2. Duplicate Time Points

```python
# Common mistake: duplicate time values
try:
    t_duplicate = [0, 1, 1, 2, 3]  # Duplicate at t=1
    q_points = [0, 1, 1.5, 2, 3]
    spline = CubicSpline(t_duplicate, q_points)
except ValueError as e:
    print(f"❌ Error: {e}")

# Solution: Remove duplicates or add small offset
def remove_duplicates(t_points, q_points, min_spacing=1e-6):
    """Remove duplicate time points or add small spacing."""
    t_clean, q_clean = [], []
    
    for i, (t, q) in enumerate(zip(t_points, q_points)):
        if i == 0 or t > t_clean[-1] + min_spacing:
            t_clean.append(t)
            q_clean.append(q)
        else:
            # Add small offset to avoid duplicate
            t_clean.append(t_clean[-1] + min_spacing)
            q_clean.append(q)
    
    return t_clean, q_clean

t_clean, q_clean = remove_duplicates(t_duplicate, q_points)
spline = CubicSpline(t_clean, q_clean)
print("✅ Fixed: Duplicates handled")
```

### 3. Choosing Smoothing Parameters

```python
def analyze_smoothing_effect(t_points, q_noisy, mu_values):
    """Analyze the effect of different smoothing parameters."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    metrics = {'mu': [], 'max_deviation': [], 'curvature': [], 'rms_error': []}
    
    for mu in mu_values:
        spline = CubicSmoothingSpline(t_points, q_noisy, mu=mu)
        
        # Evaluate
        t_eval = np.linspace(min(t_points), max(t_points), 200)
        q_smooth = spline.evaluate(t_eval)
        
        # Calculate metrics
        deviations = [abs(spline.evaluate(t) - q) for t, q in zip(t_points, q_noisy)]
        max_deviation = max(deviations)
        
        # Curvature (second derivative)
        acc = spline.evaluate_acceleration(t_eval)
        avg_curvature = np.mean(np.abs(acc))
        
        metrics['mu'].append(mu)
        metrics['max_deviation'].append(max_deviation)
        metrics['curvature'].append(avg_curvature)
    
    # Plot trade-off curves
    axes[0, 0].loglog(metrics['mu'], metrics['max_deviation'], 'o-')
    axes[0, 0].set_xlabel('Smoothing Parameter μ')
    axes[0, 0].set_ylabel('Maximum Deviation')
    axes[0, 0].set_title('Fidelity vs Smoothing')
    axes[0, 0].grid(True)
    
    axes[0, 1].loglog(metrics['mu'], metrics['curvature'], 's-', color='orange')
    axes[0, 1].set_xlabel('Smoothing Parameter μ')
    axes[0, 1].set_ylabel('Average Curvature')
    axes[0, 1].set_title('Curvature vs Smoothing')
    axes[0, 1].grid(True)
    
    # L-curve (deviation vs curvature)
    axes[1, 0].loglog(metrics['max_deviation'], metrics['curvature'], '^-', color='red')
    axes[1, 0].set_xlabel('Maximum Deviation')
    axes[1, 0].set_ylabel('Average Curvature')
    axes[1, 0].set_title('L-Curve (Optimal at Corner)')
    axes[1, 0].grid(True)
    
    # Show optimal region
    optimal_idx = len(mu_values) // 2  # Rough estimate
    axes[1, 0].scatter(metrics['max_deviation'][optimal_idx], 
                      metrics['curvature'][optimal_idx], 
                      color='green', s=100, zorder=5, label='Optimal region')
    axes[1, 0].legend()
    
    plt.tight_layout()
    plt.show()
    
    return metrics

# Test with noisy data
np.random.seed(42)
t_test = np.linspace(0, 10, 20)
q_test = np.sin(t_test) + 0.3 * np.random.randn(len(t_test))
mu_test = np.logspace(-4, 1, 20)

analyze_smoothing_effect(t_test.tolist(), q_test.tolist(), mu_test)
```

## Summary

This tutorial covered:

✅ **Basic cubic splines** with boundary conditions  
✅ **Smoothing splines** for noisy data  
✅ **Advanced splines** with acceleration constraints  
✅ **B-spline interpolation** for flexible curve design  
✅ **Practical applications** in robotics  
✅ **Performance optimization** techniques  
✅ **Common pitfalls** and solutions  

### Key Takeaways

1. **Choose the right algorithm**:
   - Clean data → `CubicSpline`
   - Noisy data → `CubicSmoothingSpline`
   - Need acceleration control → `CubicSplineWithAcceleration1/2`
   - Flexible curves → `BSplineInterpolator`

2. **Boundary conditions matter**: Zero velocities create natural-looking curves

3. **Always use vectorized evaluation** for performance

4. **Validate your data**: Check for monotonic time sequences and duplicates

5. **Smoothing is a trade-off**: Balance fidelity vs smoothness based on your application

### Next Steps

- **[Motion Profiles Tutorial](motion-profiles.md)**: Learn about bounded-derivative trajectories
- **[Quaternion Tutorial](../api-reference.md#quaternion-interpolation)**: Master 3D rotation interpolation  
- **[API Reference](../api-reference.md)**: Complete function documentation