# User Guide

Welcome to the comprehensive InterpolatePy user guide! This guide covers everything you need to know to effectively use InterpolatePy for trajectory planning and interpolation in your projects.

## Overview

InterpolatePy is designed around four core principles:

1. **Smoothness**: All algorithms provide mathematically guaranteed continuity
2. **Performance**: Vectorized operations optimized for real-time applications
3. **Flexibility**: Extensive boundary condition and constraint support
4. **Consistency**: Uniform APIs across all algorithm families

## Algorithm Categories

### 🔵 Spline Interpolation

Splines create smooth curves through waypoints with guaranteed continuity properties.

#### When to Use Splines
- **Waypoint interpolation**: You have specific points the trajectory must pass through
- **Curve fitting**: Need smooth curves through noisy or sparse data
- **C² continuity**: Require continuous acceleration (important for robotics)

#### Key Algorithms
- **[CubicSpline](api-reference.md#cubic-spline)**: Natural cubic splines with boundary conditions
- **[CubicSmoothingSpline](api-reference.md#cubic-smoothing-spline)**: Noise-robust with smoothing parameter
- **[BSplineInterpolator](api-reference.md#b-spline-interpolator)**: Flexible degree and local control

#### Example: Multi-Axis Robot Trajectory
```python
import numpy as np
import matplotlib.pyplot as plt
from interpolatepy import CubicSpline

# Define joint waypoints for 3-DOF robot
time_points = [0, 2, 4, 6, 8]
joint_trajectories = {}

# Joint angles at each waypoint
waypoints = {
    'joint1': [0, 45, 90, 45, 0],      # degrees
    'joint2': [0, -30, 60, -30, 0],
    'joint3': [0, 20, -45, 20, 0]
}

# Create splines for each joint
for joint, angles in waypoints.items():
    joint_trajectories[joint] = CubicSpline(
        time_points, 
        np.radians(angles),  # Convert to radians
        v0=0.0, vn=0.0      # Zero velocity at endpoints
    )

# Evaluate synchronized trajectory
t_eval = np.linspace(0, 8, 200)
robot_trajectory = {}

for joint, spline in joint_trajectories.items():
    robot_trajectory[joint] = {
        'position': spline.evaluate(t_eval),
        'velocity': spline.evaluate_velocity(t_eval),
        'acceleration': spline.evaluate_acceleration(t_eval)
    }

# Plot multi-joint trajectory
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
for i, joint in enumerate(['joint1', 'joint2', 'joint3']):
    axes[i, 0].plot(t_eval, np.degrees(robot_trajectory[joint]['position']))
    axes[i, 0].set_title(f'{joint} Position')
    axes[i, 0].set_ylabel('Angle (deg)')
    
    axes[i, 1].plot(t_eval, np.degrees(robot_trajectory[joint]['velocity']))
    axes[i, 1].set_title(f'{joint} Velocity') 
    axes[i, 1].set_ylabel('Angular Velocity (deg/s)')
    
    axes[i, 2].plot(t_eval, np.degrees(robot_trajectory[joint]['acceleration']))
    axes[i, 2].set_title(f'{joint} Acceleration')
    axes[i, 2].set_ylabel('Angular Acceleration (deg/s²)')

for ax in axes.flat:
    ax.grid(True)
    
plt.tight_layout()
plt.show()
```

### ⚡ Motion Profiles

Motion profiles generate time-optimal trajectories with bounded derivatives.

#### When to Use Motion Profiles
- **Industrial automation**: CNC machines, conveyor systems
- **Robotics**: Point-to-point motion with velocity/acceleration limits
- **Smooth acceleration**: Minimize jerk for passenger comfort or mechanical stress

#### Key Algorithms
- **[DoubleSTrajectory](api-reference.md#double-s-trajectory)**: Jerk-limited S-curve profiles
- **[TrapezoidalTrajectory](api-reference.md#trapezoidal-trajectory)**: Classic acceleration-limited profiles
- **[PolynomialTrajectory](api-reference.md#polynomial-trajectory)**: Exact boundary condition matching

#### Example: Elevator Motion Planning
```python
from interpolatepy import DoubleSTrajectory, StateParams, TrajectoryBounds
import numpy as np
import matplotlib.pyplot as plt

def plan_elevator_motion(floors, floor_height=3.0):
    """Plan smooth elevator motion between floors."""
    trajectories = []
    
    for i in range(len(floors) - 1):
        start_floor = floors[i]
        end_floor = floors[i + 1]
        
        # State parameters
        state = StateParams(
            q_0=start_floor * floor_height,    # Start position (m)
            q_1=end_floor * floor_height,      # End position (m)
            v_0=0.0,                           # Start at rest
            v_1=0.0                            # End at rest
        )
        
        # Elevator constraints (comfortable for passengers)
        bounds = TrajectoryBounds(
            v_bound=2.0,    # Max velocity: 2 m/s
            a_bound=1.0,    # Max acceleration: 1 m/s² (0.1g)
            j_bound=2.0     # Max jerk: 2 m/s³
        )
        
        trajectory = DoubleSTrajectory(state, bounds)
        trajectories.append({
            'trajectory': trajectory,
            'start_floor': start_floor,
            'end_floor': end_floor,
            'duration': trajectory.get_duration()
        })
    
    return trajectories

# Plan elevator path: Ground (0) → 5th floor → 2nd floor → 8th floor
floors = [0, 5, 2, 8]
elevator_plan = plan_elevator_motion(floors)

# Visualize complete journey
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
current_time = 0

for i, segment in enumerate(elevator_plan):
    traj = segment['trajectory']
    duration = segment['duration']
    
    # Time array for this segment
    t_segment = np.linspace(0, duration, 100)
    t_absolute = t_segment + current_time
    
    # Evaluate trajectory
    results = [traj.evaluate(t) for t in t_segment]
    positions = [r[0] for r in results]
    velocities = [r[1] for r in results]
    accelerations = [r[2] for r in results]
    
    # Plot
    label = f"Floor {segment['start_floor']} → {segment['end_floor']}"
    axes[0].plot(t_absolute, positions, label=label, linewidth=2)
    axes[1].plot(t_absolute, velocities, label=label, linewidth=2)
    axes[2].plot(t_absolute, accelerations, label=label, linewidth=2)
    
    current_time += duration

# Formatting
axes[0].set_ylabel('Position (m)')
axes[0].set_title('Elevator Position vs Time')
axes[0].legend()
axes[0].grid(True)

axes[1].set_ylabel('Velocity (m/s)')
axes[1].set_title('Elevator Velocity vs Time')
axes[1].legend()
axes[1].grid(True)

axes[2].set_ylabel('Acceleration (m/s²)')
axes[2].set_xlabel('Time (s)')
axes[2].set_title('Elevator Acceleration vs Time')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()

# Print journey summary
total_time = sum(seg['duration'] for seg in elevator_plan)
print(f"Total journey time: {total_time:.1f} seconds")
for i, seg in enumerate(elevator_plan):
    print(f"Segment {i+1}: Floor {seg['start_floor']} → {seg['end_floor']} ({seg['duration']:.1f}s)")
```

### 🔄 Quaternion Interpolation

Quaternions provide singularity-free rotation interpolation for 3D orientations.

#### When to Use Quaternions
- **3D orientation control**: Robot end-effectors, camera orientation
- **Animation**: Smooth rotation keyframes without gimbal lock
- **Aerospace**: Attitude control for satellites and aircraft

#### Key Algorithms
- **[Quaternion](api-reference.md#quaternion)**: Core operations with SLERP
- **[SquadC2](api-reference.md#squad-c2)**: C²-continuous rotation trajectories
- **[LogQuaternionInterpolation](api-reference.md#log-quaternion-interpolation)**: Advanced B-spline methods

#### Example: Drone Camera Gimbal Control
```python
from interpolatepy import SquadC2, Quaternion
import numpy as np
import matplotlib.pyplot as plt

def plan_camera_sweep(waypoints, times):
    """Plan smooth camera orientation for aerial cinematography."""
    
    # Convert waypoints to quaternions
    orientations = []
    for pitch, yaw, roll in waypoints:
        # Create quaternion from Euler angles (ZYX convention)
        q = Quaternion.from_euler_angles(roll, pitch, yaw)
        orientations.append(q)
    
    # Create smooth quaternion trajectory
    gimbal_trajectory = SquadC2(times, orientations)
    
    return gimbal_trajectory

# Define camera waypoints for cinematic sweep
# Format: (pitch, yaw, roll) in radians
camera_waypoints = [
    (0, 0, 0),                    # Level forward
    (-np.pi/6, np.pi/4, 0),      # Look down-right
    (-np.pi/3, np.pi/2, np.pi/8), # Look down-right with tilt
    (0, np.pi, 0),                # Look backward
    (np.pi/6, 3*np.pi/2, 0),     # Look up-left
    (0, 2*np.pi, 0)               # Return to forward
]

waypoint_times = [0, 2, 4, 6, 8, 10]

# Plan gimbal trajectory
gimbal = plan_camera_sweep(camera_waypoints, waypoint_times)

# Evaluate trajectory
t_eval = np.linspace(0, 10, 200)
orientations = [gimbal.evaluate(t) for t in t_eval]
angular_velocities = [gimbal.evaluate_velocity(t) for t in t_eval]

# Convert back to Euler angles for visualization
euler_angles = []
for q in orientations:
    roll, pitch, yaw = q.to_euler_angles()
    euler_angles.append([roll, pitch, yaw])

euler_angles = np.array(euler_angles)

# Plot orientation trajectory
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Euler angles
for i, label in enumerate(['Roll', 'Pitch', 'Yaw']):
    axes[0, 0].plot(t_eval, np.degrees(euler_angles[:, i]), label=label, linewidth=2)

axes[0, 0].set_title('Camera Orientation (Euler Angles)')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Angle (degrees)')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Angular velocity magnitude
omega_magnitude = [np.linalg.norm(omega) for omega in angular_velocities]
axes[0, 1].plot(t_eval, np.degrees(omega_magnitude), 'r-', linewidth=2)
axes[0, 1].set_title('Angular Velocity Magnitude')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Angular Speed (deg/s)')
axes[0, 1].grid(True)

# 3D trajectory visualization (quaternion components)
quaternion_components = np.array([[q.w, q.x, q.y, q.z] for q in orientations])
for i, label in enumerate(['w', 'x', 'y', 'z']):
    axes[1, 0].plot(t_eval, quaternion_components[:, i], label=f'q_{label}', linewidth=2)

axes[1, 0].set_title('Quaternion Components')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Component Value')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Angular velocity components
omega_components = np.array(angular_velocities)
for i, label in enumerate(['ωx', 'ωy', 'ωz']):
    axes[1, 1].plot(t_eval, np.degrees(omega_components[:, i]), label=label, linewidth=2)

axes[1, 1].set_title('Angular Velocity Components')
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Angular Velocity (deg/s)')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# Print trajectory analysis
max_angular_speed = max(omega_magnitude)
print(f"Maximum angular speed: {np.degrees(max_angular_speed):.1f} deg/s")
print(f"Total trajectory duration: {waypoint_times[-1]:.1f} seconds")
```

### 🎯 Path Planning

Geometric path primitives and coordinate frame computation for spatial trajectories.

#### When to Use Path Planning
- **Robotic path following**: Mobile robots, manipulator end-effector paths
- **CNC machining**: Tool path generation with proper orientation
- **Animation**: Camera paths and object motion along curves

#### Key Algorithms
- **[LinearPath](api-reference.md#linear-path)**, **[CircularPath](api-reference.md#circular-path)**: Geometric primitives
- **[FrenetFrame](api-reference.md#frenet-frame)**: Coordinate frames along curves
- **[ParabolicBlendTrajectory](api-reference.md#parabolic-blend-trajectory)**: Via-point trajectories

#### Example: CNC Tool Path with Frenet Frames
```python
from interpolatepy import compute_trajectory_frames, CubicSpline
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_machining_path(waypoints, tool_orientation=(0, 0, 0)):
    """Create 3D tool path with proper orientation frames."""
    
    # Extract coordinates
    t_points = [p[0] for p in waypoints]  # Parameter values
    x_points = [p[1] for p in waypoints]  # X coordinates
    y_points = [p[2] for p in waypoints]  # Y coordinates  
    z_points = [p[3] for p in waypoints]  # Z coordinates
    
    # Create splines for each coordinate
    x_spline = CubicSpline(t_points, x_points, v0=0, vn=0)
    y_spline = CubicSpline(t_points, y_points, v0=0, vn=0)
    z_spline = CubicSpline(t_points, z_points, v0=0, vn=0)
    
    def path_function(u):
        """Combined 3D path function with derivatives."""
        position = np.array([
            x_spline.evaluate(u),
            y_spline.evaluate(u), 
            z_spline.evaluate(u)
        ])
        
        first_derivative = np.array([
            x_spline.evaluate_velocity(u),
            y_spline.evaluate_velocity(u),
            z_spline.evaluate_velocity(u)
        ])
        
        second_derivative = np.array([
            x_spline.evaluate_acceleration(u),
            y_spline.evaluate_acceleration(u),
            z_spline.evaluate_acceleration(u)
        ])
        
        return position, first_derivative, second_derivative
    
    return path_function

# Define 3D machining waypoints (parameter, x, y, z)
waypoints = [
    (0.0, 0, 0, 0),      # Start
    (1.0, 2, 1, 0.5),    # Rise
    (2.0, 4, 3, 1.0),    # Peak
    (3.0, 6, 2, 0.8),    # Descent
    (4.0, 8, 0, 0.2),    # Valley
    (5.0, 10, 1, 0.0)    # End
]

# Create path function
path_func = create_machining_path(waypoints)

# Compute Frenet frames along path
u_values = np.linspace(0, 5, 50)
points, frames = compute_trajectory_frames(
    path_func, 
    u_values,
    tool_orientation=(0.1, -0.2, 0)  # Tool tilt (roll, pitch, yaw)
)

# Visualization
fig = plt.figure(figsize=(15, 10))

# 3D path with frames
ax1 = fig.add_subplot(221, projection='3d')

# Plot path
ax1.plot(points[:, 0], points[:, 1], points[:, 2], 'b-', linewidth=3, label='Tool path')

# Plot Frenet frames (every 5th frame for clarity)
skip = 5
for i in range(0, len(points), skip):
    origin = points[i]
    tangent = frames[i, 0] * 0.3    # Red: tangent (tool direction)
    normal = frames[i, 1] * 0.3     # Green: normal
    binormal = frames[i, 2] * 0.3   # Blue: binormal
    
    # Draw frame vectors
    ax1.quiver(origin[0], origin[1], origin[2], 
              tangent[0], tangent[1], tangent[2], 
              color='red', arrow_length_ratio=0.1)
    ax1.quiver(origin[0], origin[1], origin[2],
              normal[0], normal[1], normal[2],
              color='green', arrow_length_ratio=0.1)
    ax1.quiver(origin[0], origin[1], origin[2],
              binormal[0], binormal[1], binormal[2], 
              color='blue', arrow_length_ratio=0.1)

# Mark waypoints
waypoint_coords = np.array([[p[1], p[2], p[3]] for p in waypoints])
ax1.scatter(waypoint_coords[:, 0], waypoint_coords[:, 1], waypoint_coords[:, 2],
           c='red', s=100, alpha=0.8, label='Waypoints')

ax1.set_xlabel('X (mm)')
ax1.set_ylabel('Y (mm)')
ax1.set_zlabel('Z (mm)')
ax1.set_title('3D Tool Path with Frenet Frames')
ax1.legend()

# Path curvature analysis
ax2 = fig.add_subplot(222)
curvatures = []
for i in range(len(u_values)):
    # Compute curvature from frame vectors
    tangent = frames[i, 0]
    normal = frames[i, 1]
    
    # Curvature magnitude (simplified)
    if i < len(u_values) - 1:
        dt = u_values[i+1] - u_values[i]
        tangent_next = frames[i+1, 0] if i+1 < len(frames) else tangent
        dtangent_dt = (tangent_next - tangent) / dt
        curvature = np.linalg.norm(dtangent_dt)
    else:
        curvature = curvatures[-1] if curvatures else 0
    
    curvatures.append(curvature)

ax2.plot(u_values, curvatures, 'g-', linewidth=2)
ax2.set_xlabel('Path Parameter u')
ax2.set_ylabel('Curvature')
ax2.set_title('Path Curvature Analysis')
ax2.grid(True)

# Velocity profile along path
ax3 = fig.add_subplot(223)
path_velocities = []
for u in u_values:
    _, velocity, _ = path_func(u)
    speed = np.linalg.norm(velocity)
    path_velocities.append(speed)

ax3.plot(u_values, path_velocities, 'r-', linewidth=2)
ax3.set_xlabel('Path Parameter u')
ax3.set_ylabel('Path Velocity Magnitude')
ax3.set_title('Velocity Profile')
ax3.grid(True)

# Tool orientation angles
ax4 = fig.add_subplot(224)
roll_angles = []
pitch_angles = []
yaw_angles = []

for i in range(len(frames)):
    # Extract orientation from frame (simplified)
    tangent = frames[i, 0]
    normal = frames[i, 1]
    binormal = frames[i, 2]
    
    # Compute Euler angles from rotation matrix
    R = np.column_stack([tangent, normal, binormal])
    
    # Extract roll, pitch, yaw (ZYX convention)
    pitch = np.arcsin(-R[2, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    
    roll_angles.append(roll)
    pitch_angles.append(pitch)
    yaw_angles.append(yaw)

ax4.plot(u_values, np.degrees(roll_angles), label='Roll', linewidth=2)
ax4.plot(u_values, np.degrees(pitch_angles), label='Pitch', linewidth=2)  
ax4.plot(u_values, np.degrees(yaw_angles), label='Yaw', linewidth=2)
ax4.set_xlabel('Path Parameter u')
ax4.set_ylabel('Orientation (degrees)')
ax4.set_title('Tool Orientation Along Path')
ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.show()

# Analysis summary
total_path_length = np.sum([np.linalg.norm(points[i+1] - points[i]) 
                           for i in range(len(points)-1)])
max_curvature = max(curvatures)
max_speed = max(path_velocities)

print(f"Path Analysis:")
print(f"  Total path length: {total_path_length:.2f} mm")
print(f"  Maximum curvature: {max_curvature:.4f}")
print(f"  Maximum speed: {max_speed:.2f} units/s")
print(f"  Orientation range: Roll±{np.degrees(max(roll_angles)-min(roll_angles)):.1f}°, "
      f"Pitch±{np.degrees(max(pitch_angles)-min(pitch_angles)):.1f}°, "
      f"Yaw±{np.degrees(max(yaw_angles)-min(yaw_angles)):.1f}°")
```

## Advanced Topics

### Boundary Condition Handling

Different algorithms support various boundary conditions:

```python
from interpolatepy import CubicSpline

# Natural boundaries (zero second derivative)
spline1 = CubicSpline(t_points, q_points)  # Default: v0=0, vn=0

# Velocity boundaries
spline2 = CubicSpline(t_points, q_points, v0=2.0, vn=-1.0)

# Acceleration boundaries (requires special algorithms)
from interpolatepy import CubicSplineWithAcceleration1
spline3 = CubicSplineWithAcceleration1(
    t_points, q_points,
    v0=0.0, vn=0.0,
    a0=1.0, an=-0.5
)

# Complete boundary specification for polynomials
from interpolatepy import PolynomialTrajectory, BoundaryCondition, TimeInterval
initial = BoundaryCondition(position=0, velocity=0, acceleration=0, jerk=0)
final = BoundaryCondition(position=1, velocity=0, acceleration=0, jerk=0)
poly_traj = PolynomialTrajectory.order_7_trajectory(initial, final, TimeInterval(0, 1))
```

### Performance Optimization

#### Vectorized Evaluation
```python
import numpy as np

# Efficient: single vectorized call (assuming spline was created)
t_array = np.linspace(0, 10, 1000)
positions = spline.evaluate(t_array)

# Inefficient: multiple scalar calls
positions = [spline.evaluate(t) for t in t_array]
```

#### Algorithm Selection by Complexity

| Algorithm | Setup | Evaluation | Memory | Best For |
|-----------|-------|------------|---------|----------|
| Linear | O(1) | O(1) | O(1) | Simple interpolation |
| CubicSpline | O(n) | O(log n) | O(n) | Smooth waypoints |
| DoubleSTrajectory | O(1) | O(1) | O(1) | Motion profiles |
| BSpline | O(n²) | O(p) | O(n) | High-degree curves |
| Quaternion | O(1) | O(1) | O(1) | 3D rotations |

### Error Handling and Validation

```python
from interpolatepy import CubicSpline

try:
    # Potentially problematic input
    spline = CubicSpline([0, 1, 1, 2], [0, 1, 2, 3])  # Non-monotonic times
except ValueError as e:
    print(f"Input validation failed: {e}")

try:
    # Evaluation outside domain
    result = spline.evaluate(100)  # Time beyond trajectory
except ValueError as e:
    print(f"Evaluation error: {e}")

# Safe evaluation with bounds checking
def safe_evaluate(trajectory, t, t_min=None, t_max=None):
    """Safely evaluate trajectory with bounds checking."""
    if t_min is not None and t < t_min:
        return trajectory.evaluate(t_min)
    if t_max is not None and t > t_max:
        return trajectory.evaluate(t_max)
    return trajectory.evaluate(t)
```

## Best Practices

### 1. Choose the Right Algorithm

```python
# For waypoint interpolation with noise
if data_has_noise:
    spline = CubicSmoothingSpline(t_points, q_points, mu=smoothing_param)
else:
    spline = CubicSpline(t_points, q_points)

# For bounded motion
if need_velocity_limits:
    trajectory = DoubleSTrajectory(state_params, trajectory_bounds)
else:
    trajectory = PolynomialTrajectory.order_5_trajectory(initial, final, interval)

# For 3D rotations  
if working_with_rotations:
    quat_traj = SquadC2(times, quaternions)
else:
    coord_traj = CubicSpline(times, coordinates)
```

### 2. Validate Your Data

```python
def validate_trajectory_data(t_points, q_points):
    """Validate trajectory input data."""
    if len(t_points) != len(q_points):
        raise ValueError("Time and position arrays must have same length")
    
    if len(t_points) < 2:
        raise ValueError("Need at least 2 points for trajectory")
    
    if not all(t_points[i] < t_points[i+1] for i in range(len(t_points)-1)):
        raise ValueError("Time points must be strictly increasing")
    
    if any(not np.isfinite(q) for q in q_points):
        raise ValueError("Position points must be finite")
    
    return True
```

### 3. Handle Edge Cases

```python
def robust_trajectory_evaluation(trajectory, t_eval):
    """Robust trajectory evaluation with error handling."""
    results = []
    
    for t in np.atleast_1d(t_eval):
        try:
            # Clamp to trajectory domain
            t_clamped = np.clip(t, trajectory.t_points[0], trajectory.t_points[-1])
            result = trajectory.evaluate(t_clamped)
            results.append(result)
        except Exception as e:
            print(f"Warning: evaluation failed at t={t}: {e}")
            results.append(np.nan)
    
    return np.array(results)
```

## Next Steps

- **[Tutorials](tutorials/spline-interpolation.md)**: Dive deep into specific algorithm families
- **[API Reference](api-reference.md)**: Complete function documentation
- **[Examples](examples.md)**: Real-world applications and use cases
- **[Algorithms](algorithms.md)**: Mathematical theory and implementation details

## Getting Help

- **GitHub Issues**: [Report bugs and request features](https://github.com/GiorgioMedico/InterpolatePy/issues)
- **Discussions**: [Ask questions and share examples](https://github.com/GiorgioMedico/InterpolatePy/discussions)
- **Documentation**: Search this documentation for specific topics
- **Examples**: Check the `examples/` directory in the repository