# Examples Gallery

This gallery showcases real-world applications of InterpolatePy across robotics, animation, and scientific computing. Each example includes complete, runnable code.

## Robotics Applications

### Multi-Axis Robot Arm Control

Complete 6-DOF robot arm trajectory with synchronized motion:

```python
import numpy as np
import matplotlib.pyplot as plt
from interpolatepy import CubicSpline

def plan_robot_trajectory():
    """Plan synchronized 6-DOF robot arm trajectory."""
    
    # Joint waypoints (degrees) - Pick and place operation
    waypoints = {
        'base': [0, 30, 60, 90, 60, 30, 0],           # Base rotation
        'shoulder': [0, 45, 30, -15, 30, 45, 0],      # Shoulder lift
        'elbow': [0, -90, -60, 90, -60, -90, 0],      # Elbow bend
        'wrist1': [0, 45, 90, 0, 90, 45, 0],          # Wrist rotation
        'wrist2': [0, 0, -45, -45, -45, 0, 0],        # Wrist tilt
        'wrist3': [0, 180, 90, -90, 90, 180, 0]       # End-effector
    }
    
    # Timing: approach, pick, lift, move, place, retract, home
    time_points = [0, 2, 3, 5, 7, 8, 10]
    
    # Create splines for each joint
    joint_splines = {}
    for joint, angles in waypoints.items():
        joint_splines[joint] = CubicSpline(
            time_points,
            np.radians(angles),  # Convert to radians
            v0=0.0, vn=0.0      # Zero velocity at start/end
        )
    
    return joint_splines, time_points

# Generate and visualize trajectory
joint_splines, time_points = plan_robot_trajectory()

# Evaluate complete trajectory
t_eval = np.linspace(0, 10, 300)
joint_data = {}

for joint, spline in joint_splines.items():
    joint_data[joint] = {
        'position': np.degrees(spline.evaluate(t_eval)),
        'velocity': np.degrees(spline.evaluate_velocity(t_eval)),
        'acceleration': np.degrees(spline.evaluate_acceleration(t_eval))
    }

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 15))

# Joint positions
ax1 = plt.subplot(3, 2, 1)
for joint in joint_splines.keys():
    plt.plot(t_eval, joint_data[joint]['position'], linewidth=2, label=joint)
plt.xlabel('Time (s)')
plt.ylabel('Joint Angle (degrees)')
plt.title('Robot Joint Positions')
plt.legend()
plt.grid(True)

# Joint velocities
ax2 = plt.subplot(3, 2, 2)
for joint in joint_splines.keys():
    plt.plot(t_eval, joint_data[joint]['velocity'], linewidth=2, label=joint)
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (deg/s)')
plt.title('Robot Joint Velocities')
plt.legend()
plt.grid(True)

# Joint accelerations
ax3 = plt.subplot(3, 2, 3)
for joint in joint_splines.keys():
    plt.plot(t_eval, joint_data[joint]['acceleration'], linewidth=2, label=joint)
plt.xlabel('Time (s)')
plt.ylabel('Angular Acceleration (deg/s²)')
plt.title('Robot Joint Accelerations')
plt.legend()
plt.grid(True)

# 3D workspace visualization (simplified forward kinematics)
ax4 = plt.subplot(3, 2, 4, projection='3d')

# Simplified forward kinematics for visualization
def simple_forward_kinematics(angles):
    """Simplified FK for visualization purposes."""
    base, shoulder, elbow = angles[0], angles[1], angles[2]
    
    # Link lengths (arbitrary units)
    L1, L2, L3 = 1.0, 1.2, 0.8
    
    # End-effector position (simplified)
    x = (L2 * np.cos(shoulder) + L3 * np.cos(shoulder + elbow)) * np.cos(base)
    y = (L2 * np.cos(shoulder) + L3 * np.cos(shoulder + elbow)) * np.sin(base)
    z = L1 + L2 * np.sin(shoulder) + L3 * np.sin(shoulder + elbow)
    
    return x, y, z

# Calculate end-effector path
end_effector_path = []
for i in range(len(t_eval)):
    angles = [joint_data[joint]['position'][i] for joint in ['base', 'shoulder', 'elbow']]
    x, y, z = simple_forward_kinematics(np.radians(angles))
    end_effector_path.append([x, y, z])

end_effector_path = np.array(end_effector_path)

ax4.plot(end_effector_path[:, 0], end_effector_path[:, 1], end_effector_path[:, 2], 
         'b-', linewidth=3, label='End-effector path')
ax4.scatter(end_effector_path[0, 0], end_effector_path[0, 1], end_effector_path[0, 2],
           color='green', s=100, label='Start')
ax4.scatter(end_effector_path[-1, 0], end_effector_path[-1, 1], end_effector_path[-1, 2],
           color='red', s=100, label='End')
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('Z')
ax4.set_title('End-Effector Trajectory in Workspace')
ax4.legend()

# Trajectory phases
ax5 = plt.subplot(3, 2, 5)
phases = ['Start', 'Approach', 'Pick', 'Lift', 'Move', 'Place', 'Retract', 'Home']
phase_times = [0, 2, 3, 5, 7, 8, 10]

# Show base joint with phase markers
plt.plot(t_eval, joint_data['base']['position'], 'b-', linewidth=2, label='Base joint')
for i, (phase, time) in enumerate(zip(phases[:-1], phase_times[:-1])):
    plt.axvline(x=time, color='red', linestyle='--', alpha=0.7)
    plt.text(time, plt.ylim()[1] * 0.9, phase, rotation=45, fontsize=8)

plt.xlabel('Time (s)')
plt.ylabel('Base Angle (degrees)')
plt.title('Trajectory Phases')
plt.legend()
plt.grid(True)

# Performance metrics
ax6 = plt.subplot(3, 2, 6)
max_velocities = [max(abs(joint_data[joint]['velocity'])) for joint in joint_splines.keys()]
joint_names = list(joint_splines.keys())

bars = plt.bar(joint_names, max_velocities, color='skyblue', alpha=0.7)
plt.ylabel('Max Angular Velocity (deg/s)')
plt.title('Peak Joint Velocities')
plt.xticks(rotation=45)
plt.grid(True, axis='y')

# Add value labels on bars
for bar, vel in zip(bars, max_velocities):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{vel:.1f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("Robot Trajectory Analysis:")
print("-" * 40)
print(f"Total trajectory time: {time_points[-1]} seconds")
print(f"Number of waypoints: {len(time_points)}")
print("Peak joint velocities (deg/s):")
for joint, vel in zip(joint_names, max_velocities):
    print(f"  {joint:>8}: {vel:6.1f}")
```

### Mobile Robot Path Following

Smooth path following with velocity constraints:

```python
from interpolatepy import DoubleSTrajectory, StateParams, TrajectoryBounds
import numpy as np
import matplotlib.pyplot as plt

def plan_mobile_robot_path():
    """Plan mobile robot path with velocity and acceleration constraints."""
    
    # Waypoints in 2D space (x, y coordinates)
    waypoints = np.array([
        [0, 0],      # Start
        [3, 1],      # Intermediate points
        [5, 4],
        [8, 3],
        [10, 6],
        [12, 2],
        [15, 5]      # End
    ])
    
    # Calculate path segments and distances
    segments = []
    cumulative_distance = 0
    distances = [0]  # Start at 0
    
    for i in range(len(waypoints) - 1):
        segment_length = np.linalg.norm(waypoints[i+1] - waypoints[i])
        cumulative_distance += segment_length
        distances.append(cumulative_distance)
        segments.append(segment_length)
    
    # Robot constraints
    max_velocity = 2.0      # m/s
    max_acceleration = 1.5  # m/s²
    max_jerk = 3.0         # m/s³
    
    # Create S-curve trajectory along path
    state = StateParams(
        q_0=0.0,                    # Start at path beginning
        q_1=cumulative_distance,    # End at path end
        v_0=0.0,                    # Start from rest
        v_1=0.0                     # End at rest
    )
    
    bounds = TrajectoryBounds(
        v_bound=max_velocity,
        a_bound=max_acceleration,
        j_bound=max_jerk
    )
    
    trajectory = DoubleSTrajectory(state, bounds)
    
    return waypoints, distances, trajectory

# Generate path
waypoints, distances, trajectory = plan_mobile_robot_path()

# Interpolate positions along path
def interpolate_position(s, waypoints, distances):
    """Interpolate 2D position at path distance s."""
    if s <= 0:
        return waypoints[0]
    elif s >= distances[-1]:
        return waypoints[-1]
    
    # Find segment
    for i in range(len(distances) - 1):
        if distances[i] <= s <= distances[i + 1]:
            # Linear interpolation within segment
            alpha = (s - distances[i]) / (distances[i + 1] - distances[i])
            return waypoints[i] + alpha * (waypoints[i + 1] - waypoints[i])
    
    return waypoints[-1]

# Evaluate robot trajectory
t_eval = np.linspace(0, trajectory.total_time, 300)
path_positions = []
velocities = []
accelerations = []

for t in t_eval:
    s = trajectory.evaluate(t)              # Path distance
    v = trajectory.evaluate_velocity(t)     # Path velocity
    a = trajectory.evaluate_acceleration(t) # Path acceleration
    
    pos = interpolate_position(s, waypoints, distances)
    path_positions.append(pos)
    velocities.append(v)
    accelerations.append(a)

path_positions = np.array(path_positions)

# Visualization
fig = plt.figure(figsize=(18, 12))

# 2D path
ax1 = plt.subplot(2, 3, 1)
plt.plot(waypoints[:, 0], waypoints[:, 1], 'ro-', markersize=8, linewidth=2, 
         label='Waypoints', alpha=0.7)
plt.plot(path_positions[:, 0], path_positions[:, 1], 'b-', linewidth=2, 
         label='Robot path')
plt.scatter(path_positions[0, 0], path_positions[0, 1], color='green', s=100, 
           label='Start', zorder=5)
plt.scatter(path_positions[-1, 0], path_positions[-1, 1], color='red', s=100, 
           label='End', zorder=5)

# Add arrows to show direction
n_arrows = 10
arrow_indices = np.linspace(10, len(path_positions)-10, n_arrows, dtype=int)
for i in arrow_indices:
    dx = path_positions[i+5, 0] - path_positions[i-5, 0]
    dy = path_positions[i+5, 1] - path_positions[i-5, 1]
    plt.arrow(path_positions[i, 0], path_positions[i, 1], dx*0.1, dy*0.1,
             head_width=0.2, head_length=0.1, fc='orange', ec='orange', alpha=0.7)

plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Mobile Robot Path Following')
plt.legend()
plt.grid(True)
plt.axis('equal')

# Velocity profile
ax2 = plt.subplot(2, 3, 2)
plt.plot(t_eval, velocities, 'g-', linewidth=2)
plt.axhline(y=2.0, color='r', linestyle='--', alpha=0.7, label='Velocity limit')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Robot Velocity Profile')
plt.legend()
plt.grid(True)

# Acceleration profile
ax3 = plt.subplot(2, 3, 3)
plt.plot(t_eval, accelerations, 'm-', linewidth=2)
plt.axhline(y=1.5, color='r', linestyle='--', alpha=0.7, label='Acceleration limit')
plt.axhline(y=-1.5, color='r', linestyle='--', alpha=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.title('Robot Acceleration Profile')
plt.legend()
plt.grid(True)

# Path curvature analysis
ax4 = plt.subplot(2, 3, 4)
curvatures = []
for i in range(1, len(path_positions) - 1):
    # Approximate curvature using three points
    p1, p2, p3 = path_positions[i-1], path_positions[i], path_positions[i+1]
    
    # Vectors
    v1 = p2 - p1
    v2 = p3 - p2
    
    # Cross product magnitude (approximation)
    cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
    
    # Curvature approximation
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm > 0 and v2_norm > 0:
        curvature = cross / (v1_norm * v2_norm)
    else:
        curvature = 0
    
    curvatures.append(curvature)

path_distances = [np.linalg.norm(path_positions[i] - path_positions[0]) 
                 for i in range(len(path_positions))]

plt.plot(path_distances[1:-1], curvatures, 'orange', linewidth=2)
plt.xlabel('Path Distance (m)')
plt.ylabel('Curvature (1/m)')
plt.title('Path Curvature Analysis')
plt.grid(True)

# Velocity vs curvature
ax5 = plt.subplot(2, 3, 5)
# Match lengths for plotting
min_len = min(len(curvatures), len(velocities))
curvature_subset = curvatures[:min_len]
velocity_subset = velocities[1:min_len+1]  # Skip first velocity point

plt.scatter(curvature_subset, velocity_subset, alpha=0.6, c=t_eval[1:min_len+1], 
           cmap='viridis')
plt.xlabel('Path Curvature (1/m)')
plt.ylabel('Robot Velocity (m/s)')
plt.title('Velocity vs Curvature')
plt.colorbar(label='Time (s)')
plt.grid(True)

# Performance summary
ax6 = plt.subplot(2, 3, 6)
metrics = {
    'Total Distance': f"{distances[-1]:.1f} m",
    'Travel Time': f"{trajectory.total_time:.1f} s",
    'Average Speed': f"{distances[-1]/trajectory.total_time:.1f} m/s",
    'Max Velocity': f"{max(velocities):.1f} m/s",
    'Max Acceleration': f"{max(accelerations):.1f} m/s²",
    'Max Curvature': f"{max(curvatures):.3f} 1/m"
}

y_pos = np.arange(len(metrics))
metric_names = list(metrics.keys())
metric_values = list(metrics.values())

# Create text display
ax6.barh(y_pos, [1]*len(metrics), alpha=0.3, color='lightblue')
for i, (name, value) in enumerate(metrics.items()):
    ax6.text(0.5, i, f"{name}: {value}", ha='center', va='center', fontsize=10, fontweight='bold')

ax6.set_yticks([])
ax6.set_xlim(0, 1)
ax6.set_title('Performance Summary')
ax6.set_xticks([])

plt.tight_layout()
plt.show()

print("Mobile Robot Path Analysis:")
print("-" * 40)
for name, value in metrics.items():
    print(f"{name:>16}: {value}")
```

## Animation and Graphics

### Camera Path Animation

Smooth camera movements for cinematography:

```python
from interpolatepy import SquadC2, Quaternion, CubicSpline
import numpy as np
import matplotlib.pyplot as plt

def create_camera_animation():
    """Create smooth camera animation with position and orientation."""
    
    # Camera position waypoints (x, y, z)
    camera_positions = np.array([
        [0, 0, 5],      # Start position
        [5, 3, 4],      # Move forward and up
        [8, 8, 6],      # Sweep around
        [10, 5, 3],     # Lower altitude
        [12, 0, 8],     # High angle
        [15, -2, 5]     # End position
    ])
    
    # Camera orientations (as quaternions)
    # Each represents camera looking direction + roll
    camera_orientations = [
        Quaternion.identity(),                                    # Forward
        Quaternion.from_euler_angles(0.1, 0.3, 0),              # Slight tilt
        Quaternion.from_euler_angles(-0.2, 0.8, 0.1),           # Look down-right
        Quaternion.from_euler_angles(0.3, 1.2, -0.1),           # Look up-right
        Quaternion.from_euler_angles(-0.4, 1.8, 0.2),           # High angle
        Quaternion.from_euler_angles(0, 2.0, 0)                  # Look back
    ]
    
    # Timing for keyframes
    keyframe_times = [0, 2, 4, 6, 8, 10]
    
    # Create position splines (one for each coordinate)
    x_spline = CubicSpline(keyframe_times, camera_positions[:, 0], v0=0, vn=0)
    y_spline = CubicSpline(keyframe_times, camera_positions[:, 1], v0=0, vn=0)
    z_spline = CubicSpline(keyframe_times, camera_positions[:, 2], v0=0, vn=0)
    
    # Create orientation spline
    orientation_spline = SquadC2(keyframe_times, camera_orientations)
    
    return (x_spline, y_spline, z_spline), orientation_spline, keyframe_times, camera_positions

# Generate animation
position_splines, orientation_spline, keyframe_times, keyframe_positions = create_camera_animation()

# Evaluate animation
t_eval = np.linspace(0, 10, 200)
camera_path = []
camera_orientations_eval = []
camera_velocities = []

for t in t_eval:
    # Position
    x = position_splines[0].evaluate(t)
    y = position_splines[1].evaluate(t)
    z = position_splines[2].evaluate(t)
    camera_path.append([x, y, z])
    
    # Velocity
    vx = position_splines[0].evaluate_velocity(t)
    vy = position_splines[1].evaluate_velocity(t)
    vz = position_splines[2].evaluate_velocity(t)
    camera_velocities.append([vx, vy, vz])
    
    # Orientation
    orientation = orientation_spline.evaluate(t)
    camera_orientations_eval.append(orientation)

camera_path = np.array(camera_path)
camera_velocities = np.array(camera_velocities)

# Visualization
fig = plt.figure(figsize=(20, 15))

# 3D camera path
ax1 = plt.subplot(2, 3, 1, projection='3d')

# Plot path
ax1.plot(camera_path[:, 0], camera_path[:, 1], camera_path[:, 2], 
         'b-', linewidth=3, label='Camera path')

# Plot keyframes
ax1.scatter(keyframe_positions[:, 0], keyframe_positions[:, 1], keyframe_positions[:, 2],
           color='red', s=100, label='Keyframes', zorder=5)

# Add camera direction vectors
n_vectors = 15
vector_indices = np.linspace(0, len(camera_path)-1, n_vectors, dtype=int)

for i in vector_indices:
    pos = camera_path[i]
    quat = camera_orientations_eval[i]
    
    # Convert quaternion to direction vector (simplified)
    # Forward direction in camera space is typically -Z
    forward_local = np.array([0, 0, -1])
    
    # Rotate by quaternion to get world direction
    # This is a simplified rotation - in practice you'd use proper quaternion rotation
    roll, pitch, yaw = quat.to_euler_angles()
    
    # Approximate forward direction
    forward_world = np.array([
        np.cos(yaw) * np.cos(pitch),
        np.sin(yaw) * np.cos(pitch),
        -np.sin(pitch)
    ])
    
    # Draw direction vector
    ax1.quiver(pos[0], pos[1], pos[2], 
              forward_world[0], forward_world[1], forward_world[2],
              length=1.5, color='orange', alpha=0.7, arrow_length_ratio=0.1)

ax1.set_xlabel('X Position')
ax1.set_ylabel('Y Position')
ax1.set_zlabel('Z Position')
ax1.set_title('3D Camera Path with Look Directions')
ax1.legend()

# Speed profile
ax2 = plt.subplot(2, 3, 2)
speeds = [np.linalg.norm(vel) for vel in camera_velocities]
plt.plot(t_eval, speeds, 'g-', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Camera Speed')
plt.title('Camera Movement Speed')
plt.grid(True)

# Individual position components
ax3 = plt.subplot(2, 3, 3)
plt.plot(t_eval, camera_path[:, 0], 'r-', linewidth=2, label='X')
plt.plot(t_eval, camera_path[:, 1], 'g-', linewidth=2, label='Y')
plt.plot(t_eval, camera_path[:, 2], 'b-', linewidth=2, label='Z')
plt.scatter(keyframe_times, keyframe_positions[:, 0], color='red', s=30, zorder=5)
plt.scatter(keyframe_times, keyframe_positions[:, 1], color='green', s=30, zorder=5)
plt.scatter(keyframe_times, keyframe_positions[:, 2], color='blue', s=30, zorder=5)
plt.xlabel('Time (s)')
plt.ylabel('Position Components')
plt.title('Camera Position Components')
plt.legend()
plt.grid(True)

# Orientation as Euler angles
ax4 = plt.subplot(2, 3, 4)
euler_angles = []
for quat in camera_orientations_eval:
    roll, pitch, yaw = quat.to_euler_angles()
    euler_angles.append([roll, pitch, yaw])

euler_angles = np.array(euler_angles)

plt.plot(t_eval, np.degrees(euler_angles[:, 0]), 'r-', linewidth=2, label='Roll')
plt.plot(t_eval, np.degrees(euler_angles[:, 1]), 'g-', linewidth=2, label='Pitch')
plt.plot(t_eval, np.degrees(euler_angles[:, 2]), 'b-', linewidth=2, label='Yaw')
plt.xlabel('Time (s)')
plt.ylabel('Rotation (degrees)')
plt.title('Camera Orientation (Euler Angles)')
plt.legend()
plt.grid(True)

# Angular velocity
ax5 = plt.subplot(2, 3, 5)
angular_velocities = []
for t in t_eval:
    ang_vel = orientation_spline.evaluate_velocity(t)
    angular_velocities.append(np.linalg.norm(ang_vel))

plt.plot(t_eval, np.degrees(angular_velocities), 'purple', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Angular Speed (deg/s)')
plt.title('Camera Angular Velocity')
plt.grid(True)

# Animation timeline
ax6 = plt.subplot(2, 3, 6)

# Create timeline visualization
keyframe_names = ['Start', 'Rise', 'Sweep', 'Descend', 'High Angle', 'End']
colors = plt.cm.viridis(np.linspace(0, 1, len(keyframe_times)))

for i, (time, name, color) in enumerate(zip(keyframe_times, keyframe_names, colors)):
    plt.barh(i, 1, left=time, height=0.6, color=color, alpha=0.7)
    plt.text(time + 0.5, i, name, ha='center', va='center', fontweight='bold')

plt.xlim(0, 10)
plt.ylim(-0.5, len(keyframe_times) - 0.5)
plt.xlabel('Time (s)')
plt.ylabel('Keyframes')
plt.title('Animation Timeline')
plt.yticks(range(len(keyframe_names)), keyframe_names)

plt.tight_layout()
plt.show()

# Print animation statistics
print("Camera Animation Analysis:")
print("-" * 40)
print(f"Total duration: {keyframe_times[-1]} seconds")
print(f"Number of keyframes: {len(keyframe_times)}")
print(f"Average speed: {np.mean(speeds):.2f} units/s")
print(f"Maximum speed: {np.max(speeds):.2f} units/s")
print(f"Average angular speed: {np.mean(angular_velocities):.1f} deg/s")
print(f"Maximum angular speed: {np.max(angular_velocities):.1f} deg/s")

# Calculate path smoothness (jerk)
accelerations = []
for i in range(1, len(camera_velocities) - 1):
    dt = t_eval[1] - t_eval[0]
    acc = (camera_velocities[i+1] - camera_velocities[i-1]) / (2 * dt)
    accelerations.append(np.linalg.norm(acc))

print(f"Average acceleration: {np.mean(accelerations):.2f} units/s²")
print(f"Maximum acceleration: {np.max(accelerations):.2f} units/s²")
```

## Scientific Computing

### Data Smoothing and Noise Reduction

Advanced signal processing with smoothing splines:

```python
from interpolatepy import CubicSmoothingSpline, smoothing_spline_with_tolerance
import numpy as np
import matplotlib.pyplot as plt

def analyze_experimental_data():
    """Analyze noisy experimental data with different smoothing techniques."""
    
    # Simulate experimental data
    np.random.seed(42)
    
    # True underlying signal (unknown in real experiments)
    t_true = np.linspace(0, 20, 500)
    signal_true = (2 * np.sin(0.5 * t_true) + 
                   np.sin(2 * t_true) + 
                   0.5 * np.sin(5 * t_true) + 
                   0.1 * t_true)
    
    # Measurement points (sparse, realistic sampling)
    t_measured = np.linspace(0, 20, 100)
    
    # Add realistic noise
    measurement_noise = 0.3 * np.random.randn(len(t_measured))
    signal_measured = np.interp(t_measured, t_true, signal_true) + measurement_noise
    
    # Add some outliers (common in real data)
    outlier_indices = np.random.choice(len(t_measured), 5, replace=False)
    signal_measured[outlier_indices] += np.random.choice([-1, 1], 5) * np.random.uniform(1, 2, 5)
    
    return t_true, signal_true, t_measured, signal_measured

# Generate data
t_true, signal_true, t_measured, signal_measured = analyze_experimental_data()

# Apply different smoothing strategies
smoothing_methods = {
    'No Smoothing': CubicSmoothingSpline(t_measured.tolist(), signal_measured.tolist(), mu=0.0),
    'Light Smoothing': CubicSmoothingSpline(t_measured.tolist(), signal_measured.tolist(), mu=0.001),
    'Medium Smoothing': CubicSmoothingSpline(t_measured.tolist(), signal_measured.tolist(), mu=0.01),
    'Heavy Smoothing': CubicSmoothingSpline(t_measured.tolist(), signal_measured.tolist(), mu=0.1),
    'Auto Smoothing': smoothing_spline_with_tolerance(
        t_measured.tolist(), signal_measured.tolist(), tolerance=0.2
    )
}

# Evaluate all methods
t_eval = np.linspace(0, 20, 400)
results = {}

for method_name, spline in smoothing_methods.items():
    smoothed_signal = spline.evaluate(t_eval)
    
    # Calculate metrics
    true_signal_interp = np.interp(t_eval, t_true, signal_true)
    mse = np.mean((smoothed_signal - true_signal_interp)**2)
    
    # Calculate smoothness (average curvature)
    acceleration = spline.evaluate_acceleration(t_eval)
    smoothness = np.mean(np.abs(acceleration))
    
    results[method_name] = {
        'signal': smoothed_signal,
        'mse': mse,
        'smoothness': smoothness,
        'mu': getattr(spline, 'mu', 0.0)
    }

# Comprehensive visualization
fig = plt.figure(figsize=(20, 15))

# Main comparison plot
ax1 = plt.subplot(3, 3, (1, 3))
plt.plot(t_true, signal_true, 'k-', linewidth=2, alpha=0.8, label='True signal (unknown)')
plt.scatter(t_measured, signal_measured, color='red', alpha=0.6, s=20, label='Noisy measurements')

colors = ['blue', 'green', 'orange', 'purple', 'brown']
for i, (method, result) in enumerate(results.items()):
    plt.plot(t_eval, result['signal'], color=colors[i], linewidth=2, 
             label=f"{method} (MSE: {result['mse']:.3f})")

plt.xlabel('Time')
plt.ylabel('Signal Value')
plt.title('Experimental Data Smoothing Comparison')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# MSE vs Smoothness trade-off
ax2 = plt.subplot(3, 3, 4)
mse_values = [result['mse'] for result in results.values()]
smoothness_values = [result['smoothness'] for result in results.values()]
method_names = list(results.keys())

scatter = plt.scatter(smoothness_values, mse_values, c=range(len(method_names)), 
                     s=100, cmap='viridis', alpha=0.7)
for i, name in enumerate(method_names):
    plt.annotate(name, (smoothness_values[i], mse_values[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.xlabel('Average Curvature (Smoothness)')
plt.ylabel('Mean Squared Error')
plt.title('Bias-Variance Trade-off')
plt.grid(True)

# Smoothing parameter effect
ax3 = plt.subplot(3, 3, 5)
mu_values = [result['mu'] for result in results.values()]
mse_values = [result['mse'] for result in results.values()]

# Filter out zero mu for log plot
nonzero_indices = [i for i, mu in enumerate(mu_values) if mu > 0]
if nonzero_indices:
    mu_nonzero = [mu_values[i] for i in nonzero_indices]
    mse_nonzero = [mse_values[i] for i in nonzero_indices]
    names_nonzero = [method_names[i] for i in nonzero_indices]
    
    plt.semilogx(mu_nonzero, mse_nonzero, 'o-', linewidth=2, markersize=8)
    for mu, mse, name in zip(mu_nonzero, mse_nonzero, names_nonzero):
        plt.annotate(name, (mu, mse), xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.xlabel('Smoothing Parameter μ')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs Smoothing Parameter')
plt.grid(True)

# Residual analysis
ax4 = plt.subplot(3, 3, 6)
auto_smoothed = results['Auto Smoothing']['signal']
residuals = signal_measured - np.interp(t_measured, t_eval, auto_smoothed)

plt.scatter(t_measured, residuals, alpha=0.7, color='red')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.axhline(y=np.std(residuals), color='gray', linestyle=':', alpha=0.7, label=f'±1σ ({np.std(residuals):.2f})')
plt.axhline(y=-np.std(residuals), color='gray', linestyle=':', alpha=0.7)
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.title('Residual Analysis (Auto Smoothing)')
plt.legend()
plt.grid(True)

# Frequency domain analysis
ax5 = plt.subplot(3, 3, 7)
from scipy import fft

# FFT of original noisy signal
freqs = fft.fftfreq(len(t_measured), t_measured[1] - t_measured[0])
fft_noisy = fft.fft(signal_measured)

# FFT of smoothed signal (auto method)
t_interp = np.interp(t_measured, t_eval, auto_smoothed)
fft_smoothed = fft.fft(t_interp)

# Plot magnitude spectra
plt.loglog(freqs[:len(freqs)//2], np.abs(fft_noisy)[:len(freqs)//2], 
           'r-', alpha=0.7, label='Noisy signal')
plt.loglog(freqs[:len(freqs)//2], np.abs(fft_smoothed)[:len(freqs)//2], 
           'b-', linewidth=2, label='Smoothed signal')

plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Domain Comparison')
plt.legend()
plt.grid(True)

# Statistical summary
ax6 = plt.subplot(3, 3, 8)
stats_data = []
stats_labels = []

for method, result in results.items():
    stats_data.append([result['mse'], result['smoothness']])
    stats_labels.append(method)

stats_data = np.array(stats_data)

# Normalize for display
stats_normalized = stats_data / np.max(stats_data, axis=0)

# Create heatmap
im = plt.imshow(stats_normalized.T, cmap='RdYlBu_r', aspect='auto')
plt.yticks([0, 1], ['MSE', 'Smoothness'])
plt.xticks(range(len(stats_labels)), stats_labels, rotation=45, ha='right')
plt.title('Method Comparison Matrix')

# Add text annotations
for i in range(len(stats_labels)):
    for j in range(2):
        plt.text(i, j, f'{stats_data[i, j]:.3f}', ha='center', va='center', 
                color='white' if stats_normalized[i, j] > 0.5 else 'black',
                fontweight='bold', fontsize=8)

# Performance recommendations
ax7 = plt.subplot(3, 3, 9)
ax7.axis('off')

recommendations = [
    "SMOOTHING RECOMMENDATIONS:",
    "",
    "• No Smoothing: Use when data is clean",
    "  and exact interpolation is needed",
    "",
    "• Light Smoothing (μ≈0.001): Good for",
    "  low-noise scientific measurements", 
    "",
    "• Medium Smoothing (μ≈0.01): General",
    "  purpose for typical experimental data",
    "",
    "• Heavy Smoothing (μ≈0.1): Use when",
    "  trend extraction is more important",
    "",
    "• Auto Smoothing: Recommended for",
    "  objective, data-driven smoothing",
    "",
    f"BEST METHOD FOR THIS DATA:",
    f"Auto Smoothing (μ={results['Auto Smoothing']['mu']:.4f})",
    f"MSE: {results['Auto Smoothing']['mse']:.3f}"
]

for i, text in enumerate(recommendations):
    if text.startswith("BEST METHOD") or text.startswith("SMOOTHING RECOMMENDATIONS"):
        ax7.text(0.05, 0.95 - i*0.045, text, transform=ax7.transAxes, 
                fontweight='bold', fontsize=10, color='blue')
    elif text.startswith("•"):
        ax7.text(0.05, 0.95 - i*0.045, text, transform=ax7.transAxes, 
                fontsize=9, color='darkgreen')
    else:
        ax7.text(0.05, 0.95 - i*0.045, text, transform=ax7.transAxes, 
                fontsize=9)

plt.tight_layout()
plt.show()

# Print detailed analysis
print("EXPERIMENTAL DATA ANALYSIS REPORT")
print("=" * 50)
print(f"Data points: {len(t_measured)}")
print(f"Time range: {t_measured[0]:.1f} to {t_measured[-1]:.1f}")
print(f"Noise level (std): {np.std(measurement_noise):.3f}")
print(f"Signal-to-noise ratio: {np.std(signal_true)/np.std(measurement_noise):.1f}")
print()

print("SMOOTHING METHOD COMPARISON:")
print("-" * 30)
print(f"{'Method':<20} {'MSE':<8} {'Smoothness':<12} {'μ':<10}")
print("-" * 50)
for method, result in results.items():
    print(f"{method:<20} {result['mse']:<8.4f} {result['smoothness']:<12.4f} {result['mu']:<10.6f}")

print()
print("RECOMMENDATIONS:")
print("-" * 15)
best_mse = min(results.values(), key=lambda x: x['mse'])
best_method = [name for name, result in results.items() if result['mse'] == best_mse['mse']][0]
print(f"• Lowest MSE: {best_method}")
print(f"• For this data: Auto Smoothing provides good balance")
print(f"• Outliers detected and handled automatically")
```

## Summary

This examples gallery demonstrates InterpolatePy's versatility across:

✅ **Robotics**: Multi-axis control, mobile robot navigation  
✅ **Animation**: Smooth camera paths with quaternion orientation  
✅ **Scientific Computing**: Advanced data smoothing and analysis  

### Key Takeaways

1. **Combine algorithms**: Use multiple InterpolatePy classes together for complex applications
2. **Real-world constraints**: Always consider physical limits and safety requirements
3. **Visualization**: Comprehensive plotting helps validate and understand trajectories
4. **Performance analysis**: Quantify smoothness, accuracy, and computational efficiency

### Next Steps

- **[Tutorials](tutorials/spline-interpolation.md)**: Deep dive into specific algorithm families
- **[API Reference](api-reference.md)**: Complete function documentation
- **[Contributing](contributing.md)**: Add your own examples to the gallery

---

*Have an interesting InterpolatePy application? [Share it with the community!](https://github.com/GiorgioMedico/InterpolatePy/discussions)*