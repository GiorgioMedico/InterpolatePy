# Motion Profiles Tutorial

Motion profiles are essential for robotics and automation, providing trajectories with bounded velocity, acceleration, and jerk. This tutorial covers S-curve, trapezoidal, and polynomial profiles with practical applications.

## Why Motion Profiles?

Motion profiles solve critical problems in robotics:

- **Smooth acceleration**: Reduce mechanical stress and vibration
- **Bounded derivatives**: Respect actuator limits  
- **Time optimization**: Reach targets as quickly as possible
- **Passenger comfort**: Smooth elevator and vehicle motion

## S-Curve Trajectories (Jerk-Limited)

S-curve profiles provide the smoothest motion by limiting jerk (rate of acceleration change).

### Basic S-Curve Generation

```python
import numpy as np
import matplotlib.pyplot as plt
from interpolatepy import DoubleSTrajectory, StateParams, TrajectoryBounds

# Define motion parameters
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

# Generate S-curve trajectory
trajectory = DoubleSTrajectory(state, bounds)

print(f"Total duration: {trajectory.total_time:.2f} seconds")

# Plot the complete profile
trajectory.plot()
plt.show()
```

### Understanding the 7 Phases

S-curve trajectories can have up to 7 phases:

```python
# Create detailed phase analysis
fig, axes = plt.subplots(4, 1, figsize=(14, 12))

# Evaluate trajectory
t_eval = np.linspace(0, trajectory.total_time, 1000)
positions = [trajectory.evaluate(t) for t in t_eval]
velocities = [trajectory.evaluate_velocity(t) for t in t_eval] 
accelerations = [trajectory.evaluate_acceleration(t) for t in t_eval]
jerks = [trajectory.evaluate_jerk(t) for t in t_eval]

# Position
axes[0].plot(t_eval, positions, 'b-', linewidth=2)
axes[0].set_ylabel('Position')
axes[0].set_title('S-Curve Motion Profile - All Derivatives')
axes[0].grid(True)

# Velocity (S-shaped)
axes[1].plot(t_eval, velocities, 'g-', linewidth=2)
axes[1].axhline(y=bounds.v_bound, color='r', linestyle='--', alpha=0.7, label='Velocity limit')
axes[1].axhline(y=-bounds.v_bound, color='r', linestyle='--', alpha=0.7)
axes[1].set_ylabel('Velocity')
axes[1].legend()
axes[1].grid(True)

# Acceleration (trapezoidal)
axes[2].plot(t_eval, accelerations, 'm-', linewidth=2)
axes[2].axhline(y=bounds.a_bound, color='r', linestyle='--', alpha=0.7, label='Acceleration limit')
axes[2].axhline(y=-bounds.a_bound, color='r', linestyle='--', alpha=0.7)
axes[2].set_ylabel('Acceleration')
axes[2].legend()
axes[2].grid(True)

# Jerk (bang-bang)
axes[3].plot(t_eval, jerks, 'orange', linewidth=2)
axes[3].axhline(y=bounds.j_bound, color='r', linestyle='--', alpha=0.7, label='Jerk limit')
axes[3].axhline(y=-bounds.j_bound, color='r', linestyle='--', alpha=0.7)
axes[3].set_xlabel('Time (s)')
axes[3].set_ylabel('Jerk')
axes[3].legend()
axes[3].grid(True)

plt.tight_layout()
plt.show()

# Identify phases by analyzing jerk
print("\\nPhase Analysis:")
print("Phase 1: Jerk-up (acceleration increases)")
print("Phase 2: Constant acceleration") 
print("Phase 3: Jerk-down (acceleration decreases to 0)")
print("Phase 4: Constant velocity")
print("Phase 5: Jerk-down (deceleration begins)")
print("Phase 6: Constant deceleration")
print("Phase 7: Jerk-up (deceleration decreases to 0)")
```

### Effect of Different Constraints

```python
# Compare different constraint combinations
constraint_sets = [
    {'v_bound': 3.0, 'a_bound': 5.0, 'j_bound': 15.0, 'label': 'Conservative'},
    {'v_bound': 5.0, 'a_bound': 10.0, 'j_bound': 30.0, 'label': 'Moderate'},
    {'v_bound': 8.0, 'a_bound': 20.0, 'j_bound': 60.0, 'label': 'Aggressive'}
]

# Same motion (0 to 10 units)
base_state = StateParams(q_0=0.0, q_1=10.0, v_0=0.0, v_1=0.0)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

for i, constraints in enumerate(constraint_sets):
    bounds = TrajectoryBounds(**{k: v for k, v in constraints.items() if k != 'label'})
    traj = DoubleSTrajectory(base_state, bounds)
    
    t_eval = np.linspace(0, traj.total_time, 200)
    positions = [traj.evaluate(t) for t in t_eval]
    velocities = [traj.evaluate_velocity(t) for t in t_eval]
    
    color = ['blue', 'green', 'red'][i]
    label = constraints['label']
    
    # Position
    axes[0, 0].plot(t_eval, positions, color=color, linewidth=2, 
                    label=f'{label} ({traj.total_time:.1f}s)')
    
    # Velocity
    axes[0, 1].plot(t_eval, velocities, color=color, linewidth=2, label=label)

# Formatting
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Position')
axes[0, 0].set_title('Position Profiles')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Velocity')
axes[0, 1].set_title('Velocity Profiles')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Create bar chart of durations
labels = [c['label'] for c in constraint_sets]
durations = []
for constraints in constraint_sets:
    bounds = TrajectoryBounds(**{k: v for k, v in constraints.items() if k != 'label'})
    traj = DoubleSTrajectory(base_state, bounds)
    durations.append(traj.total_time)

axes[1, 0].bar(labels, durations, color=['blue', 'green', 'red'], alpha=0.7)
axes[1, 0].set_ylabel('Duration (s)')
axes[1, 0].set_title('Trajectory Duration Comparison')
axes[1, 0].grid(True, axis='y')

# Create constraint comparison table
constraint_table = np.array([[c['v_bound'], c['a_bound'], c['j_bound']] for c in constraint_sets])
im = axes[1, 1].imshow(constraint_table.T, cmap='viridis', aspect='auto')
axes[1, 1].set_xticks(range(len(labels)))
axes[1, 1].set_xticklabels(labels)
axes[1, 1].set_yticks(range(3))
axes[1, 1].set_yticklabels(['Velocity', 'Acceleration', 'Jerk'])
axes[1, 1].set_title('Constraint Values')

# Add text annotations
for i in range(len(labels)):
    for j in range(3):
        axes[1, 1].text(i, j, f'{constraint_table[i, j]:.0f}', 
                        ha='center', va='center', color='white', fontweight='bold')

plt.tight_layout()
plt.show()

print("Key Insight: More aggressive constraints = faster trajectories")
```

### Handling Non-Zero Initial/Final Velocities

```python
# Moving between conveyor belts with different speeds
scenarios = [
    {'v_0': 0.0, 'v_1': 0.0, 'label': 'Stop-to-stop'},
    {'v_0': 0.0, 'v_1': 2.0, 'label': 'Start-to-cruise'},
    {'v_0': 2.0, 'v_1': 0.0, 'label': 'Cruise-to-stop'},
    {'v_0': 1.0, 'v_1': 3.0, 'label': 'Speed change'}
]

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

bounds = TrajectoryBounds(v_bound=5.0, a_bound=8.0, j_bound=20.0)

for i, scenario in enumerate(scenarios):
    state = StateParams(q_0=0.0, q_1=10.0, v_0=scenario['v_0'], v_1=scenario['v_1'])
    traj = DoubleSTrajectory(state, bounds)
    
    t_eval = np.linspace(0, traj.total_time, 200)
    positions = [traj.evaluate(t) for t in t_eval]
    velocities = [traj.evaluate_velocity(t) for t in t_eval]
    
    # Position
    axes[i].plot(t_eval, positions, 'b-', linewidth=2, label='Position')
    ax2 = axes[i].twinx()
    ax2.plot(t_eval, velocities, 'r-', linewidth=2, label='Velocity')
    
    # Highlight boundary conditions
    axes[i].axhline(y=0, color='blue', linestyle=':', alpha=0.5)
    axes[i].axhline(y=10, color='blue', linestyle=':', alpha=0.5)
    ax2.axhline(y=scenario['v_0'], color='red', linestyle=':', alpha=0.5)
    ax2.axhline(y=scenario['v_1'], color='red', linestyle=':', alpha=0.5)
    
    axes[i].set_xlabel('Time (s)')
    axes[i].set_ylabel('Position', color='blue')
    ax2.set_ylabel('Velocity', color='red')
    axes[i].set_title(f"{scenario['label']}\\n(v₀={scenario['v_0']}, v₁={scenario['v_1']})")
    axes[i].grid(True)

plt.tight_layout()
plt.show()

# Print duration comparison
print("Duration Analysis:")
for scenario in scenarios:
    state = StateParams(q_0=0.0, q_1=10.0, v_0=scenario['v_0'], v_1=scenario['v_1'])
    traj = DoubleSTrajectory(state, bounds)
    print(f"{scenario['label']:>15}: {traj.total_time:.2f}s")
```

## Trapezoidal Trajectories

Trapezoidal profiles are simpler than S-curves but still provide bounded acceleration.

### Basic Trapezoidal Profile

```python
from interpolatepy import TrapezoidalTrajectory, TrajectoryParams

# Define trajectory parameters
params = TrajectoryParams(
    q0=0.0,       # Initial position
    q1=15.0,      # Final position
    v0=0.0,       # Initial velocity
    v1=0.0,       # Final velocity
    amax=5.0,     # Maximum acceleration
    vmax=8.0      # Maximum velocity
)

# Generate trajectory
traj_func, duration = TrapezoidalTrajectory.generate_trajectory(params)

print(f"Trajectory duration: {duration:.2f} seconds")

# Evaluate trajectory
t_eval = np.linspace(0, duration, 300)
results = [traj_func(t) for t in t_eval]
positions = [r[0] for r in results]
velocities = [r[1] for r in results]
accelerations = [r[2] for r in results]

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(t_eval, positions, 'b-', linewidth=2)
axes[0].set_ylabel('Position')
axes[0].set_title('Trapezoidal Motion Profile')
axes[0].grid(True)

axes[1].plot(t_eval, velocities, 'g-', linewidth=2)
axes[1].axhline(y=params.vmax, color='r', linestyle='--', alpha=0.7, label='Velocity limit')
axes[1].set_ylabel('Velocity')
axes[1].legend()
axes[1].grid(True)

axes[2].plot(t_eval, accelerations, 'm-', linewidth=2)
axes[2].axhline(y=params.amax, color='r', linestyle='--', alpha=0.7, label='Acceleration limit')
axes[2].axhline(y=-params.amax, color='r', linestyle='--', alpha=0.7)
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Acceleration')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()
```

### Trapezoidal vs Triangular Profiles

```python
# Compare trapezoidal vs triangular profiles
distances = [5, 10, 20, 50]  # Different travel distances
params_base = TrajectoryParams(q0=0.0, v0=0.0, v1=0.0, amax=5.0, vmax=8.0)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, distance in enumerate(distances):
    params = TrajectoryParams(
        q0=0.0, q1=distance, v0=0.0, v1=0.0, 
        amax=5.0, vmax=8.0
    )
    
    traj_func, duration = TrapezoidalTrajectory.generate_trajectory(params)
    
    t_eval = np.linspace(0, duration, 200)
    results = [traj_func(t) for t in t_eval]
    velocities = [r[1] for r in results]
    
    max_vel_reached = max(velocities)
    profile_type = "Trapezoidal" if max_vel_reached >= 7.9 else "Triangular"
    
    axes[i].plot(t_eval, velocities, 'b-', linewidth=2, 
                 label=f'{profile_type} (d={distance})')
    axes[i].axhline(y=params.vmax, color='r', linestyle='--', alpha=0.7, 
                    label='Velocity limit')
    axes[i].set_xlabel('Time (s)')
    axes[i].set_ylabel('Velocity')
    axes[i].set_title(f'Distance = {distance} units\\n'
                      f'Max velocity reached: {max_vel_reached:.1f}')
    axes[i].legend()
    axes[i].grid(True)

plt.tight_layout()
plt.show()

print("Profile Type Analysis:")
print("- Short distances → Triangular (never reach vmax)")
print("- Long distances → Trapezoidal (reach and maintain vmax)")
```

### Duration vs Velocity Constrained

```python
# Compare duration-constrained vs velocity-constrained planning
base_params = {'q0': 0.0, 'q1': 20.0, 'v0': 0.0, 'v1': 0.0, 'amax': 4.0}

# Velocity-constrained (find minimum time)
params_vel = TrajectoryParams(**base_params, vmax=6.0)
traj_vel, duration_vel = TrapezoidalTrajectory.generate_trajectory(params_vel)

# Duration-constrained (find required velocity)
target_duration = 8.0  # Longer than minimum time
params_dur = TrajectoryParams(**base_params, duration=target_duration)
traj_dur, duration_dur = TrapezoidalTrajectory.generate_trajectory(params_dur)

# Compare results
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Evaluate both trajectories
t_eval_vel = np.linspace(0, duration_vel, 200)
t_eval_dur = np.linspace(0, duration_dur, 200)

results_vel = [traj_vel(t) for t in t_eval_vel]
results_dur = [traj_dur(t) for t in t_eval_dur]

pos_vel = [r[0] for r in results_vel]
vel_vel = [r[1] for r in results_vel]
pos_dur = [r[0] for r in results_dur]
vel_dur = [r[1] for r in results_dur]

# Position comparison
axes[0].plot(t_eval_vel, pos_vel, 'b-', linewidth=2, 
             label=f'Velocity-constrained ({duration_vel:.1f}s)')
axes[0].plot(t_eval_dur, pos_dur, 'r--', linewidth=2,
             label=f'Duration-constrained ({duration_dur:.1f}s)')
axes[0].set_ylabel('Position')
axes[0].set_title('Duration vs Velocity Constrained Planning')
axes[0].legend()
axes[0].grid(True)

# Velocity comparison
axes[1].plot(t_eval_vel, vel_vel, 'b-', linewidth=2, label='Velocity-constrained')
axes[1].plot(t_eval_dur, vel_dur, 'r--', linewidth=2, label='Duration-constrained')
axes[1].axhline(y=6.0, color='blue', linestyle=':', alpha=0.7, label='Velocity limit')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Velocity')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

print(f"Velocity-constrained: {duration_vel:.1f}s (minimum time)")
print(f"Duration-constrained: {duration_dur:.1f}s (specified time)")
print(f"Duration-constrained uses lower peak velocity: {max(vel_dur):.1f} vs {max(vel_vel):.1f}")
```

## Polynomial Trajectories

Polynomial trajectories provide exact boundary condition matching.

### Different Polynomial Orders

```python
from interpolatepy import PolynomialTrajectory, BoundaryCondition, TimeInterval

# Define boundary conditions
initial = BoundaryCondition(position=0.0, velocity=0.0, acceleration=0.0, jerk=0.0)
final = BoundaryCondition(position=10.0, velocity=0.0, acceleration=0.0, jerk=0.0)
interval = TimeInterval(t0=0.0, t1=5.0)

# Generate different order polynomials
poly_3 = PolynomialTrajectory.order_3_trajectory(
    BoundaryCondition(position=0.0, velocity=0.0),
    BoundaryCondition(position=10.0, velocity=0.0), 
    interval
)

poly_5 = PolynomialTrajectory.order_5_trajectory(
    BoundaryCondition(position=0.0, velocity=0.0, acceleration=0.0),
    BoundaryCondition(position=10.0, velocity=0.0, acceleration=0.0),
    interval
)

poly_7 = PolynomialTrajectory.order_7_trajectory(initial, final, interval)

# Evaluate all trajectories
t_eval = np.linspace(0, 5, 200)

results_3 = [poly_3(t) for t in t_eval]
results_5 = [poly_5(t) for t in t_eval]
results_7 = [poly_7(t) for t in t_eval]

# Extract derivatives
pos_3 = [r[0] for r in results_3]
vel_3 = [r[1] for r in results_3]
acc_3 = [r[2] for r in results_3]

pos_5 = [r[0] for r in results_5]
vel_5 = [r[1] for r in results_5]
acc_5 = [r[2] for r in results_5]

pos_7 = [r[0] for r in results_7]
vel_7 = [r[1] for r in results_7]
acc_7 = [r[2] for r in results_7]
jerk_7 = [r[3] for r in results_7]

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Position
axes[0, 0].plot(t_eval, pos_3, 'b-', linewidth=2, label='3rd order (cubic)')
axes[0, 0].plot(t_eval, pos_5, 'g-', linewidth=2, label='5th order (quintic)')
axes[0, 0].plot(t_eval, pos_7, 'r-', linewidth=2, label='7th order (septic)')
axes[0, 0].set_ylabel('Position')
axes[0, 0].set_title('Position Profiles')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Velocity
axes[0, 1].plot(t_eval, vel_3, 'b-', linewidth=2, label='3rd order')
axes[0, 1].plot(t_eval, vel_5, 'g-', linewidth=2, label='5th order')
axes[0, 1].plot(t_eval, vel_7, 'r-', linewidth=2, label='7th order')
axes[0, 1].set_ylabel('Velocity')
axes[0, 1].set_title('Velocity Profiles')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Acceleration
axes[1, 0].plot(t_eval, acc_3, 'b-', linewidth=2, label='3rd order')
axes[1, 0].plot(t_eval, acc_5, 'g-', linewidth=2, label='5th order')
axes[1, 0].plot(t_eval, acc_7, 'r-', linewidth=2, label='7th order')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Acceleration')
axes[1, 0].set_title('Acceleration Profiles')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Jerk (only for 7th order)
axes[1, 1].plot(t_eval, jerk_7, 'r-', linewidth=2, label='7th order jerk')
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Jerk')
axes[1, 1].set_title('Jerk Profile (7th order only)')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

print("Polynomial Order Analysis:")
print("3rd order: Matches position and velocity boundaries")
print("5th order: Matches position, velocity, and acceleration boundaries")
print("7th order: Matches position, velocity, acceleration, and jerk boundaries")
```

## Practical Applications

### Elevator Motion Planning

```python
# Realistic elevator control system
def plan_elevator_journey(floors, floor_height=3.0):
    """Plan complete elevator journey with passenger comfort constraints."""
    
    # Passenger comfort constraints
    bounds = TrajectoryBounds(
        v_bound=2.0,    # 2 m/s max speed
        a_bound=1.2,    # 1.2 m/s² max acceleration (0.12g)
        j_bound=2.5     # 2.5 m/s³ max jerk
    )
    
    segments = []
    current_time = 0.0
    
    for i in range(len(floors) - 1):
        start_floor = floors[i]
        end_floor = floors[i + 1]
        
        # Calculate positions
        start_pos = start_floor * floor_height
        end_pos = end_floor * floor_height
        
        # Create trajectory segment
        state = StateParams(
            q_0=start_pos, q_1=end_pos,
            v_0=0.0, v_1=0.0  # Always stop at floors
        )
        
        trajectory = DoubleSTrajectory(state, bounds)
        
        segments.append({
            'start_floor': start_floor,
            'end_floor': end_floor,
            'start_time': current_time,
            'duration': trajectory.total_time,
            'trajectory': trajectory,
            'direction': 'Up' if end_floor > start_floor else 'Down'
        })
        
        current_time += trajectory.total_time
    
    return segments, current_time

# Plan elevator journey: Lobby → 15th floor → 3rd floor → 22nd floor
floors = [0, 15, 3, 22]
segments, total_time = plan_elevator_journey(floors)

# Visualize complete journey
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

current_time = 0.0
all_times, all_positions, all_velocities, all_accelerations = [], [], [], []

for segment in segments:
    traj = segment['trajectory']
    duration = segment['duration']
    
    # Time array for this segment
    t_segment = np.linspace(0, duration, 100)
    t_absolute = t_segment + current_time
    
    # Evaluate trajectory
    positions = [traj.evaluate(t) for t in t_segment]
    velocities = [traj.evaluate_velocity(t) for t in t_segment]
    accelerations = [traj.evaluate_acceleration(t) for t in t_segment]
    
    # Store for continuous plot
    all_times.extend(t_absolute)
    all_positions.extend(positions)
    all_velocities.extend(velocities)
    all_accelerations.extend(accelerations)
    
    current_time += duration

# Plot complete journey
axes[0].plot(all_times, all_positions, 'b-', linewidth=2)
# Mark floor levels
for floor in range(23):
    axes[0].axhline(y=floor * 3.0, color='gray', linestyle=':', alpha=0.3)
# Mark stops
for i, floor in enumerate(floors):
    stop_time = sum(seg['duration'] for seg in segments[:i])
    axes[0].scatter(stop_time, floor * 3.0, color='red', s=100, zorder=5)
    axes[0].annotate(f'Floor {floor}', (stop_time, floor * 3.0), 
                     xytext=(5, 5), textcoords='offset points')

axes[0].set_ylabel('Height (m)')
axes[0].set_title('Elevator Journey Profile')
axes[0].grid(True)

# Velocity
axes[1].plot(all_times, all_velocities, 'g-', linewidth=2)
axes[1].axhline(y=2.0, color='r', linestyle='--', alpha=0.7, label='Speed limit')
axes[1].axhline(y=-2.0, color='r', linestyle='--', alpha=0.7)
axes[1].set_ylabel('Velocity (m/s)')
axes[1].legend()
axes[1].grid(True)

# Acceleration
axes[2].plot(all_times, all_accelerations, 'm-', linewidth=2)
axes[2].axhline(y=1.2, color='r', linestyle='--', alpha=0.7, label='Acceleration limit')
axes[2].axhline(y=-1.2, color='r', linestyle='--', alpha=0.7)
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Acceleration (m/s²)')
axes[2].legend()
axes[2].grid(True)

# Add segment markers
for i, segment in enumerate(segments):
    start_time = segment['start_time']
    end_time = start_time + segment['duration']
    
    for ax in axes:
        ax.axvspan(start_time, end_time, alpha=0.1, 
                   color=['blue', 'green', 'orange'][i % 3])

plt.tight_layout()
plt.show()

# Journey summary
print("Elevator Journey Summary:")
print("=" * 50)
for i, segment in enumerate(segments):
    distance = abs(segment['end_floor'] - segment['start_floor']) * 3.0
    avg_speed = distance / segment['duration']
    print(f"Segment {i+1}: Floor {segment['start_floor']} → {segment['end_floor']} "
          f"({segment['direction']})")
    print(f"  Duration: {segment['duration']:.1f}s, "
          f"Distance: {distance:.1f}m, "
          f"Avg Speed: {avg_speed:.1f}m/s")

print(f"\\nTotal journey time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
```

### CNC Machine Tool Path

```python
# CNC machining with different motion profiles
def compare_machining_profiles(distance=100, cutting_speed=50):
    """Compare motion profiles for CNC machining."""
    
    profiles = {
        'S-curve (Jerk-limited)': {
            'generator': lambda: DoubleSTrajectory(
                StateParams(q_0=0, q_1=distance, v_0=0, v_1=0),
                TrajectoryBounds(v_bound=cutting_speed, a_bound=200, j_bound=1000)
            ),
            'color': 'blue'
        },
        'Trapezoidal': {
            'generator': lambda: TrapezoidalTrajectory.generate_trajectory(
                TrajectoryParams(q0=0, q1=distance, v0=0, v1=0, 
                               amax=200, vmax=cutting_speed)
            ),
            'color': 'green'
        }
    }
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    for profile_name, profile_info in profiles.items():
        if 'S-curve' in profile_name:
            # S-curve trajectory
            traj = profile_info['generator']()
            duration = traj.total_time
            t_eval = np.linspace(0, duration, 300)
            
            positions = [traj.evaluate(t) for t in t_eval]
            velocities = [traj.evaluate_velocity(t) for t in t_eval]
            accelerations = [traj.evaluate_acceleration(t) for t in t_eval]
            
        else:
            # Trapezoidal trajectory
            traj_func, duration = profile_info['generator']()
            t_eval = np.linspace(0, duration, 300)
            
            results = [traj_func(t) for t in t_eval]
            positions = [r[0] for r in results]
            velocities = [r[1] for r in results]
            accelerations = [r[2] for r in results]
        
        color = profile_info['color']
        
        # Plot profiles
        axes[0].plot(t_eval, positions, color=color, linewidth=2, 
                     label=f'{profile_name} ({duration:.2f}s)')
        axes[1].plot(t_eval, velocities, color=color, linewidth=2, label=profile_name)
        axes[2].plot(t_eval, accelerations, color=color, linewidth=2, label=profile_name)
    
    # Formatting
    axes[0].set_ylabel('Position (mm)')
    axes[0].set_title('CNC Tool Path Motion Profiles')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].set_ylabel('Velocity (mm/s)')
    axes[1].axhline(y=cutting_speed, color='red', linestyle='--', alpha=0.7, 
                    label='Cutting speed')
    axes[1].legend()
    axes[1].grid(True)
    
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Acceleration (mm/s²)')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()

# Compare for different cutting operations
cutting_operations = [
    {'name': 'Fine finishing', 'distance': 50, 'speed': 20},
    {'name': 'Roughing', 'distance': 200, 'speed': 100},
    {'name': 'Long traverse', 'distance': 500, 'speed': 200}
]

for operation in cutting_operations:
    print(f"\\n{operation['name'].upper()}:")
    print("-" * 30)
    compare_machining_profiles(operation['distance'], operation['speed'])
```

## Choosing the Right Motion Profile

### Decision Matrix

```python
import pandas as pd

# Create decision matrix
decision_data = {
    'Algorithm': ['S-Curve (DoubleSTrajectory)', 'Trapezoidal', 'Polynomial (3rd)', 'Polynomial (5th)', 'Polynomial (7th)'],
    'Smoothness': ['Excellent (C²)', 'Good (C¹)', 'Good (C¹)', 'Excellent (C²)', 'Excellent (C²)'],
    'Jerk Limiting': ['Yes', 'No', 'No', 'No', 'Yes'],
    'Setup Time': ['Fast', 'Fast', 'Fast', 'Fast', 'Fast'],
    'Evaluation Speed': ['Fast', 'Fast', 'Fast', 'Fast', 'Fast'],
    'Boundary Conditions': ['Position, Vel', 'Position, Vel', 'Position, Vel', 'Position, Vel, Acc', 'Position, Vel, Acc, Jerk'],
    'Best For': ['Robotics, Automation', 'Simple Point-to-Point', 'Basic Trajectories', 'Smooth Motion', 'Maximum Control']
}

df = pd.DataFrame(decision_data)
print("Motion Profile Selection Guide:")
print("=" * 80)
print(df.to_string(index=False))

# Usage recommendations
recommendations = {
    'Industrial Robotics': 'S-Curve (smooth, respects actuator limits)',
    'CNC Machining': 'S-Curve or Trapezoidal (depends on precision needs)',
    'Elevator Control': 'S-Curve (passenger comfort)',
    'Simple Pick-and-Place': 'Trapezoidal (fast, adequate smoothness)',
    'High-Precision Positioning': 'Polynomial 5th/7th order',
    'Conveyor Systems': 'Trapezoidal (simple, reliable)',
    'Spacecraft Attitude': 'Polynomial 7th order (maximum control)',
    'Mobile Robot Navigation': 'S-Curve (smooth, energy efficient)'
}

print("\\n\\nApplication Recommendations:")
print("=" * 50)
for application, recommendation in recommendations.items():
    print(f"{application:>25}: {recommendation}")
```

### Performance Comparison

```python
import time

# Performance benchmark
algorithms = {
    'S-Curve': lambda: DoubleSTrajectory(
        StateParams(q_0=0, q_1=100, v_0=0, v_1=0),
        TrajectoryBounds(v_bound=50, a_bound=100, j_bound=500)
    ),
    'Trapezoidal': lambda: TrapezoidalTrajectory.generate_trajectory(
        TrajectoryParams(q0=0, q1=100, v0=0, v1=0, amax=100, vmax=50)
    ),
    'Polynomial 5th': lambda: PolynomialTrajectory.order_5_trajectory(
        BoundaryCondition(position=0, velocity=0, acceleration=0),
        BoundaryCondition(position=100, velocity=0, acceleration=0),
        TimeInterval(0, 2)
    )
}

n_iterations = 1000
results = {}

print("Performance Benchmark (1000 iterations):")
print("-" * 50)

for name, generator in algorithms.items():
    # Time setup
    start_time = time.time()
    for _ in range(n_iterations):
        if name == 'S-Curve':
            traj = generator()
        elif name == 'Trapezoidal':
            traj_func, duration = generator()
        else:  # Polynomial
            traj_func = generator()
    setup_time = time.time() - start_time
    
    # Time evaluation (using first generated trajectory/function)
    if name == 'S-Curve':
        traj = generator()
        start_time = time.time()
        for _ in range(n_iterations):
            _ = traj.evaluate(1.0)
        eval_time = time.time() - start_time
        
    elif name == 'Trapezoidal':
        traj_func, duration = generator()
        start_time = time.time()
        for _ in range(n_iterations):
            _ = traj_func(1.0)
        eval_time = time.time() - start_time
        
    else:  # Polynomial
        traj_func = generator()
        start_time = time.time()
        for _ in range(n_iterations):
            _ = traj_func(1.0)
        eval_time = time.time() - start_time
    
    results[name] = {'setup': setup_time, 'eval': eval_time}
    print(f"{name:>15}: Setup {setup_time*1000:.1f}ms, "
          f"Eval {eval_time*1000:.1f}ms")

print("\\n💡 All algorithms are highly optimized for real-time applications!")
```

## Summary

This tutorial covered:

✅ **S-curve trajectories** for smooth, jerk-limited motion  
✅ **Trapezoidal profiles** for simple, bounded-acceleration motion  
✅ **Polynomial trajectories** for precise boundary condition control  
✅ **Practical applications** in elevators, CNC machining, and robotics  
✅ **Algorithm selection** based on requirements  
✅ **Performance considerations** for real-time systems  

### Key Takeaways

1. **Choose based on smoothness needs**:
   - Maximum smoothness → S-curve
   - Good smoothness → Trapezoidal  
   - Precise control → Polynomial

2. **Consider constraints**: Jerk limiting is crucial for mechanical systems and passenger comfort

3. **Performance**: All algorithms are O(1) evaluation time - suitable for real-time control

4. **Boundary conditions**: Higher-order polynomials provide more control but may have unwanted oscillations

### Next Steps

- **[Quaternion Interpolation](../api-reference.md#quaternion-interpolation)**: Learn 3D rotation trajectories
- **[Path Planning](../api-reference.md#path-planning)**: Combine motion profiles with geometric paths
- **[API Reference](../api-reference.md)**: Complete function documentation