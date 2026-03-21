# Quaternion Interpolation Tutorial

Quaternions provide singularity-free rotation interpolation for 3D orientations. This tutorial covers InterpolatePy's quaternion algorithms, from basic SLERP to C²-continuous rotation splines.

## Why Quaternions?

Euler angles suffer from gimbal lock, and rotation matrices are redundant (9 numbers for 3 DOF). Quaternions are:

- **Compact**: 4 numbers represent any 3D rotation
- **Singularity-free**: No gimbal lock
- **Smooth interpolation**: SLERP gives constant angular velocity
- **Numerically stable**: Easy to normalize

## Basic Quaternion Operations

The `Quaternion` class provides core operations:

```python
from interpolatepy import Quaternion
import numpy as np

# Create quaternions
q_identity = Quaternion.identity()                              # No rotation
q_from_axis = Quaternion.from_angle_axis(np.pi/2, np.array([0, 0, 1]))  # 90 deg about Z
q_from_euler = Quaternion.from_euler_angles(0.3, 0.5, 0.1)    # roll, pitch, yaw

# Operations
q_product = q_from_axis * q_from_euler        # Compose rotations
q_inverse = q_from_axis.inverse()              # Inverse rotation
q_normalized = q_from_axis.unit()              # Normalize

# Convert back
axis, angle = q_from_axis.to_axis_angle()
roll, pitch, yaw = q_from_euler.to_euler_angles()

# Access components
scalar = q_from_axis.s_          # Scalar part
vector = q_from_axis.v_          # Vector part (numpy array of length 3)
rotation_matrix = q_from_axis.R()  # 3x3 rotation matrix
```

## SLERP: Two-Point Interpolation

SLERP (Spherical Linear Interpolation) smoothly interpolates between two orientations at constant angular velocity:

```python
from interpolatepy import Quaternion
import numpy as np

# Two orientations
q_start = Quaternion.identity()
q_end = Quaternion.from_angle_axis(np.pi / 2, np.array([0, 0, 1]))

# Interpolate at parameter t in [0, 1]
q_mid = q_start.slerp(q_end, 0.5)   # Halfway rotation (45 deg about Z)
q_quarter = q_start.slerp(q_end, 0.25)  # Quarter way

print(f"Start: {q_start.to_euler_angles()}")
print(f"Quarter: {q_quarter.to_euler_angles()}")
print(f"Mid: {q_mid.to_euler_angles()}")
print(f"End: {q_end.to_euler_angles()}")
```

SLERP is ideal for interpolating between exactly two orientations. For multiple waypoints, use one of the spline methods below.

## QuaternionSpline: Multi-Point Interpolation

`QuaternionSpline` handles multiple orientation waypoints with automatic method selection:

```python
from interpolatepy import QuaternionSpline, Quaternion
import numpy as np

# Define orientation keyframes
times = [0.0, 1.0, 2.0, 3.0, 4.0]
orientations = [
    Quaternion.identity(),
    Quaternion.from_euler_angles(0.3, 0.2, 0.0),
    Quaternion.from_euler_angles(0.6, 0.5, 0.4),
    Quaternion.from_euler_angles(0.2, 0.8, 0.7),
    Quaternion.from_euler_angles(0.0, 0.4, 1.0),
]

# Create spline (auto-selects SQUAD for 4+ points, SLERP for 2)
quat_spline = QuaternionSpline(times, orientations, interpolation_method="auto")

# Evaluate at any time
q, segment = quat_spline.interpolate_at_time(2.5)
print(f"Orientation at t=2.5: euler={q.to_euler_angles()}")

# With angular velocity
q, angular_velocity, segment = quat_spline.interpolate_with_velocity(2.5)
```

## SquadC2: C²-Continuous Rotation Splines

`SquadC2` is the most advanced quaternion interpolator. It provides:

- **C² continuity**: Smooth angular velocity and acceleration
- **Zero-clamped boundaries**: Zero angular velocity/acceleration at endpoints
- **Non-uniform timing**: Supports variable segment durations

```python
from interpolatepy import SquadC2, Quaternion
import numpy as np
import matplotlib.pyplot as plt

# Define rotation keyframes
times = [0.0, 1.0, 2.0, 3.0, 4.0]
orientations = [
    Quaternion.identity(),
    Quaternion.from_angle_axis(np.pi / 4, np.array([1, 0, 0])),
    Quaternion.from_angle_axis(np.pi / 2, np.array([0, 1, 0])),
    Quaternion.from_angle_axis(np.pi / 3, np.array([0, 0, 1])),
    Quaternion.identity(),
]

# Create C² continuous spline
squad = SquadC2(times, orientations)

# Evaluate trajectory
t_eval = np.linspace(0, 4, 200)
euler_angles = []
angular_velocities = []

for t in t_eval:
    q = squad.evaluate(t)
    omega = squad.evaluate_velocity(t)
    euler_angles.append(q.to_euler_angles())
    angular_velocities.append(omega)

euler_angles = np.array(euler_angles)
angular_velocities = np.array(angular_velocities)

# Plot orientation and angular velocity
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

for i, label in enumerate(["Roll", "Pitch", "Yaw"]):
    axes[0].plot(t_eval, np.degrees(euler_angles[:, i]), label=label)
axes[0].set_ylabel("Angle (deg)")
axes[0].set_title("SquadC2: Orientation")
axes[0].legend()
axes[0].grid(True)

for i, label in enumerate(["wx", "wy", "wz"]):
    axes[1].plot(t_eval, angular_velocities[:, i], label=label)
axes[1].set_ylabel("Angular velocity (rad/s)")
axes[1].set_xlabel("Time (s)")
axes[1].set_title("SquadC2: Angular Velocity (smooth, zero at endpoints)")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
```

## Log-Quaternion Interpolation

`LogQuaternionInterpolation` maps quaternions to 3D logarithmic space, applies cubic B-spline interpolation there, and maps back. This provides smooth, continuously differentiable rotational motion:

```python
from interpolatepy import LogQuaternionInterpolation, Quaternion
import numpy as np

times = [0.0, 1.0, 2.0, 3.0]
quaternions = [
    Quaternion.identity(),
    Quaternion.from_angle_axis(np.pi / 4, np.array([1, 0, 0])),
    Quaternion.from_angle_axis(np.pi / 2, np.array([0, 1, 0])),
    Quaternion.from_angle_axis(np.pi / 3, np.array([0, 0, 1])),
]

log_interp = LogQuaternionInterpolation(times, quaternions)

# Evaluate
q = log_interp.evaluate(1.5)
omega = log_interp.evaluate_velocity(1.5)
alpha = log_interp.evaluate_acceleration(1.5)
```

The `ModifiedLogQuaternionInterpolation` variant decouples angle and axis interpolation, which can give better behavior near 180-degree rotations.

## Choosing the Right Method

| Method | Continuity | Best For |
|--------|-----------|----------|
| `Quaternion.slerp()` | C⁰ | Two orientations, constant angular velocity |
| `QuaternionSpline` (SQUAD) | C¹ | Multiple waypoints, general use |
| `SquadC2` | C² | Robotics, smooth angular acceleration needed |
| `LogQuaternionInterpolation` | C² | Large rotations, logarithmic space benefits |

## Common Pitfalls

### Double Cover

Quaternions `q` and `-q` represent the same rotation. InterpolatePy handles this automatically in `SquadC2`, but be aware when constructing quaternions manually:

```python
# These represent the SAME rotation
q1 = Quaternion(1, 0, 0, 0)
q2 = Quaternion(-1, 0, 0, 0)
```

### Near-180-Degree Rotations

Interpolation near 180-degree rotations can be problematic because the interpolation path becomes ambiguous. `SquadC2` handles this better than basic SLERP.

## Next Steps

- **[API Reference](../api-reference.md#quaternion)** for complete method documentation
- **[Algorithms Guide](../algorithms.md)** for mathematical foundations
- **Example scripts**: `examples/squad_c2_ex.py`, `examples/log_quat_ex.py`, `examples/quat_visualization_ex.py`
