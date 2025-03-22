# InterpolatePy

InterpolatePy is a comprehensive Python library for generating smooth trajectories and curves with precise control over position, velocity, acceleration, and jerk profiles. Designed for robotics, motion planning, computer graphics, and scientific computing applications, it provides a wide range of interpolation techniques from simple linear interpolation to advanced B-splines and motion profiles.

## Key Features

### Spline Interpolation

- **B-Splines**: Versatile implementation with customizable degree and knot vectors
- **Cubic Splines**: Standard cubic spline with velocity constraints at endpoints
- **Smoothing Splines**: Cubic splines with μ parameter for smoothness control
- **Acceleration Constraints**: Two distinct methods for implementing cubic splines with endpoint acceleration constraints

### Motion Profiles

- **Double-S Trajectory**: S-curve motion profile with bounded velocity, acceleration, and jerk
- **Linear Trajectory**: Simple linear interpolation with constant velocity
- **Polynomial Trajectory**: Trajectory generation using polynomials of orders 3, 5, and 7
- **Trapezoidal Trajectory**: Trapezoidal velocity profiles with various constraint options

### Path Generation

- **Linear Path**: Simple linear paths with constant velocity
- **Circular Path**: Circular arcs and paths in 3D
- **Frenet Frames**: Tools for computing and visualizing Frenet frames along parametric curves

## Installation

```bash
# Clone the repository
git clone https://github.com/GiorgioMedico/InterpolatePy.git
cd InterpolatePy

# Install with all dependencies
pip install -e ".[all]"
```

## Quick Start

Here's a simple example to create a cubic spline trajectory:

```python
from interpolatepy.cubic_spline import CubicSpline

# Define waypoints
t_points = [0.0, 5.0, 7.0, 8.0, 10.0, 15.0, 18.0]
q_points = [3.0, -2.0, -5.0, 0.0, 6.0, 12.0, 8.0]

# Create cubic spline with initial and final velocities
spline = CubicSpline(t_points, q_points, v0=2.0, vn=-3.0)

# Evaluate at specific time
position = spline.evaluate(6.0)
velocity = spline.evaluate_velocity(6.0)
acceleration = spline.evaluate_acceleration(6.0)

# Plot the trajectory with position, velocity, and acceleration profiles
spline.plot()
```

For more advanced examples, see the [API documentation](api.md) or explore the examples directory in the repository.

## Requirements

- Python 3.10+
- NumPy 2.0.0+
- Matplotlib 3.10.1+
- SciPy 1.15.2+

## License

InterpolatePy is released under the MIT License. See the [LICENSE](https://github.com/GiorgioMedico/InterpolatePy/blob/main/LICENSE) file for more details.
