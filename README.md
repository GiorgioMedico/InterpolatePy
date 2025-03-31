# InterpolatePy

![Python](https://img.shields.io/badge/python-3.11+-blue)
[![pre-commit](https://github.com/GiorgioMedico/InterpolatePy/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/GiorgioMedico/InterpolatePy/actions/workflows/pre-commit.yml)
[![ci-test](https://github.com/GiorgioMedico/InterpolatePy/actions/workflows/test.yml/badge.svg)](https://github.com/GiorgioMedico/InterpolatePy/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Support the Project

If you find InterpolatePy useful for your work, please consider giving it a star on GitHub! ⭐

Your stars help make the project more visible to others who might benefit from these tools and encourage continued development and maintenance.

[![GitHub stars](https://img.shields.io/github/stars/GiorgioMedico/InterpolatePy.svg?style=social&label=Star&maxAge=2592000)](https://github.com/GiorgioMedico/InterpolatePy/stargazers/)

Have you implemented something cool with InterpolatePy? We'd love to hear about it in the Issues section or as a feature on our wiki!

## Table of Contents

- [InterpolatePy](#interpolatepy)
  - [Support the Project](#support-the-project)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Future Implementations](#future-implementations)
  - [Key Features](#key-features)
    - [Spline Interpolation](#spline-interpolation)
      - [B-Splines](#b-splines)
      - [Cubic Splines](#cubic-splines)
        - [Imposing Acceleration Constraints at Endpoints](#imposing-acceleration-constraints-at-endpoints)
    - [Motion Profiles](#motion-profiles)
    - [Path Generation](#path-generation)
    - [Utility Functions](#utility-functions)
  - [Installation](#installation)
    - [Using pip](#using-pip)
    - [From Source](#from-source)
    - [Optional Dependencies](#optional-dependencies)
  - [Usage Examples](#usage-examples)
    - [Cubic Spline Trajectory](#cubic-spline-trajectory)
    - [Cubic Spline with Acceleration Constraints](#cubic-spline-with-acceleration-constraints)
    - [Double-S Trajectory](#double-s-trajectory)
    - [B-Spline Curve](#b-spline-curve)
    - [Trapezoidal Trajectory with Waypoints](#trapezoidal-trajectory-with-waypoints)
    - [3D Path with Frenet Frames](#3d-path-with-frenet-frames)
  - [Mathematical Concepts](#mathematical-concepts)
    - [B-splines](#b-splines-1)
    - [Cubic Splines](#cubic-splines-1)
    - [Smoothing Splines](#smoothing-splines)
    - [Trapezoidal Velocity Profiles](#trapezoidal-velocity-profiles)
    - [Double-S Trajectories](#double-s-trajectories)
    - [Frenet Frames](#frenet-frames)
  - [Requirements](#requirements)
  - [Development](#development)
    - [Running Tests](#running-tests)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)
  - [Citation](#citation)

## Overview

InterpolatePy is a comprehensive Python library for generating smooth trajectories and curves with precise control over position, velocity, acceleration, and jerk profiles. Designed for robotics, motion planning, computer graphics, and scientific computing applications, it provides a wide range of interpolation techniques from simple linear interpolation to advanced B-splines and motion profiles.

Whether you need to generate smooth robotic joint motions, create path planning for autonomous vehicles, or design animation curves with specific dynamic properties, InterpolatePy offers the tools to create trajectories that maintain continuity while adhering to physical constraints.

## Future Implementations

InterpolatePy is continuously evolving, with several exciting features planned for future releases:

- [ ] **Bezier Curves**: Implementation of parametric Bezier curves with arbitrary degree control
- [ ] **Linear Interpolation with Polynomial Blend**: Enhanced linear interpolation using quintic Bezier curves for smooth transitions between line segments
- [ ] **Linear Interpolation with Parabolic Blends**: An alternative blending approach using parabolic segments for smooth connections
- [ ] **Spherical Path**: Tools for interpolation along spherical paths
- [ ] **LERP (Linear Interpolation)**: Linear interpolation functions for quaternions
- [ ] **SLERP (Spherical Linear Interpolation)**: Algorithms for smooth interpolation between orientations represented as quaternions
- [ ] **SQUAD (Spherical Quadrangle Interpolation)**: Advanced spherical interpolation with cubic-like smoothness for quaternion interpolation
- [ ] **B-spline Quaternion Curves**: Extension of B-spline techniques to quaternion space for smooth orientation interpolation

These implementations will significantly enhance InterpolatePy's capabilities for orientation interpolation and complex curve generation, particularly for robotic applications, character animation, and camera path planning.

## Key Features

### Spline Interpolation

#### B-Splines

- **BSpline**: Versatile implementation with customizable degree and knot vectors
- **ApproximationBSpline**: Efficiently approximates sets of points with a B-spline curve
- **CubicBSplineInterpolation**: Specialized cubic B-spline interpolation that passes through all points
- **BSplineInterpolator**: General B-spline interpolation with controllable continuity (C²-C⁴)
- **SmoothingCubicBSpline**: B-splines with adjustable smoothness-vs-accuracy tradeoff

#### Cubic Splines

- **CubicSpline**: Standard cubic spline with velocity constraints at endpoints
- **CubicSplineWithAcceleration1**: Cubic spline with velocity and acceleration constraints (extra points method)
- **CubicSplineWithAcceleration2**: Alternative cubic spline with acceleration constraints (quintic segments method)
- **CubicSmoothingSpline**: Cubic splines with μ parameter for smoothness control
- **SplineConfig/smoothing_spline_with_tolerance**: Tools for finding optimal smoothing parameters

##### Imposing Acceleration Constraints at Endpoints

InterpolatePy offers two distinct methods for implementing cubic splines with endpoint acceleration constraints:

1. **Extra Points Method (`CubicSplineWithAcceleration1`)**: This approach adds two extra points in the first and last segments to satisfy the acceleration constraints while maintaining C² continuity throughout the entire curve. The extra points are placed at the midpoints of the first and last segments, with positions calculated to ensure the specified accelerations at endpoints are achieved.

2. **Quintic Segments Method (`CubicSplineWithAcceleration2`)**: This approach uses standard cubic polynomials for interior segments, but replaces the first and last segments with quintic (5th degree) polynomials. The higher degree provides the additional degrees of freedom needed to satisfy the acceleration constraints at endpoints while maintaining overall C² continuity at all knot points.

### Motion Profiles

- **DoubleSTrajectory**: S-curve motion profile with bounded velocity, acceleration, and jerk
- **linear_traj**: Simple linear interpolation with constant velocity
- **PolynomialTrajectory**: Trajectory generation using polynomials of orders 3, 5, and 7
- **TrapezoidalTrajectory**: Trapezoidal velocity profiles with various constraint options

### Path Generation

- **LinearPath**: Simple linear paths with constant velocity
- **CircularPath**: Circular arcs and paths in 3D
- **Frenet Frames**: Tools for computing and visualizing Frenet frames along parametric curves

### Utility Functions

- **solve_tridiagonal**: Efficient tridiagonal matrix solver (Thomas algorithm)

## Installation

### Using pip

```bash
pip install InterpolatePy
```

### From Source

To install the latest development version with all dependencies:

```bash
# Clone the repository
git clone https://github.com/GiorgioMedico/InterpolatePy.git
cd InterpolatePy

# Install with development dependencies
pip install -e ".[all]"
```

### Optional Dependencies

You can install specific dependency groups:

```bash
# For testing dependencies only
pip install -e ".[test]"

# For development tools only
pip install -e ".[dev]"
```

## Usage Examples

### Cubic Spline Trajectory

Create a smooth trajectory through waypoints with velocity constraints:

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

### Cubic Spline with Acceleration Constraints

Create a smooth trajectory with both velocity and acceleration constraints at endpoints:

```python
from interpolatepy.c_s_with_acc2 import CubicSplineWithAcceleration2, SplineParameters

# Define waypoints
t_points = [0.0, 5.0, 7.0, 8.0, 10.0, 15.0, 18.0]
q_points = [3.0, -2.0, -5.0, 0.0, 6.0, 12.0, 8.0]

# Create parameters with velocity and acceleration constraints
params = SplineParameters(
    v0=2.0,    # Initial velocity
    vn=-3.0,   # Final velocity
    a0=0.0,    # Initial acceleration
    an=0.0     # Final acceleration
)

# Create spline with quintic segments at endpoints
spline = CubicSplineWithAcceleration2(t_points, q_points, params)

# Evaluate at specific time
position = spline.evaluate(6.0)
velocity = spline.evaluate_velocity(6.0)
acceleration = spline.evaluate_acceleration(6.0)

# Plot the trajectory
spline.plot()
```

### Double-S Trajectory

Generate a trajectory with bounded jerk for smooth motion profiles:

```python
from interpolatepy.double_s import DoubleSTrajectory, StateParams, TrajectoryBounds

# Create parameters for trajectory
state = StateParams(q_0=0.0, q_1=10.0, v_0=0.0, v_1=0.0)
bounds = TrajectoryBounds(v_bound=5.0, a_bound=10.0, j_bound=30.0)

# Create trajectory
trajectory = DoubleSTrajectory(state, bounds)

# Get trajectory information
duration = trajectory.get_duration()
phases = trajectory.get_phase_durations()

# Generate trajectory points
import numpy as np
time_points = np.linspace(0, duration, 100)
positions, velocities, accelerations, jerks = trajectory.evaluate(time_points)
```

### B-Spline Curve

Create and manipulate a B-spline curve with control points:

```python
import numpy as np
from interpolatepy.b_spline import BSpline

# Define control points, degree, and knot vector
control_points = np.array([[0, 0], [1, 2], [3, 1], [4, 0]])
degree = 3
knots = BSpline.create_uniform_knots(degree, len(control_points))

# Create B-spline
bspline = BSpline(degree, knots, control_points)

# Evaluate at parameter value
point = bspline.evaluate(0.5)

# Generate curve points for plotting
u_values, curve_points = bspline.generate_curve_points(100)

# Plot the curve with control polygon
bspline.plot_2d(show_control_polygon=True)
```

### Trapezoidal Trajectory with Waypoints

Generate a trajectory with trapezoidal velocity profile through multiple points:

```python
from interpolatepy.trapezoidal import TrapezoidalTrajectory, InterpolationParams

# Define waypoints
points = [0.0, 5.0, 3.0, 8.0, 2.0]

# Create interpolation parameters
params = InterpolationParams(
    points=points, 
    v0=0.0,         # Initial velocity
    vn=0.0,         # Final velocity 
    amax=10.0,      # Maximum acceleration
    vmax=5.0        # Maximum velocity
)

# Generate trajectory
traj_func, duration = TrapezoidalTrajectory.interpolate_waypoints(params)

# Evaluate at specific time
position, velocity, acceleration = traj_func(2.5)
```

### 3D Path with Frenet Frames

Create and visualize a trajectory with coordinate frames along the path:

```python
import numpy as np
import matplotlib.pyplot as plt
from interpolatepy.frenet_frame import (
    helicoidal_trajectory_with_derivatives,
    compute_trajectory_frames,
    plot_frames
)

# Create a helicoidal path
u_values = np.linspace(0, 4 * np.pi, 100)
def helix_func(u):
    return helicoidal_trajectory_with_derivatives(u, r=2.0, d=0.5)

# Compute Frenet frames along the path
points, frames = compute_trajectory_frames(helix_func, u_values)

# Visualize
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plot_frames(ax, points, frames, scale=0.5, skip=10)
plt.show()
```

## Mathematical Concepts

InterpolatePy implements several key mathematical concepts for trajectory generation:

### B-splines

Piecewise parametric curves defined by control points and a knot vector. B-splines offer local control (changes to a control point only affect the curve locally) and customizable continuity.

### Cubic Splines

Piecewise polynomials with C² continuity (continuous position, velocity, and acceleration) that interpolate a given set of points.

### Smoothing Splines

Splines with a controllable balance between accuracy (passing through points exactly) and smoothness (minimizing curvature). The μ parameter controls this tradeoff.

### Trapezoidal Velocity Profiles

Trajectories with linear segments of constant acceleration and velocity, creating a trapezoidal shape in the velocity profile.

### Double-S Trajectories

Motion profiles with bounded jerk, acceleration, and velocity, creating smooth S-curves in the acceleration profile. These are ideal for robotic motion to reduce stress on mechanical systems.

### Frenet Frames

Local coordinate systems defined by tangent, normal, and binormal vectors along a curve, useful for tool orientation and trajectory tracking.

## Requirements

- Python 3.10+
- NumPy 2.0.0+
- Matplotlib 3.10.1+
- SciPy 1.15.2+

## Development

InterpolatePy uses modern Python tooling for development:

- **Code Quality**: Black and isort for formatting, Ruff and mypy for linting and type checking
- **Testing**: pytest for unit tests and benchmarks

To set up the development environment:

```bash
pip install -e ".[all]"
pre-commit install
```

### Running Tests

```bash
python -m pytest tests
```

## Contributing

Contributions to InterpolatePy are welcome! To contribute:

1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Run tests to ensure they pass
5. Submit a pull request

Please follow the existing code style and include appropriate tests for new features.

## License

InterpolatePy is released under the MIT License. See the [LICENSE](https://github.com/GiorgioMedico/InterpolatePy/blob/main/LICENSE) file for more details.

## Acknowledgments

InterpolatePy implements algorithms and mathematical concepts primarily from the following authoritative textbooks:

- Biagiotti, L., & Melchiorri, C. (2008). *Trajectory Planning for Automatic Machines and Robots*. Springer.
- Siciliano, B., Sciavicco, L., Villani, L., & Oriolo, G. (2010). *Robotics: Modelling, Planning and Control*. Springer.

The library's implementation draws heavily from the theoretical frameworks, mathematical formulations, and algorithms presented in these works.

I express my gratitude to these authors for their significant contributions to the field of trajectory planning and robotics, which have made this library possible.

## Citation

If you use InterpolatePy in your research or project, please cite it as follows:

```bibtex
@software{InterpolatePy,
  author = {Medico, Giorgio},
  title = {InterpolatePy: A Comprehensive Python Library for Trajectory Planning and Interpolation},
  year = {2025},
  url = {https://github.com/GiorgioMedico/InterpolatePy}
}
```

For specific methods or algorithms implemented in InterpolatePy, please also consider citing the original research papers or textbooks referenced in the documentation and acknowledgments section.
