# InterpolatePy

![Python](https://img.shields.io/badge/python-3.10+-blue)
[![pre-commit](https://github.com/GiorgioMedico/InterpolatePy/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/GiorgioMedico/InterpolatePy/actions/workflows/pre-commit.yml)
[![ci-test](https://github.com/GiorgioMedico/InterpolatePy/actions/workflows/test.yml/badge.svg)](https://github.com/GiorgioMedico/InterpolatePy/actions/workflows/test.yml)

## Overview

InterpolatePy provides a collection of algorithms for generating smooth trajectories and curves with precise control over position, velocity, acceleration, and jerk. The library implements numerous interpolation techniques ranging from simple linear interpolation to advanced B-splines and motion profiles.

## Key Features

### Spline Interpolation

#### B-Splines
- **BSpline**: Base implementation of B-splines with customizable degree and knot vectors
- **ApproximationBSpline**: B-spline curve approximation of sets of points
- **CubicBSplineInterpolation**: Specialized cubic B-spline interpolation
- **BSplineInterpolator**: General B-spline interpolation with controllable continuity
- **SmoothingCubicBSpline**: B-splines with smoothness-vs-accuracy tradeoff

#### Cubic Splines
- **CubicSpline**: Basic cubic spline with velocity constraints
- **CubicSplineWithAcceleration1**: Cubic spline with velocity and acceleration constraints (extra points method)
- **CubicSplineWithAcceleration2**: Alternative cubic spline with acceleration constraints (quintic segments method)
- **CubicSmoothingSpline**: Cubic splines with μ parameter for smoothness control
- **SplineConfig/smoothing_spline_with_tolerance**: Tools for finding optimal smoothing parameters

### Motion Profiles

- **DoubleSTrajectory**: S-curve motion profile with bounded velocity, acceleration, and jerk
- **linear_traj**: Linear interpolation with constant velocity
- **PolynomialTrajectory**: Trajectory generation using polynomials of orders 3, 5, and 7
- **TrapezoidalTrajectory**: Trapezoidal velocity profiles with various constraints

### Path Generation

- **LinearPath**: Simple linear paths with constant velocity
- **CircularPath**: Circular arcs and paths
- **Frenet Frames**: Tools for computing and visualizing Frenet frames along parametric curves

### Utility Functions

- **solve_tridiagonal**: Efficient tridiagonal matrix solver (Thomas algorithm)

## Installation

```bash
# Install directly from PyPI
pip install interpolatepy

# Install from source with development dependencies
pip install -e ".[all]"
```

## Usage Examples

### Creating a Cubic Spline Trajectory

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

# Plot the trajectory
spline.plot()
```

### Generating a Double-S Trajectory

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

### Creating a B-Spline Curve

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

# Plot the curve
bspline.plot_2d(show_control_polygon=True)
```

### Trapezoidal Trajectory with Waypoints

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

## Mathematical Concepts

The library implements several key mathematical concepts for trajectory generation:

- **B-splines**: Parametric curves defined by control points and a knot vector, offering local control and customizable continuity
- **Cubic splines**: Piecewise polynomials with C2 continuity (continuous position, velocity, and acceleration)
- **Smoothing splines**: Splines with a controllable balance between accuracy and smoothness
- **Trapezoidal velocity profiles**: Trajectories with linear segments of constant acceleration and velocity
- **Double-S trajectories**: Motions with bounded jerk, acceleration, and velocity, creating smooth S-curves
- **Frenet frames**: Local coordinate systems defined by tangent, normal, and binormal vectors along a curve

## Requirements

- Python 3.10+
- NumPy 2.0.0+
- Matplotlib 3.10.1+
- SciPy 1.15.2+

## Development

InterpolatePy uses modern Python tools for development:

- Black and isort for code formatting
- Ruff and mypy for linting and type checking
- pytest for testing
- mkdocs for documentation

To set up the development environment:

```bash
pip install -e ".[all]"
pre-commit install
```

## License

MIT License
