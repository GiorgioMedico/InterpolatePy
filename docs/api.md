# API Reference

This document provides a comprehensive reference to the classes and functions available in InterpolatePy.

## Table of Contents

- [API Reference](#api-reference)
  - [Table of Contents](#table-of-contents)
  - [Spline Interpolation](#spline-interpolation)
    - [B-Splines](#b-splines)
      - [`BSpline`](#bspline)
      - [`ApproximationBSpline`](#approximationbspline)
      - [`CubicBSplineInterpolation`](#cubicbsplineinterpolation)
      - [`BSplineInterpolator`](#bsplineinterpolator)
      - [`SmoothingCubicBSpline`](#smoothingcubicbspline)
    - [Cubic Splines](#cubic-splines)
      - [`CubicSpline`](#cubicspline)
      - [`CubicSplineWithAcceleration1`](#cubicsplinewithacceleration1)
      - [`CubicSplineWithAcceleration2`](#cubicsplinewithacceleration2)
    - [Smoothing Splines](#smoothing-splines)
      - [`CubicSmoothingSpline`](#cubicsmoothingspline)
      - [`smoothing_spline_with_tolerance`](#smoothing_spline_with_tolerance)
  - [Motion Profiles](#motion-profiles)
    - [Double-S Trajectory](#double-s-trajectory)
      - [`DoubleSTrajectory`](#doublestrajectory)
    - [Linear Trajectory](#linear-trajectory)
      - [`linear_traj`](#linear_traj)
    - [Polynomial Trajectories](#polynomial-trajectories)
      - [`PolynomialTrajectory`](#polynomialtrajectory)
    - [Trapezoidal Trajectory](#trapezoidal-trajectory)
      - [`TrapezoidalTrajectory`](#trapezoidaltrajectory)
  - [Path Generation](#path-generation)
    - [Linear Path](#linear-path)
      - [`LinearPath`](#linearpath)
    - [Circular Path](#circular-path)
      - [`CircularPath`](#circularpath)
    - [Frenet Frames](#frenet-frames)
      - [`compute_trajectory_frames`](#compute_trajectory_frames)
      - [`plot_frames`](#plot_frames)
  - [Utilities](#utilities)
    - [Tridiagonal Solver](#tridiagonal-solver)
      - [`solve_tridiagonal`](#solve_tridiagonal)

## Spline Interpolation

### B-Splines

#### `BSpline`

```python
class BSpline:
    def __init__(self, degree, knots, control_points)
```

A class for representing and evaluating B-spline curves.

**Parameters:**
- `degree` (int): The degree of the B-spline
- `knots` (array_like): The knot vector
- `control_points` (array_like): The control points defining the B-spline

**Methods:**
- `evaluate(u)`: Evaluate the B-spline curve at parameter value u
- `evaluate_derivative(u, order=1)`: Evaluate the derivative of the B-spline curve
- `basis_functions(u, span_index)`: Calculate all non-zero basis functions at parameter u
- `find_knot_span(u)`: Find the knot span index for a given parameter value u
- `plot_2d(num_points=100, show_control_polygon=True, show_knots=False, ax=None)`: Plot a 2D B-spline curve
- `plot_3d(num_points=100, show_control_polygon=True, ax=None)`: Plot a 3D B-spline curve

**Static Methods:**
- `create_uniform_knots(degree, num_control_points, domain_min=0.0, domain_max=1.0)`: Create a uniform knot vector
- `create_periodic_knots(degree, num_control_points, domain_min=0.0, domain_max=1.0)`: Create a periodic knot vector

#### `ApproximationBSpline`

```python
class ApproximationBSpline(BSpline):
    def __init__(self, points, num_control_points, *, degree=3, weights=None, method="chord_length", debug=False)
```

A class for B-spline curve approximation of a set of points. Inherits from BSpline class.

**Parameters:**
- `points` (array_like): The points to approximate
- `num_control_points` (int): The number of control points to use
- `degree` (int, optional): The degree of the B-spline. Default is 3
- `weights` (array_like, optional): Weights for each data point
- `method` (str, optional): Method for parameter calculation ('equally_spaced', 'chord_length', or 'centripetal')
- `debug` (bool, optional): Whether to print debug information

**Methods:**
- `calculate_approximation_error(points=None, u_bar=None)`: Calculate the approximation error
- `refine(max_error=0.1, max_control_points=100)`: Refine the approximation by adding more control points

#### `CubicBSplineInterpolation`

```python
class CubicBSplineInterpolation(BSpline):
    def __init__(self, points, v0=None, vn=None, method="chord_length", auto_derivatives=False)
```

A class for cubic B-spline interpolation of a set of points.

**Parameters:**
- `points` (array_like): The points to interpolate
- `v0` (array_like, optional): Initial endpoint derivative vector
- `vn` (array_like, optional): Final endpoint derivative vector
- `method` (str, optional): Method for parameter calculation
- `auto_derivatives` (bool, optional): Whether to automatically calculate derivatives

#### `BSplineInterpolator`

```python
class BSplineInterpolator(BSpline):
    def __init__(self, degree, points, times=None, initial_velocity=None, final_velocity=None, 
                 initial_acceleration=None, final_acceleration=None, cyclic=False)
```

A B-spline that interpolates a set of points with specified degrees of continuity.

**Parameters:**
- `degree` (int): The degree of the B-spline (3, 4, or 5)
- `points` (array_like): The points to be interpolated
- `times` (array_like, optional): The time instants for each point
- `initial_velocity` (array_like, optional): Initial velocity constraint
- `final_velocity` (array_like, optional): Final velocity constraint
- `initial_acceleration` (array_like, optional): Initial acceleration constraint
- `final_acceleration` (array_like, optional): Final acceleration constraint
- `cyclic` (bool, optional): Whether to use cyclic (periodic) conditions

**Methods:**
- `plot_with_points(num_points=100, show_control_polygon=True, ax=None)`: Plot the 2D B-spline curve with interpolation points
- `plot_with_points_3d(num_points=100, show_control_polygon=True, ax=None)`: Plot the 3D B-spline curve with interpolation points

#### `SmoothingCubicBSpline`

```python
class SmoothingCubicBSpline(BSpline):
    def __init__(self, points, params=None)
```

A class for creating smoothing cubic B-splines that approximate a set of points.

**Parameters:**
- `points` (array_like): The points to approximate
- `params` (BSplineParams, optional): Configuration parameters

**Methods:**
- `calculate_approximation_error()`: Calculate the approximation error for each point
- `calculate_total_error()`: Calculate the total weighted approximation error
- `calculate_smoothness_measure(num_points=100)`: Calculate the smoothness measure

### Cubic Splines

#### `CubicSpline`

```python
class CubicSpline:
    def __init__(self, t_points, q_points, v0=0.0, vn=0.0, debug=False)
```

Cubic spline trajectory planning implementation.

**Parameters:**
- `t_points` (array_like): List or array of time points (t0, t1, ..., tn)
- `q_points` (array_like): List or array of position points (q0, q1, ..., qn)
- `v0` (float, optional): Initial velocity at t0. Default is 0.0
- `vn` (float, optional): Final velocity at tn. Default is 0.0
- `debug` (bool, optional): Whether to print debug information. Default is False

**Methods:**
- `evaluate(t)`: Evaluate the spline at time t
- `evaluate_velocity(t)`: Evaluate the velocity at time t
- `evaluate_acceleration(t)`: Evaluate the acceleration at time t
- `plot(num_points=1000)`: Plot the spline trajectory with velocity and acceleration profiles

#### `CubicSplineWithAcceleration1`

```python
class CubicSplineWithAcceleration1:
    def __init__(self, t_points, q_points, v0=0.0, vn=0.0, a0=0.0, an=0.0, debug=False)
```

Cubic spline trajectory planning with both velocity and acceleration constraints at endpoints using extra points method.

**Parameters:**
- `t_points` (array_like): Time points
- `q_points` (array_like): Position points
- `v0` (float, optional): Initial velocity
- `vn` (float, optional): Final velocity
- `a0` (float, optional): Initial acceleration
- `an` (float, optional): Final acceleration
- `debug` (bool, optional): Whether to print debug information

**Methods:**
- `evaluate(t)`: Evaluate the spline at time t
- `evaluate_velocity(t)`: Evaluate the velocity at time t
- `evaluate_acceleration(t)`: Evaluate the acceleration at time t
- `plot(num_points=1000)`: Plot the trajectory with velocity and acceleration profiles

#### `CubicSplineWithAcceleration2`

```python
class CubicSplineWithAcceleration2(CubicSpline):
    def __init__(self, t_points, q_points, params=None)
```

Cubic spline trajectory planning with initial and final acceleration constraints using quintic segments method.

**Parameters:**
- `t_points` (array_like): Array of time points
- `q_points` (array_like): Array of position points
- `params` (SplineParameters, optional): Spline parameters including initial/final velocities and accelerations

**Methods:**
- Inherits methods from CubicSpline plus specialized versions for handling quintic segments

### Smoothing Splines

#### `CubicSmoothingSpline`

```python
class CubicSmoothingSpline:
    def __init__(self, t_points, q_points, mu=0.5, weights=None, v0=0.0, vn=0.0, debug=False)
```

Cubic smoothing spline trajectory planning with control over the smoothness versus waypoint accuracy trade-off.

**Parameters:**
- `t_points` (array_like): Time points for the spline knots
- `q_points` (array_like): Position points at each time point
- `mu` (float, optional): Trade-off parameter between accuracy (μ=1) and smoothness (μ=0)
- `weights` (array_like, optional): Individual point weights
- `v0` (float, optional): Initial velocity constraint. Default is 0.0
- `vn` (float, optional): Final velocity constraint. Default is 0.0
- `debug` (bool, optional): Whether to print debug information. Default is False

**Methods:**
- `evaluate(t)`: Evaluate the spline at time t
- `evaluate_velocity(t)`: Evaluate the velocity at time t
- `evaluate_acceleration(t)`: Evaluate the acceleration at time t
- `plot(num_points=1000)`: Plot the spline trajectory

#### `smoothing_spline_with_tolerance`

```python
def smoothing_spline_with_tolerance(t_points, q_points, tolerance, config)
```

Find a cubic smoothing spline with a maximum approximation error smaller than a given tolerance.

**Parameters:**
- `t_points` (array_like): Time points
- `q_points` (array_like): Position points
- `tolerance` (float): Maximum allowed approximation error
- `config` (SplineConfig): Configuration object with optional parameters

**Returns:**
- `spline` (CubicSmoothingSpline): The final CubicSmoothingSpline object
- `mu` (float): The found value of μ parameter
- `e_max` (float): The maximum approximation error achieved
- `iterations` (int): Number of iterations performed

## Motion Profiles

### Double-S Trajectory

#### `DoubleSTrajectory`

```python
class DoubleSTrajectory:
    def __init__(self, state_params, bounds)
```

Double S-Trajectory Planner Class that generates smooth motion profiles with bounded velocity, acceleration, and jerk.

**Parameters:**
- `state_params` (StateParams): Start and end states for trajectory planning
- `bounds` (TrajectoryBounds): Velocity, acceleration, and jerk bounds for trajectory planning

**Methods:**
- `evaluate(t)`: Evaluate the double-S trajectory at time t (position, velocity, acceleration, jerk)
- `get_duration()`: Returns the total duration of the trajectory
- `get_phase_durations()`: Returns the durations of each phase in the trajectory

**Static Methods:**
- `create_trajectory(state_params, bounds)`: Static factory method to create a trajectory function and return its duration

### Linear Trajectory

#### `linear_traj`

```python
def linear_traj(p0, p1, t0, t1, time_array)
```

Generate points along a linear trajectory using NumPy vectorization.

**Parameters:**
- `p0` (float or array_like): Starting position
- `p1` (float or array_like): Ending position
- `t0` (float): Start time of the trajectory
- `t1` (float): End time of the trajectory
- `time_array` (array_like): Array of time points at which to calculate the trajectory

**Returns:**
- `positions` (array_like): Array of positions at each time point
- `velocities` (array_like): Constant velocity at each time point
- `accelerations` (array_like): Zero acceleration at each time point

### Polynomial Trajectories

#### `PolynomialTrajectory`

```python
class PolynomialTrajectory:
    @staticmethod
    def order_3_trajectory(initial, final, time)
    
    @staticmethod
    def order_5_trajectory(initial, final, time)
    
    @staticmethod
    def order_7_trajectory(initial, final, time)
    
    @classmethod
    def multipoint_trajectory(cls, params)
```

A class for generating polynomial trajectories with specified boundary conditions.

**Static Methods:**
- `order_3_trajectory(initial, final, time)`: Generate a 3rd order polynomial trajectory
- `order_5_trajectory(initial, final, time)`: Generate a 5th order polynomial trajectory
- `order_7_trajectory(initial, final, time)`: Generate a 7th order polynomial trajectory
- `heuristic_velocities(points, times)`: Compute intermediate velocities using the heuristic rule

**Class Methods:**
- `multipoint_trajectory(params)`: Generate a trajectory through a sequence of points with specified times

### Trapezoidal Trajectory

#### `TrapezoidalTrajectory`

```python
class TrapezoidalTrajectory:
    @staticmethod
    def generate_trajectory(params)
    
    @classmethod
    def interpolate_waypoints(cls, params)
```

Generate trapezoidal velocity profiles for trajectory planning.

**Static Methods:**
- `generate_trajectory(params)`: Generate a trapezoidal trajectory with non-null initial and final velocities
- `calculate_heuristic_velocities(q_list, v0, vn, v_max=None, amax=None)`: Calculate velocities based on height differences

**Class Methods:**
- `interpolate_waypoints(params)`: Generate a trajectory through a sequence of points using trapezoidal velocity profiles

## Path Generation

### Linear Path

#### `LinearPath`

```python
class LinearPath:
    def __init__(self, pi, pf)
```

Initialize a linear path from point pi to point pf.

**Parameters:**
- `pi` (array_like): Initial point coordinates [x, y, z]
- `pf` (array_like): Final point coordinates [x, y, z]

**Methods:**
- `position(s)`: Calculate position at arc length s
- `velocity(s=None)`: Calculate first derivative with respect to arc length
- `acceleration(s=None)`: Calculate second derivative with respect to arc length
- `evaluate_at(s_values)`: Evaluate position, velocity, and acceleration at specific arc length values
- `all_traj(num_points=100)`: Generate a complete trajectory along the entire linear path

### Circular Path

#### `CircularPath`

```python
class CircularPath:
    def __init__(self, r, d, pi)
```

Initialize a circular path.

**Parameters:**
- `r` (array_like): Unit vector of circle axis
- `d` (array_like): Position vector of a point on the circle axis
- `pi` (array_like): Position vector of a point on the circle

**Methods:**
- `position(s)`: Calculate position at arc length s
- `velocity(s)`: Calculate first derivative with respect to arc length
- `acceleration(s)`: Calculate second derivative with respect to arc length
- `evaluate_at(s_values)`: Evaluate position, velocity, and acceleration at specific arc length values
- `all_traj(num_points=100)`: Generate a complete trajectory around the entire circular path

### Frenet Frames

#### `compute_trajectory_frames`

```python
def compute_trajectory_frames(position_func, u_values, tool_orientation=None)
```

Compute Frenet frames along a parametric curve and optionally apply tool orientation.

**Parameters:**
- `position_func` (callable): A function that takes a parameter u and returns position and derivatives
- `u_values` (array_like): Parameter values at which to compute the frames
- `tool_orientation` (float or tuple, optional): Angle or angles to rotate the frames

**Returns:**
- `points` (array_like): Points on the curve
- `frames` (array_like): Frames at each point

#### `plot_frames`

```python
def plot_frames(ax, points, frames, scale=0.5, skip=5, colors=None)
```

Plot the trajectory and frames.

**Parameters:**
- `ax` (matplotlib.axes.Axes): The 3D axes to plot on
- `points` (array_like): Points on the trajectory
- `frames` (array_like): Frames at each point
- `scale` (float, optional): Scale factor for the frame vectors
- `skip` (int, optional): Number of frames to skip between plotted frames
- `colors` (list of str, optional): Colors for the three vectors

## Utilities

### Tridiagonal Solver

#### `solve_tridiagonal`

```python
def solve_tridiagonal(lower_diagonal, main_diagonal, upper_diagonal, right_hand_side)
```

Solve a tridiagonal system using the Thomas algorithm.

**Parameters:**
- `lower_diagonal` (array_like): Lower diagonal elements (first element is not used)
- `main_diagonal` (array_like): Main diagonal elements
- `upper_diagonal` (array_like): Upper diagonal elements (last element is not used)
- `right_hand_side` (array_like): Right-hand side vector of the equation

**Returns:**
- Solution vector x
