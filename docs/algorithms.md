# Algorithm Reference

This comprehensive reference covers the mathematical foundations, implementation details, and theoretical background for all trajectory planning and interpolation algorithms in InterpolatePy.

## Overview

InterpolatePy implements state-of-the-art trajectory planning algorithms based on mathematical optimization and approximation theory. All algorithms provide:

- **Smoothness Guarantees**: C⁰, C¹, or C² continuity as specified
- **Boundary Condition Support**: Position, velocity, and acceleration constraints
- **Efficient Evaluation**: Optimized polynomial and spline evaluation
- **Comprehensive API**: Position, velocity, acceleration, and jerk evaluation
- **Visualization Support**: Built-in plotting for analysis and debugging

### Mathematical Notation

Throughout this document, we use the following conventions:

- **q(t)**: Position trajectory as a function of time
- **q̇(t)**: Velocity (first derivative)  
- **q̈(t)**: Acceleration (second derivative)
- **q⃛(t)**: Jerk (third derivative)
- **C^n**: n-times continuously differentiable
- **[t₀, tₙ]**: Time domain interval
- **ωᵢ**: Acceleration values at waypoints (spline notation)

## Algorithm Categories

InterpolatePy organizes algorithms into five main categories:

### 1. Spline Interpolation
Advanced piecewise polynomial methods with global smoothness constraints.

### 2. Motion Profiles  
Classical trajectory profiles optimized for robotics and automation.

### 3. Polynomial Trajectories
Direct polynomial fitting with boundary conditions.

### 4. Quaternion Interpolation
Specialized methods for smooth rotation interpolation.

### 5. Specialized Algorithms
Path-following, frame computation, and geometric utilities.

---

## Spline Interpolation

Spline methods construct piecewise polynomials that maintain continuity across segment boundaries. InterpolatePy implements cubic splines (C² continuous) and B-splines with various constraint handling approaches.

### Cubic Spline

**Class**: [`CubicSpline`](api-reference.md#cubic-spline)  
**File**: `cubic_spline.py`

#### Mathematical Theory

Implements natural cubic spline interpolation with configurable boundary conditions. Given waypoints {(tᵢ, qᵢ)}ᵢ₌₀ⁿ, constructs piecewise cubic polynomials:

$$q_k(t) = a_{k0} + a_{k1}(t-t_k) + a_{k2}(t-t_k)^2 + a_{k3}(t-t_k)^3$$

for $t \in [t_k, t_{k+1}]$. The spline satisfies:

- **C² continuity**: Position, velocity, and acceleration continuous at interior points
- **Interpolation**: $q_k(t_i) = q_i$ for all waypoints
- **Boundary conditions**: Configurable velocity/acceleration at endpoints

The algorithm solves a tridiagonal linear system for accelerations $\omega_i = \ddot{q}(t_i)$:

$$A\omega = c$$

where A is tridiagonal with entries based on time intervals $T_i = t_{i+1} - t_i$.

#### System Matrix Construction

The tridiagonal system has the form:

$$\begin{bmatrix}
1 & 0 & 0 & \cdots & 0 \\
T_0 & 2(T_0+T_1) & T_1 & \cdots & 0 \\
0 & T_1 & 2(T_1+T_2) & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & 1
\end{bmatrix}
\begin{bmatrix}
\omega_0 \\ \omega_1 \\ \omega_2 \\ \vdots \\ \omega_n
\end{bmatrix}
=
\begin{bmatrix}
bc_0 \\ c_1 \\ c_2 \\ \vdots \\ bc_n
\end{bmatrix}$$

Where:
- $bc_0$, $bc_n$ are boundary condition values
- $c_i = 6\left(\frac{q_{i+1}-q_i}{T_i} - \frac{q_i-q_{i-1}}{T_{i-1}}\right)$ for interior points

#### Boundary Conditions

**Natural Spline** (default):
- $\omega_0 = 0$ (zero curvature at start)
- $\omega_n = 0$ (zero curvature at end)

**Clamped Spline**:
- $\omega_0 = \frac{6}{T_0}\left(\frac{q_1-q_0}{T_0} - v_0\right)$
- $\omega_n = \frac{6}{T_{n-1}}\left(v_n - \frac{q_n-q_{n-1}}{T_{n-1}}\right)$

#### Performance Characteristics

- **Setup**: O(n) - Solve tridiagonal system
- **Evaluation**: O(log n) - Binary search + polynomial evaluation
- **Memory**: O(n) - Store coefficients for each segment
- **Smoothness**: C² continuous

!!! example "Cubic Spline Example"
    ```python
    from interpolatepy import CubicSpline
    import numpy as np
    
    # Define waypoints
    t_points = [0, 1, 2, 3, 4]
    q_points = [0, 1, 0, -1, 0]
    
    # Create spline with velocity constraints
    spline = CubicSpline(t_points, q_points, v0=0.0, vn=0.0)
    
    # Evaluate at specific time
    t = 1.5
    position = spline.evaluate(t)
    velocity = spline.evaluate_velocity(t)
    acceleration = spline.evaluate_acceleration(t)
    
    print(f"At t={t}: pos={position:.3f}, vel={velocity:.3f}, acc={acceleration:.3f}")
    ```

### Cubic Smoothing Spline

**Class**: [`CubicSmoothingSpline`](api-reference.md#cubic-smoothing-spline)  
**File**: `c_s_smoothing.py`

#### Mathematical Theory

Implements cubic smoothing splines that balance interpolation accuracy with smoothness. Instead of exact interpolation, minimizes the objective function:

$$J(q) = \mu\sum_{i=0}^n(q_i - \hat{q}_i)^2 + \int_{t_0}^{t_n} (\ddot{q}(t))^2 dt$$

where:
- **μ**: Smoothing parameter (0 ≤ μ ≤ ∞)
- **$\hat{q}_i$**: Target positions at waypoints
- **First term**: Penalizes deviation from waypoints (fidelity)
- **Second term**: Penalizes curvature (smoothness)

#### Parameter Selection

Special cases:
- **μ → 0**: Natural cubic spline (exact interpolation)
- **μ → ∞**: Linear least squares fit
- **Optimal μ**: Balance between fidelity and smoothness

The modified tridiagonal system becomes:

$$(\mathbf{A} + \mu\mathbf{I})\boldsymbol{\omega} = \mathbf{c}$$

where $\mathbf{I}$ is the identity matrix scaled by the smoothing parameter.

#### Automatic Parameter Selection

InterpolatePy provides `smoothing_spline_with_tolerance()` which uses binary search to find optimal μ:

```python
from interpolatepy import smoothing_spline_with_tolerance

# Automatically determine smoothing parameter
spline = smoothing_spline_with_tolerance(
    t_points, q_noisy, 
    tolerance=0.01,     # Maximum allowed deviation
    max_iterations=50   # Binary search limit
)
```

!!! tip "Smoothing Guidelines"
    - **Low noise**: μ ∈ [0.001, 0.01]
    - **Medium noise**: μ ∈ [0.01, 0.1] 
    - **High noise**: μ ∈ [0.1, 1.0]
    - **Very noisy**: Use automatic tolerance-based selection

### Cubic Spline with Acceleration Constraints

Two methods are provided for handling acceleration boundary conditions:

#### Method 1: Virtual Waypoint Insertion

**Class**: [`CubicSplineWithAcceleration1`](api-reference.md#cubic-spline-with-acceleration1)  
**File**: `c_s_with_acc1.py`

**Approach**:
1. **Add virtual waypoints**: Insert points $t_1$ and $t_{n-1}$ at segment midpoints
2. **Expanded system**: Original waypoints become part of larger system
3. **Constraint equations**: Virtual point positions determined by boundary conditions

Virtual point positions computed using:

$$q_1 = q_0 + T_0 \cdot v_0 + \frac{T_0^2}{3} \cdot a_0 + \frac{T_0^2}{6} \cdot \omega_1$$

$$q_{n-1} = q_n - T_{n-1} \cdot v_n + \frac{T_{n-1}^2}{3} \cdot a_n + \frac{T_{n-1}^2}{6} \cdot \omega_{n-1}$$

This creates a larger tridiagonal system that can satisfy 4 boundary conditions total (v₀, vₙ, a₀, aₙ).

#### Method 2: Quintic End Segments

**Class**: [`CubicSplineWithAcceleration2`](api-reference.md#cubic-spline-with-acceleration2)  
**File**: `c_s_with_acc2.py`

**Approach**:
1. **First segment**: Quintic polynomial $q_0(t)$ with 6 conditions
2. **Last segment**: Quintic polynomial $q_{n-1}(t)$ with 6 conditions  
3. **Interior segments**: Standard cubic polynomials

The quintic segments have the form:

$$q(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5$$

This approach provides more flexibility while maintaining computational efficiency.

!!! algorithm "Acceleration Constraint Comparison"
    **Method 1 (Virtual Waypoints)**:
    - ✅ Uniform cubic segments
    - ✅ Consistent tridiagonal solve
    - ⚠️ Introduces virtual waypoints
    
    **Method 2 (Quintic Ends)**:
    - ✅ Direct boundary condition satisfaction
    - ✅ No virtual waypoints
    - ⚠️ Mixed polynomial degrees

### B-Spline Family

B-splines provide a more general framework for curve construction using basis functions.

#### Mathematical Foundation

A B-spline curve of degree p is defined as:

$$\mathbf{q}(u) = \sum_{i=0}^n \mathbf{P}_i N_{i,p}(u)$$

where:
- **$\mathbf{P}_i$**: Control points
- **$N_{i,p}(u)$**: B-spline basis functions of degree p
- **u**: Parameter (can be mapped from time t)

#### B-Spline Basis Functions

The basis functions are defined recursively:

$$N_{i,0}(u) = \begin{cases} 
1 & \text{if } u_i \leq u < u_{i+1} \\
0 & \text{otherwise}
\end{cases}$$

$$N_{i,p}(u) = \frac{u - u_i}{u_{i+p} - u_i} N_{i,p-1}(u) + \frac{u_{i+p+1} - u}{u_{i+p+1} - u_{i+1}} N_{i+1,p-1}(u)$$

#### Key Properties

- **Local Support**: Changing one control point affects only a local region
- **Convex Hull Property**: Curve lies within convex hull of control points
- **Variation Diminishing**: Curve doesn't oscillate more than control polygon
- **Degree Flexibility**: Support for degrees 3, 4, and 5

#### B-Spline Variants

**[`BSplineInterpolator`](api-reference.md#b-spline-interpolator)**: Passes through all waypoints
```python
from interpolatepy import BSplineInterpolator
import numpy as np

points = np.array([[0, 0], [1, 2], [3, 1], [4, 3]])
times = np.array([0, 1, 2, 3])

bspline = BSplineInterpolator(degree=3, points=points, times=times)
```

**[`ApproximationBSpline`](api-reference.md#approximation-b-spline)**: Least-squares fitting to waypoints
**[`SmoothingCubicBSpline`](api-reference.md#smoothing-cubic-b-spline)**: Includes smoothing parameter

---

## Motion Profiles

Motion profiles generate time-optimal trajectories with bounded derivatives, essential for robotics and automation.

### Double-S Trajectory (Jerk-Limited)

**Class**: [`DoubleSTrajectory`](api-reference.md#double-s-trajectory)  
**File**: `double_s.py`

#### Mathematical Theory

Implements double-S (jerk-bounded) trajectories that limit the rate of acceleration change. The velocity profile follows an S-curve shape with up to 7 distinct phases:

1. **Jerk-up phase**: $\dddot{q} = +j_{max}$ (acceleration increases)
2. **Constant acceleration**: $\dddot{q} = 0$, $\ddot{q} = a_{max}$
3. **Jerk-down phase**: $\dddot{q} = -j_{max}$ (acceleration decreases to 0)
4. **Constant velocity**: $\dddot{q} = 0$, $\ddot{q} = 0$, $\dot{q} = v_{max}$
5. **Jerk-down phase**: $\dddot{q} = -j_{max}$ (deceleration begins)
6. **Constant acceleration**: $\dddot{q} = 0$, $\ddot{q} = -a_{max}$
7. **Jerk-up phase**: $\dddot{q} = +j_{max}$ (deceleration decreases to 0)

#### Constraint Hierarchy

The trajectory respects the following constraint hierarchy:

1. **Jerk limit**: $|\dddot{q}(t)| \leq j_{max}$
2. **Acceleration limit**: $|\ddot{q}(t)| \leq a_{max}$
3. **Velocity limit**: $|\dot{q}(t)| \leq v_{max}$
4. **Boundary conditions**: Configurable initial/final position, velocity, acceleration

#### Phase Duration Calculation

The algorithm determines phase durations based on:

**Acceleration Time**:
$$t_{acc} = \min\left(\frac{v_{max}}{a_{max}}, \sqrt{\frac{v_{max}}{j_{max}}}\right)$$

**Jerk Time**:
$$t_{jerk} = \min\left(\frac{a_{max}}{j_{max}}, \sqrt{\frac{v_{max}}{j_{max}}}\right)$$

**Total Time Computation**:
The algorithm solves for total time considering all constraints and boundary conditions.

#### Performance Characteristics

- **Setup**: O(1) - Analytical solution
- **Evaluation**: O(1) - Piecewise analytical formulas
- **Memory**: O(1) - Store only phase parameters
- **Smoothness**: C² continuous (bounded jerk)

!!! example "S-Curve Trajectory Example"
    ```python
    from interpolatepy import DoubleSTrajectory, StateParams, TrajectoryBounds
    
    # Define motion parameters
    state = StateParams(q_0=0, q_1=10, v_0=0, v_1=0)
    bounds = TrajectoryBounds(v_bound=5.0, a_bound=10.0, j_bound=30.0)
    
    # Generate trajectory
    trajectory = DoubleSTrajectory(state, bounds)
    
    print(f"Duration: {trajectory.total_time:.2f}s")
    
    # Evaluate at midpoint
    t_mid = trajectory.total_time / 2
    pos = trajectory.evaluate(t_mid)
    vel = trajectory.evaluate_velocity(t_mid) 
    acc = trajectory.evaluate_acceleration(t_mid)
    jerk = trajectory.evaluate_jerk(t_mid)
    ```

### Trapezoidal Trajectory

**Class**: [`TrapezoidalTrajectory`](api-reference.md#trapezoidal-trajectory)  
**File**: `trapezoidal.py`

#### Mathematical Theory

Classical trapezoidal velocity profiles consist of three phases:

1. **Acceleration phase**: Linear increase to cruise velocity
2. **Constant velocity phase**: Maintain maximum velocity  
3. **Deceleration phase**: Linear decrease to final velocity

#### Trajectory Types

**Trapezoidal Profile**: Reaches maximum velocity $v_{max}$

$$v(t) = \begin{cases}
v_0 + a_{max} \cdot t & 0 \leq t \leq t_1 \\
v_{max} & t_1 \leq t \leq t_2 \\
v_{max} - a_{max} \cdot (t - t_2) & t_2 \leq t \leq t_3
\end{cases}$$

**Triangular Profile**: Never reaches $v_{max}$ (shorter distances)

The peak velocity is:
$$v_p = \sqrt{a_{max} \cdot |q_1 - q_0| + \frac{v_0^2 + v_1^2}{2}}$$

#### Constraint Handling

The algorithm operates in two modes:

**Velocity-Constrained**: Given $v_{max}$, find minimum time
**Duration-Constrained**: Given time T, find velocity profile

For velocity-constrained mode:
$$v_v = \min\left(v_{max}, \sqrt{h \cdot a_{max} + \frac{v_0^2 + v_1^2}{2}}\right)$$

where $h = q_1 - q_0$ is the displacement.

### Polynomial Trajectories

**Class**: [`PolynomialTrajectory`](api-reference.md#polynomial-trajectory)  
**File**: `polynomials.py`

#### Mathematical Theory

Generates polynomial trajectories of orders 3, 5, and 7 to satisfy boundary conditions:

**3rd Order (Cubic)**:
- 4 constraints: $q(t_0)$, $q(t_1)$, $\dot{q}(t_0)$, $\dot{q}(t_1)$
- Form: $q(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3$

**5th Order (Quintic)**:
- 6 constraints: positions, velocities, and accelerations at endpoints
- Form: $q(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5$

**7th Order (Septic)**:
- 8 constraints: positions, velocities, accelerations, and jerks at endpoints
- Form: $q(t) = \sum_{i=0}^7 a_i t^i$

#### Coefficient Determination

The coefficients are determined by solving the linear system $\mathbf{A}\mathbf{c} = \mathbf{b}$ where:

- $\mathbf{c}$ contains the polynomial coefficients
- $\mathbf{A}$ is the constraint matrix (Vandermonde-like)
- $\mathbf{b}$ contains the boundary condition values

For a 5th-order polynomial with boundary conditions at $t_0$ and $t_1$:

$$\mathbf{A} = \begin{bmatrix}
1 & t_0 & t_0^2 & t_0^3 & t_0^4 & t_0^5 \\
0 & 1 & 2t_0 & 3t_0^2 & 4t_0^3 & 5t_0^4 \\
0 & 0 & 2 & 6t_0 & 12t_0^2 & 20t_0^3 \\
1 & t_1 & t_1^2 & t_1^3 & t_1^4 & t_1^5 \\
0 & 1 & 2t_1 & 3t_1^2 & 4t_1^3 & 5t_1^4 \\
0 & 0 & 2 & 6t_1 & 12t_1^2 & 20t_1^3
\end{bmatrix}$$

!!! warning "Numerical Stability"
    Higher-order polynomials can suffer from numerical instability. For orders > 7, consider using spline methods instead.

---

## Quaternion Interpolation

Quaternions provide singularity-free rotation interpolation for 3D orientations, essential for robotics and animation.

### Quaternion Mathematics

**Class**: [`Quaternion`](api-reference.md#quaternion)  
**File**: `quat_core.py`

#### Mathematical Foundation

A unit quaternion $\mathbf{q} = w + x\mathbf{i} + y\mathbf{j} + z\mathbf{k}$ represents a rotation, where:

**Key Properties**:
- **Unit constraint**: $|\mathbf{q}| = w^2 + x^2 + y^2 + z^2 = 1$
- **Double cover**: $\mathbf{q}$ and $-\mathbf{q}$ represent the same rotation
- **Composition**: Quaternion multiplication corresponds to rotation composition
- **Interpolation**: SLERP provides shortest path on unit sphere

#### Core Operations

**SLERP (Spherical Linear Interpolation)**:
$$\text{slerp}(\mathbf{q}_0, \mathbf{q}_1, t) = \frac{\sin((1-t)\theta)}{\sin\theta}\mathbf{q}_0 + \frac{\sin(t\theta)}{\sin\theta}\mathbf{q}_1$$

where $\theta = \arccos(|\mathbf{q}_0 \cdot \mathbf{q}_1|)$ is the angular distance.

**Logarithm**:
$$\log(\mathbf{q}) = \log(|\mathbf{q}|) + \frac{\mathbf{v}}{|\mathbf{v}|} \arccos\left(\frac{w}{|\mathbf{q}|}\right)$$

where $\mathbf{v} = (x, y, z)$ is the vector part.

**Exponential**:
$$\exp(\mathbf{q}) = e^w \left(\cos|\mathbf{v}| + \frac{\mathbf{v}}{|\mathbf{v}|}\sin|\mathbf{v}|\right)$$

### SQUAD C² Interpolation

**Class**: [`SquadC2`](api-reference.md#squad-c2)  
**File**: `squad_c2.py`

#### Mathematical Theory

Implements C²-continuous SQUAD interpolation using the method from Wittmann et al. (ICRA 2023). Key innovations:

**Extended Quaternion Sequence**:
- Original: $[\mathbf{q}_1, \mathbf{q}_2, \ldots, \mathbf{q}_n]$
- Extended: $[\mathbf{q}_1, \mathbf{q}_1^{virt}, \mathbf{q}_2, \ldots, \mathbf{q}_{n-1}^{virt}, \mathbf{q}_n]$
- Virtual waypoints: $\mathbf{q}_1^{virt} = \mathbf{q}_1$, $\mathbf{q}_{n-1}^{virt} = \mathbf{q}_n$

**Corrected Intermediate Quaternions**:
Uses corrected formula (Equation 5) that properly handles non-uniform time spacing:

$$\mathbf{s}_i = \mathbf{q}_i \otimes \exp\left[\frac{\log(\mathbf{q}_i^{-1} \otimes \mathbf{q}_{i+1})}{-2(1+h_i/h_{i-1})} + \frac{\log(\mathbf{q}_i^{-1} \otimes \mathbf{q}_{i-1})}{-2(1+h_{i-1}/h_i)}\right]$$

where $h_i = t_{i+1} - t_i$ are the time intervals.

**Quintic Polynomial Parameterization**:
- Maps time $t$ to parameter $u(t)$ using 5th-order polynomial
- Zero-clamped boundaries: $u(t_0) = 0$, $u'(t_0) = 0$, $u''(t_0) = 0$
- Ensures C² continuity and zero angular velocity/acceleration at endpoints

#### SQUAD Evaluation

The SQUAD interpolation is defined as:

$$\text{squad}(\mathbf{q}_i, \mathbf{s}_i, \mathbf{s}_{i+1}, \mathbf{q}_{i+1}, u) = \text{slerp}(\text{slerp}(\mathbf{q}_i, \mathbf{q}_{i+1}, u), \text{slerp}(\mathbf{s}_i, \mathbf{s}_{i+1}, u), 2u(1-u))$$

### Logarithmic Quaternion Interpolation (LQI)

**Class**: [`LogQuaternionInterpolation`](api-reference.md#log-quaternion-interpolation)  
**File**: `log_quat.py`

#### Mathematical Theory

Implements the Logarithmic Quaternion Interpolation (LQI) method from Parker et al. (2023). The algorithm:

1. **Transform to axis-angle**: Convert quaternions to $\mathbf{r} = \theta\hat{\mathbf{n}}$ representation
2. **Handle discontinuities**: Use Algorithm 1 for continuous axis-angle recovery
3. **B-spline interpolation**: Interpolate axis-angle vectors using cubic B-splines
4. **Transform back**: Convert interpolated axis-angle to unit quaternions

#### Algorithm 1: Continuous Recovery

The continuous recovery algorithm addresses:

- **Quaternion double-cover**: Choose sign to minimize angular distance
- **Axis discontinuity**: Flip axis direction for continuity
- **Phase unwrapping**: Handle ±2π jumps in angle
- **Singularity handling**: Special cases where $\theta \approx 0$

**Quaternion Sign Selection**:
Choose $\mathbf{q}_i$ or $-\mathbf{q}_i$ to minimize:
$$d(\mathbf{q}_{i-1}, \mathbf{q}_i) = \arccos(|\mathbf{q}_{i-1} \cdot \mathbf{q}_i|)$$

**Axis Continuity**:
For consecutive axes $\hat{\mathbf{n}}_{i-1}$ and $\hat{\mathbf{n}}_i$, choose sign such that:
$$\hat{\mathbf{n}}_{i-1} \cdot \hat{\mathbf{n}}_i \geq 0$$

**Phase Unwrapping**:
Unwrap angle sequence $\{\theta_i\}$ to ensure:
$$|\theta_i - \theta_{i-1}| < \pi$$

#### Modified LQI (mLQI)

**Class**: [`ModifiedLogQuaternionInterpolation`](api-reference.md#modified-log-quaternion-interpolation)

Enhanced version that decouples angle and axis interpolation:

1. **Decompose**: Separate quaternions into $(\theta, X, Y, Z)$ where $X^2+Y^2+Z^2=1$
2. **Separate interpolation**: Use different B-splines for $\theta$ and $(X,Y,Z)$
3. **Optional normalization**: Maintain unit constraint on interpolated axes
4. **Reconstruct**: Build quaternions as $\mathbf{q} = [\cos(\theta/2), \sin(\theta/2) \cdot (X,Y,Z)]$

**Benefits**:
- Better numerical stability for large rotations
- Independent control over angle and axis smoothness
- Flexible boundary condition specification
- Reduced coupling between rotation magnitude and direction

---

## Path Planning and Geometric Primitives

### Linear Path

**Class**: [`LinearPath`](api-reference.md#linear-path)  
**File**: `simple_paths.py`

#### Mathematical Theory

Analytical representation of straight line segments in 3D space.

**Position**: $\mathbf{p}(s) = \mathbf{p}_i + \frac{s}{L}(\mathbf{p}_f - \mathbf{p}_i)$ where $s$ is arc length

**Velocity**: $\frac{d\mathbf{p}}{ds} = \frac{\mathbf{p}_f - \mathbf{p}_i}{L}$ (constant unit tangent)

**Acceleration**: $\frac{d^2\mathbf{p}}{ds^2} = \mathbf{0}$ (no curvature)

Total path length: $L = |\mathbf{p}_f - \mathbf{p}_i|$

### Circular Path

**Class**: [`CircularPath`](api-reference.md#circular-path)  
**File**: `simple_paths.py`

#### Mathematical Theory

Analytical representation of circular arcs in 3D space.

**Parameterization**: Uses axis $\mathbf{r}$, center point $\mathbf{d}$, and initial point $\mathbf{p}_i$

**Local Coordinate System**:
- **$\mathbf{e}_1$**: $(\mathbf{p}_i - \mathbf{d}) / |\mathbf{p}_i - \mathbf{d}|$ (radial direction)
- **$\mathbf{e}_2$**: $\mathbf{r} \times \mathbf{e}_1$ (tangential direction)
- **$\mathbf{e}_3$**: $\mathbf{r}$ (axis direction)

**Position**: $\mathbf{p}(s) = \mathbf{d} + R[\mathbf{e}_1 \cos(s/R) + \mathbf{e}_2 \sin(s/R)]$

**Curvature**: $\kappa = 1/R$ (constant)

where $R$ is the circle radius and $s$ is the arc length parameter.

### Frenet Frame Computation

**Function**: [`compute_trajectory_frames`](api-reference.md#frenet-frame-computation)  
**File**: `frenet_frame.py`

#### Mathematical Theory

Computes Frenet frames (moving coordinate systems) along parametric curves. For a curve $\mathbf{p}(u)$, the Frenet frame consists of:

**Tangent Vector**: $\mathbf{T} = \frac{\mathbf{p}'(u)}{|\mathbf{p}'(u)|}$ (direction of motion)

**Normal Vector**: $\mathbf{N} = \frac{\mathbf{T}'(u)}{|\mathbf{T}'(u)|}$ (direction of curvature)

**Binormal Vector**: $\mathbf{B} = \mathbf{T} \times \mathbf{N}$ (completes right-handed frame)

#### Curvature and Torsion

**Curvature**: $\kappa = \frac{|\mathbf{p}'(u) \times \mathbf{p}''(u)|}{|\mathbf{p}'(u)|^3}$

**Torsion**: $\tau = \frac{(\mathbf{p}'(u) \times \mathbf{p}''(u)) \cdot \mathbf{p}'''(u)}{|\mathbf{p}'(u) \times \mathbf{p}''(u)|^2}$

#### Frenet-Serret Formulas

The frame evolution is governed by:

$$\frac{d}{du}\begin{bmatrix} \mathbf{T} \\ \mathbf{N} \\ \mathbf{B} \end{bmatrix} = \begin{bmatrix} 0 & \kappa & 0 \\ -\kappa & 0 & \tau \\ 0 & -\tau & 0 \end{bmatrix} \begin{bmatrix} \mathbf{T} \\ \mathbf{N} \\ \mathbf{B} \end{bmatrix}$$

#### Tool Orientation

Additional tool rotations (roll, pitch, yaw) can be applied to the Frenet frame:

$$\mathbf{R}_{tool} = \mathbf{R}_z(\psi) \mathbf{R}_y(\theta) \mathbf{R}_x(\phi)$$

where $\phi$, $\theta$, $\psi$ are roll, pitch, and yaw angles respectively.

!!! example "Frenet Frame Example"
    ```python
    from interpolatepy import compute_trajectory_frames
    import numpy as np
    
    def helix_path(u):
        """Parametric helix with derivatives."""
        r, pitch = 2.0, 0.5
        
        position = np.array([r*np.cos(u), r*np.sin(u), pitch*u])
        first_derivative = np.array([-r*np.sin(u), r*np.cos(u), pitch])
        second_derivative = np.array([-r*np.cos(u), -r*np.sin(u), 0])
        
        return position, first_derivative, second_derivative
    
    # Compute frames along helix
    u_values = np.linspace(0, 4*np.pi, 100)
    points, frames = compute_trajectory_frames(
        helix_path, u_values, tool_orientation=(0.1, 0.2, 0.0)
    )
    
    # frames[i] = [tangent, normal, binormal] at points[i]
    ```

---

## Utilities and Supporting Algorithms

### Tridiagonal System Solver

**Function**: [`solve_tridiagonal`](api-reference.md#tridiagonal-solver)  
**File**: `tridiagonal_inv.py`

#### Mathematical Theory

Efficient solver for tridiagonal linear systems using the Thomas algorithm:

$$\begin{bmatrix}
b_1 & c_1 & 0 & \cdots & 0 \\
a_2 & b_2 & c_2 & \cdots & 0 \\
0 & a_3 & b_3 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & b_n
\end{bmatrix}
\begin{bmatrix}
x_1 \\ x_2 \\ x_3 \\ \vdots \\ x_n
\end{bmatrix}
=
\begin{bmatrix}
d_1 \\ d_2 \\ d_3 \\ \vdots \\ d_n
\end{bmatrix}$$

#### Thomas Algorithm

**Forward Elimination**:
```
for i = 2 to n:
    w = a[i] / b[i-1]
    b[i] = b[i] - w * c[i-1]
    d[i] = d[i] - w * d[i-1]
```

**Back Substitution**:
```
x[n] = d[n] / b[n]
for i = n-1 down to 1:
    x[i] = (d[i] - c[i] * x[i+1]) / b[i]
```

**Complexity**: O(n) vs O(n³) for general Gaussian elimination

#### Applications in InterpolatePy

- Cubic spline acceleration computation
- B-spline control point calculation
- Smoothing spline parameter estimation

### Linear Interpolation

**Function**: [`linear_traj`](api-reference.md#linear-interpolation)  
**File**: `linear.py`

#### Mathematical Theory

Simple linear interpolation between two points with constant velocity:

**Position**: $\mathbf{q}(t) = \mathbf{q}_0 + \frac{\mathbf{q}_1 - \mathbf{q}_0}{t_1 - t_0} (t - t_0)$

**Velocity**: $\dot{\mathbf{q}}(t) = \frac{\mathbf{q}_1 - \mathbf{q}_0}{t_1 - t_0}$ (constant)

**Acceleration**: $\ddot{\mathbf{q}}(t) = \mathbf{0}$

Supports both scalar and vector positions with proper broadcasting.

---

## Implementation Notes

### Numerical Stability

#### Condition Number Considerations

- **Spline matrices**: Tridiagonal systems are well-conditioned for reasonable time spacing
- **B-spline basis**: Normalized basis functions maintain numerical stability
- **Quaternion operations**: Use double precision for rotation compositions

#### Robust Evaluation

```python
def safe_evaluate(trajectory, t):
    """Safe trajectory evaluation with bounds checking."""
    t_min, t_max = trajectory.t_points[0], trajectory.t_points[-1]
    t_clamped = np.clip(t, t_min, t_max)
    return trajectory.evaluate(t_clamped)
```

### Performance Optimization

#### Memory Layout

- **Contiguous arrays**: Use C-contiguous numpy arrays for cache efficiency
- **Vectorized operations**: Leverage BLAS/LAPACK through numpy
- **Minimal allocations**: Reuse arrays when possible

#### Algorithmic Complexity

| Operation | Cubic Spline | B-Spline | Motion Profile | Quaternion |
|-----------|--------------|----------|----------------|------------|
| Setup | O(n) | O(n²) | O(1) | O(n) |
| Single Eval | O(log n) | O(p) | O(1) | O(1) |
| Vector Eval | O(k log n) | O(kp) | O(k) | O(k) |

Where n = waypoints, k = evaluation points, p = B-spline degree.

### Validation and Testing

#### Input Validation

```python
def validate_time_sequence(t_points):
    """Validate monotonic time sequence."""
    if not all(t_points[i] < t_points[i+1] for i in range(len(t_points)-1)):
        raise ValueError("Time points must be strictly increasing")
    
def validate_finite_values(values):
    """Check for NaN/Inf values."""
    if not np.all(np.isfinite(values)):
        raise ValueError("All values must be finite")
```

#### Unit Tests

InterpolatePy includes comprehensive test coverage:

- **Analytical verification**: Compare with known analytical solutions
- **Continuity tests**: Verify C⁰, C¹, C² properties at boundaries
- **Boundary condition tests**: Ensure constraints are satisfied
- **Performance benchmarks**: Regression testing for speed

---

## References

This library implements algorithms from the following research:

### Robotics & Trajectory Planning
- Biagiotti, L., & Melchiorri, C. (2008). *Trajectory Planning for Automatic Machines and Robots*. Springer.
- Siciliano, B., Sciavicco, L., Villani, L., & Oriolo, G. (2010). *Robotics: Modelling, Planning and Control*. Springer.

### Quaternion Interpolation  
- Parker, J. K., et al. (2023). "Logarithm-Based Methods for Interpolating Quaternion Time Series." *IEEE Transactions on Robotics*.
- Wittmann, D., et al. (2023). "Spherical Cubic Blends: C²-Continuous Quaternion Interpolation." *IEEE International Conference on Robotics and Automation (ICRA)*.

### Spline Theory
- de Boor, C. (2001). *A Practical Guide to Splines*. Springer.
- Schumaker, L. L. (2007). *Spline Functions: Basic Theory*. Cambridge University Press.

### Numerical Methods
- Press, W. H., et al. (2007). *Numerical Recipes: The Art of Scientific Computing*. Cambridge University Press.
- Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations*. Johns Hopkins University Press.

---

For practical examples and usage patterns, see:
- **[API Reference](api-reference.md)**: Complete function documentation
- **[Examples](examples.md)**: Real-world applications  
- **[Tutorials](tutorials/spline-interpolation.md)**: Step-by-step guides