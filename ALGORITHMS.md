# InterpolatePy Algorithms Reference

This document provides comprehensive technical documentation for all trajectory planning and interpolation algorithms implemented in InterpolatePy. The library is designed for robotics, animation, and scientific computing applications requiring smooth trajectory generation with precise control over position, velocity, acceleration, and jerk profiles.

## Table of Contents

1. [Introduction](#introduction)
2. [Algorithm Classification](#algorithm-classification)
3. [Spline Interpolation](#spline-interpolation)
4. [Motion Profiles](#motion-profiles)
5. [Polynomial Trajectories](#polynomial-trajectories)
6. [Quaternion Interpolation](#quaternion-interpolation)
7. [Specialized Algorithms](#specialized-algorithms)
8. [Utilities](#utilities)

## Introduction

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

## Algorithm Classification

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

### CubicSpline

**File**: `cubic_spline.py`  
**Class**: `CubicSpline`

#### Theory

Implements natural cubic spline interpolation with configurable boundary conditions. Given waypoints {(tᵢ, qᵢ)}ᵢ₌₀ⁿ, constructs piecewise cubic polynomials:

```
qₖ(t) = aₖ₀ + aₖ₁(t-tₖ) + aₖ₂(t-tₖ)² + aₖ₃(t-tₖ)³
```

for t ∈ [tₖ, tₖ₊₁]. The spline satisfies:
- **C² continuity**: Position, velocity, and acceleration continuous at interior points
- **Interpolation**: qₖ(tᵢ) = qᵢ for all waypoints
- **Boundary conditions**: Configurable velocity/acceleration at endpoints

**Mathematical Derivation**:

1. **Hermite Form**: Express each segment using accelerations ωᵢ = q̈(tᵢ):
   ```
   qₖ(t) = Aₖ + Bₖ(t-tₖ) + (Cₖ/6)(t-tₖ)³ + (Dₖ/6)(t-tₖ)²
   ```
   where Cₖ = ωₖ, Dₖ = ωₖ₊₁ - ωₖ

2. **Continuity Constraints**: At interior points tᵢ, enforce:
   - Position: qᵢ₋₁(tᵢ) = qᵢ₊₁(tᵢ) = qᵢ
   - Velocity: q̇ᵢ₋₁(tᵢ) = q̇ᵢ₊₁(tᵢ)

3. **Tridiagonal System**: These constraints yield the linear system Aω = c where:
   ```
   Aᵢᵢ = (Tᵢ₋₁ + Tᵢ)/3,  Aᵢ,ᵢ₋₁ = Tᵢ₋₁/6,  Aᵢ,ᵢ₊₁ = Tᵢ/6
   cᵢ = (qᵢ₊₁ - qᵢ)/Tᵢ - (qᵢ - qᵢ₋₁)/Tᵢ₋₁
   ```
   with Tᵢ = tᵢ₊₁ - tᵢ being the time intervals.

4. **Boundary Conditions**:
   - **Natural**: ω₀ = ωₙ = 0 (zero curvature at endpoints)
   - **Clamped**: Specified derivatives v₀, vₙ modify first/last equations
   - **Not-a-knot**: Force C³ continuity at second/penultimate points

#### API Reference

```python
class CubicSpline:
    def __init__(self, t_points: list[float], q_points: list[float], 
                 v0: float = 0.0, vn: float = 0.0, 
                 a0: float | None = None, an: float | None = None)
```

**Parameters**:
- `t_points`: Time waypoints [t₀, t₁, ..., tₙ] (strictly increasing)
- `q_points`: Position waypoints [q₀, q₁, ..., qₙ] (same length as t_points)
- `v0`: Initial velocity constraint (default: 0.0)
- `vn`: Final velocity constraint (default: 0.0)  
- `a0`: Initial acceleration constraint (default: None → natural spline)
- `an`: Final acceleration constraint (default: None → natural spline)

**Key Methods**:

```python
def evaluate(self, t: float | np.ndarray) -> float | np.ndarray
```
Evaluate position q(t) at time(s) t.

```python
def evaluate_velocity(self, t: float | np.ndarray) -> float | np.ndarray
```
Evaluate velocity q̇(t) at time(s) t.

```python
def evaluate_acceleration(self, t: float | np.ndarray) -> float | np.ndarray  
```
Evaluate acceleration q̈(t) at time(s) t.

```python
def plot(self, num_points: int = 1000) -> None
```
Generate visualization with position, velocity, and acceleration profiles.

#### Example

```python
import numpy as np
from interpolatepy import CubicSpline

# Define waypoints
t_points = [0, 1, 2, 3, 4]
q_points = [0, 1, 0, -1, 0]

# Create spline with velocity constraints
spline = CubicSpline(t_points, q_points, v0=0.0, vn=0.0)

# Evaluate at specific time
position = spline.evaluate(1.5)
velocity = spline.evaluate_velocity(1.5)
acceleration = spline.evaluate_acceleration(1.5)

# Generate trajectory
t_eval = np.linspace(0, 4, 100)
trajectory = spline.evaluate(t_eval)

# Visualize
spline.plot()
```

### CubicSmoothingSpline

**File**: `c_s_smoothing.py`  
**Class**: `CubicSmoothingSpline`

#### Theory

Implements cubic smoothing splines that balance interpolation accuracy with smoothness. Instead of exact interpolation, minimizes the objective function:

```
J(q) = μ∑ᵢ(qᵢ - q̂ᵢ)² + ∫[t₀,tₙ] (q̈(t))² dt
```

where:
- **μ**: Smoothing parameter (0 ≤ μ ≤ ∞)
- **q̂ᵢ**: Target positions at waypoints
- **First term**: Penalizes deviation from waypoints
- **Second term**: Penalizes curvature (acceleration)

Special cases:
- **μ → 0**: Natural cubic spline (exact interpolation)
- **μ → ∞**: Linear least squares fit

The solution involves solving a modified tridiagonal system where the diagonal is augmented by the smoothing parameter.

#### API Reference

```python
class CubicSmoothingSpline:
    def __init__(self, t_points: list[float], q_points: list[float],
                 mu: float, v0: float = 0.0, vn: float = 0.0)
```

**Parameters**:
- `t_points`: Time waypoints
- `q_points`: Target position waypoints  
- `mu`: Smoothing parameter (≥ 0)
- `v0`, `vn`: Boundary velocity conditions

**Key Methods**: Same as `CubicSpline`

#### Example

```python
from interpolatepy import CubicSmoothingSpline

# Noisy data points
t_points = [0, 0.5, 1.0, 1.5, 2.0]
q_noisy = [0.1, 1.05, -0.02, -0.98, 0.05]  # Noisy sine wave

# Smooth the data
spline = CubicSmoothingSpline(t_points, q_noisy, mu=0.1)
spline.plot()
```

### smoothing_spline_with_tolerance

**File**: `c_s_smoot_search.py`  
**Function**: `smoothing_spline_with_tolerance`

#### Theory

Automatically determines the optimal smoothing parameter μ using binary search to achieve a specified tolerance. The algorithm iteratively adjusts μ until:

```
max|qᵢ - q̂ᵢ| ≤ tolerance
```

This provides an objective way to balance smoothness and fidelity without manual parameter tuning.

#### API Reference

```python
def smoothing_spline_with_tolerance(
    t_points: np.ndarray,
    q_points: np.ndarray,
    tolerance: float,
    config: SplineConfig,
) -> tuple[CubicSmoothingSpline, float, float, int]
```

**Parameters**:
- `t_points`: Time points as numpy array
- `q_points`: Position points as numpy array  
- `tolerance`: Maximum allowed deviation from waypoints
- `config`: SplineConfig with search parameters

#### Example

```python
from interpolatepy import smoothing_spline_with_tolerance, SplineConfig
import numpy as np

# Automatically find optimal smoothing
config = SplineConfig(max_iterations=50)
spline, mu, error, iterations = smoothing_spline_with_tolerance(
    np.array(t_points), np.array(q_noisy), tolerance=0.01, config=config
)
print(f"Optimal μ = {mu:.6f}, error = {error:.6f}")
```

### CubicSplineWithAcceleration1

**File**: `c_s_with_acc1.py`  
**Class**: `CubicSplineWithAcceleration1`

#### Theory

Extends cubic spline interpolation to handle both velocity AND acceleration boundary conditions simultaneously. This requires adding extra degrees of freedom to the system.

**Method 1 Approach**:
1. **Add virtual waypoints**: Insert points t₁ and tₙ₋₁ at segment midpoints
2. **Expanded system**: Original waypoints [q₀, q₂, q₃, ..., qₙ₋₂, qₙ] become [q₀, q₁, q₂, q₃, ..., qₙ₋₂, qₙ₋₁, qₙ]
3. **Constraint equations**: Virtual point positions determined by boundary conditions

The positions of virtual points are computed using:

```
q₁ = q₀ + T₀·v₀ + (T₀²/3)·a₀ + (T₀²/6)·ω₁
qₙ₋₁ = qₙ - Tₙ₋₁·vₙ + (Tₙ₋₁²/3)·aₙ + (Tₙ₋₁²/6)·ωₙ₋₁
```

This creates a larger tridiagonal system that can satisfy 4 boundary conditions total.

#### API Reference

```python
class CubicSplineWithAcceleration1:
    def __init__(self, t_points: list[float], q_points: list[float],
                 v0: float = 0.0, vn: float = 0.0, 
                 a0: float = 0.0, an: float = 0.0)
```

**Parameters**:
- `t_points`: Original time waypoints
- `q_points`: Original position waypoints
- `v0`, `vn`: Initial and final velocity
- `a0`, `an`: Initial and final acceleration

**Key Methods**: Same as `CubicSpline`, plus:

```python
@property
def original_indices(self) -> list[int]
```
Returns indices of original waypoints in expanded arrays (useful for plotting).

#### Example

```python
from interpolatepy import CubicSplineWithAcceleration1

# Trajectory with specific boundary conditions
spline = CubicSplineWithAcceleration1(
    t_points=[0, 1, 2, 3],
    q_points=[0, 1, 0, 1], 
    v0=1.0, vn=-0.5,      # Velocity constraints
    a0=0.0, an=0.0        # Acceleration constraints
)

spline.plot()  # Shows original vs virtual waypoints
```

### CubicSplineWithAcceleration2

**File**: `c_s_with_acc2.py`  
**Class**: `CubicSplineWithAcceleration2`

#### Theory

Alternative approach to handling acceleration boundary conditions using quintic (5th-order) polynomials for the first and last segments while maintaining cubic segments in between.

**Method 2 Approach**:
1. **First segment**: Quintic polynomial q₀(t) with 6 conditions
   - q₀(t₀) = q₀, q₀(t₁) = q₁
   - q̇₀(t₀) = v₀, q̈₀(t₀) = a₀  
   - Continuity: q̇₀(t₁) = q̇₁(t₁), q̈₀(t₁) = q̈₁(t₁)

2. **Last segment**: Quintic polynomial qₙ₋₁(t) with 6 conditions
   - Similar structure with final boundary conditions

3. **Interior segments**: Standard cubic polynomials

This approach provides more flexibility in satisfying boundary conditions while maintaining computational efficiency.

#### API Reference

```python
class CubicSplineWithAcceleration2:
    def __init__(self, t_points: list[float], q_points: list[float],
                 v0: float = 0.0, vn: float = 0.0,
                 a0: float = 0.0, an: float = 0.0)
```

**Parameters**: Same as Method 1

**Key Methods**: Same as `CubicSpline`

### BSpline and Variants

**Files**: `b_spline.py`, `b_spline_*.py`  
**Classes**: `BSpline`, `BSplineApproximation`, `BSplineInterpolator`, etc.

#### Theory

B-splines provide a more general framework for curve construction using basis functions. A B-spline curve of degree p is defined as:

```
q(u) = ∑ᵢ₌₀ⁿ Pᵢ Nᵢ,ₚ(u)
```

where:
- **Pᵢ**: Control points
- **Nᵢ,ₚ(u)**: B-spline basis functions of degree p
- **u**: Parameter (can be mapped from time t)

**Key Properties**:
- **Local support**: Changing one control point affects only a local region
- **Convex hull property**: Curve lies within convex hull of control points
- **Variation diminishing**: Curve doesn't oscillate more than control polygon
- **Degree flexibility**: Support for degrees 3, 4, and 5

**Variants Implemented**:
- **BSplineInterpolator**: Passes through all waypoints
- **BSplineApproximation**: Least-squares fitting to waypoints
- **BSplineCubic**: Specialized cubic implementation
- **BSplineSmoothing**: Includes smoothing parameter

#### API Reference

```python
class BSplineInterpolator:
    def __init__(self, degree: int, points: np.ndarray, 
                 times: np.ndarray | None = None,
                 initial_velocity: np.ndarray | None = None,
                 final_velocity: np.ndarray | None = None)
```

**Parameters**:
- `degree`: B-spline degree (3, 4, or 5)
- `points`: Control/waypoints (N×d array for d-dimensional curves)
- `times`: Time values (if None, uses uniform parameterization)
- `initial_velocity`, `final_velocity`: Boundary conditions

---

## Motion Profiles

Motion profiles are classical trajectory planning methods optimized for robotics applications. They focus on bounded acceleration/jerk and smooth velocity transitions.

### DoubleSTrajectory

**File**: `double_s.py`  
**Class**: `DoubleSTrajectory`

#### Theory

Implements double-S (jerk-bounded) trajectories that limit the rate of acceleration change. The velocity profile follows an S-curve shape with up to 7 distinct phases:

1. **Jerk-up phase**: q⃛ = +jₘₐₓ (acceleration increases)
2. **Constant acceleration**: q⃛ = 0, q̈ = aₘₐₓ  
3. **Jerk-down phase**: q⃛ = -jₘₐₓ (acceleration decreases to 0)
4. **Constant velocity**: q⃛ = 0, q̈ = 0, q̇ = vₘₐₓ
5. **Jerk-down phase**: q⃛ = -jₘₐₓ (deceleration begins)
6. **Constant acceleration**: q⃛ = 0, q̈ = -aₘₐₓ
7. **Jerk-up phase**: q⃛ = +jₘₐₓ (deceleration decreases to 0)

**Mathematical Model**:

The trajectory is computed by solving the time-optimal control problem:
```
minimize T
subject to: |q̇(t)| ≤ vₘₐₓ, |q̈(t)| ≤ aₘₐₓ, |q⃛(t)| ≤ jₘₐₓ
           q(0) = q₀, q(T) = q₁, q̇(0) = v₀, q̇(T) = v₁
```

**Phase Duration Calculation**:
1. **Acceleration phases**: T₁ = min(aₘₐₓ/jₘₐₓ, √((v₁-v₀)/(2jₘₐₓ)))
2. **Constant acceleration**: T₂ determined by velocity limit constraints
3. **Velocity phase**: T₄ calculated from displacement requirements

**Kinematic Equations**: For each phase i with jerk jᵢ, acceleration a₀ᵢ, velocity v₀ᵢ:
```
q̈ᵢ(t) = a₀ᵢ + jᵢt
q̇ᵢ(t) = v₀ᵢ + a₀ᵢt + ½jᵢt²
qᵢ(t) = q₀ᵢ + v₀ᵢt + ½a₀ᵢt² + ⅙jᵢt³
```

**Constraints**:
- **Jerk limit**: |q⃛(t)| ≤ jₘₐₓ
- **Acceleration limit**: |q̈(t)| ≤ aₘₐₓ  
- **Velocity limit**: |q̇(t)| ≤ vₘₐₓ
- **Boundary conditions**: Configurable initial/final position, velocity, acceleration

#### API Reference

```python
@dataclass
class StateParams:
    q_0: float = 0.0      # Initial position
    q_1: float = 1.0      # Final position  
    v_0: float = 0.0      # Initial velocity
    v_1: float = 0.0      # Final velocity

@dataclass  
class TrajectoryBounds:
    v_bound: float = 1.0   # Velocity bound
    a_bound: float = 1.0   # Acceleration bound
    j_bound: float = 1.0   # Jerk bound

class DoubleSTrajectory:
    def __init__(self, state_params: StateParams, bounds: TrajectoryBounds)
```

**Key Methods**:

```python
def evaluate(self, t: float) -> float
def evaluate_velocity(self, t: float) -> float  
def evaluate_acceleration(self, t: float) -> float
def evaluate_jerk(self, t: float) -> float
def get_duration(self) -> float
def plot(self) -> None
```

#### Example

```python
from interpolatepy import DoubleSTrajectory, StateParams, TrajectoryBounds

# Define trajectory parameters
state = StateParams(q_0=0, q_1=10, v_0=0, v_1=0)
bounds = TrajectoryBounds(v_bound=2.0, a_bound=1.0, j_bound=0.5)

# Create trajectory
traj = DoubleSTrajectory(state, bounds)

# Evaluate
t_eval = np.linspace(0, traj.get_duration(), 100)
positions = [traj.evaluate(t) for t in t_eval]
velocities = [traj.evaluate_velocity(t) for t in t_eval]

# Visualize all profiles
traj.plot()
```

### TrapezoidalTrajectory

**File**: `trapezoidal.py`  
**Class**: `TrapezoidalTrajectory`

#### Theory

Classical trapezoidal velocity profiles consist of three phases:
1. **Acceleration phase**: Linear increase to cruise velocity
2. **Constant velocity phase**: Maintain maximum velocity
3. **Deceleration phase**: Linear decrease to final velocity

Two trajectory types are supported:
- **Trapezoidal**: Reaches maximum velocity vₘₐₓ
- **Triangular**: Never reaches vₘₐₓ (shorter distances)

The algorithm can operate in two modes:
- **Duration-constrained**: Given time T, find velocity profile
- **Velocity-constrained**: Given vₘₐₓ, find minimum time

**Key Equations**:
For velocity-constrained mode:
```
vᵥ = min(vₘₐₓ, √(h·aₘₐₓ + (v₀² + v₁²)/2))
```

where h = q₁ - q₀ is the displacement.

#### API Reference

```python
@dataclass
class TrajectoryParams:
    q0: float                    # Initial position
    q1: float                    # Final position
    t0: float = 0.0             # Start time
    v0: float = 0.0             # Initial velocity
    v1: float = 0.0             # Final velocity
    amax: float | None = None    # Maximum acceleration
    vmax: float | None = None    # Maximum velocity  
    duration: float | None = None # Total duration

class TrapezoidalTrajectory:
    @staticmethod
    def generate_trajectory(params: TrajectoryParams) -> 
        tuple[Callable[[float], tuple[float, float, float]], float]
```

**Returns**: (trajectory_function, duration)

The trajectory function takes time t and returns (position, velocity, acceleration).

#### Example

```python
from interpolatepy import TrapezoidalTrajectory, TrajectoryParams

# Velocity-constrained trajectory
params = TrajectoryParams(
    q0=0, q1=5, v0=0, v1=0,
    amax=2.0, vmax=1.5
)

traj_func, duration = TrapezoidalTrajectory.generate_trajectory(params)

# Evaluate trajectory
t = 1.0  
pos, vel, acc = traj_func(t)
print(f"At t={t}: pos={pos:.2f}, vel={vel:.2f}, acc={acc:.2f}")
```

### ParabolicBlendTrajectory

**File**: `lin_poly_parabolic.py`  
**Class**: `ParabolicBlendTrajectory`

#### Theory

Constructs trajectories from linear segments connected by parabolic blends at via points. This approach:

1. **Linear segments**: Constant velocity between via points
2. **Parabolic blends**: Smooth transitions with bounded acceleration
3. **Zero boundaries**: Initial and final velocities are zero

Each blend has duration dt_blend[i] with acceleration:
```
aᵢ = (v_after[i] - v_before[i]) / dt_blend[i]
```

The total trajectory duration is extended by:
```
T_total = T_nominal + (dt_blend[0] + dt_blend[-1]) / 2
```

#### API Reference

```python
class ParabolicBlendTrajectory:
    def __init__(self, q: list | np.ndarray, t: list | np.ndarray,
                 dt_blend: list | np.ndarray, dt: float = 0.01)
    
    def generate(self) -> tuple[Callable[[float], tuple[float, float, float]], float]
```

**Parameters**:
- `q`: Position waypoints
- `t`: Nominal arrival times  
- `dt_blend`: Blend duration at each waypoint
- `dt`: Sampling interval for plotting

---

## Polynomial Trajectories

Direct polynomial fitting methods for trajectory generation with specified boundary conditions.

### PolynomialTrajectory

**File**: `polynomials.py`  
**Class**: `PolynomialTrajectory`

#### Theory

Generates polynomial trajectories of orders 3, 5, and 7 to satisfy boundary conditions:

**3rd Order (Cubic)**:
- 4 constraints: q(t₀), q(t₁), q̇(t₀), q̇(t₁)
- Form: q(t) = a₀ + a₁t + a₂t² + a₃t³

**5th Order (Quintic)**:  
- 6 constraints: positions, velocities, and accelerations at endpoints
- Form: q(t) = a₀ + a₁t + a₂t² + a₃t³ + a₄t⁴ + a₅t⁵

**7th Order (Septic)**:
- 8 constraints: positions, velocities, accelerations, and jerks at endpoints
- Form: q(t) = ∑ᵢ₌₀⁷ aᵢtⁱ

The coefficients are determined by solving the linear system Ac = b where c contains the polynomial coefficients.

#### API Reference

```python
@dataclass
class BoundaryCondition:
    position: float
    velocity: float | None = None
    acceleration: float | None = None  
    jerk: float | None = None

@dataclass
class TimeInterval:
    t0: float
    t1: float

class PolynomialTrajectory:
    @staticmethod
    def order_3_trajectory(initial: BoundaryCondition, final: BoundaryCondition,
                          interval: TimeInterval) -> Callable[[float], tuple[float, float, float, float]]
    
    @staticmethod  
    def order_5_trajectory(initial: BoundaryCondition, final: BoundaryCondition,
                          interval: TimeInterval) -> Callable[[float], tuple[float, float, float, float]]
    
    @staticmethod
    def order_7_trajectory(initial: BoundaryCondition, final: BoundaryCondition, 
                          interval: TimeInterval) -> Callable[[float], tuple[float, float, float, float]]
```

**Returns**: Function that takes time t and returns (position, velocity, acceleration, jerk)

#### Example

```python
from interpolatepy import PolynomialTrajectory, BoundaryCondition, TimeInterval

# Define boundary conditions
initial = BoundaryCondition(position=0, velocity=0, acceleration=0)
final = BoundaryCondition(position=1, velocity=0, acceleration=0)
interval = TimeInterval(start=0, end=2)

# Generate quintic trajectory
traj_func = PolynomialTrajectory.order_5_trajectory(initial, final, interval)

# Evaluate
pos, vel, acc, jerk = traj_func(1.0)
```

### linear_traj

**File**: `linear.py`  
**Function**: `linear_traj`

#### Theory

Simple linear interpolation between two points with constant velocity and zero acceleration:

```
q(t) = q₀ + (q₁ - q₀) * (t - t₀) / (t₁ - t₀)
q̇(t) = (q₁ - q₀) / (t₁ - t₀)  
q̈(t) = 0
```

Supports both scalar and vector positions with proper broadcasting.

#### API Reference

```python
def linear_traj(p0: float | list[float] | np.ndarray,
                p1: float | list[float] | np.ndarray, 
                t0: float, t1: float,
                time_array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

**Returns**: (positions, velocities, accelerations) arrays

---

## Quaternion Interpolation

Specialized algorithms for smooth rotation interpolation using unit quaternions. These methods handle the double-cover property and provide C² continuous orientation trajectories.

### Quaternion (Core Operations)

**File**: `quat_core.py`  
**Class**: `Quaternion`

#### Theory

Implements fundamental quaternion operations for 3D rotations. A unit quaternion q = w + xi + yj + zk represents a rotation, where:

**Key Properties**:
- **Unit constraint**: |q| = w² + x² + y² + z² = 1
- **Double cover**: q and -q represent the same rotation  
- **Composition**: Quaternion multiplication corresponds to rotation composition
- **Interpolation**: SLERP provides shortest path on unit sphere

**Core Operations Implemented**:
- **SLERP**: Spherical Linear Interpolation
- **SQUAD**: Spherical Quadrangle (cubic interpolation)
- **Logarithm/Exponential**: For advanced interpolation methods
- **Axis-angle conversion**: to_axis_angle(), from_angle_axis()

#### API Reference

```python
class Quaternion:
    def __init__(self, w: float, x: float, y: float, z: float)
    
    # Basic operations
    def norm(self) -> float
    def unit(self) -> 'Quaternion' 
    def inverse(self) -> 'Quaternion'
    def conjugate(self) -> 'Quaternion'
    
    # Interpolation  
    def slerp(self, other: 'Quaternion', t: float) -> 'Quaternion'
    def squad(self, s1: 'Quaternion', s2: 'Quaternion', 
              other: 'Quaternion', t: float) -> 'Quaternion'
    
    # Advanced operations
    def Log(self) -> 'Quaternion'
    def exp(self) -> 'Quaternion' 
    def to_axis_angle(self) -> tuple[np.ndarray, float]
    
    @classmethod
    def from_angle_axis(cls, angle: float, axis: np.ndarray) -> 'Quaternion'
```

### LogQuaternionInterpolation (LQI)

**File**: `log_quat.py`  
**Class**: `LogQuaternionInterpolation`

> **Note**: This class is available in the `log_quat.py` module but not exported in the main `__init__.py`. Import directly: `from interpolatepy.log_quat import LogQuaternionInterpolation`

#### Theory

Implements the Logarithmic Quaternion Interpolation (LQI) method from Parker et al. (2023). The algorithm:

1. **Transform to axis-angle**: Convert quaternions to r = θn̂ representation
2. **Handle discontinuities**: Use Algorithm 1 for continuous axis-angle recovery
3. **B-spline interpolation**: Interpolate axis-angle vectors using cubic B-splines  
4. **Transform back**: Convert interpolated axis-angle to unit quaternions

**Key Advantages**:
- Handles quaternion double-cover automatically
- Resolves axis-angle discontinuities (θ → θ + 2π)
- Provides C² continuous quaternion trajectories
- Supports non-uniform time spacing

**Algorithm 1 (Continuous Recovery)**:
- Choose quaternion sign to minimize angular distance  
- Flip axis direction if needed for continuity
- Apply phase unwrapping to handle ±2π jumps
- Handle special cases where θ ≈ 0

#### API Reference

```python
class LogQuaternionInterpolation:
    def __init__(self, time_points: list[float], quaternions: list[Quaternion],
                 degree: int = 3, 
                 initial_velocity: np.ndarray | None = None,
                 final_velocity: np.ndarray | None = None)
    
    def evaluate(self, t: float) -> Quaternion
    def evaluate_velocity(self, t: float) -> np.ndarray  # Angular velocity
    def evaluate_acceleration(self, t: float) -> np.ndarray  # Angular acceleration
```

### ModifiedLogQuaternionInterpolation (mLQI)

**File**: `log_quat.py`  
**Class**: `ModifiedLogQuaternionInterpolation`

> **Note**: This class is available in the `log_quat.py` module but not exported in the main `__init__.py`. Import directly: `from interpolatepy.log_quat import ModifiedLogQuaternionInterpolation`

#### Theory

Enhanced version of LQI that decouples angle and axis interpolation:

1. **Decompose**: Separate quaternions into (θ, X, Y, Z) where X²+Y²+Z²=1
2. **Separate interpolation**: Use different B-splines for θ and (X,Y,Z)
3. **Optional normalization**: Maintain unit constraint on interpolated axes
4. **Reconstruct**: Build quaternions as q = [cos(θ/2), sin(θ/2)·(X,Y,Z)]

**Benefits**:
- Better numerical stability for large rotations
- Independent control over angle and axis smoothness
- Flexible boundary condition specification
- Reduced coupling between rotation magnitude and direction

#### API Reference

```python
class ModifiedLogQuaternionInterpolation:
    def __init__(self, time_points: list[float], quaternions: list[Quaternion],
                 degree: int = 3, normalize_axis: bool = True,
                 initial_velocity: np.ndarray | None = None,  # 4D: [θ̇, Ẋ, Ẏ, Ż]
                 final_velocity: np.ndarray | None = None)
    
    def evaluate(self, t: float) -> Quaternion
    def evaluate_velocity(self, t: float) -> np.ndarray    # 4D derivative vector
    def evaluate_acceleration(self, t: float) -> np.ndarray # 4D second derivative
```

### SquadC2

**File**: `squad_c2.py`  
**Class**: `SquadC2`

#### Theory

Implements C²-continuous SQUAD interpolation using the method from Wittmann et al. (ICRA 2023). Key innovations:

**Extended Quaternion Sequence**:
- Original: [q₁, q₂, ..., qₙ]  
- Extended: [q₁, q₁ᵛⁱʳᵗ, q₂, ..., qₙ₋₁ᵛⁱʳᵗ, qₙ]
- Virtual waypoints: q₁ᵛⁱʳᵗ = q₁, qₙ₋₁ᵛⁱʳᵗ = qₙ

**Corrected Intermediate Quaternions**:
Uses corrected formula (Equation 5) that properly handles non-uniform time spacing:

```
sᵢ = qᵢ ⊗ exp[log(qᵢ⁻¹⊗qᵢ₊₁)/(-2(1+hᵢ/hᵢ₋₁)) + log(qᵢ⁻¹⊗qᵢ₋₁)/(-2(1+hᵢ₋₁/hᵢ))]
```

**Quintic Polynomial Parameterization**:
- Maps time t to parameter u(t) using 5th-order polynomial
- Zero-clamped boundaries: u(t₀) = 0, u'(t₀) = 0, u''(t₀) = 0
- Ensures C² continuity and zero angular velocity/acceleration at endpoints

#### API Reference

```python
class SquadC2:
    def __init__(self, time_points: list[float], quaternions: list[Quaternion],
                 normalize_quaternions: bool = True,
                 validate_continuity: bool = True)
                 
    def evaluate(self, t: float) -> Quaternion
    def evaluate_velocity(self, t: float) -> np.ndarray
    def evaluate_acceleration(self, t: float) -> np.ndarray
    
    def get_waypoints(self) -> tuple[list[float], list[Quaternion]]  # Original
    def get_extended_waypoints(self) -> tuple[list[float], list[Quaternion]]  # With virtual
```

#### Example

```python
from interpolatepy import SquadC2, Quaternion
import numpy as np

# Define rotation waypoints
times = [0, 1, 2, 3]
quats = [
    Quaternion.identity(),
    Quaternion.from_angle_axis(np.pi/2, np.array([1, 0, 0])),  # 90° about X
    Quaternion.from_angle_axis(np.pi, np.array([0, 1, 0])),    # 180° about Y  
    Quaternion.from_angle_axis(np.pi/4, np.array([0, 0, 1]))   # 45° about Z
]

# Create C² continuous interpolator
squad = SquadC2(times, quats)

# Evaluate smooth trajectory
t_eval = np.linspace(0, 3, 100)
trajectory = [squad.evaluate(t) for t in t_eval]
angular_vels = [squad.evaluate_velocity(t) for t in t_eval]
```

---

## Specialized Algorithms

Utility algorithms for path-following, coordinate frame computation, and geometric operations.

### Frenet Frame Computation

**File**: `frenet_frame.py`  
**Function**: `compute_trajectory_frames`

#### Theory

Computes Frenet frames (moving coordinate systems) along parametric curves. For a curve p(u), the Frenet frame consists of:

1. **Tangent vector**: T = p'(u)/|p'(u)| (direction of motion)
2. **Normal vector**: N = T'(u)/|T'(u)| (direction of curvature)  
3. **Binormal vector**: B = T × N (completes right-handed frame)

**Applications**:
- Robot end-effector orientation along paths
- Camera orientation for flythrough animations
- Tool orientation in manufacturing

**Extensions**:
- **Tool orientation**: Apply additional rotations (roll, pitch, yaw) to Frenet frame
- **Non-arc-length parameterization**: Handle general parameter u ≠ arc length s

#### API Reference

```python
def compute_trajectory_frames(
    position_func: Callable[[float], tuple[np.ndarray, np.ndarray, np.ndarray]],
    u_values: np.ndarray,
    tool_orientation: float | tuple[float, float, float] | None = None
) -> tuple[np.ndarray, np.ndarray]
```

**Parameters**:
- `position_func`: Function returning (position, first_derivative, second_derivative)
- `u_values`: Parameter values for frame computation
- `tool_orientation`: Additional rotation (angle or RPY tuple)

**Returns**: (points, frames) where frames[i] = [tangent, normal, binormal]

**Helper Functions**:
```python
def helicoidal_trajectory_with_derivatives(u: float, r: float = 2.0, d: float = 0.5)
def circular_trajectory_with_derivatives(u: float, r: float = 2.0)  
def plot_frames(ax, points: np.ndarray, frames: np.ndarray, scale: float = 0.5)
```

#### Example

```python
from interpolatepy import compute_trajectory_frames, helicoidal_trajectory_with_derivatives
import numpy as np

# Define helix parameters
u_values = np.linspace(0, 4*np.pi, 100)

# Compute Frenet frames with tool orientation
points, frames = compute_trajectory_frames(
    helicoidal_trajectory_with_derivatives, 
    u_values,
    tool_orientation=(0.1, 0.2, 0.3)  # Roll, pitch, yaw
)

# Visualize (requires matplotlib)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plot_frames(ax, points, frames, skip=10)
```

### Simple Paths

**File**: `simple_paths.py`  
**Classes**: `LinearPath`, `CircularPath`

#### Theory

Provides analytical path representations for common geometric primitives.

**LinearPath**:
- **Position**: p(s) = pᵢ + (s/L)(pf - pᵢ) where s is arc length
- **Velocity**: dp/ds = (pf - pᵢ)/L (constant unit tangent)
- **Acceleration**: d²p/ds² = 0 (no curvature)

**CircularPath**:  
- **Parameterization**: Uses axis r, center point d, and initial point pᵢ
- **Local coordinates**: Circular motion in plane perpendicular to axis
- **Position**: p(s) = center + R·[R cos(s/R), R sin(s/R), 0]ᵀ
- **Curvature**: κ = 1/R (constant)

#### API Reference

```python
class LinearPath:
    def __init__(self, pi: np.ndarray, pf: np.ndarray)
    
    def position(self, s: float) -> np.ndarray
    def velocity(self, s: float) -> np.ndarray  
    def acceleration(self, s: float) -> np.ndarray
    def evaluate_at(self, s_values: np.ndarray) -> dict[str, np.ndarray]

class CircularPath:
    def __init__(self, r: np.ndarray, d: np.ndarray, pi: np.ndarray)
    # r: axis direction, d: point on axis, pi: point on circle
    
    def position(self, s: float) -> np.ndarray
    def velocity(self, s: float) -> np.ndarray
    def acceleration(self, s: float) -> np.ndarray  
    def evaluate_at(self, s_values: np.ndarray) -> dict[str, np.ndarray]
```

---

## Utilities

Supporting numerical methods and shared functionality.

### Tridiagonal System Solver

**File**: `tridiagonal_inv.py`  
**Function**: `solve_tridiagonal`

#### Theory

Efficient solver for tridiagonal linear systems of the form:

```
[b₁ c₁  0  ...  0 ] [x₁]   [d₁]
[a₂ b₂ c₂  ...  0 ] [x₂]   [d₂]  
[0  a₃ b₃  ...  0 ] [x₃] = [d₃]
[⋮   ⋮  ⋮   ⋱  ⋮ ] [⋮ ]   [⋮ ]
[0   0  0  ... bₙ] [xₙ]   [dₙ]
```

Uses Thomas algorithm (specialized Gaussian elimination) with O(n) complexity instead of O(n³) for general systems.

**Applications in InterpolatePy**:
- Cubic spline acceleration computation
- B-spline control point calculation  
- Smoothing spline parameter estimation

#### API Reference

```python
def solve_tridiagonal(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray
```

**Parameters**:
- `a`: Lower diagonal (length n, first element unused)
- `b`: Main diagonal (length n)  
- `c`: Upper diagonal (length n, last element unused)
- `d`: Right-hand side vector (length n)

**Returns**: Solution vector x

**Note**: Modifies input arrays during computation for memory efficiency.

---

This completes the comprehensive algorithm reference for InterpolatePy. Each algorithm is designed to integrate seamlessly with the others while providing specialized functionality for different trajectory planning scenarios.

For implementation examples and advanced usage patterns, see the `examples/` directory in the repository.