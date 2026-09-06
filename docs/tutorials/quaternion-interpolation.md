# Quaternion interpolation

Quaternions avoid Euler-angle singularities, but introduce a double cover: `q`
and `-q` represent the same orientation. InterpolatePy's trajectory classes
choose consistent signs across keyframes.

## Construct and convert orientations

```python
import numpy as np

from interpolatepy import Quaternion

identity = Quaternion.identity()
about_z = Quaternion.from_angle_axis(
    np.pi / 2.0, np.array([0.0, 0.0, 1.0])
)
from_rpy = Quaternion.from_euler_angles(roll=0.2, pitch=0.3, yaw=0.4)

matrix = from_rpy.to_rotation_matrix()
roll, pitch, yaw = from_rpy.to_euler_angles()
axis, angle = about_z.to_axis_angle()

assert matrix.shape == (3, 3)
assert np.isclose(angle, np.pi / 2.0)
assert np.allclose(axis, [0.0, 0.0, 1.0])
```

Notice the asymmetric naming: `from_angle_axis(angle, axis)` accepts angle
first, while `to_axis_angle()` returns axis first.

Normalize arbitrary components with `.unit()` before treating them as a
rotation.

## SLERP between two orientations

```python
import numpy as np

from interpolatepy import Quaternion

start = Quaternion.identity()
end = Quaternion.from_angle_axis(np.pi, np.array([0.0, 0.0, 1.0]))
halfway = start.slerp(end, 0.5)
axis, angle = halfway.to_axis_angle()

assert np.isclose(angle, np.pi / 2.0)
```

The interpolation parameter is dimensionless and lies between zero and one.

## QuaternionSpline keyframes

```python
from interpolatepy import Quaternion
from interpolatepy import QuaternionSpline

times = [0.0, 1.0, 2.0, 3.0, 4.0]
orientations = [
    Quaternion.identity(),
    Quaternion.from_euler_angles(0.1, 0.2, 0.1),
    Quaternion.from_euler_angles(0.3, 0.5, 0.4),
    Quaternion.from_euler_angles(0.2, 0.7, 0.8),
    Quaternion.from_euler_angles(0.0, 0.4, 1.0),
]
spline = QuaternionSpline(times, orientations, interpolation_method="auto")

q = spline.evaluate(2.5)
omega = spline.evaluate_velocity(2.5)
alpha = spline.evaluate_acceleration(2.5)
```

The available method strings are `"slerp"`, `"squad"`, and `"auto"`. In
SQUAD or automatic mode, boundary segments use SLERP and eligible interior
segments use SQUAD. With too few keyframes for SQUAD, evaluation falls back to
SLERP.

`interpolate_at_time()` also returns a status code for legacy callers;
`evaluate()` is the protocol-compatible interface.

## SquadC2

Use `SquadC2` when zero-clamped boundaries and smoother angular derivatives are
important:

```python
import numpy as np

from interpolatepy import Quaternion
from interpolatepy import SquadC2

times = [0.0, 1.0, 2.0, 3.0]
orientations = [
    Quaternion.identity(),
    Quaternion.from_angle_axis(0.4, np.array([1.0, 0.0, 0.0])),
    Quaternion.from_angle_axis(0.8, np.array([0.0, 1.0, 0.0])),
    Quaternion.from_angle_axis(1.0, np.array([0.0, 0.0, 1.0])),
]
spline = SquadC2(times, orientations)

t = 1.5
q = spline.evaluate(t)
omega = spline.evaluate_velocity(t)
alpha = spline.evaluate_acceleration(t)
assert omega.shape == (3,)
assert alpha.shape == (3,)
```

Input quaternions are normalized by default. `validate_continuity` controls the
implementation's construction-time continuity diagnostics.

## Logarithmic quaternion interpolation

LQI unwraps keyframes into continuous rotation vectors and interpolates those
vectors with a B-spline:

```python
import numpy as np

from interpolatepy import LogQuaternionInterpolation
from interpolatepy import Quaternion

times = [0.0, 2.0]
orientations = [
    Quaternion.identity(),
    Quaternion.from_angle_axis(1.2, np.array([0.0, 0.0, 1.0])),
]
interpolator = LogQuaternionInterpolation(times, orientations, degree=5)

q = interpolator.evaluate(1.0)
rotation_vector_rate = interpolator.evaluate_velocity(1.0)
assert rotation_vector_rate.shape == (3,)
```

Degrees 3, 4, and 5 all accept two or more keyframes in 3.2.0.

On the Python backend, distinguish rotation-vector derivatives from physical
angular kinematics:

```python
import numpy as np

from interpolatepy import Quaternion
from interpolatepy.quaternion.logarithmic import LogQuaternionInterpolation

times = [0.0, 1.0, 2.0]
orientations = [
    Quaternion.identity(),
    Quaternion.from_euler_angles(0.2, 0.3, 0.1),
    Quaternion.from_euler_angles(0.5, 0.4, 0.7),
]
interpolator = LogQuaternionInterpolation(times, orientations)
omega, alpha = interpolator.get_physical_kinematics(1.0)
assert omega.shape == (3,)
assert alpha.shape == (3,)
```

`get_physical_kinematics()` and acceleration boundary arguments are currently
Python-backend helpers. Force the fallback for code that relies on them.

## Modified logarithmic interpolation

mLQI splines the angle and axis separately. Its internal derivative vector has
four components: angle plus three axis components.

```python
import numpy as np

from interpolatepy import ModifiedLogQuaternionInterpolation
from interpolatepy import Quaternion

times = [0.0, 1.0, 2.0]
orientations = [
    Quaternion.identity(),
    Quaternion.from_angle_axis(0.5, np.array([1.0, 0.0, 0.0])),
    Quaternion.from_angle_axis(1.0, np.array([0.0, 1.0, 0.0])),
]
interpolator = ModifiedLogQuaternionInterpolation(
    times, orientations, degree=3, normalize_axis=True
)

q = interpolator.evaluate(0.75)
coordinate_rate = interpolator.evaluate_velocity(0.75)
assert coordinate_rate.shape == (4,)
```

Keep `normalize_axis=True` unless the interpolated axis is already known to
stay unit length.

## SPRING minimum-curvature interpolation

`SpringQuaternionInterpolation` implements SPRING (*Spherical Interpolation
using Numerical Gradient descent*) from Dam, Koch, and Lillholm's 1998 report
*Quaternions, Interpolation and Animation*. It starts with piecewise SLERP and
numerically relaxes the in-between frames to minimize tangential curvature on
the unit-quaternion sphere. Keyframes remain fixed.

```python
import numpy as np

from interpolatepy import Quaternion
from interpolatepy import SpringConfig
from interpolatepy import SpringQuaternionInterpolation

times = [0.0, 1.0, 2.0, 3.0]
orientations = [
    Quaternion.identity(),
    Quaternion.from_angle_axis(0.8, np.array([1.0, 0.0, 0.0])),
    Quaternion.from_angle_axis(1.0, np.array([0.0, 1.0, 0.0])),
    Quaternion.from_angle_axis(1.2, np.array([0.0, 0.0, 1.0])),
]
spring = SpringQuaternionInterpolation(
    times,
    orientations,
    SpringConfig(num_samples=81, iterations=300),
)

q = spring.evaluate(1.5)
omega = spring.evaluate_velocity(1.5)
alpha = spring.evaluate_acceleration(1.5)
assert np.isclose(q.norm(), 1.0)
assert spring.final_energy <= spring.initial_energy
```

SPRING is a global, iterative method: changing one keyframe can affect the
whole curve, and increasing `num_samples` or `iterations` increases setup
cost. Intermediate values are SLERP evaluations of the optimized discrete
frames; angular derivatives are numerical estimates. The default three-level
solver refines a coarse curve approximately fivefold at each level. Optimized
coarse frames are held fixed at the next level, following the report's
multi-step procedure. Set `refinement_levels=1` for a one-stage solve.
Refinement uses the actual spacing between selected frames when computing
second differences. Convergence is checked on the final grid before starting
coarse optimization, so an already-converged curve is preserved.
If refinement increases curvature energy on the final grid, the final stage
restarts from the original piecewise-SLERP curve with only keyframes fixed.

The compiled extension runs SPRING in C++ automatically when `HAS_CPP` is
true. Set `INTERPOLATEPY_NO_CPP=1` to select the NumPy reference implementation.

`refinement_sample_counts` and `stage_energy_history` are immutable tuples for
inspecting convergence. `energy_history` is the final-stage history, while
`initial_energy` and `final_energy` compare the original piecewise-SLERP curve
and the final optimized curve on the same final grid.

Run the visual and timing comparison against multiple shooting, piecewise
SLERP, SQUAD, and SQUAD-C2:

```bash
uv run python examples/spring_quaternion_ex.py
```

The example plots orientation paths, physical angular speed, and a common
discrete tangential-curvature measure. It reports median construction and
1,000-evaluation batch times separately and labels the active backend. Treat
the numbers as measurements of the current machine, not universal performance
claims.

### A faster solver for the same discrete SPRING problem

Use `SpringConfig(solver="gauss_newton")` to accelerate SPRING without
changing its sample lattice, chord-based parametrization, curvature weights,
norm penalty, fixed coarse samples, or final starting state:

```python
from interpolatepy import SpringConfig, SpringQuaternionInterpolation

spring = SpringQuaternionInterpolation(
    times, orientations,
    SpringConfig(num_samples=1001, solver="gauss_newton"),
)
print(spring.converged, spring.stage_gradient_norms[-1])
```

The original `"gradient_descent"` solver remains the default. Both options
run **identical coarse solves**; only the final-grid minimizer changes.
Gauss-Newton uses analytic Jacobians of the same weighted tangential-curvature
and norm-penalty residuals. Each residual involves at most three neighboring
quaternions, so its normal matrix has a fixed bandwidth. A banded factorization
avoids a dense solve. The number of optimization variables still depends on
the sample count: this is not the continuous multiple-shooting algorithm below.

Both solvers cache each grid's curvature coefficients and skip gradient
accumulation for rejected line-search trials, except when a near-roundoff
acceptance decision needs the gradient. The Python implementation also batches
the existing log/exp SLERP operations while copying fixed coarse samples
unchanged. These implementation optimizations retain the objective, iteration
budgets, stopping tolerances, and step-acceptance rules.

`converged` checks the Euclidean norm of the free gradient on the **final grid
with its fixed anchors**, before output normalization. It does not imply that
all coarse solves converged or that a global minimum was found. Inspect
`stage_gradient_norms` for each level. `energy_history` includes the full
optimization energy, including the norm penalty; `final_energy` reports
curvature after normalizing the output quaternions.

Two solvers need not agree after the same finite iteration count. Compare them
after convergence, and measure the actual angular difference. The optional
`final_iterations` overrides only the final-grid budget without changing the
coarse stages. With this override, `iterations_run` may exceed `iterations`.
Setting it to `None` (the default) retains the original shared budget; `0`
disables final-grid updates. This makes an equal-input comparison possible
without inadvertently changing the coarse anchors when increasing the final
solve budget.

```bash
uv run python examples/spring_solver_benchmark.py
INTERPOLATEPY_NO_CPP=1 uv run python examples/spring_solver_benchmark.py
```

The benchmark uses identical inputs and a common final gradient tolerance.
It verifies identical coarse histories and starting energy, and measures
orientation differences at 2,001 times. It reports a speedup only when both
solvers converge and satisfy the specified angular-agreement limit. This
checks numerical agreement for that case, not universal uniqueness of the
nonconvex minimum. The continuous shooting comparison below is a different
experiment and does **not** establish equivalence to discrete SPRING.

## Multiple shooting for natural Riemannian cubics

`ShootingQuaternionInterpolation` solves a continuous rotation interpolation
problem with only nine unknowns per keyframe interval, independently of the
number of output frames. It is available in C++ and NumPy/SciPy with the same
public API:

```python
from interpolatepy import Quaternion, ShootingConfig, ShootingQuaternionInterpolation

times = [0.0, 1.0, 2.0, 3.0]
orientations = [
    Quaternion.identity(),
    Quaternion.from_euler_angles(0.9, 0.1, 0.2),
    Quaternion.from_euler_angles(0.2, 1.1, 0.5),
    Quaternion.from_euler_angles(-0.4, 0.3, 1.4),
]
curve = ShootingQuaternionInterpolation(times, orientations, ShootingConfig(tolerance=1e-8))
q = curve.evaluate(1.5)
omega = curve.evaluate_velocity(1.5)      # body angular velocity, rad/s
alpha = curve.evaluate_acceleration(1.5) # body angular acceleration, rad/s^2
sample_times, samples = curve.generate_trajectory(1001)  # no new solve
assert curve.num_variables == 27
assert curve.residual_norm <= curve.config.tolerance
```

The bi-invariant Riemannian cubic equation is
`V''' + [V, V''] = 0`, or after integration,
`V'' + [V, V'] = C` on each interval. Here `q' = q V`, and the
pure-quaternion bracket is twice the vector cross product. See
[*Riemannian cubics in quadratic matrix Lie groups*](https://www.sciencedirect.com/science/article/abs/pii/S0096300320300515)
for the continuous cubic equations.

For a segment of duration `h`, the implementation uses local time
`u = (t - t_i)/h` and unknowns `(v_0, a_0, c) = (h V_0, h^2 V'_0, h^3 C)`.
It integrates `q_u = q v`, `v_u = a`, `a_u = c - 2 cross(v, a)`.
Each segment starts at its prescribed keyframe. Quaternion-log endpoint
residuals enforce orientation matching; derivative residuals match physical
velocity and acceleration across knots. Zero acceleration at both outer
endpoints supplies the natural boundary conditions. Constants `c` may differ
between segments. With `K` keyframes this gives a square `9*(K-1)` system,
solved by sparse Newton with backtracking and analytic ODE sensitivities.

RK4 starts at `integration_steps=16` steps per segment. Each converged solve
is checked by reintegration on a finer grid, refining up to
`max_integration_steps=1024`. `tolerance` bounds the maximum component of the
dimensionless matching residual (orientation uses half-angle quaternion logs;
derivatives use local time scaling), not a uniform physical-unit error bound
on the entire curve. `max_iterations=30` limits accepted Newton steps across
all grids. Nonconvergence or failed accuracy verification raises `RuntimeError`;
increase the appropriate limit or inspect difficult/near-antipodal data.
`iterations_run`, `integration_steps`, and `residual_norm` report the solve;
`acceleration_energy` approximates the integral of squared physical angular
acceleration. Cached integration states provide evaluation and derivatives
without finite differences or another optimization.

This is a **local stationary curve** of the continuous squared covariant
acceleration functional with free endpoint velocities. It is not guaranteed
to be the global minimizer, and its objective differs from sampled SPRING.
Construction can be faster than dense SPRING with the compiled backend;
coarse SPRING or closed-form SQUAD can still be cheaper. The Python reference
has additional integration-loop overhead. Measure your actual workload:

```bash
uv run python examples/shooting_quaternion_benchmark.py
INTERPOLATEPY_NO_CPP=1 uv run python examples/shooting_quaternion_benchmark.py
```

## Compare orientations correctly

Component equality rejects the equivalent pair `q` and `-q`. For unit
quaternions, use the absolute dot product:

```python
from interpolatepy import Quaternion

q1 = Quaternion.identity()
q2 = -q1
same_orientation = abs(q1.dot_prod(q2)) > 1.0 - 1e-9
assert same_orientation
```

## Choosing a method

| Need | Method |
| --- | --- |
| One segment | `Quaternion.slerp()` |
| Simple keyframe API | `QuaternionSpline` |
| Zero-clamped smooth SQUAD construction | `SquadC2` |
| Global numerical minimum-curvature curve | `SpringQuaternionInterpolation` |
| Continuous natural cubic with analytic angular derivatives | `ShootingQuaternionInterpolation` |
| Continuous rotation-vector representation | LQI |
| Separate angle/axis state | mLQI |

Rotations close to 180 degrees are intrinsically branch-sensitive. Inspect the
actual orientation path, angular velocity, and angular acceleration for the
chosen keyframes rather than selecting solely by method name.
