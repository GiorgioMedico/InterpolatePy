# Troubleshooting

## Import and environment issues

### `ModuleNotFoundError: No module named 'interpolatepy'`

Check the package with the same interpreter used to run the program:

```bash
python -m pip show InterpolatePy
python -c "import sys; print(sys.executable)"
```

Install into that environment with `python -m pip install InterpolatePy`, or run
`uv sync` from a source checkout.

### A documented top-level name cannot be imported

Check the installed version:

```bash
python -c "import interpolatepy as ip; print(ip.__version__)"
```

This documentation targets 3.2.0. Import public algorithms from `interpolatepy`.
The trapezoidal `TrajectoryParams` is the intentional exception:

```python
from interpolatepy import TrapezoidalTrajectory
from interpolatepy.trapezoidal import TrajectoryParams
```

The top-level `TrajectoryParams` configures multipoint polynomial motion.

## Backend issues

### The C++ backend is not active

```bash
python -c "import interpolatepy as ip; print(ip.HAS_CPP)"
```

`False` is not an error; all core algorithms have a Python implementation. To
activate the extension from a checkout, follow
[Optional C++ backend](installation.md#optional-c-backend). The built extension
must be copied into `interpolatepy/`, not merely left in the CMake build tree.

If a copied extension still fails to load, inspect the original import error:

```bash
python -c "import interpolatepy.interpolatecpp_py"
```

Typical causes are a Python ABI mismatch, a missing shared library, or building
for a different architecture.

### Force fallback behavior

```bash
INTERPOLATEPY_NO_CPP=1 python your_program.py
```

The environment variable is tested for nonemptiness, so do not use
`INTERPOLATEPY_NO_CPP=0` to mean false.

### A helper exists only on the Python backend

The portable API is the common constructor/evaluation surface. Some diagnostic,
refinement, and logarithmic-quaternion helpers are currently Python-only. See
[Adapter layer](architecture.md#adapter-layer) and force the fallback when code
needs them.

## Time and sample validation

### Time points must be strictly increasing

Sort paired data together and decide how duplicate timestamps should be
resolved; do not add arbitrary epsilon offsets without understanding the data:

```python
import numpy as np

t = np.array([0.0, 2.0, 1.0])
q = np.array([0.0, 4.0, 1.0])
order = np.argsort(t)
t = t[order]
q = q[order]

if np.any(np.diff(t) <= 0.0):
    raise ValueError("duplicate timestamps require aggregation")
```

### Time and position lengths differ

Do not silently truncate either sequence. Fix the source data and assert the
relationship before construction:

```python
if len(t_points) != len(q_points):
    raise ValueError("each timestamp needs one position")
```

### Evaluation is outside the domain

Behavior depends on the family:

- scalar cubic splines and Double-S trajectories clamp to their endpoints;
- `LinearPath.position()` clamps arc length to the finite line segment;
- `CircularPath` is periodic and accepts arc length around the circle;
- B-spline evaluation outside `[u_min, u_max]` raises.

Clamp explicitly in application code when that behavior is part of the control
policy, rather than relying on implementation details.

## Spline questions

### The endpoint slope is zero unexpectedly

`CubicSpline` defaults to `v0=0.0` and `vn=0.0`. Supply the desired endpoint
velocities. This is a clamped spline, not a natural spline.

### Smoothing does the opposite of what was expected

For `CubicSmoothingSpline`, `mu=1` gives exact interpolation and values closer
to zero give more smoothing. Zero itself is invalid. Infinite point weights pin
selected samples exactly.

### Method 2 rejects direct acceleration keywords

`CubicSplineWithAcceleration2` takes a parameter object:

```python
from interpolatepy import CubicSplineWithAcceleration2
from interpolatepy import SplineParameters

params = SplineParameters(v0=0.0, vn=0.0, a0=0.5, an=-0.5)
spline = CubicSplineWithAcceleration2(
    [0.0, 1.0, 2.0], [0.0, 1.0, 0.0], params
)
```

Method 1 accepts `a0` and `an` directly.

## B-spline questions

### Knot/control count is invalid

For a base B-spline of degree `p`, provide
`len(knots) == len(control_points) + p + 1`. A clamped knot vector repeats both
end knots `p + 1` times.

### Evaluation says the parameter is outside the valid range

Use the object's computed domain:

```python
import numpy as np

u = np.linspace(curve.u_min, curve.u_max, 100)
points = np.array([curve.evaluate(value) for value in u])
```

### Fewer samples than `degree + 1`

`BSplineInterpolator` supports two or more samples for degrees 3, 4, and 5 as
of 3.2.0. Upgrade if an older release reports insufficient samples.

### A constrained interpolation system is ill-conditioned

Check for coincident points, repeated times, inconsistent cyclic endpoints, and
conflicting endpoint derivative constraints. Rescale very large or small time
domains. Do not add redundant constraints merely to increase the row count.

## Motion-profile questions

### `DoubleSTrajectory.evaluate()` cannot be unpacked

It returns position only:

```python
q, qd, qdd, qddd = trajectory.evaluate_full(t)
```

Use the individual `evaluate_*` methods when only one derivative is needed.

### Bounds are invalid or the move is infeasible

Velocity, acceleration, and jerk bounds are positive magnitudes. Initial and
final speeds must be compatible with the velocity bound and the requested
motion. For trapezoidal trajectories, a requested duration also has to leave
enough time to satisfy boundary speeds and acceleration.

### The trapezoidal parameter class looks wrong

There are two classes named `TrajectoryParams`. Import the trapezoidal one from
`interpolatepy.trapezoidal`; the top-level class is for polynomial multipoint
interpolation.

## Quaternion questions

### Equivalent orientations have opposite components

`q` and `-q` encode the same rotation. Compare rotation matrices or the absolute
dot product:

```python
same_rotation = abs(q1.dot_prod(q2)) > 1.0 - 1e-9
```

### Axis and angle appear swapped

`Quaternion.to_axis_angle()` returns `(axis, angle)`, while
`Quaternion.from_angle_axis(angle, axis)` accepts angle first.

### Logarithmic interpolation has too few keyframes

LQI and mLQI support two or more quaternions for degree 3, 4, or 5 in 3.2.0.
The time list must still match the quaternion list and be strictly increasing.

### Derivatives do not match expected angular velocity

Logarithmic interpolators differentiate their internal coordinate state.
On the Python backend, call `get_physical_kinematics()` for physical angular
velocity and acceleration.

## Plotting and examples

In a server, CI job, or SSH session without a display, select a noninteractive
backend:

```bash
MPLBACKEND=Agg python examples/cubic_spline_ex.py
```

A warning about many open figures can occur when every example is run in one
batch. Run scripts in separate processes, as shown on the
[Examples](examples.md) page.

## Reporting a bug

Include:

- InterpolatePy, Python, NumPy, and SciPy versions;
- `interpolatepy.HAS_CPP` and whether fallback mode changes the result;
- operating system and architecture;
- a minimal reproducible input;
- the complete exception and expected behavior.

Open an issue at <https://github.com/GiorgioMedico/InterpolatePy/issues>.
