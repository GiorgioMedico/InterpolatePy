# InterpolatePy

[![PyPI Downloads](https://static.pepy.tech/badge/interpolatepy)](https://pepy.tech/projects/interpolatepy)
[![CI Tests](https://github.com/GiorgioMedico/InterpolatePy/actions/workflows/test.yml/badge.svg)](https://github.com/GiorgioMedico/InterpolatePy/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/GiorgioMedico/InterpolatePy/blob/main/LICENSE)

InterpolatePy 3.2.0 provides trajectory planning and interpolation for scalar
motion, parametric curves, 3D paths, and rotations. It runs with a NumPy/SciPy
implementation by default and can transparently use an optional C++20 backend.

## Install

```bash
python -m pip install InterpolatePy
```

Python 3.11 or newer is required. See [Installation](installation.md) for exact
dependency floors, development setup, and native build instructions.

## First trajectory

```python
import numpy as np

from interpolatepy import CubicSpline

spline = CubicSpline(
    [0.0, 1.0, 2.0, 3.0],
    [0.0, 1.5, -0.5, 2.0],
    v0=0.0,
    vn=0.0,
)

t = np.linspace(0.0, 3.0, 100)
q = spline.evaluate(t)
qd = spline.evaluate_velocity(t)
qdd = spline.evaluate_acceleration(t)
```

The endpoint velocity arguments make this a clamped spline. They do not select
natural, zero-curvature boundary conditions.

## Find the right tool

| Need | Recommended API |
| --- | --- |
| Scalar interpolation | `CubicSpline` |
| Noisy scalar data | `CubicSmoothingSpline` |
| Parametric B-spline interpolation | `BSplineInterpolator` |
| Curve approximation or smoothing | `ApproximationBSpline`, `SmoothingCubicBSpline` |
| Jerk-limited motion | `DoubleSTrajectory` |
| Acceleration-limited motion | `TrapezoidalTrajectory` |
| Exact derivative boundary conditions | `PolynomialTrajectory` |
| Orientation keyframes | `QuaternionSpline`, `SquadC2`, logarithmic interpolators |
| 3D line/circle geometry | `LinearPath`, `CircularPath` |
| Moving frames | `compute_trajectory_frames()` |

Read the [User guide](user-guide.md) for API conventions, the
[Algorithms guide](algorithms.md) for selection and theory, or go directly to
the [API reference](api-reference.md).

## Version 3.2 highlights

`BSplineInterpolator`, `LogQuaternionInterpolation`, and
`ModifiedLogQuaternionInterpolation` now support two-point input for degrees 3,
4, and 5. The generated boundary constraints keep the interpolation systems
well posed even when the number of samples is smaller than `degree + 1`.

## Backend selection

```python
import interpolatepy

print(interpolatepy.__version__)
print(interpolatepy.HAS_CPP)
```

Set `INTERPOLATEPY_NO_CPP=1` before starting Python to force the fallback. Code
should import classes from `interpolatepy` so backend routing remains
transparent. See [Architecture](architecture.md) for the import flow and known
backend-surface differences.

## Next steps

- [Quick start](quickstart.md): four small, runnable examples
- [Tutorials](user-guide.md#tutorials): splines, motion, rotations, and paths
- [Examples](examples.md): map of the checked scripts in `examples/`
- [Troubleshooting](troubleshooting.md): common input and build errors
- [Contributing](contributing.md): test, style, docs, and C++ workflows
