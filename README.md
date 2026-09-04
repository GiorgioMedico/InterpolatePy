# InterpolatePy

![Python](https://img.shields.io/badge/python-3.11+-blue)
[![PyPI Downloads](https://static.pepy.tech/badge/interpolatepy)](https://pepy.tech/projects/interpolatepy)
[![pre-commit](https://github.com/GiorgioMedico/InterpolatePy/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/GiorgioMedico/InterpolatePy/actions/workflows/pre-commit.yml)
[![ci-test](https://github.com/GiorgioMedico/InterpolatePy/actions/workflows/test.yml/badge.svg)](https://github.com/GiorgioMedico/InterpolatePy/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

InterpolatePy is a trajectory-planning and interpolation library for robotics,
animation, and scientific computing. It includes scalar splines, B-spline curve
tools, bounded motion profiles, quaternion interpolation, and 3D path utilities.

The package has a NumPy/SciPy implementation and an optional C++20 backend. Use
the public names exported by `interpolatepy`; they select the available backend
at import time.

## Installation

```bash
python -m pip install InterpolatePy
```

InterpolatePy 3.2.1 requires Python 3.11 or newer, NumPy 1.26 or newer,
SciPy 1.11 or newer, and Matplotlib 3.6 or newer.

## Quick start

```python
import numpy as np

from interpolatepy import CubicSpline
from interpolatepy import DoubleSTrajectory
from interpolatepy import StateParams
from interpolatepy import TrajectoryBounds

# A clamped cubic spline: v0 and vn are endpoint velocities.
spline = CubicSpline(
    [0.0, 1.0, 2.0, 3.0],
    [0.0, 1.5, -0.5, 2.0],
    v0=0.0,
    vn=0.0,
)
times = np.linspace(0.0, 3.0, 100)
positions = spline.evaluate(times)
velocities = spline.evaluate_velocity(times)

# A jerk-limited Double-S motion profile.
state = StateParams(q_0=0.0, q_1=10.0, v_0=0.0, v_1=0.0)
bounds = TrajectoryBounds(v_bound=5.0, a_bound=10.0, j_bound=30.0)
motion = DoubleSTrajectory(state, bounds)
sample_times = np.linspace(0.0, motion.get_duration(), 100)
q, qd, qdd, qddd = motion.evaluate_full(sample_times)
```

`DoubleSTrajectory.evaluate()` returns position only. Use
`evaluate_velocity()`, `evaluate_acceleration()`, and `evaluate_jerk()` for one
component, or `evaluate_full()` for all four.

## What is included

| Area | Public APIs | Typical use |
| --- | --- | --- |
| Scalar splines | `CubicSpline`, `CubicSmoothingSpline`, `CubicSplineWithAcceleration1`, `CubicSplineWithAcceleration2` | Smooth scalar waypoint trajectories and noisy data |
| B-splines | `BSpline`, `BSplineInterpolator`, `CubicBSplineInterpolation`, `ApproximationBSpline`, `SmoothingCubicBSpline` | Parametric curves, interpolation, approximation, and smoothing |
| Motion profiles | `DoubleSTrajectory`, `TrapezoidalTrajectory`, `PolynomialTrajectory`, `ParabolicBlendTrajectory` | Bounded or boundary-conditioned scalar motion |
| Rotations | `Quaternion`, `QuaternionSpline`, `SquadC2`, `LogQuaternionInterpolation`, `ModifiedLogQuaternionInterpolation` | Orientation interpolation without Euler-angle singularities |
| Paths and utilities | `LinearPath`, `CircularPath`, Frenet-frame helpers, `linear_traj`, `solve_tridiagonal` | Geometric paths, moving frames, and numerical helpers |

Degrees 3, 4, and 5 of `BSplineInterpolator`,
`LogQuaternionInterpolation`, and `ModifiedLogQuaternionInterpolation` support
as few as two waypoints in version 3.2.0.

See the [algorithm selection guide](ALGORITHMS.md), the
[full documentation](https://giorgiomedico.github.io/InterpolatePy/), and the
[runnable examples](examples/).

## Optional C++ backend

The package falls back to Python automatically when the extension is absent:

```python
import interpolatepy

print(interpolatepy.HAS_CPP)
```

Set `INTERPOLATEPY_NO_CPP=1` before importing the package to force the Python
implementation. The standard Python package does not need a compiler. Building
the native library or Python extension from a source checkout requires CMake
3.21+, a C++20 compiler, and network access for CMake's fetched dependencies;
see the [installation guide](docs/installation.md#optional-c-backend).

## Development

This project uses [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/GiorgioMedico/InterpolatePy.git
cd InterpolatePy
uv sync
uv run pytest
uv run pre-commit run --all-files
```

Documentation dependencies are in a separate group:

```bash
uv sync --group docs
uv run mkdocs serve
```

Every Python program in `examples/` can also be run directly, for example:

```bash
uv run python examples/double_s_ex.py
```

For the complete workflow, see [Contributing](docs/contributing.md).

## License and citation

InterpolatePy is distributed under the [MIT License](LICENSE).

```bibtex
@misc{InterpolatePy,
  author = {Giorgio Medico},
  title  = {InterpolatePy: Trajectory Planning and Interpolation for Python},
  year   = {2026},
  url    = {https://github.com/GiorgioMedico/InterpolatePy}
}
```
