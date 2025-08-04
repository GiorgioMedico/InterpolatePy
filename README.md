# InterpolatePy

![Python](https://img.shields.io/badge/python-3.10+-blue)
[![PyPI Downloads](https://static.pepy.tech/badge/interpolatepy)](https://pepy.tech/projects/interpolatepy)
[![pre-commit](https://github.com/GiorgioMedico/InterpolatePy/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/GiorgioMedico/InterpolatePy/actions/workflows/pre-commit.yml)
[![ci-test](https://github.com/GiorgioMedico/InterpolatePy/actions/workflows/test.yml/badge.svg)](https://github.com/GiorgioMedico/InterpolatePy/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Smooth trajectories, precise motion — one library.**
> InterpolatePy brings together classic and modern interpolation techniques for robotics, animation, and scientific computing in a single, easy‑to‑use Python package.

---

## ⭐️ Support the Project

If InterpolatePy saves you time or powers your research, please consider **starring** the repo – it helps others discover the project and motivates future development!


Have you built something cool on top of InterpolatePy? Open an issue or start a discussion – we’d love to showcase community projects.


## Overview

InterpolatePy is a **comprehensive collection of trajectory‑generation algorithms** – from simple linear blends to high‑order B‑splines – with a consistent, NumPy‑friendly API. Designed for robotics, animation, path planning, and data smoothing, it lets you craft trajectories that respect position, velocity, acceleration **and** jerk constraints.

Key design goals:

* **Breadth** – one package for splines *and* motion profiles.
* **Visualization‑ready** – every spline exposes `plot()` helpers built on Matplotlib.
* **Pure Python ≥ 3.10** – no compiled extensions; installs quickly everywhere.

---

## Roadmap

Upcoming features (✅ done, 🚧 planned):

| Status | Feature                                                                   |
| ------ | ------------------------------------------------------------------------- |
| 🚧     | **Bezier curves** – arbitrary degree                                      |
| 🚧     | **Quaternion interpolation**: LERP / SLERP / SQUAD & B‑spline‑quaternions |
| 🚧     | **Linear blends** with quintic/parabolic smoothing                        |
| 🚧     | **Spherical paths** & great‑circle splines                                |

---

## Key Features

### 1 · Spline Interpolation

* **B‑Splines** – cubic, approximating, smoothing.
* **Cubic Splines** – with optional velocity/acceleration endpoint constraints.
* **Global B‑Spline Interpolation** – C², C³, C⁴ continuity (degree 3–5).

### 2 · Motion Profiles

* **Double‑S** (S‑curve) – bounded jerk.
* **Trapezoidal** – classic industrial profile.
* **Polynomial** – 3/5/7‑order with boundary conditions.

### 3 · Path Utilities

* **Linear & Circular paths** in 3‑D.
* **Frenet frames** helper for tool orientation along curves.

---

## Installation

InterpolatePy lives on PyPI. Install the latest stable release with:

```bash
pip install InterpolatePy
```

Development version (with test & dev extras):

```bash
git clone https://github.com/GiorgioMedico/InterpolatePy.git
cd InterpolatePy
pip install -e '.[all]'
```

Optional extras:

```bash
pip install InterpolatePy[test]   # testing only
pip install InterpolatePy[dev]    # linting & build tools
```

---

## Quick Start

```python
from interpolatepy.cubic_spline import CubicSpline

t = [0, 5, 10]
q = [0, 2, 0]

spline = CubicSpline(t, q, v0=0.0, vn=0.0)
print(spline.evaluate(7.5))  # position at t = 7.5 s
spline.plot()                # visualize position/velocity/acceleration
```

---

## Usage Examples

<details>
<summary>Cubic spline with velocity constraints</summary>

```python
from interpolatepy.cubic_spline import CubicSpline

t_points = [0.0, 5.0, 7.0, 10.0]
q_points = [1.0, 3.0, -1.0, 2.0]

s = CubicSpline(t_points, q_points, v0=1.0, vn=0.0)
position = s.evaluate(6.0)
```

</details>

<details>
<summary>Double‑S trajectory</summary>

```python
from interpolatepy.double_s import DoubleSTrajectory, StateParams, TrajectoryBounds

state  = StateParams(q_0=0, q_1=10, v_0=0, v_1=0)
bounds = TrajectoryBounds(v_bound=5, a_bound=10, j_bound=30)
traj   = DoubleSTrajectory(state, bounds)
```

</details>

For more, see the [examples folder](examples/) or the full API docs (coming soon).

---

## Requirements

* Python ≥ 3.10
* NumPy ≥ 2.0
* SciPy ≥ 1.15
* Matplotlib ≥ 3.10

---

## Development

InterpolatePy uses modern Python tooling for development:

* **Code Quality**: Black and isort for formatting, Ruff and mypy for linting and type checking
* **Testing**: pytest for unit tests and benchmarks

To set up the development environment:

```bash
pip install -e '.[all]'
pre-commit install
```

### Running Tests

```bash
python -m pytest tests
```

### Coverage Reports

Generate HTML coverage report to see test coverage:

```bash
python -m pytest tests --cov=interpolatepy --cov-report=html --cov-report=term
```

Then open `htmlcov/index.html` in your browser for an interactive coverage report.

## Contributing

We love pull requests — thanks for helping improve **InterpolatePy**!

1. **Fork** the repository and create a descriptive branch (`feat/my-feature`).
2. **Install** dev dependencies:

   ```bash
   pip install -e '.[all]'
   pre-commit install
   ```
3. **Code** your change, following our style (Black, isort, Ruff, mypy).
4. **Test** with `pytest` and run `pre-commit run --all-files`.
5. **Open** a pull request and explain *why* & *how* your change helps.

For larger ideas, open an issue first so we can discuss direction and scope.

---

## Acknowledgments

InterpolatePy implements algorithms and mathematical concepts primarily from the following authoritative textbooks:

* **Biagiotti, L., & Melchiorri, C.** (2008). *Trajectory Planning for Automatic Machines and Robots*. Springer.
* **Siciliano, B., Sciavicco, L., Villani, L., & Oriolo, G.** (2010). *Robotics: Modelling, Planning and Control*. Springer.

The library's implementation draws heavily from the theoretical frameworks, mathematical formulations, and algorithms presented in these works.

I express my gratitude to these authors for their significant contributions to the field of trajectory planning and robotics, which have made this library possible.

---

## License

InterpolatePy is released under the MIT License – do whatever you want, but please give credit.

---

## Citation

If InterpolatePy contributes to your academic work, consider citing it:

```text
@misc{InterpolatePy,
  author       = {Giorgio Medico},
  title        = {InterpolatePy: Trajectory and Spline Library},
  year         = {2025},
  howpublished = {\url{https://github.com/GiorgioMedico/InterpolatePy}}
}
```
