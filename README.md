# InterpolatePy

![Python](https://img.shields.io/badge/python-3.10+-blue)
[![PyPI Downloads](https://static.pepy.tech/badge/interpolatepy)](https://pepy.tech/projects/interpolatepy)
[![pre-commit](https://github.com/GiorgioMedico/InterpolatePy/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/GiorgioMedico/InterpolatePy/actions/workflows/pre-commit.yml)
[![ci-test](https://github.com/GiorgioMedico/InterpolatePy/actions/workflows/test.yml/badge.svg)](https://github.com/GiorgioMedico/InterpolatePy/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Production-ready trajectory planning and interpolation for robotics, animation, and scientific computing.**

InterpolatePy provides 20+ algorithms for smooth trajectory generation with precise control over position, velocity, acceleration, and jerk. From cubic splines and B-curves to quaternion interpolation and S-curve motion profiles — everything you need for professional motion control.

**⚡ Fast:** Vectorized NumPy operations, ~1ms for 1000-point cubic splines  
**🎯 Precise:** Research-grade algorithms with C² continuity and bounded derivatives  
**📊 Visual:** Built-in plotting for every algorithm  
**🔧 Complete:** Splines, motion profiles, quaternions, and path planning in one library

---

## Installation

```bash
pip install InterpolatePy
```

**Requirements:** Python ≥3.10, NumPy ≥2.0, SciPy ≥1.15, Matplotlib ≥3.10

<details>
<summary><strong>Development Installation</strong></summary>

```bash
git clone https://github.com/GiorgioMedico/InterpolatePy.git
cd InterpolatePy
pip install -e '.[all]'  # Includes testing and development tools
```
</details>

## Quick Start

```python
import numpy as np
import matplotlib.pyplot as plt
from interpolatepy import CubicSpline, DoubleSTrajectory, StateParams, TrajectoryBounds

# Smooth spline through waypoints
t_points = [0.0, 5.0, 10.0, 15.0]
q_points = [0.0, 2.0, -1.0, 3.0]
spline = CubicSpline(t_points, q_points, v0=0.0, vn=0.0)

# Evaluate at any time
position = spline.evaluate(7.5)
velocity = spline.evaluate_velocity(7.5)
acceleration = spline.evaluate_acceleration(7.5)

# Built-in visualization
spline.plot()

# S-curve motion profile (jerk-limited)
state = StateParams(q_0=0.0, q_1=10.0, v_0=0.0, v_1=0.0)
bounds = TrajectoryBounds(v_bound=5.0, a_bound=10.0, j_bound=30.0)
trajectory = DoubleSTrajectory(state, bounds)

print(f"Duration: {trajectory.get_duration():.2f}s")
trajectory.plot()

plt.show()
```

---

## Algorithm Overview

| Category | Algorithms | Key Features | Use Cases |
|----------|------------|--------------|-----------|
| **🔵 Splines** | Cubic, B-Spline, Smoothing | C² continuity, noise-robust | Waypoint interpolation, curve fitting |
| **⚡ Motion Profiles** | S-curves, Trapezoidal, Polynomial | Bounded derivatives, time-optimal | Industrial automation, robotics |
| **🔄 Quaternions** | SLERP, SQUAD, Splines | Smooth rotations, no gimbal lock | 3D orientation control, animation |
| **🎯 Path Planning** | Linear, Circular, Frenet frames | Geometric primitives, tool orientation | Path following, machining |

📚 **[Complete Algorithms Reference →](ALGORITHMS.md)**  
*Detailed technical documentation, mathematical foundations, and implementation details for all 22 algorithms*

<details>
<summary><strong>Complete Algorithm List</strong></summary>

### Spline Interpolation
- `CubicSpline` – Natural cubic splines with boundary conditions
- `CubicSmoothingSpline` – Noise-robust splines with smoothing parameter  
- `CubicSplineWithAcceleration1/2` – Bounded acceleration constraints
- `BSpline` – General B-spline curves with configurable degree
- `ApproximationBSpline`, `CubicBSpline`, `InterpolationBSpline`, `SmoothingCubicBSpline`

### Motion Profiles
- `DoubleSTrajectory` – S-curve profiles with bounded jerk
- `TrapezoidalTrajectory` – Classic trapezoidal velocity profiles
- `PolynomialTrajectory` – 3rd, 5th, 7th order polynomials
- `LinearPolyParabolicTrajectory` – Linear segments with parabolic blends

### Quaternion Interpolation  
- `Quaternion` – Core quaternion operations with SLERP
- `QuaternionSpline` – C²-continuous rotation trajectories
- `SquadC2` – Enhanced SQUAD with zero-clamped boundaries
- `LogQuaternion` – Logarithmic quaternion methods

### Path Planning & Utilities
- `SimpleLinearPath`, `SimpleCircularPath` – 3D geometric primitives
- `FrenetFrame` – Frenet-Serret frame computation along curves
- `LinearInterpolation` – Basic linear interpolation
- `TridiagonalInverse` – Efficient tridiagonal system solver

</details>

## Advanced Examples

<details>
<summary><strong>Quaternion Rotation Interpolation</strong></summary>

```python
import matplotlib.pyplot as plt
from interpolatepy import QuaternionSpline, Quaternion

# Define rotation waypoints
orientations = [
    Quaternion.identity(),
    Quaternion.from_euler_angles(0.5, 0.3, 0.1),
    Quaternion.from_euler_angles(1.0, -0.2, 0.8)
]
times = [0.0, 2.0, 5.0]

# Smooth quaternion trajectory with C² continuity
quat_spline = QuaternionSpline(times, orientations, method="squad")

# Evaluate at any time
orientation = quat_spline.evaluate(3.5)
angular_velocity = quat_spline.evaluate_angular_velocity(3.5)

quat_spline.plot()
plt.show()
```
</details>

<details>
<summary><strong>B-Spline Curve Fitting</strong></summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from interpolatepy import SmoothingCubicBSpline

# Fit smooth curve to noisy data
t = np.linspace(0, 10, 50)
q = np.sin(t) + 0.1 * np.random.randn(50)

bspline = SmoothingCubicBSpline(t, q, smoothing=0.01)
bspline.plot()
plt.show()
```
</details>

<details>
<summary><strong>Industrial Motion Planning</strong></summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from interpolatepy import DoubleSTrajectory, StateParams, TrajectoryBounds

# Jerk-limited S-curve for smooth industrial motion
state = StateParams(q_0=0.0, q_1=50.0, v_0=0.0, v_1=0.0)
bounds = TrajectoryBounds(v_bound=10.0, a_bound=5.0, j_bound=2.0)

trajectory = DoubleSTrajectory(state, bounds)
print(f"Duration: {trajectory.get_duration():.2f}s")

# Evaluate trajectory
t_eval = np.linspace(0, trajectory.get_duration(), 1000)
positions = [trajectory.evaluate(t) for t in t_eval]
velocities = [trajectory.evaluate_velocity(t) for t in t_eval]

trajectory.plot()
plt.show()
```
</details>

## Who Should Use InterpolatePy?

**🤖 Robotics Engineers:** Motion planning, trajectory optimization, smooth control  
**🎬 Animation Artists:** Smooth keyframe interpolation, camera paths, character motion  
**🔬 Scientists:** Data smoothing, curve fitting, experimental trajectory analysis  
**🏭 Automation Engineers:** Industrial motion control, CNC machining, conveyor systems  

---

## Performance & Quality

- **Fast:** Vectorized NumPy operations, optimized algorithms
- **Reliable:** 85%+ test coverage, continuous integration
- **Modern:** Python 3.10+, strict typing, dataclass-based APIs
- **Research-grade:** Peer-reviewed algorithms from robotics literature

**Typical Performance:**
- Cubic spline (1000 points): ~1ms
- B-spline evaluation (10k points): ~5ms
- S-curve trajectory planning: ~0.5ms

---

## Development

<details>
<summary><strong>Development Setup</strong></summary>

```bash
git clone https://github.com/GiorgioMedico/InterpolatePy.git
cd InterpolatePy
pip install -e '.[all]'
pre-commit install

# Run tests
python -m pytest tests/

# Run tests with coverage
python -m pytest tests/ --cov=interpolatepy --cov-report=html --cov-report=term

# Code quality
ruff format interpolatepy/
ruff check interpolatepy/
mypy interpolatepy/

```
</details>

---

## Contributing

Contributions welcome! Please:

1. Fork the repo and create a feature branch
2. Install dev dependencies: `pip install -e '.[all]'`
3. Follow existing patterns and add tests
4. Run `pre-commit run --all-files` before submitting
5. Open a pull request with clear description

For major changes, open an issue first to discuss the approach.

---

## Support the Project

⭐ **Star the repo** if InterpolatePy helps your work!  
🐛 **Report issues** on [GitHub Issues](https://github.com/GiorgioMedico/InterpolatePy/issues)  
💬 **Join discussions** to share your use cases and feedback  

---

## License & Citation

**MIT License** – Free for commercial and academic use. See [LICENSE](LICENSE) for details.

If you use InterpolatePy in research, please cite:

```bibtex
@misc{InterpolatePy,
  author = {Giorgio Medico},
  title  = {InterpolatePy: Trajectory Planning and Interpolation for Python},
  year   = {2025},
  url    = {https://github.com/GiorgioMedico/InterpolatePy}
}
```

<details>
<summary><strong>Academic References</strong></summary>

This library implements algorithms from:

**Robotics & Trajectory Planning:**
- Biagiotti & Melchiorri (2008). *Trajectory Planning for Automatic Machines and Robots*
- Siciliano et al. (2010). *Robotics: Modelling, Planning and Control*

**Quaternion Interpolation:**
- Parker et al. (2023). "Logarithm-Based Methods for Interpolating Quaternion Time Series"
- Wittmann et al. (2023). "Spherical Cubic Blends: C²-Continuous Quaternion Interpolation"
- Dam, E. B., Koch, M., & Lillholm, M. (1998). "Quaternions, Interpolation and Animation"

</details>

---

*Built with ❤️ for the robotics and scientific computing community.*
