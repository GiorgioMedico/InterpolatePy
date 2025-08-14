# InterpolatePy

[![PyPI Downloads](https://static.pepy.tech/badge/interpolatepy)](https://pepy.tech/projects/interpolatepy)
[![CI Tests](https://github.com/GiorgioMedico/InterpolatePy/actions/workflows/test.yml/badge.svg)](https://github.com/GiorgioMedico/InterpolatePy/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/downloads/)

**Production-ready trajectory planning and interpolation for robotics, animation, and scientific computing.**

InterpolatePy provides 20+ algorithms for smooth trajectory generation with precise control over position, velocity, acceleration, and jerk. From cubic splines and B-curves to quaternion interpolation and S-curve motion profiles — everything you need for professional motion control.

## ✨ Key Features

<div class="feature-grid">
  <div class="feature-card">
    <h3>⚡ Fast Performance</h3>
    <p>Vectorized NumPy operations with ~1ms evaluation for 1000-point cubic splines. Optimized algorithms for real-time applications.</p>
  </div>
  <div class="feature-card">
    <h3>🎯 Research-Grade Precision</h3>
    <p>C² continuity and bounded derivatives with peer-reviewed algorithms from robotics literature.</p>
  </div>
  <div class="feature-card">
    <h3>📊 Built-in Visualization</h3>
    <p>Comprehensive plotting for every algorithm with position, velocity, acceleration, and jerk profiles.</p>
  </div>
  <div class="feature-card">
    <h3>🔧 Complete Toolkit</h3>
    <p>Splines, motion profiles, quaternions, and path planning unified in one library with consistent APIs.</p>
  </div>
</div>

## 🚀 Quick Start

### Installation

```bash
pip install InterpolatePy
```

### Basic Example

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

# Manual plotting (DoubleSTrajectory doesn't have built-in plot method)
t_eval = np.linspace(0, trajectory.get_duration(), 100)
results = [trajectory.evaluate(t) for t in t_eval]
positions = [r[0] for r in results]
velocities = [r[1] for r in results]

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t_eval, positions)
plt.ylabel('Position')
plt.title('Double-S Trajectory')
plt.subplot(2, 1, 2)
plt.plot(t_eval, velocities)
plt.ylabel('Velocity')
plt.xlabel('Time (s)')

plt.show()
```

## 📚 Algorithm Categories

| Category | Algorithms | Key Features | Use Cases |
|----------|------------|--------------|-----------|
| **🔵 Splines** | Cubic, B-Spline, Smoothing | C² continuity, noise-robust | Waypoint interpolation, curve fitting |
| **⚡ Motion Profiles** | S-curves, Trapezoidal, Polynomial | Bounded derivatives, time-optimal | Industrial automation, robotics |
| **🔄 Quaternions** | SLERP, SQUAD, Splines | Smooth rotations, no gimbal lock | 3D orientation control, animation |
| **🎯 Path Planning** | Linear, Circular, Frenet frames | Geometric primitives, tool orientation | Path following, machining |

!!! tip "Complete Algorithm Reference"
    See our [**Algorithms Guide**](algorithms.md) for detailed mathematical foundations and implementation details for all 22 algorithms.

## 🎯 Who Should Use InterpolatePy?

<div class="feature-grid">
  <div class="feature-card">
    <h3>🤖 Robotics Engineers</h3>
    <p>Motion planning, trajectory optimization, smooth control systems with bounded derivatives.</p>
  </div>
  <div class="feature-card">
    <h3>🎬 Animation Artists</h3>
    <p>Smooth keyframe interpolation, camera paths, character motion with C² continuity.</p>
  </div>
  <div class="feature-card">
    <h3>🔬 Scientists</h3>
    <p>Data smoothing, curve fitting, experimental trajectory analysis with noise robustness.</p>
  </div>
  <div class="feature-card">
    <h3>🏭 Automation Engineers</h3>
    <p>Industrial motion control, CNC machining, conveyor systems with jerk-limited profiles.</p>
  </div>
</div>

## 🔍 Algorithm Overview

### Spline Interpolation
- [`CubicSpline`](api-reference.md#cubic-spline) – Natural cubic splines with boundary conditions
- [`CubicSmoothingSpline`](api-reference.md#cubic-smoothing-spline) – Noise-robust splines with smoothing parameter  
- [`CubicSplineWithAcceleration1/2`](api-reference.md#cubic-spline-with-acceleration) – Bounded acceleration constraints
- [`BSpline`](api-reference.md#b-spline) family – General B-spline curves with configurable degree

### Motion Profiles
- [`DoubleSTrajectory`](api-reference.md#double-s-trajectory) – S-curve profiles with bounded jerk
- [`TrapezoidalTrajectory`](api-reference.md#trapezoidal-trajectory) – Classic trapezoidal velocity profiles
- [`PolynomialTrajectory`](api-reference.md#polynomial-trajectory) – 3rd, 5th, 7th order polynomials

### Quaternion Interpolation  
- [`Quaternion`](api-reference.md#quaternion) – Core quaternion operations with SLERP
- [`QuaternionSpline`](api-reference.md#quaternion-spline) – C²-continuous rotation trajectories
- [`SquadC2`](api-reference.md#squad-c2) – Enhanced SQUAD with zero-clamped boundaries

### Path Planning & Utilities
- [`SimpleLinearPath`](api-reference.md#simple-linear-path), [`SimpleCircularPath`](api-reference.md#simple-circular-path) – 3D geometric primitives
- [`FrenetFrame`](api-reference.md#frenet-frame) – Frenet-Serret frame computation along curves

## 📊 Performance Benchmarks

**Typical Performance on Modern Hardware:**

| Algorithm | Operation | Performance |
|-----------|-----------|-------------|
| Cubic Spline | 1000 points evaluation | ~1ms |
| B-Spline | 10k points evaluation | ~5ms |
| S-Curve Planning | Trajectory generation | ~0.5ms |
| Quaternion SLERP | Interpolation | ~0.1ms |

## 🛠️ Development & Quality

- **Modern Python**: 3.10+ with strict typing and dataclass-based APIs
- **High Test Coverage**: 85%+ coverage with continuous integration
- **Research-Grade**: Peer-reviewed algorithms from robotics literature
- **Production-Ready**: Used in industrial applications and research projects

## 📖 Getting Started

1. **[Installation Guide](installation.md)** - Get up and running quickly
2. **[Quick Start](quickstart.md)** - Your first trajectories in minutes
3. **[User Guide](user-guide.md)** - Comprehensive tutorials and examples
4. **[API Reference](api-reference.md)** - Complete function documentation
5. **[Algorithms](algorithms.md)** - Mathematical foundations and theory

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guide](contributing.md) for details on:

- Setting up the development environment
- Running tests and quality checks
- Submitting pull requests
- Reporting issues

## 📄 License & Citation

InterpolatePy is released under the **MIT License** – free for commercial and academic use.

If you use InterpolatePy in research, please cite:

```bibtex
@misc{InterpolatePy,
  author = {Giorgio Medico},
  title  = {InterpolatePy: Trajectory Planning and Interpolation for Python},
  year   = {2025},
  url    = {https://github.com/GiorgioMedico/InterpolatePy}
}
```

---

*Built with ❤️ for the robotics and scientific computing community.*