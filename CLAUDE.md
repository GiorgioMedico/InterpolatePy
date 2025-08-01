# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## AI Guidance

* After receiving tool results, carefully reflect on their quality and determine optimal next steps before proceeding. Use your thinking to plan and iterate based on this new information, and then take the best next action.
* For maximum efficiency, whenever you need to perform multiple independent operations, invoke all relevant tools simultaneously rather than sequentially.
* Before you finish, please verify your solution
* Do what has been asked; nothing more, nothing less.
* NEVER create files unless they're absolutely necessary for achieving your goal.
* ALWAYS prefer editing an existing file to creating a new one.
* NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
* When you update or modify core context files, also update markdown documentation and memory bank
* When asked to commit changes, exclude CLAUDE.md and CLAUDE-*.md referenced memory bank system files from any commits. Never delete these files.

## Memory Bank System

This project uses a structured memory bank system with specialized context files. Always check these files for relevant information before starting work:

### Core Context Files

* **CLAUDE-activeContext.md** - Current session state, goals, and progress (if exists)
* **CLAUDE-patterns.md** - Established code patterns and conventions (if exists)
* **CLAUDE-decisions.md** - Architecture decisions and rationale (if exists)
* **CLAUDE-troubleshooting.md** - Common issues and proven solutions (if exists)
* **CLAUDE-config-variables.md** - Configuration variables reference (if exists)
* **CLAUDE-temp.md** - Temporary scratch pad (only read when referenced)

**Important:** Always reference the active context file first to understand what's currently being worked on and maintain session continuity.

### Memory Bank System Backups

When asked to backup Memory Bank System files, you will copy the core context files above and @.claude settings directory to directory @/path/to/backup-directory. If files already exist in the backup directory, you will overwrite them.

## Project Overview

InterpolatePy is a comprehensive Python library for trajectory planning and interpolation, designed for robotics, animation, and scientific computing. The library provides smooth trajectory generation with precise control over position, velocity, acceleration, and jerk profiles.

## Development Commands

### Setup
```bash
# Install development dependencies
pip install -e '.[all]'
pre-commit install
```

### Testing
```bash
# Run all tests
python -m pytest tests

# Run specific test file
python -m pytest tests/inv_test.py
```

### Code Quality
```bash
# Check code with Ruff
ruff check interpolatepy/

# Format code with Ruff
ruff format interpolatepy/

# Type checking with mypy
mypy interpolatepy/

# Run all pre-commit hooks
pre-commit run --all-files
```

### Running Examples
```bash
# Run main example script
python examples/main.py

# Run specific examples
python examples/cubic_spline_ex.py
python examples/double_s_ex.py
```

## Architecture Overview

InterpolatePy is organized into distinct categories of interpolation and trajectory planning algorithms:

### Module Categories

1. **Spline Interpolation**
   - `cubic_spline.py` - C2 continuous cubic splines with boundary conditions
   - `b_spline.py` - General B-spline curves with configurable degree
   - `b_spline_*` variants - Approximation, cubic, interpolation, smoothing
   
2. **Motion Profiles**
   - `double_s.py` - S-curve trajectories with bounded jerk
   - `trapezoidal.py` - Classic trapezoidal velocity profiles
   - `lin_poly_parabolic.py` - Linear segments with parabolic blends

3. **Polynomial Trajectories**
   - `polynomials.py` - 3rd, 5th, 7th order polynomials with boundary conditions
   - `linear.py` - Simple linear interpolation

4. **Specialized Algorithms**
   - `quat_interp.py` - Quaternion interpolation (SLERP, SQUAD)
   - `frenet_frame.py` - Path-following with Frenet-Serret frames
   - `simple_paths.py` - Linear and circular path utilities

5. **Utilities**
   - `tridiagonal_inv.py` - Tridiagonal system solver (shared utility)

### API Design Patterns

All algorithms follow consistent patterns:

**Initialization Pattern:**
```python
# Constructor takes problem definition parameters
spline = CubicSpline(t_points, q_points, v0=0.0, vn=0.0)
trajectory = DoubleSTrajectory(state_params, bounds)
```

**Evaluation Pattern:**
```python
# Primary evaluation method
position = algorithm.evaluate(t)
velocity = algorithm.evaluate_velocity(t)
acceleration = algorithm.evaluate_acceleration(t)
```

**Configuration with Dataclasses:**
- Modern use of `@dataclass` for parameter organization
- Examples: `TrajectoryBounds`, `StateParams`, `BoundaryCondition`

**Built-in Plotting:**
- Every algorithm class includes `plot()` methods
- Consistent multi-subplot layouts (position, velocity, acceleration)
- Matplotlib integration with customizable visualization

### Code Quality Standards

- **Formatting**: Ruff formatter (Black-compatible)
- **Linting**: Ruff with extensive rule set (F, E, W, I, etc.)
- **Type Checking**: mypy with strict configuration
- **Testing**: pytest with benchmark support
- **Pre-commit**: Automated checks on commit

### Dependencies

- **Core**: NumPy ≥ 2.0, SciPy ≥ 1.15, Matplotlib ≥ 3.10
- **Python**: ≥ 3.10 (uses modern features like type hints, dataclasses)
- **Development**: pytest, ruff, mypy, pre-commit

### Key Implementation Notes

- Each algorithm is self-contained in its own module
- Shared utilities are properly separated (tridiagonal solver)
- Numerical stability considerations (epsilon handling, input validation)
- Comprehensive examples directory with usage patterns
- Extensible design for adding new interpolation methods
