# Changelog

All notable changes to InterpolatePy will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - 2025-08-14

### Added
- **C++ backend** with pybind11 bindings for performance-critical algorithms
  - Eigen-based C++20 implementation of all spline, B-spline, motion profile, path, and quaternion algorithms
  - Transparent adapter layer (`_adapters/`) preserving the existing Python API
  - Runtime backend detection via `HAS_CPP` flag
  - Pure-Python fallback when C++ extension is unavailable
  - `INTERPOLATEPY_NO_CPP=1` environment variable to force pure-Python mode
  - 142 C++ unit tests with Catch2 framework
  - 16 C++ example programs
- **Protocol definitions** (`protocols.py`) for structural typing
  - `ScalarTrajectory`, `CurveEvaluator`, `GeometricPath`, `QuaternionTrajectory`, `TrajectoryFunction`
- **Duration-based `TrapezoidalTrajectory` constructor**

### Changed
- Minimum Python version updated to 3.11
- Updated dependencies: NumPy >=2.3.0, SciPy >=1.16.0, Matplotlib >=3.10.5
- Backend-aware import system via `_api.py` conditionally routes to C++ or pure-Python

## [2.0.0] - 2025-08-06

### Added
- **Complete MkDocs documentation** with Material theme
  - Comprehensive API reference with auto-generated docs
  - Step-by-step tutorials for all algorithm categories
  - Real-world examples gallery
  - Algorithm theory and mathematical foundations
- **Enhanced quaternion interpolation**
  - `SquadC2` class for C²-continuous rotation trajectories
  - Improved `LogQuaternionInterpolation` with better continuity handling
  - `ModifiedLogQuaternionInterpolation` for decoupled angle/axis interpolation
- **Advanced spline variants**
  - `CubicSplineWithAcceleration1` and `CubicSplineWithAcceleration2`
  - Enhanced B-spline family with multiple degrees and boundary conditions
  - Automatic smoothing parameter selection with `smoothing_spline_with_tolerance`
- **Motion profile improvements**
  - Enhanced `DoubleSTrajectory` with better phase detection
  - `TrapezoidalTrajectory` with duration and velocity-constrained modes
  - Complete polynomial trajectory family (3rd, 5th, 7th order)
- **Path planning utilities**
  - `LinearPath` and `CircularPath` geometric primitives
  - `FrenetFrame` computation for tool orientation along curves
  - `ParabolicBlendTrajectory` for via-point trajectories
- **Developer experience**
  - Comprehensive type hints throughout codebase
  - 85%+ test coverage with performance benchmarks
  - Pre-commit hooks and automated code quality checks
  - Detailed contributing guidelines

### Changed
- **API consistency improvements**
  - Standardized evaluation methods across all algorithms
  - Uniform parameter naming and boundary condition handling
  - Better error messages and input validation
- **Performance optimizations**
  - Vectorized evaluation for all algorithms
  - Memory-efficient coefficient storage
  - Optimized tridiagonal solver implementation
- **Documentation overhaul**
  - NumPy-style docstrings for all public methods
  - Comprehensive examples for every algorithm
  - Mathematical theory documentation
  - Performance comparison guides

### Fixed
- Numerical stability issues in edge cases
- Boundary condition handling for acceleration-constrained splines
- Quaternion double-cover handling in interpolation
- Memory leaks in large trajectory evaluations

### Breaking Changes
- Some internal API changes for consistency
- Updated minimum Python version to 3.11
- Renamed some utility functions for clarity

## [1.1.0] - 2025-05-17

### Added
- Initial quaternion interpolation support
- B-spline approximation methods
- Enhanced smoothing spline functionality

### Changed
- Improved error handling and validation
- Better plotting functionality

### Fixed
- Edge cases in cubic spline interpolation
- Performance issues with large datasets

## [1.0.1] - 2025-03-26

### Added
- Initial release of InterpolatePy
- Core cubic spline interpolation
- Basic motion profiles (S-curve, trapezoidal)
- Fundamental polynomial trajectories
- Essential utility functions

### Features
- Clean, intuitive API design
- Comprehensive test suite
- Basic documentation and examples
- MIT license for open-source use

---

## Development Guidelines

### Version Numbering

InterpolatePy follows [Semantic Versioning](https://semver.org/):

- **MAJOR.MINOR.PATCH** (e.g., 2.1.3)
- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Pre-release Versions

- **Alpha**: `2.1.0a1` - Early development, API may change
- **Beta**: `2.1.0b1` - Feature complete, testing phase
- **Release Candidate**: `2.1.0rc1` - Final testing before release

### Release Process

1. **Feature Development**: Work on `dev` branch
2. **Testing**: Comprehensive test suite must pass
3. **Documentation**: Update docs for all changes
4. **Changelog**: Document all notable changes
5. **Version Bump**: Update version in `interpolatepy/version.py`
6. **Release**: Create GitHub release and publish to PyPI

### Deprecation Policy

- **Deprecation Warning**: Mark features as deprecated one major version before removal
- **Migration Guide**: Provide clear migration paths for breaking changes
- **Backward Compatibility**: Maintain compatibility within major versions when possible

---

## Migration Guides

### Migrating from 1.x to 2.0

#### API Changes

**Old (1.x)**:
```python
from interpolatepy import CubicSpline
spline = CubicSpline(t_points, q_points)
# Evaluation method names varied
result = spline.eval(t)
```

**New (2.0)**:
```python
from interpolatepy import CubicSpline
spline = CubicSpline(t_points, q_points, v0=0.0, vn=0.0)
# Consistent evaluation methods
result = spline.evaluate(t)
```

#### Parameter Changes

- Boundary conditions now explicit (v0, vn parameters)
- Consistent parameter naming across all algorithms
- Better default values for optional parameters

#### New Features to Adopt

1. **Enhanced Documentation**: Check new tutorials and examples
2. **Quaternion Interpolation**: Use for 3D rotation trajectories
3. **Advanced Splines**: Try acceleration-constrained variants
4. **Vectorized Evaluation**: Use array inputs for better performance

#### Performance Improvements

- All algorithms now support vectorized evaluation
- Memory usage reduced by 20-30% for large trajectories
- Evaluation speed improved by 50-100% in many cases

---

## Roadmap

### Completed

- [x] **C++ Backend**: Native C++20 implementation with pybind11 bindings (v3.0.0)
- [x] **Protocol Definitions**: Structural typing for algorithm interchangeability (v3.0.0)

### Planned Features

#### Version 3.1 (Next Minor Release)
- [ ] **Enhanced B-splines**: NURBS support and surface interpolation
- [ ] **Optimization Integration**: Interface with scipy.optimize for trajectory optimization
- [ ] **Extended Examples**: More industry-specific use cases
- [ ] **Pre-built Wheels**: Distribute pre-compiled C++ extensions via PyPI

#### Version 3.2
- [ ] **Multi-dimensional Trajectories**: Native support for N-dimensional spaces
- [ ] **Constraint Handling**: General inequality constraints on derivatives
- [ ] **Adaptive Algorithms**: Automatic degree/parameter selection

#### Version 3.0 (Future Major Release)
- [ ] **Modern Python Features**: Python 3.12+ features and optimizations
- [ ] **API Modernization**: Potential breaking changes for consistency
- [ ] **Advanced Algorithms**: Research-grade methods from latest literature
- [ ] **Integration Ecosystem**: Better integration with robotics frameworks

### Community Contributions

We welcome contributions in these areas:

- **New Algorithms**: Implement methods from recent research papers
- **Performance**: Optimization and benchmarking improvements
- **Examples**: Real-world applications and use cases
- **Documentation**: Tutorials, guides, and explanations
- **Testing**: Edge cases, performance tests, and validation

See our [Contributing Guide](contributing.md) for details on how to contribute.

---

## Support and Migration Help

### Getting Help

- **Documentation**: Check our comprehensive [documentation](index.md)
- **GitHub Issues**: [Report bugs or request features](https://github.com/GiorgioMedico/InterpolatePy/issues)
- **Discussions**: [Ask questions and share examples](https://github.com/GiorgioMedico/InterpolatePy/discussions)

### Professional Support

For commercial support, custom development, or consulting services, please contact the maintainers through GitHub.

---

*This changelog is automatically updated with each release. For the most current information, check the [GitHub releases page](https://github.com/GiorgioMedico/InterpolatePy/releases).*