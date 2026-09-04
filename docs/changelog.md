# Changelog

This changelog records released InterpolatePy behavior. Dates and versions match
the repository tags. The project follows [Semantic Versioning](https://semver.org/).

## 3.2.1 — 2026-09-04

### Added

- CI smoke checks for all Python and C++ examples on the supported backends.
- Executable validation for standalone Python snippets in the documentation.
- Native-backend regression tests for public adapter behavior.

### Fixed

- Native adapters now accept the documented Python parameter objects, scalar
  inputs, parameterization names, and helper return types.
- Native spline, B-spline, motion-profile, and quaternion adapters now expose
  the plotting, diagnostic, and compatibility helpers used by the public API.
- Examples now import exported algorithms through `interpolatepy`, ensuring
  backend selection is exercised instead of bypassed.
- Static typing errors and contextless source references were removed.

### Changed

- Documentation tooling now constrains MkDocs to the supported 1.x series.

## 3.2.0 — 2026-09-04

### Added

- `BSplineInterpolator` now supports two or more samples for degrees 3, 4, and
  5, including explicit endpoint velocity/acceleration constraints and cyclic
  interpolation.
- `LogQuaternionInterpolation` and
  `ModifiedLogQuaternionInterpolation` now support two or more quaternion
  samples for degrees 3, 4, and 5.

### Fixed

- Generated natural B-spline boundary rows no longer duplicate explicitly
  supplied acceleration constraints.
- Python and C++ interpolation behavior is covered through the public API for
  small waypoint sets.

## 3.1.0 — 2026-05-21

### Changed

- Removed the deprecated `LogQuaternionBSpline` compatibility name. Use
  `LogQuaternionInterpolation`.
- `Quaternion.to_axis_angle()` now returns a canonical axis-angle
  representation.
- Added LQI and mLQI examples that interpolate Frenet-frame orientations along
  a helical path.
- Updated GitHub Actions versions and repaired the documentation workflow.

## 3.0.1 — 2026-05-14

### Fixed

- Restored compatibility with NumPy 1.x behavior used by the declared lower
  bound.
- Relaxed runtime dependency floors to NumPy 1.26, SciPy 1.11, and Matplotlib
  3.6.
- Migrated development dependency management and CI commands to uv.

## 3.0.0 — 2026-03-21

### Added

- Optional C++20 implementation with pybind11 bindings and Python adapters.
- Runtime backend detection through `interpolatepy.HAS_CPP`.
- `INTERPOLATEPY_NO_CPP` fallback override.
- Standalone C++ examples and Catch2 tests.
- Runtime-checkable protocols: `ScalarTrajectory`, `CurveEvaluator`,
  `GeometricPath`, `QuaternionTrajectory`, and `TrajectoryFunction`.
- Duration-based trapezoidal trajectory generation.

### Changed

- The package root now routes supported algorithms through `_api.py`.
- The minimum Python version is 3.11.
- Development, tests, docs, and examples use dependency groups in
  `pyproject.toml`.

## 2.0.0 — 2025-08-06

### Added

- MkDocs documentation, tutorials, algorithm notes, and an API reference.
- `SquadC2`, logarithmic quaternion interpolation, and modified logarithmic
  quaternion interpolation.
- Two acceleration-constrained cubic spline constructions.
- B-spline interpolation, approximation, and smoothing variants.
- Prescribed-tolerance cubic smoothing search.
- `LinearPath`, `CircularPath`, Frenet-frame helpers, and parabolic blends.
- Expanded Double-S, trapezoidal, and polynomial trajectory APIs.

### Changed

- Standardized the main scalar evaluation method names to `evaluate()`,
  `evaluate_velocity()`, and `evaluate_acceleration()`.
- Added array evaluation to scalar spline algorithms.
- Expanded validation and NumPy-style docstrings.

## 1.1.0 — 2025-05-17

- Added quaternion interpolation.
- Added B-spline approximation and smoothing functionality.
- Improved validation and plotting.

## 1.0.1 — 2025-03-26

- Published the initial installable release with cubic splines, motion
  profiles, polynomial trajectories, and numerical utilities.

For commits after the latest release, see the
[repository history](https://github.com/GiorgioMedico/InterpolatePy/commits/main/).
