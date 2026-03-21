"""Protocol definitions for InterpolatePy trajectory and curve interfaces.

Defines structural typing protocols (PEP 544) that group classes by behavior
without requiring inheritance. Classes conform by having the right method
signatures — no modifications to existing classes needed.

Five protocol groups:
- ScalarTrajectory: 1D position/velocity/acceleration evaluation
- CurveEvaluator: Parametric curve with derivative evaluation
- GeometricPath: 3D path with position/velocity/acceleration by arc length
- QuaternionTrajectory: Quaternion-valued trajectory evaluation
- TrajectoryFunction: Callable returning (position, velocity, acceleration) tuple
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

    from .quat_core import Quaternion


@runtime_checkable
class ScalarTrajectory(Protocol):
    """Protocol for scalar (1D) trajectory evaluation.

    Conforming classes provide position, velocity, and acceleration
    as functions of time. Used by spline-based trajectory planners.

    Conforming Classes
    ------------------
    CubicSpline, CubicSplineWithAcceleration1, CubicSplineWithAcceleration2,
    CubicSmoothingSpline, DoubleSTrajectory
    """

    def evaluate(self, t: float | np.ndarray) -> float | np.ndarray: ...

    def evaluate_velocity(self, t: float | np.ndarray) -> float | np.ndarray: ...

    def evaluate_acceleration(self, t: float | np.ndarray) -> float | np.ndarray: ...


@runtime_checkable
class CurveEvaluator(Protocol):
    """Protocol for parametric curve evaluation with derivative support.

    Conforming classes evaluate a curve at parameter u and compute
    derivatives of arbitrary order.

    Conforming Classes
    ------------------
    BSpline, BSplineInterpolator, CubicBSplineInterpolation,
    SmoothingCubicBSpline, ApproximationBSpline
    """

    def evaluate(self, u: float) -> np.ndarray: ...

    def evaluate_derivative(self, u: float, order: int = 1) -> np.ndarray: ...


@runtime_checkable
class GeometricPath(Protocol):
    """Protocol for 3D geometric path evaluation by arc length.

    Conforming classes provide position, velocity, and acceleration
    vectors as functions of arc length parameter s.

    Conforming Classes
    ------------------
    LinearPath, CircularPath
    """

    def position(self, s: float | np.ndarray) -> np.ndarray: ...

    def velocity(self, s: float | np.ndarray) -> np.ndarray: ...

    def acceleration(self, s: float | np.ndarray) -> np.ndarray: ...


@runtime_checkable
class QuaternionTrajectory(Protocol):
    """Protocol for quaternion-valued trajectory evaluation.

    Conforming classes provide quaternion interpolation with angular
    velocity and acceleration as functions of time.

    Conforming Classes
    ------------------
    SquadC2, LogQuaternionInterpolation, ModifiedLogQuaternionInterpolation,
    QuaternionSpline
    """

    def evaluate(self, t: float) -> Quaternion: ...

    def evaluate_velocity(self, t: float) -> np.ndarray: ...

    def evaluate_acceleration(self, t: float) -> np.ndarray: ...


@runtime_checkable
class TrajectoryFunction(Protocol):
    """Protocol for callable trajectory functions returning (pos, vel, acc).

    Conforming objects are callables (typically returned by factory methods)
    that accept a time parameter and return a tuple of position, velocity,
    and acceleration.

    Conforming Classes
    ------------------
    Output callables from TrapezoidalTrajectory, ParabolicBlendTrajectory,
    PolynomialTrajectory factories
    """

    def __call__(self, t: float) -> tuple[float, float, float]: ...
