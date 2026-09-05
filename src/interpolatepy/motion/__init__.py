"""Backend-neutral motion-profile API."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .double_s import DoubleSTrajectory
    from .double_s import StateParams
    from .double_s import TrajectoryBounds
    from .parabolic import ParabolicBlendTrajectory
    from .polynomial import BoundaryCondition
    from .polynomial import PolynomialTrajectory
    from .polynomial import TimeInterval
    from .polynomial import TrajectoryParams as PolynomialTrajectoryParams
    from .trapezoidal import CalculationParams
    from .trapezoidal import InterpolationParams
    from .trapezoidal import TrajectoryParams as TrapezoidalTrajectoryParams
    from .trapezoidal import TrapezoidalTrajectory

_ROUTED = {
    "BoundaryCondition",
    "DoubleSTrajectory",
    "ParabolicBlendTrajectory",
    "PolynomialTrajectory",
    "StateParams",
    "TimeInterval",
    "TrajectoryBounds",
    "TrapezoidalTrajectory",
}
_LOCAL = {
    "CalculationParams": ("interpolatepy.motion.trapezoidal", "CalculationParams"),
    "InterpolationParams": ("interpolatepy.motion.trapezoidal", "InterpolationParams"),
    "PolynomialTrajectoryParams": ("interpolatepy.motion.polynomial", "TrajectoryParams"),
    "TrapezoidalTrajectoryParams": ("interpolatepy.motion.trapezoidal", "TrajectoryParams"),
}
__all__ = [
    "BoundaryCondition",
    "CalculationParams",
    "DoubleSTrajectory",
    "InterpolationParams",
    "ParabolicBlendTrajectory",
    "PolynomialTrajectory",
    "PolynomialTrajectoryParams",
    "StateParams",
    "TimeInterval",
    "TrajectoryBounds",
    "TrapezoidalTrajectory",
    "TrapezoidalTrajectoryParams",
]


def __getattr__(name: str) -> Any:
    """Resolve public names lazily and through the active backend when applicable."""
    if name in _ROUTED:
        value = getattr(import_module("interpolatepy._api"), name)
    elif name in _LOCAL:
        module_name, attribute = _LOCAL[name]
        value = getattr(import_module(module_name), attribute)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return the public domain API for interactive discovery."""
    return list(__all__)
