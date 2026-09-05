"""Backend-neutral spline interpolation API."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .acceleration1 import CubicSplineWithAcceleration1
    from .acceleration2 import CubicSplineWithAcceleration2
    from .acceleration2 import SplineParameters
    from .cubic import CubicSpline
    from .search import SplineConfig
    from .search import smoothing_spline_with_tolerance
    from .smoothing import CubicSmoothingSpline

__all__ = [
    "CubicSmoothingSpline",
    "CubicSpline",
    "CubicSplineWithAcceleration1",
    "CubicSplineWithAcceleration2",
    "SplineConfig",
    "SplineParameters",
    "smoothing_spline_with_tolerance",
]


def __getattr__(name: str) -> Any:
    """Resolve public names through the active backend."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("interpolatepy._api"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return the public domain API for interactive discovery."""
    return sorted(__all__)
