"""Backend-neutral B-spline API."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .approximation import ApproximationBSpline
    from .core import BSpline
    from .cubic import CubicBSplineInterpolation
    from .interpolation import BSplineInterpolator
    from .smoothing import BSplineParams
    from .smoothing import SmoothingCubicBSpline

__all__ = [
    "ApproximationBSpline",
    "BSpline",
    "BSplineInterpolator",
    "BSplineParams",
    "CubicBSplineInterpolation",
    "SmoothingCubicBSpline",
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
