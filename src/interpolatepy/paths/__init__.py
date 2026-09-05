"""Backend-neutral geometric path API."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .frenet import circular_trajectory_with_derivatives
    from .frenet import compute_trajectory_frames
    from .frenet import helicoidal_trajectory_with_derivatives
    from .frenet import plot_frames
    from .geometric import CircularPath
    from .geometric import LinearPath
    from .linear import linear_traj

_ROUTED = {
    "CircularPath",
    "LinearPath",
    "circular_trajectory_with_derivatives",
    "compute_trajectory_frames",
    "helicoidal_trajectory_with_derivatives",
    "linear_traj",
}
_LOCAL = {"plot_frames": ("interpolatepy.paths.frenet", "plot_frames")}
__all__ = [
    "CircularPath",
    "LinearPath",
    "circular_trajectory_with_derivatives",
    "compute_trajectory_frames",
    "helicoidal_trajectory_with_derivatives",
    "linear_traj",
    "plot_frames",
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
