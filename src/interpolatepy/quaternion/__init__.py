"""Backend-neutral quaternion interpolation API."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .core import Quaternion
    from .logarithmic import LogQuaternionInterpolation
    from .logarithmic import ModifiedLogQuaternionInterpolation
    from .spline import QuaternionSpline
    from .spring import SpringConfig
    from .spring import SpringQuaternionInterpolation
    from .squad import SquadC2
    from .squad import SquadC2Config

_ROUTED = {
    "LogQuaternionInterpolation",
    "ModifiedLogQuaternionInterpolation",
    "QuaternionSpline",
    "SpringQuaternionInterpolation",
    "SquadC2",
}
_LOCAL = {
    "Quaternion": ("interpolatepy.quaternion.core", "Quaternion"),
    "SpringConfig": ("interpolatepy.quaternion.spring", "SpringConfig"),
    "SquadC2Config": ("interpolatepy.quaternion.squad", "SquadC2Config"),
}
__all__ = [
    "LogQuaternionInterpolation",
    "ModifiedLogQuaternionInterpolation",
    "Quaternion",
    "QuaternionSpline",
    "SpringConfig",
    "SpringQuaternionInterpolation",
    "SquadC2",
    "SquadC2Config",
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
