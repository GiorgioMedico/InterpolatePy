"""Installed distribution version."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

try:
    __version__ = version("InterpolatePy")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"
