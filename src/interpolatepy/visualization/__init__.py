"""Optional plotting utilities."""

from .bspline import plot_bspline_2d
from .bspline import plot_bspline_3d
from .quaternion import PlotStyle
from .quaternion import QuaternionTrajectoryVisualizer

__all__ = [
    "PlotStyle",
    "QuaternionTrajectoryVisualizer",
    "plot_bspline_2d",
    "plot_bspline_3d",
]
