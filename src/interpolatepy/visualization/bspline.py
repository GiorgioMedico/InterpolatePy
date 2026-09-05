"""Plotting helpers for B-spline curves."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from mpl_toolkits.mplot3d import Axes3D

    from interpolatepy.bsplines.core import BSpline


def plot_bspline_2d(
    spline: BSpline,
    num_points: int = 100,
    show_control_polygon: bool = True,
    show_knots: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """Plot a two-dimensional B-spline and optional construction geometry."""
    if spline.dimension != spline.DIM_2:
        raise ValueError(f"Control points must be 2D for this plot function, got {spline.dimension}D")

    if ax is None:
        import matplotlib.pyplot as plt  # noqa: PLC0415

        _, ax = plt.subplots(figsize=(10, 6))

    _, curve_points = spline.generate_curve_points(num_points)
    ax.plot(
        curve_points[:, 0],
        curve_points[:, 1],
        color="blue",
        linewidth=2,
        label="B-spline curve",
    )

    if show_control_polygon:
        ax.plot(
            spline.control_points[:, 0],
            spline.control_points[:, 1],
            color="red",
            linestyle="--",
            marker="o",
            linewidth=1,
            markersize=8,
            label="Control polygon",
        )

    if show_knots:
        valid_knots = [knot for knot in spline.knots if spline.u_min <= knot <= spline.u_max]
        unique_knots = np.unique(valid_knots)
        knot_points = np.array([spline.evaluate(knot) for knot in unique_knots])
        ax.plot(
            knot_points[:, 0],
            knot_points[:, 1],
            color="green",
            marker="x",
            markersize=10,
            linestyle="none",
            label="Knot points",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"B-spline curve of degree {spline.degree}")
    ax.legend()
    ax.grid(True)
    return ax


def plot_bspline_3d(
    spline: BSpline,
    num_points: int = 100,
    show_control_polygon: bool = True,
    ax: Axes3D | None = None,
) -> Axes3D:
    """Plot a three-dimensional B-spline and optional control polygon."""
    if spline.dimension != spline.DIM_3:
        raise ValueError(f"Control points must be 3D for this plot function, got {spline.dimension}D")

    if ax is None:
        import matplotlib.pyplot as plt  # noqa: PLC0415

        figure = plt.figure(figsize=(10, 8))
        ax = figure.add_subplot(111, projection="3d")

    _, curve_points = spline.generate_curve_points(num_points)
    ax.plot(
        curve_points[:, 0],
        curve_points[:, 1],
        curve_points[:, 2],
        color="blue",
        linewidth=2,
        label="B-spline curve",
    )

    if show_control_polygon:
        ax.plot(
            spline.control_points[:, 0],
            spline.control_points[:, 1],
            spline.control_points[:, 2],
            color="red",
            linestyle="--",
            marker="o",
            linewidth=1,
            markersize=8,
            label="Control polygon",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"3D B-spline curve of degree {spline.degree}")
    ax.legend()
    return ax


__all__ = ["plot_bspline_2d", "plot_bspline_3d"]
