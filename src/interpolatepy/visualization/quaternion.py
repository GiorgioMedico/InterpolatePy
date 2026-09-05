"""
Quaternion Trajectory Visualization using Stereographic Projection.

This module provides tools for visualizing quaternion trajectories by projecting them
from 4D quaternion space to 3D space using stereographic projection. The implementation
uses Modified Rodrigues Parameters (MRPs) as the mathematical foundation.

Mathematical Background:
- Unit quaternions live on a 3-sphere (S³) in 4D space
- Stereographic projection maps points from S³ to R³ (3D Euclidean space)
- For quaternion q = [w, x, y, z], the MRP is: [x/(1+w), y/(1+w), z/(1+w)]
- This projection is from pole (-1,0,0,0) to the equatorial 3D hyperplane

References:
- Modified Rodrigues Parameters: An Efficient Representation of Orientation
- Stereographic Projection for Quaternion Visualization
- 3Blue1Brown: Visualizing Quaternions
"""

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from interpolatepy.quaternion.core import Quaternion
from interpolatepy.visualization.quaternion_math import (
    inverse_stereographic_projection as _inverse_stereographic_projection,
)
from interpolatepy.visualization.quaternion_math import project_trajectory as _project_trajectory
from interpolatepy.visualization.quaternion_math import quaternion_distance
from interpolatepy.visualization.quaternion_math import stereographic_projection as _stereographic_projection
from interpolatepy.visualization.quaternion_math import velocity_magnitudes


@dataclass
class PlotStyle:
    """Configuration for plot styling."""

    color: str = "blue"
    line_width: float = 2.0
    point_size: float = 20.0
    figsize: tuple[float, float] = (10, 8)
    title: str = "Quaternion Plot"


class QuaternionTrajectoryVisualizer:
    """
    Visualizer for quaternion trajectories using stereographic projection.

    This class provides methods to project quaternion trajectories from 4D space
    to 3D space using Modified Rodrigues Parameters and create various visualizations.
    """

    SINGULARITY_THRESHOLD = 1e-6
    DEFAULT_TRAJECTORY_COLOR = "blue"
    DEFAULT_POINT_SIZE = 20
    DEFAULT_LINE_WIDTH = 2

    def __init__(self) -> None:
        """Initialize the quaternion trajectory visualizer."""

    @staticmethod
    def stereographic_projection(q: Quaternion) -> np.ndarray:
        """
        Project a unit quaternion to 3D space using stereographic projection.

        Uses Modified Rodrigues Parameters (MRPs) formula:
        MRP = [x/(1+w), y/(1+w), z/(1+w)]

        Args:
            q: Unit quaternion to project

        Returns:
            3D point in stereographic projection space

        Raises:
            ValueError: If quaternion is too close to the singularity at w = -1
        """
        return _stereographic_projection(
            q, QuaternionTrajectoryVisualizer.SINGULARITY_THRESHOLD
        )

    @staticmethod
    def inverse_stereographic_projection(mrp: np.ndarray) -> Quaternion:
        """
        Convert Modified Rodrigues Parameters back to quaternion.

        Args:
            mrp: 3D point in stereographic projection space

        Returns:
            Unit quaternion corresponding to the MRP
        """
        return _inverse_stereographic_projection(mrp)

    def project_trajectory(self, quaternions: list[Quaternion]) -> np.ndarray:
        """
        Project a trajectory of quaternions to 3D space.

        Args:
            quaternions: List of unit quaternions forming a trajectory

        Returns:
            Array of 3D points (N x 3) in stereographic projection space
        """
        return _project_trajectory(quaternions, self.SINGULARITY_THRESHOLD)

    def plot_3d_trajectory(
        self,
        quaternions: list[Quaternion],
        waypoints: list[Quaternion] | None = None,
        waypoint_times: list[float] | None = None,
        **kwargs: Any,
    ) -> Figure:
        """
        Create a simple 3D plot of quaternion trajectory using stereographic projection.

        Args:
            quaternions: List of quaternions to plot
            waypoints: Optional list of waypoint quaternions to highlight
            waypoint_times: Optional time values for waypoint labels
            **kwargs: Optional styling parameters
                (title, color, line_width, point_size, show_points, show_waypoints,
                waypoint_size, waypoint_color, show_waypoint_labels, figsize)

        Returns:
            Matplotlib figure object
        """
        # Extract parameters with defaults
        title = kwargs.get("title", "Quaternion Trajectory (3D Stereographic Projection)")
        color = kwargs.get("color", "blue")
        line_width = kwargs.get("line_width", 2.0)
        point_size = kwargs.get("point_size", 20.0)
        show_points = kwargs.get("show_points", True)
        show_waypoints = kwargs.get("show_waypoints", True)
        waypoint_size = kwargs.get("waypoint_size", 80.0)
        waypoint_color = kwargs.get("waypoint_color", "red")
        show_waypoint_labels = kwargs.get("show_waypoint_labels", True)
        figsize = kwargs.get("figsize", (10, 8))

        if not quaternions:
            raise ValueError("Empty quaternion list provided")

        # Project quaternions to 3D space
        projected_points = self.project_trajectory(quaternions)

        if len(projected_points) == 0:
            raise ValueError("No valid projected points (all quaternions near singularity)")

        # Create 3D plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        # Plot trajectory line
        ax.plot(
            projected_points[:, 0],
            projected_points[:, 1],
            projected_points[:, 2],
            color=color,
            linewidth=line_width,
            alpha=0.8,
            label="Trajectory"
        )

        # Plot individual points if requested
        if show_points and len(projected_points) > 1:
            # Color points by progression
            colors = plt.colormaps["viridis"](np.linspace(0, 1, len(projected_points)))
            ax.scatter(
                projected_points[:, 0],
                projected_points[:, 1],
                projected_points[:, 2],  # pyright: ignore[reportArgumentType]
                c=colors,
                s=point_size,
                alpha=0.7
            )

        # Plot waypoints if provided
        if waypoints and show_waypoints:
            waypoint_points = self.project_trajectory(waypoints)
            if len(waypoint_points) > 0:
                ax.scatter(
                    waypoint_points[:, 0],
                    waypoint_points[:, 1],
                    waypoint_points[:, 2],  # pyright: ignore[reportArgumentType]
                    color=waypoint_color,
                    s=waypoint_size,
                    marker="D",
                    alpha=0.9,
                    edgecolors="black",
                    linewidth=1.5,
                    label="Waypoints",
                    zorder=10
                )

                # Add waypoint labels if requested
                if show_waypoint_labels:
                    for i, point in enumerate(waypoint_points):
                        if waypoint_times is not None and i < len(waypoint_times):
                            label = f"t={waypoint_times[i]:.1f}"
                        else:
                            label = f"W{i}"
                        ax.text(
                            point[0], point[1], point[2],
                            label,
                            fontsize=8,
                            ha="center",
                            va="bottom"
                        )

        # Mark start and end points (only if no waypoints or waypoints don't cover start/end)
        if len(projected_points) > 1 and not (waypoints and show_waypoints):
            start_point = projected_points[0]
            end_point = projected_points[-1]
            ax.scatter(
                start_point[0], start_point[1], start_point[2],
                color="green", s=point_size * 2, marker="o", label="Start"
            )
            ax.scatter(
                end_point[0], end_point[1], end_point[2],
                color="red", s=point_size * 2, marker="s", label="End"
            )

        # Set labels and title
        ax.set_xlabel("MRP X")
        ax.set_ylabel("MRP Y")
        ax.set_zlabel("MRP Z")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_waypoints_only(
        self,
        waypoints: list[Quaternion],
        waypoint_times: list[float] | None = None,
        **kwargs: Any,
    ) -> Figure:
        """
        Create a 3D plot showing only waypoints (no trajectory lines).

        Args:
            waypoints: List of waypoint quaternions to plot
            waypoint_times: Optional time values for waypoint labels
            **kwargs: Optional styling parameters
                (title, waypoint_color, waypoint_size, show_waypoint_labels,
                show_connections, figsize)

        Returns:
            Matplotlib figure object
        """
        # Extract parameters with defaults
        title = kwargs.get("title", "Quaternion Waypoints (3D Stereographic Projection)")
        waypoint_color = kwargs.get("waypoint_color", "red")
        waypoint_size = kwargs.get("waypoint_size", 100.0)
        show_waypoint_labels = kwargs.get("show_waypoint_labels", True)
        show_connections = kwargs.get("show_connections", True)
        figsize = kwargs.get("figsize", (10, 8))

        if not waypoints:
            raise ValueError("Empty waypoints list provided")

        # Project waypoints to 3D space
        waypoint_points = self.project_trajectory(waypoints)

        if len(waypoint_points) == 0:
            raise ValueError("No valid projected waypoints (all quaternions near singularity)")

        # Create 3D plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        # Plot connection lines between waypoints if requested
        if show_connections and len(waypoint_points) > 1:
            ax.plot(
                waypoint_points[:, 0],
                waypoint_points[:, 1],
                waypoint_points[:, 2],
                color="gray",
                linewidth=1.5,
                alpha=0.6,
                linestyle="--",
                label="Waypoint Connections"
            )

        # Plot waypoints
        ax.scatter(
            waypoint_points[:, 0],
            waypoint_points[:, 1],
            waypoint_points[:, 2],  # pyright: ignore[reportArgumentType]
            color=waypoint_color,
            s=waypoint_size,
            marker="D",
            alpha=0.9,
            edgecolors="black",
            linewidth=2.0,
            label="Waypoints",
            zorder=10
        )

        # Add waypoint labels if requested
        if show_waypoint_labels:
            for i, point in enumerate(waypoint_points):
                if waypoint_times is not None and i < len(waypoint_times):
                    label = f"t={waypoint_times[i]:.1f}"
                else:
                    label = f"W{i}"
                ax.text(
                    point[0], point[1], point[2],
                    label,
                    fontsize=10,
                    ha="center",
                    va="bottom",
                    weight="bold"
                )

        # Highlight start and end waypoints
        if len(waypoint_points) > 1:
            start_point = waypoint_points[0]
            end_point = waypoint_points[-1]
            ax.scatter(
                start_point[0], start_point[1], start_point[2],
                color="green", s=waypoint_size * 1.5, marker="o",
                alpha=0.8, edgecolors="darkgreen", linewidth=2,
                label="Start", zorder=11
            )
            ax.scatter(
                end_point[0], end_point[1], end_point[2],
                color="red", s=waypoint_size * 1.5, marker="s",
                alpha=0.8, edgecolors="darkred", linewidth=2,
                label="End", zorder=11
            )

        # Set labels and title
        ax.set_xlabel("MRP X")
        ax.set_ylabel("MRP Y")
        ax.set_zlabel("MRP Z")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def _quaternion_distance(q1: Quaternion, q2: Quaternion) -> float:
        """
        Calculate the distance between two quaternions using quaternion norm.

        Args:
            q1: First quaternion
            q2: Second quaternion

        Returns:
            Distance as ||q1 - q2||
        """
        return quaternion_distance(q1, q2)

    @staticmethod
    def compute_velocity_magnitudes(
        quaternions: list[Quaternion],
        time_points: list[float] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute velocity magnitudes using the formula: V(qi) = [||qi - qi-1|| + ||qi - qi+1||] / 2

        Args:
            quaternions: List of quaternions
            time_points: Optional time points (if None, uses indices)

        Returns:
            Tuple of (time_array, velocity_magnitudes)
        """
        return velocity_magnitudes(quaternions, time_points)

    def plot_angular_velocity(
        self,
        quaternions: list[Quaternion],
        time_points: list[float] | None = None,
        **kwargs: Any,
    ) -> Figure:
        """
        Plot quaternion velocity magnitudes over time using custom formula.

        Uses the formula: V(qi) = [||qi - qi-1|| + ||qi - qi+1||] / 2

        Args:
            quaternions: List of quaternions
            time_points: Time values (if None, uses indices)
            **kwargs: Optional styling parameters (title, color, line_width, figsize)

        Returns:
            Matplotlib figure object
        """
        # Extract parameters with defaults
        title = kwargs.get("title", "Quaternion Velocity Magnitude")
        color = kwargs.get("color", "red")
        line_width = kwargs.get("line_width", 2.0)
        figsize = kwargs.get("figsize", (10, 6))
        if not quaternions:
            raise ValueError("Empty quaternion list provided")

        # Compute velocity magnitudes
        times, velocities = self.compute_velocity_magnitudes(quaternions, time_points)

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(times, velocities, color=color, linewidth=line_width, label="Velocity Magnitude")
        ax.fill_between(times, velocities, alpha=0.3, color=color)

        # Mark peaks
        min_points_for_peaks = 2
        if len(velocities) > min_points_for_peaks:
            max_idx = np.argmax(velocities)
            ax.scatter(times[max_idx], velocities[max_idx], color="darkred", s=50, zorder=5,
                      marker="^", label=f"Max: {velocities[max_idx]:.3f}")

        ax.set_xlabel("Time" if time_points is not None else "Index")
        ax.set_ylabel("Velocity Magnitude")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_trajectory_with_velocity(
        self,
        quaternions: list[Quaternion],
        time_points: list[float] | None = None,
        **kwargs: Any,
    ) -> Figure:
        """
        Create combined plot showing 3D trajectory and velocity magnitude.

        Args:
            quaternions: List of quaternions
            time_points: Time values (if None, uses indices)
            **kwargs: Optional styling parameters (title, trajectory_color, velocity_color, figsize)

        Returns:
            Matplotlib figure object with two subplots
        """
        # Extract parameters with defaults
        title = kwargs.get("title", "Quaternion Trajectory with Velocity Analysis")
        trajectory_color = kwargs.get("trajectory_color", "blue")
        velocity_color = kwargs.get("velocity_color", "red")
        figsize = kwargs.get("figsize", (15, 6))
        if not quaternions:
            raise ValueError("Empty quaternion list provided")

        # Create figure with two subplots
        fig = plt.figure(figsize=figsize)
        fig.suptitle(title, fontsize=14)

        # Left subplot: 3D trajectory
        ax1 = fig.add_subplot(121, projection="3d")

        # Project quaternions to 3D space
        projected_points = self.project_trajectory(quaternions)

        if len(projected_points) == 0:
            raise ValueError("No valid projected points (all quaternions near singularity)")

        # Plot 3D trajectory
        ax1.plot(
            projected_points[:, 0],
            projected_points[:, 1],
            projected_points[:, 2],
            color=trajectory_color,
            linewidth=2.0,
            alpha=0.8,
            label="Trajectory"
        )

        # Color points by progression
        if len(projected_points) > 1:
            colors = plt.colormaps["viridis"](np.linspace(0, 1, len(projected_points)))
            ax1.scatter(
                projected_points[:, 0],
                projected_points[:, 1],
                projected_points[:, 2],  # pyright: ignore[reportArgumentType]
                c=colors,
                s=20,
                alpha=0.7
            )

        # Mark start and end
        start_point = projected_points[0]
        end_point = projected_points[-1]
        ax1.scatter(start_point[0], start_point[1], start_point[2], color="green", s=50, marker="o", label="Start")
        ax1.scatter(end_point[0], end_point[1], end_point[2], color="red", s=50, marker="s", label="End")

        ax1.set_xlabel("MRP X")
        ax1.set_ylabel("MRP Y")
        ax1.set_zlabel("MRP Z")
        ax1.set_title("3D Trajectory\n(Stereographic Projection)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Right subplot: Velocity magnitude
        ax2 = fig.add_subplot(122)

        # Compute and plot velocity
        times, velocities = self.compute_velocity_magnitudes(quaternions, time_points)

        ax2.plot(times, velocities, color=velocity_color, linewidth=2.0, label="Velocity Magnitude")
        ax2.fill_between(times, velocities, alpha=0.3, color=velocity_color)

        # Mark velocity peaks
        min_points_for_peaks = 2
        if len(velocities) > min_points_for_peaks:
            max_idx = np.argmax(velocities)
            ax2.scatter(times[max_idx], velocities[max_idx], color="darkred", s=50, zorder=5,
                       marker="^", label=f"Max: {velocities[max_idx]:.3f}")

        ax2.set_xlabel("Time" if time_points is not None else "Index")
        ax2.set_ylabel("Velocity Magnitude")
        ax2.set_title("Velocity Magnitude\nV(qi) = [||qi-qi-1|| + ||qi-qi+1||]/2")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
