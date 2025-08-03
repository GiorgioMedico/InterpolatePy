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

from .quat_core import Quaternion


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
        # Ensure quaternion is normalized
        q_unit = q.unit()

        # Handle singularity at w = -1 (antipodal point)
        if abs(q_unit.w + 1.0) < QuaternionTrajectoryVisualizer.SINGULARITY_THRESHOLD:
            # Use the equivalent quaternion -q to avoid singularity
            q_unit = -q_unit

        if abs(q_unit.w + 1.0) < QuaternionTrajectoryVisualizer.SINGULARITY_THRESHOLD:
            raise ValueError(
                "Quaternion is too close to singularity at w = -1. "
                "Cannot perform stereographic projection."
            )

        # Compute Modified Rodrigues Parameters
        denominator = 1.0 + q_unit.w
        return np.array([q_unit.x / denominator, q_unit.y / denominator, q_unit.z / denominator])

    @staticmethod
    def inverse_stereographic_projection(mrp: np.ndarray) -> Quaternion:
        """
        Convert Modified Rodrigues Parameters back to quaternion.

        Args:
            mrp: 3D point in stereographic projection space

        Returns:
            Unit quaternion corresponding to the MRP
        """
        mrp_norm_sq = np.dot(mrp, mrp)

        # Compute quaternion components
        w = (1 - mrp_norm_sq) / (1 + mrp_norm_sq)
        vector_scale = 2 / (1 + mrp_norm_sq)
        x, y, z = vector_scale * mrp

        return Quaternion(w, x, y, z)

    def project_trajectory(self, quaternions: list[Quaternion]) -> np.ndarray:
        """
        Project a trajectory of quaternions to 3D space.

        Args:
            quaternions: List of unit quaternions forming a trajectory

        Returns:
            Array of 3D points (N x 3) in stereographic projection space
        """
        if not quaternions:
            return np.empty((0, 3))

        projected_points = []

        for q in quaternions:
            try:
                mrp = self.stereographic_projection(q)
                projected_points.append(mrp)
            except ValueError as e:
                print(f"Warning: Skipping quaternion due to singularity: {e}")
                continue

        return np.array(projected_points)

    def plot_3d_trajectory(
        self,
        quaternions: list[Quaternion],
        **kwargs: Any,
    ) -> plt.Figure:
        """
        Create a simple 3D plot of quaternion trajectory using stereographic projection.

        Args:
            quaternions: List of quaternions to plot
            **kwargs: Optional styling parameters
                (title, color, line_width, point_size, show_points, figsize)

        Returns:
            Matplotlib figure object
        """
        # Extract parameters with defaults
        title = kwargs.get("title", "Quaternion Trajectory (3D Stereographic Projection)")
        color = kwargs.get("color", "blue")
        line_width = kwargs.get("line_width", 2.0)
        point_size = kwargs.get("point_size", 20.0)
        show_points = kwargs.get("show_points", True)
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
                projected_points[:, 2],
                c=colors,
                s=point_size,
                alpha=0.7
            )

        # Mark start and end points
        if len(projected_points) > 1:
            ax.scatter(
                *projected_points[0], color="green", s=point_size * 2, marker="o", label="Start"
            )
            ax.scatter(
                *projected_points[-1], color="red", s=point_size * 2, marker="s", label="End"
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
        diff = q1 - q2
        return diff.norm()

    def compute_velocity_magnitudes(
        self,
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
        min_quaternions = 2
        if len(quaternions) < min_quaternions:
            raise ValueError("Need at least 2 quaternions to compute velocity")

        n = len(quaternions)
        velocities = np.zeros(n)

        # Handle first point: only forward difference
        velocities[0] = self._quaternion_distance(quaternions[1], quaternions[0])

        # Handle interior points: average of forward and backward differences
        for i in range(1, n - 1):
            dist_backward = self._quaternion_distance(quaternions[i], quaternions[i - 1])
            dist_forward = self._quaternion_distance(quaternions[i + 1], quaternions[i])
            velocities[i] = (dist_backward + dist_forward) / 2.0

        # Handle last point: only backward difference
        velocities[-1] = self._quaternion_distance(quaternions[-1], quaternions[-2])

        # Create time array
        if time_points is None:
            times = np.arange(n, dtype=float)
        else:
            if len(time_points) != n:
                raise ValueError("Time points length must match quaternions length")
            times = np.array(time_points)

        return times, velocities

    def plot_angular_velocity(
        self,
        quaternions: list[Quaternion],
        time_points: list[float] | None = None,
        **kwargs: Any,
    ) -> plt.Figure:
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
    ) -> plt.Figure:
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
                projected_points[:, 2],
                c=colors,
                s=20,
                alpha=0.7
            )

        # Mark start and end
        ax1.scatter(*projected_points[0], color="green", s=50, marker="o", label="Start")
        ax1.scatter(*projected_points[-1], color="red", s=50, marker="s", label="End")

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
