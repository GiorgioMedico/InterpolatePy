"""
Examples demonstrating the computation and visualization of Frenet frames.

This module contains implementations of examples for various trajectories
including helicoidal and circular paths, with and without tool orientation.
"""

import matplotlib.pyplot as plt
import numpy as np

from interpolatepy.frenet_frame import circular_trajectory_with_derivatives
from interpolatepy.frenet_frame import compute_trajectory_frames
from interpolatepy.frenet_frame import helicoidal_trajectory_with_derivatives
from interpolatepy.frenet_frame import plot_frames
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def example_8_5() -> None:
    """Recreate Example 8.5 using the general approach."""
    print("Example 8.5: Helicoidal trajectory with Frenet frames")

    # Parameters from the example
    r = 2.0
    d = 0.5
    u_values = np.linspace(0, 4 * np.pi, 100)

    # Get helicoidal trajectory function
    def helix_func(u: float) -> tuple:
        return helicoidal_trajectory_with_derivatives(u, r, d)

    # Compute Frenet frames
    points, frames = compute_trajectory_frames(helix_func, u_values)

    # Visualize
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    plot_frames(ax, points, frames, scale=0.5, skip=10)
    ax.set_title("Helicoidal Trajectory with Frenet Frames")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    plt.show()


def example_8_6() -> None:
    """Recreate Example 8.6."""
    print("Example 8.6: Circular trajectory with tool orientation")

    # Parameters from the example
    r = 2.0
    alpha = np.radians(30)  # 30 degrees rotation about binormal axis
    u_values = np.linspace(0, 2 * np.pi, 100)

    # Get circular trajectory function
    def circle_func(u: float) -> tuple:
        return circular_trajectory_with_derivatives(u, r)

    # Compute Frenet frames
    points, frenet_frames = compute_trajectory_frames(circle_func, u_values)

    # Compute tool frames with orientation alpha
    _, tool_frames = compute_trajectory_frames(circle_func, u_values, tool_orientation=alpha)

    # Visualize
    fig = plt.figure(figsize=(12, 6))

    # Plot Frenet frames
    ax1 = fig.add_subplot(121, projection="3d")
    plot_frames(ax1, points, frenet_frames, scale=0.5, skip=8)
    ax1.set_title("Circular Trajectory with Frenet Frames")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")

    # Plot tool frames
    ax2 = fig.add_subplot(122, projection="3d")
    plot_frames(ax2, points, tool_frames, scale=0.5, skip=8)
    ax2.set_title("Circular Trajectory with Tool Frames (α = 30°)")  # noqa: RUF001
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")

    # Set equal aspect ratio with more space for z
    for ax in [ax1, ax2]:
        ax.set_box_aspect([1, 1, 0.5])
        ax.set_zlim(-1, 1)

    plt.tight_layout()
    plt.show()


def example_rot() -> None:
    """Recreate Example Rotations."""
    print("Example with RPY rotations: Circular trajectory with tool orientation")

    # Parameters from the example
    r = 2.0
    alpha = np.radians(30)  # Roll angle
    beta = np.radians(60)  # Pitch angle
    gamma = np.radians(0)  # Yaw angle
    u_values = np.linspace(0, 2 * np.pi, 100)

    # Get circular trajectory function
    def circle_func(u: float) -> tuple:
        return circular_trajectory_with_derivatives(u, r)

    # Compute Frenet frames
    points, frenet_frames = compute_trajectory_frames(circle_func, u_values)

    # Compute tool frames with RPY orientation
    _, tool_frames = compute_trajectory_frames(
        circle_func, u_values, tool_orientation=(alpha, beta, gamma)
    )

    # Visualize
    fig = plt.figure(figsize=(12, 6))

    # Plot Frenet frames
    ax1 = fig.add_subplot(121, projection="3d")
    plot_frames(ax1, points, frenet_frames, scale=0.5, skip=8)
    ax1.set_title("Circular Trajectory with Frenet Frames")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")

    # Plot tool frames
    ax2 = fig.add_subplot(122, projection="3d")
    plot_frames(ax2, points, tool_frames, scale=0.5, skip=8)
    ax2.set_title("Circular Trajectory with Tool Frames Rotated")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")

    # Set equal aspect ratio with more space for z
    for ax in [ax1, ax2]:
        ax.set_box_aspect([1, 1, 0.5])
        ax.set_zlim(-1, 1)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print()
    print("Frenet Frame and Tool Orientation Implementation")
    print("=" * 60)
    print()

    example_8_5()
    print("=" * 60)
    print()

    example_8_6()
    print("=" * 60)
    print()

    example_rot()
    print("=" * 60)
    print()
