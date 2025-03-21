"""
Simple example demonstrating how to use geometric paths with polynomial motion laws.

This example shows how to:
1. Create a linear path and a circular path
2. Apply polynomial motion profiles to traverse these paths smoothly
3. Visualize the path and motion profiles
"""

import matplotlib.pyplot as plt
import numpy as np

from interpolatepy.polynomials import BoundaryCondition
from interpolatepy.polynomials import PolynomialTrajectory
from interpolatepy.polynomials import TimeInterval
from interpolatepy.simple_paths import CircularPath
from interpolatepy.simple_paths import LinearPath
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def plot_3d_path(
    positions: np.ndarray,
    title: str,
    sampled_points: np.ndarray | None = None,
) -> None:
    """
    Plot a 3D path with option to show sampled points and key defining points.

    Parameters:
    -----------
    positions : np.ndarray
        Array of positions along the continuous path
    title : str
        Plot title
    sampled_points : np.ndarray, optional
        Points sampled along the path based on motion law
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the continuous path
    ax.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        "b-",
        linewidth=1.3,
        alpha=0.7,
        label="Path",
    )

    # Plot sampled points from motion law if provided
    if sampled_points is not None:
        # Use a subset of points to avoid overcrowding
        step = max(1, len(sampled_points) // 25)
        ax.scatter(
            sampled_points[::step, 0],
            sampled_points[::step, 1],
            sampled_points[::step, 2],
            color="green",
            s=50,
            marker="o",
            label="Motion Points",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.set_box_aspect([1.0, 1.0, 1.0])

    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_motion_profiles(
    times: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
    title: str,
) -> None:
    """
    Plot motion profiles.

    Parameters:
    -----------
    times : np.ndarray
        Time points for evaluation
    positions : np.ndarray
        Position values at each time point
    velocities : np.ndarray
        Velocity values at each time point
    accelerations : np.ndarray
        Acceleration values at each time point
    title : str
        Title for the plot
    """
    plt.figure(figsize=(10, 8))

    # Position plot
    plt.subplot(3, 1, 1)
    plt.plot(times, positions, "b-", linewidth=2)
    plt.grid(True)
    plt.ylabel("Position")
    plt.title(title)

    # Velocity plot
    plt.subplot(3, 1, 2)
    plt.plot(times, velocities, "g-", linewidth=2)
    plt.grid(True)
    plt.ylabel("Velocity")

    # Acceleration plot
    plt.subplot(3, 1, 3)
    plt.plot(times, accelerations, "r-", linewidth=2)
    plt.grid(True)
    plt.ylabel("Acceleration")
    plt.xlabel("Time")

    plt.tight_layout()
    plt.show()


def linear_path_example() -> None:
    """Example with a linear path and polynomial motion."""
    print("Linear Path Example")
    print("-----------------")

    # Create a linear path from origin to a point
    start_point = np.array([0.0, 0.0, 0.0])
    end_point = np.array([10.0, 5.0, 2.0])
    path = LinearPath(start_point, end_point)

    # Get the path length
    path_length = path.length
    print(f"Path length: {path_length:.2f}")

    # Generate the path for visualization
    path_data = path.all_traj(num_points=100)
    path_positions = path_data["position"]

    # Plot the path with key points highlighted
    plot_3d_path(path_positions, "Linear Path with Key Points")

    # Create a polynomial motion profile to follow the path
    # We want to start and end with zero velocity and acceleration
    initial = BoundaryCondition(position=0.0, velocity=0.0, acceleration=0.0)
    final = BoundaryCondition(position=path_length, velocity=0.0, acceleration=0.0)
    time_interval = TimeInterval(start=0.0, end=5.0)  # 5 seconds duration

    # Create a 5th order polynomial trajectory
    trajectory = PolynomialTrajectory.order_5_trajectory(initial, final, time_interval)

    # Sample times for evaluation
    times = np.linspace(0.0, 5.0, 100)

    # Get position (arc length), velocity, and acceleration at each time
    s_values = np.array([trajectory(t)[0] for t in times])  # Position (arc length)
    v_values = np.array([trajectory(t)[1] for t in times])  # Velocity (ds/dt)
    a_values = np.array([trajectory(t)[2] for t in times])  # Acceleration (d²s/dt²)

    # Plot the motion profiles
    plot_motion_profiles(times, s_values, v_values, a_values, "Motion Profiles for Linear Path")

    # Now let's map the motion profile to 3D positions along the path
    positions_3d = np.zeros((len(times), 3))

    # For each time point, get the corresponding position on the path
    for i, s in enumerate(s_values):
        # Evaluate the path at arc length s
        pos = path.position(s)
        positions_3d[i] = pos

    # Plot the path with the motion profile applied, showing key points and sampled points
    plot_3d_path(
        path_positions,
        "Linear Path with Polynomial Motion",
        sampled_points=positions_3d,
    )


def circular_path_example() -> None:
    """Example with a circular path and polynomial motion."""
    print("\nCircular Path Example")
    print("-------------------")

    # Create a circular path in the XY plane
    axis = np.array([0.0, 0.0, 1.0])  # Z-axis
    center = np.array([0.0, 0.0, 0.0])  # Origin
    point = np.array([3.0, 0.0, 0.0])  # Point on circle

    path = CircularPath(axis, center, point)

    # Get circle properties
    radius = path.radius
    print(f"Circle radius: {radius:.2f}")

    # We'll traverse half the circle
    arc_length = np.pi * radius
    print(f"Half circle arc length: {arc_length:.2f}")

    # Generate the complete circle for visualization
    path_data = path.all_traj(num_points=100)
    path_positions = path_data["position"]

    # Plot the circle with key points
    # For circular path, let's show the center and points along the circle
    angle_points = np.linspace(0, 2 * np.pi, 5)[:-1]  # 4 points at 90° intervals
    circle_points = []

    # Add the center point
    circle_points.append(center)

    # Add points along the circle at regular intervals
    for angle in angle_points:
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        z = center[2]
        circle_points.append(np.array([x, y, z]))

    # Plot the path with key points highlighted
    plot_3d_path(path_positions, "Circular Path with Key Points")

    # Create a polynomial motion profile for the half circle
    initial = BoundaryCondition(position=0.0, velocity=0.0, acceleration=0.0)
    final = BoundaryCondition(position=arc_length, velocity=0.0, acceleration=0.0)
    time_interval = TimeInterval(start=0.0, end=4.0)  # 4 seconds duration

    # Create a 5th order polynomial trajectory
    trajectory = PolynomialTrajectory.order_5_trajectory(initial, final, time_interval)

    # Sample times for evaluation
    times = np.linspace(0.0, 4.0, 100)

    # Get position (arc length), velocity, and acceleration at each time
    s_values = np.array([trajectory(t)[0] for t in times])
    v_values = np.array([trajectory(t)[1] for t in times])
    a_values = np.array([trajectory(t)[2] for t in times])

    # Plot the motion profiles
    plot_motion_profiles(times, s_values, v_values, a_values, "Motion Profiles for Circular Path")

    # Now let's map the motion profile to 3D positions along the path
    positions_3d = np.zeros((len(times), 3))

    # For each time point, get the corresponding position on the path
    for i, s in enumerate(s_values):
        # Evaluate the path at arc length s
        pos = path.position(s)
        positions_3d[i] = pos

    # Plot the path with the motion profile applied, showing key points and sampled points
    plot_3d_path(
        path_positions,
        "Half Circular Path with Polynomial Motion",
        sampled_points=positions_3d,
    )


def main() -> None:
    """Run examples."""
    print("Simple Path Examples with Polynomial Motion\n")

    # Linear path example
    linear_path_example()

    # Circular path example
    circular_path_example()


if __name__ == "__main__":
    main()
