"""
Example file demonstrating the use of linear_traj function.

This example shows:
1. How to generate a linear trajectory between two points
2. How to use the function with both scalar and vector positions
3. Visualization of the generated trajectories
"""

import matplotlib.pyplot as plt
import numpy as np

from interpolatepy.linear import linear_traj


def scalar_trajectory_example() -> None:
    """Demonstrate linear_traj with scalar positions."""
    print("Scalar Trajectory Example")
    print("--------------------------")

    # Define start and end points
    p0 = 0.0
    p1 = 10.0

    # Define time interval
    t0 = 0.0
    t1 = 2.0

    # Generate time array
    num_points = 100
    time_array = np.linspace(t0, t1, num_points)

    # Calculate trajectory
    positions, velocities, accelerations = linear_traj(p0, p1, t0, t1, time_array)

    # Print some values
    print(f"Start position: {positions[0]}")
    print(f"End position: {positions[-1]}")
    print(f"Constant velocity: {velocities[0]}")
    print(f"Acceleration: {accelerations[0]}")

    # Plot results
    plt.figure(figsize=(12, 8))

    # Position plot
    plt.subplot(3, 1, 1)
    plt.plot(time_array, positions, "b-", linewidth=2)
    plt.grid(True)
    plt.ylabel("Position")
    plt.title("Linear Trajectory - Scalar Case")

    # Velocity plot
    plt.subplot(3, 1, 2)
    plt.plot(time_array, velocities, "g-", linewidth=2)
    plt.grid(True)
    plt.ylabel("Velocity")

    # Acceleration plot
    plt.subplot(3, 1, 3)
    plt.plot(time_array, accelerations, "r-", linewidth=2)
    plt.grid(True)
    plt.ylabel("Acceleration")
    plt.xlabel("Time")

    plt.tight_layout()
    plt.show()


def vector_trajectory_example() -> None:
    """Demonstrate linear_traj with vector positions (2D points)."""
    print("\nVector Trajectory Example (2D)")
    print("------------------------------")

    # Define start and end points (2D vectors)
    p0 = [0.0, 0.0]  # Starting at origin
    p1 = [10.0, 5.0]  # Ending at (10, 5)

    # Define time interval
    t0 = 0.0
    t1 = 3.0

    # Generate time array
    num_points = 100
    time_array = np.linspace(t0, t1, num_points)

    # Calculate trajectory
    positions, velocities, accelerations = linear_traj(p0, p1, t0, t1, time_array)

    # Print some values
    print(f"Start position: {positions[0]}")
    print(f"End position: {positions[-1]}")
    print(f"Constant velocity: {velocities[0]}")
    print(f"Acceleration: {accelerations[0]}")

    # Extract x and y components
    x_positions = positions[:, 0]
    y_positions = positions[:, 1]
    x_velocities = velocities[:, 0]
    y_velocities = velocities[:, 1]

    # Plot results
    plt.figure(figsize=(12, 10))

    # Trajectory in 2D space
    plt.subplot(3, 1, 1)
    plt.plot(x_positions, y_positions, "b-", linewidth=2)
    plt.plot(p0[0], p0[1], "ro", markersize=8, label="Start")
    plt.plot(p1[0], p1[1], "go", markersize=8, label="End")
    plt.grid(True)
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.title("Linear Trajectory in 2D Space")
    plt.legend()

    # X and Y positions over time
    plt.subplot(3, 1, 2)
    plt.plot(time_array, x_positions, "b-", linewidth=2, label="X position")
    plt.plot(time_array, y_positions, "g-", linewidth=2, label="Y position")
    plt.grid(True)
    plt.ylabel("Position")
    plt.legend()

    # X and Y velocities over time
    plt.subplot(3, 1, 3)
    plt.plot(time_array, x_velocities, "b-", linewidth=2, label="X velocity")
    plt.plot(time_array, y_velocities, "g-", linewidth=2, label="Y velocity")
    plt.grid(True)
    plt.ylabel("Velocity")
    plt.xlabel("Time")
    plt.legend()

    plt.tight_layout()
    plt.show()


def main() -> None:
    """Run both examples."""
    print("Linear Trajectory Examples")
    print("=========================\n")

    # Run scalar example
    scalar_trajectory_example()

    # Run vector example
    vector_trajectory_example()


if __name__ == "__main__":
    main()
