"""
Example of using polynomial trajectories for interpolation.

This example demonstrates:
1. Two-point interpolation with 3rd, 5th, and 7th order polynomials
2. Multi-point interpolation with different polynomial orders
3. Visualization of the generated trajectories
"""

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np

from interpolatepy.polynomials import ORDER_3
from interpolatepy.polynomials import ORDER_5
from interpolatepy.polynomials import ORDER_7
from interpolatepy.polynomials import BoundaryCondition
from interpolatepy.polynomials import PolynomialTrajectory
from interpolatepy.polynomials import TimeInterval
from interpolatepy.polynomials import TrajectoryParams


def plot_trajectory(
    trajectory_func: Callable[[float], tuple[float, float, float, float]],
    time_interval: TimeInterval,
    title: str,
    num_points: int = 100,
) -> None:
    """
    Plot the generated trajectory.

    Parameters
    ----------
    trajectory_func : Callable[[float], tuple[float, float, float, float]]
        Function that computes trajectory at time t
    time_interval : TimeInterval
        Time interval for the trajectory
    title : str
        Title for the plot
    num_points : int, optional
        Number of points to plot, by default 100
    """
    times = np.linspace(time_interval.start, time_interval.end, num_points)
    positions = []
    velocities = []
    accelerations = []
    jerks = []

    for t in times:
        pos, vel, acc, jrk = trajectory_func(t)
        positions.append(pos)
        velocities.append(vel)
        accelerations.append(acc)
        jerks.append(jrk)

    plt.figure(figsize=(12, 10))

    # Position plot
    plt.subplot(4, 1, 1)
    plt.plot(times, positions, "b-", linewidth=2)
    plt.grid(True)
    plt.ylabel("Position")

    # Velocity plot
    plt.subplot(4, 1, 2)
    plt.plot(times, velocities, "g-", linewidth=2)
    plt.grid(True)
    plt.ylabel("Velocity")

    # Acceleration plot
    plt.subplot(4, 1, 3)
    plt.plot(times, accelerations, "r-", linewidth=2)
    plt.grid(True)
    plt.ylabel("Acceleration")

    # Jerk plot
    plt.subplot(4, 1, 4)
    plt.plot(times, jerks, "m-", linewidth=2)
    plt.grid(True)
    plt.ylabel("Jerk")
    plt.xlabel("Time")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def two_point_interpolation_example() -> None:
    """Demonstrate two-point interpolation with different polynomial orders."""
    print("Two-point Interpolation Example")
    print("----------------------------------------------------------")

    # Define common boundary conditions and time interval
    initial_pos = BoundaryCondition(position=0.0, velocity=0.0, acceleration=0.0, jerk=0.0)
    final_pos = BoundaryCondition(position=1.0, velocity=0.0, acceleration=0.0, jerk=0.0)
    time_interval = TimeInterval(start=0.0, end=2.0)

    # 3rd order polynomial
    traj_3 = PolynomialTrajectory.order_3_trajectory(initial_pos, final_pos, time_interval)

    # 5th order polynomial
    traj_5 = PolynomialTrajectory.order_5_trajectory(initial_pos, final_pos, time_interval)

    # 7th order polynomial
    traj_7 = PolynomialTrajectory.order_7_trajectory(initial_pos, final_pos, time_interval)

    # Compare the different orders on a single plot
    times = np.linspace(time_interval.start, time_interval.end, 100)

    plt.figure(figsize=(12, 10))

    # Position plot - all orders
    plt.subplot(4, 1, 1)
    positions_3 = [traj_3(t)[0] for t in times]
    positions_5 = [traj_5(t)[0] for t in times]
    positions_7 = [traj_7(t)[0] for t in times]

    plt.plot(times, positions_3, "b-", linewidth=2, label="Order 3")
    plt.plot(times, positions_5, "g-", linewidth=2, label="Order 5")
    plt.plot(times, positions_7, "r-", linewidth=2, label="Order 7")
    plt.grid(True)
    plt.ylabel("Position")
    plt.legend()

    # Velocity plot - all orders
    plt.subplot(4, 1, 2)
    velocities_3 = [traj_3(t)[1] for t in times]
    velocities_5 = [traj_5(t)[1] for t in times]
    velocities_7 = [traj_7(t)[1] for t in times]

    plt.plot(times, velocities_3, "b-", linewidth=2, label="Order 3")
    plt.plot(times, velocities_5, "g-", linewidth=2, label="Order 5")
    plt.plot(times, velocities_7, "r-", linewidth=2, label="Order 7")
    plt.grid(True)
    plt.ylabel("Velocity")
    plt.legend()

    # Acceleration plot - all orders
    plt.subplot(4, 1, 3)
    accelerations_3 = [traj_3(t)[2] for t in times]
    accelerations_5 = [traj_5(t)[2] for t in times]
    accelerations_7 = [traj_7(t)[2] for t in times]

    plt.plot(times, accelerations_3, "b-", linewidth=2, label="Order 3")
    plt.plot(times, accelerations_5, "g-", linewidth=2, label="Order 5")
    plt.plot(times, accelerations_7, "r-", linewidth=2, label="Order 7")
    plt.grid(True)
    plt.ylabel("Acceleration")
    plt.legend()

    # Jerk plot - all orders
    plt.subplot(4, 1, 4)
    jerks_3 = [traj_3(t)[3] for t in times]
    jerks_5 = [traj_5(t)[3] for t in times]
    jerks_7 = [traj_7(t)[3] for t in times]

    plt.plot(times, jerks_3, "b-", linewidth=2, label="Order 3")
    plt.plot(times, jerks_5, "g-", linewidth=2, label="Order 5")
    plt.plot(times, jerks_7, "r-", linewidth=2, label="Order 7")
    plt.grid(True)
    plt.ylabel("Jerk")
    plt.xlabel("Time")
    plt.legend()

    plt.suptitle("Comparison of Polynomial Orders")
    plt.tight_layout()
    plt.show()


def multipoint_interpolation_example() -> None:
    """Demonstrate multi-point interpolation with different polynomial orders."""
    print("Multi-point Interpolation Example")
    print("----------------------------------------------------------")

    # Define points and times for multi-point trajectory
    points = [0.0, 0.5, 0.8, 1.0, 0.7, 0.3, 0.0]
    times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    # Define velocities
    velocities = [0.0, 0.2, 0.1, 0.0, -0.1, -0.2, 0.0]

    # Define accelerations
    accelerations = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Define jerks
    jerks = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Create trajectory parameters for different orders
    params_3 = TrajectoryParams(
        points=points,
        times=times,
        velocities=velocities,
        order=ORDER_3,
    )

    params_5 = TrajectoryParams(
        points=points,
        times=times,
        velocities=velocities,
        accelerations=accelerations,
        order=ORDER_5,
    )

    params_7 = TrajectoryParams(
        points=points,
        times=times,
        velocities=velocities,
        accelerations=accelerations,
        jerks=jerks,
        order=ORDER_7,
    )

    # Create multi-point trajectories
    traj_3 = PolynomialTrajectory.multipoint_trajectory(params_3)
    traj_5 = PolynomialTrajectory.multipoint_trajectory(params_5)
    traj_7 = PolynomialTrajectory.multipoint_trajectory(params_7)

    # Time interval for plotting
    time_interval = TimeInterval(start=times[0], end=times[-1])
    detailed_times = np.linspace(time_interval.start, time_interval.end, 500)

    # Plot individual figure for 3rd order polynomial
    plt.figure(figsize=(12, 10))

    # Position plot
    plt.subplot(4, 1, 1)
    positions_3 = [traj_3(t)[0] for t in detailed_times]
    plt.plot(detailed_times, positions_3, "b-", linewidth=2)
    plt.plot(times, points, "ko", markersize=8, label="Control Points")
    plt.grid(True)
    plt.ylabel("Position")
    plt.legend()

    # Velocity plot
    plt.subplot(4, 1, 2)
    velocities_3 = [traj_3(t)[1] for t in detailed_times]
    plt.plot(detailed_times, velocities_3, "b-", linewidth=2)
    plt.grid(True)
    plt.ylabel("Velocity")

    # Acceleration plot
    plt.subplot(4, 1, 3)
    accelerations_3 = [traj_3(t)[2] for t in detailed_times]
    plt.plot(detailed_times, accelerations_3, "b-", linewidth=2)
    plt.grid(True)
    plt.ylabel("Acceleration")

    # Jerk plot
    plt.subplot(4, 1, 4)
    jerks_3 = [traj_3(t)[3] for t in detailed_times]
    plt.plot(detailed_times, jerks_3, "b-", linewidth=2)
    plt.grid(True)
    plt.ylabel("Jerk")
    plt.xlabel("Time")

    plt.suptitle("3rd Order Polynomial Trajectory")
    plt.tight_layout()
    plt.show()

    # Plot individual figure for 5th order polynomial
    plt.figure(figsize=(12, 10))

    # Position plot
    plt.subplot(4, 1, 1)
    positions_5 = [traj_5(t)[0] for t in detailed_times]
    plt.plot(detailed_times, positions_5, "g-", linewidth=2)
    plt.plot(times, points, "ko", markersize=8, label="Control Points")
    plt.grid(True)
    plt.ylabel("Position")
    plt.legend()

    # Velocity plot
    plt.subplot(4, 1, 2)
    velocities_5 = [traj_5(t)[1] for t in detailed_times]
    plt.plot(detailed_times, velocities_5, "g-", linewidth=2)
    plt.grid(True)
    plt.ylabel("Velocity")

    # Acceleration plot
    plt.subplot(4, 1, 3)
    accelerations_5 = [traj_5(t)[2] for t in detailed_times]
    plt.plot(detailed_times, accelerations_5, "g-", linewidth=2)
    plt.grid(True)
    plt.ylabel("Acceleration")

    # Jerk plot
    plt.subplot(4, 1, 4)
    jerks_5 = [traj_5(t)[3] for t in detailed_times]
    plt.plot(detailed_times, jerks_5, "g-", linewidth=2)
    plt.grid(True)
    plt.ylabel("Jerk")
    plt.xlabel("Time")

    plt.suptitle("5th Order Polynomial Trajectory")
    plt.tight_layout()
    plt.show()

    # Plot individual figure for 7th order polynomial
    plt.figure(figsize=(12, 10))

    # Position plot
    plt.subplot(4, 1, 1)
    positions_7 = [traj_7(t)[0] for t in detailed_times]
    plt.plot(detailed_times, positions_7, "r-", linewidth=2)
    plt.plot(times, points, "ko", markersize=8, label="Control Points")
    plt.grid(True)
    plt.ylabel("Position")
    plt.legend()

    # Velocity plot
    plt.subplot(4, 1, 2)
    velocities_7 = [traj_7(t)[1] for t in detailed_times]
    plt.plot(detailed_times, velocities_7, "r-", linewidth=2)
    plt.grid(True)
    plt.ylabel("Velocity")

    # Acceleration plot
    plt.subplot(4, 1, 3)
    accelerations_7 = [traj_7(t)[2] for t in detailed_times]
    plt.plot(detailed_times, accelerations_7, "r-", linewidth=2)
    plt.grid(True)
    plt.ylabel("Acceleration")

    # Jerk plot
    plt.subplot(4, 1, 4)
    jerks_7 = [traj_7(t)[3] for t in detailed_times]
    plt.plot(detailed_times, jerks_7, "r-", linewidth=2)
    plt.grid(True)
    plt.ylabel("Jerk")
    plt.xlabel("Time")

    plt.suptitle("7th Order Polynomial Trajectory")
    plt.tight_layout()
    plt.show()

    # Plot the comparison figure with all three orders
    plt.figure(figsize=(12, 10))

    # Position plot - all orders
    plt.subplot(4, 1, 1)
    plt.plot(detailed_times, positions_3, "b-", linewidth=2, label="Order 3")
    plt.plot(detailed_times, positions_5, "g-", linewidth=2, label="Order 5")
    plt.plot(detailed_times, positions_7, "r-", linewidth=2, label="Order 7")
    plt.plot(times, points, "ko", markersize=8, label="Control Points")
    plt.grid(True)
    plt.ylabel("Position")
    plt.legend()

    # Velocity plot - all orders
    plt.subplot(4, 1, 2)
    plt.plot(detailed_times, velocities_3, "b-", linewidth=2, label="Order 3")
    plt.plot(detailed_times, velocities_5, "g-", linewidth=2, label="Order 5")
    plt.plot(detailed_times, velocities_7, "r-", linewidth=2, label="Order 7")
    plt.grid(True)
    plt.ylabel("Velocity")
    plt.legend()

    # Acceleration plot - all orders
    plt.subplot(4, 1, 3)
    plt.plot(detailed_times, accelerations_3, "b-", linewidth=2, label="Order 3")
    plt.plot(detailed_times, accelerations_5, "g-", linewidth=2, label="Order 5")
    plt.plot(detailed_times, accelerations_7, "r-", linewidth=2, label="Order 7")
    plt.grid(True)
    plt.ylabel("Acceleration")
    plt.legend()

    # Jerk plot - all orders
    plt.subplot(4, 1, 4)
    plt.plot(detailed_times, jerks_3, "b-", linewidth=2, label="Order 3")
    plt.plot(detailed_times, jerks_5, "g-", linewidth=2, label="Order 5")
    plt.plot(detailed_times, jerks_7, "r-", linewidth=2, label="Order 7")
    plt.grid(True)
    plt.ylabel("Jerk")
    plt.xlabel("Time")
    plt.legend()

    plt.suptitle("Comparison of Multi-point Polynomial Orders")
    plt.tight_layout()
    plt.show()


def multipoint_interpolation_no_vel_example() -> None:
    """Demonstrate multi-point interpolation with different polynomial orders."""
    print("Multi-point Interpolation with Heuristic Velocity Example")
    print("----------------------------------------------------------")

    # Define points and times for multi-point trajectory
    points = [0.0, 0.5, 0.8, 1.0, 0.7, 0.3, 0.0]
    times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    # Create trajectory parameters for different orders
    params_3 = TrajectoryParams(
        points=points,
        times=times,
        order=ORDER_3,
    )

    params_5 = TrajectoryParams(
        points=points,
        times=times,
        order=ORDER_5,
    )

    params_7 = TrajectoryParams(
        points=points,
        times=times,
        order=ORDER_7,
    )

    # Create multi-point trajectories
    traj_3 = PolynomialTrajectory.multipoint_trajectory(params_3)
    traj_5 = PolynomialTrajectory.multipoint_trajectory(params_5)
    traj_7 = PolynomialTrajectory.multipoint_trajectory(params_7)

    # Time interval for plotting
    time_interval = TimeInterval(start=times[0], end=times[-1])
    detailed_times = np.linspace(time_interval.start, time_interval.end, 500)

    # Plot individual figure for 3rd order polynomial
    plt.figure(figsize=(12, 10))

    # Position plot
    plt.subplot(4, 1, 1)
    positions_3 = [traj_3(t)[0] for t in detailed_times]
    plt.plot(detailed_times, positions_3, "b-", linewidth=2)
    plt.plot(times, points, "ko", markersize=8, label="Control Points")
    plt.grid(True)
    plt.ylabel("Position")
    plt.legend()

    # Velocity plot
    plt.subplot(4, 1, 2)
    velocities_3 = [traj_3(t)[1] for t in detailed_times]
    plt.plot(detailed_times, velocities_3, "b-", linewidth=2)
    plt.grid(True)
    plt.ylabel("Velocity")

    # Acceleration plot
    plt.subplot(4, 1, 3)
    accelerations_3 = [traj_3(t)[2] for t in detailed_times]
    plt.plot(detailed_times, accelerations_3, "b-", linewidth=2)
    plt.grid(True)
    plt.ylabel("Acceleration")

    # Jerk plot
    plt.subplot(4, 1, 4)
    jerks_3 = [traj_3(t)[3] for t in detailed_times]
    plt.plot(detailed_times, jerks_3, "b-", linewidth=2)
    plt.grid(True)
    plt.ylabel("Jerk")
    plt.xlabel("Time")

    plt.suptitle("3rd Order Polynomial Trajectory")
    plt.tight_layout()
    plt.show()

    # Plot individual figure for 5th order polynomial
    plt.figure(figsize=(12, 10))

    # Position plot
    plt.subplot(4, 1, 1)
    positions_5 = [traj_5(t)[0] for t in detailed_times]
    plt.plot(detailed_times, positions_5, "g-", linewidth=2)
    plt.plot(times, points, "ko", markersize=8, label="Control Points")
    plt.grid(True)
    plt.ylabel("Position")
    plt.legend()

    # Velocity plot
    plt.subplot(4, 1, 2)
    velocities_5 = [traj_5(t)[1] for t in detailed_times]
    plt.plot(detailed_times, velocities_5, "g-", linewidth=2)
    plt.grid(True)
    plt.ylabel("Velocity")

    # Acceleration plot
    plt.subplot(4, 1, 3)
    accelerations_5 = [traj_5(t)[2] for t in detailed_times]
    plt.plot(detailed_times, accelerations_5, "g-", linewidth=2)
    plt.grid(True)
    plt.ylabel("Acceleration")

    # Jerk plot
    plt.subplot(4, 1, 4)
    jerks_5 = [traj_5(t)[3] for t in detailed_times]
    plt.plot(detailed_times, jerks_5, "g-", linewidth=2)
    plt.grid(True)
    plt.ylabel("Jerk")
    plt.xlabel("Time")

    plt.suptitle("5th Order Polynomial Trajectory")
    plt.tight_layout()
    plt.show()

    # Plot individual figure for 7th order polynomial
    plt.figure(figsize=(12, 10))

    # Position plot
    plt.subplot(4, 1, 1)
    positions_7 = [traj_7(t)[0] for t in detailed_times]
    plt.plot(detailed_times, positions_7, "r-", linewidth=2)
    plt.plot(times, points, "ko", markersize=8, label="Control Points")
    plt.grid(True)
    plt.ylabel("Position")
    plt.legend()

    # Velocity plot
    plt.subplot(4, 1, 2)
    velocities_7 = [traj_7(t)[1] for t in detailed_times]
    plt.plot(detailed_times, velocities_7, "r-", linewidth=2)
    plt.grid(True)
    plt.ylabel("Velocity")

    # Acceleration plot
    plt.subplot(4, 1, 3)
    accelerations_7 = [traj_7(t)[2] for t in detailed_times]
    plt.plot(detailed_times, accelerations_7, "r-", linewidth=2)
    plt.grid(True)
    plt.ylabel("Acceleration")

    # Jerk plot
    plt.subplot(4, 1, 4)
    jerks_7 = [traj_7(t)[3] for t in detailed_times]
    plt.plot(detailed_times, jerks_7, "r-", linewidth=2)
    plt.grid(True)
    plt.ylabel("Jerk")
    plt.xlabel("Time")

    plt.suptitle("7th Order Polynomial Trajectory")
    plt.tight_layout()
    plt.show()

    # Plot the comparison figure with all three orders
    plt.figure(figsize=(12, 10))

    # Position plot - all orders
    plt.subplot(4, 1, 1)
    plt.plot(detailed_times, positions_3, "b-", linewidth=2, label="Order 3")
    plt.plot(detailed_times, positions_5, "g-", linewidth=2, label="Order 5")
    plt.plot(detailed_times, positions_7, "r-", linewidth=2, label="Order 7")
    plt.plot(times, points, "ko", markersize=8, label="Control Points")
    plt.grid(True)
    plt.ylabel("Position")
    plt.legend()

    # Velocity plot - all orders
    plt.subplot(4, 1, 2)
    plt.plot(detailed_times, velocities_3, "b-", linewidth=2, label="Order 3")
    plt.plot(detailed_times, velocities_5, "g-", linewidth=2, label="Order 5")
    plt.plot(detailed_times, velocities_7, "r-", linewidth=2, label="Order 7")
    plt.grid(True)
    plt.ylabel("Velocity")
    plt.legend()

    # Acceleration plot - all orders
    plt.subplot(4, 1, 3)
    plt.plot(detailed_times, accelerations_3, "b-", linewidth=2, label="Order 3")
    plt.plot(detailed_times, accelerations_5, "g-", linewidth=2, label="Order 5")
    plt.plot(detailed_times, accelerations_7, "r-", linewidth=2, label="Order 7")
    plt.grid(True)
    plt.ylabel("Acceleration")
    plt.legend()

    # Jerk plot - all orders
    plt.subplot(4, 1, 4)
    plt.plot(detailed_times, jerks_3, "b-", linewidth=2, label="Order 3")
    plt.plot(detailed_times, jerks_5, "g-", linewidth=2, label="Order 5")
    plt.plot(detailed_times, jerks_7, "r-", linewidth=2, label="Order 7")
    plt.grid(True)
    plt.ylabel("Jerk")
    plt.xlabel("Time")
    plt.legend()

    plt.suptitle("Comparison of Multi-point Polynomial Orders")
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Main function to run the examples."""
    print("Polynomial Trajectory Interpolation Examples")
    print("===============================================")

    # Two-point interpolation example
    two_point_interpolation_example()

    # Multi-point interpolation example
    multipoint_interpolation_example()

    # Heuristic Vel example
    multipoint_interpolation_no_vel_example()


if __name__ == "__main__":
    main()
