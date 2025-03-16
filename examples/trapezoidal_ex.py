"""
Example demonstrating the usage of the TrapezoidalTrajectory class for motion planning.

This file showcases various capabilities of the trapezoidal trajectory generator, including:
1. Simple point-to-point trajectories with different constraints
2. Trajectories with non-zero initial and final velocities
3. Multi-point trajectories through sequences of waypoints
4. Time-constrained trajectories with custom timing
5. Visualization of position, velocity, and acceleration profiles
"""

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np

from interpolatepy.trapezoidal import InterpolationParams
from interpolatepy.trapezoidal import TrajectoryParams
from interpolatepy.trapezoidal import TrapezoidalTrajectory


def plot_trajectory(
    trajectory_func: Callable[[float], tuple[float, float, float]],
    duration: float,
    title: str = "Trapezoidal Trajectory",
    num_points: int = 1000,
) -> None:
    """
    Plot a trajectory showing position, velocity, and acceleration profiles.

    Parameters:
    -----------
    trajectory_func : Callable[[float], tuple[float, float, float]]
        Function that returns position, velocity, and acceleration at time t
    duration : float
        Total duration of the trajectory
    title : str
        Title for the plot
    num_points : int
        Number of points to sample for the plot
    """
    # Generate time points
    t_values = np.linspace(0, duration, num_points)

    # Evaluate trajectory at each time point
    positions = []
    velocities = []
    accelerations = []

    for t in t_values:
        pos, vel, acc = trajectory_func(t)
        positions.append(pos)
        velocities.append(vel)
        accelerations.append(acc)

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(title)

    # Plot position
    ax1.plot(t_values, positions, "b-")
    ax1.set_ylabel("Position")
    ax1.grid(True)

    # Plot velocity
    ax2.plot(t_values, velocities, "g-")
    ax2.set_ylabel("Velocity")
    ax2.grid(True)

    # Plot acceleration
    ax3.plot(t_values, accelerations, "r-")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Acceleration")
    ax3.grid(True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


def example_1_basic_trajectory() -> None:
    """Demonstrates a basic trapezoidal trajectory with zero initial and final velocities."""
    print("Example 1: Basic Trapezoidal Trajectory")
    print("--------------------------------------")

    params = TrajectoryParams(q0=0.0, q1=10.0, v0=0.0, v1=0.0, amax=3.0, vmax=4.0)

    traj_func, duration = TrapezoidalTrajectory.generate_trajectory(params)

    print(f"Duration: {duration:.2f} seconds")
    print(f"Position at t=0: {traj_func(0)[0]:.2f}")
    print(f"Position at t={duration / 2:.2f}: {traj_func(duration / 2)[0]:.2f}")
    print(f"Position at t={duration:.2f}: {traj_func(duration)[0]:.2f}")

    plot_trajectory(traj_func, duration, "Basic Trapezoidal Trajectory")


def example_2_nonzero_velocities() -> None:
    """Demonstrates a trajectory with non-zero initial and final velocities."""
    print("\nExample 2: Trajectory with Non-Zero Initial and Final Velocities")
    print("-------------------------------------------------------------")

    params = TrajectoryParams(
        q0=0.0,
        q1=10.0,
        v0=1.5,  # Non-zero initial velocity
        v1=2.0,  # Non-zero final velocity
        amax=5.0,
        vmax=4.0,
    )

    traj_func, duration = TrapezoidalTrajectory.generate_trajectory(params)

    print(f"Duration: {duration:.2f} seconds")
    print(f"Initial velocity: {traj_func(0)[1]:.2f}")
    print(f"Final velocity: {traj_func(duration)[1]:.2f}")

    plot_trajectory(traj_func, duration, "Trajectory with Non-Zero Initial and Final Velocities")


def example_3_negative_displacement() -> None:
    """Demonstrates a trajectory with negative displacement."""
    print("\nExample 3: Trajectory with Negative Displacement")
    print("---------------------------------------------")

    params = TrajectoryParams(
        q0=0.0, q1=-10.0, v0=1.5, v1=2.0, amax=5.0, vmax=4.0  # Negative displacement
    )

    traj_func, duration = TrapezoidalTrajectory.generate_trajectory(params)

    print(f"Duration: {duration:.2f} seconds")
    print(f"Initial position: {traj_func(0)[0]:.2f}")
    print(f"Final position: {traj_func(duration)[0]:.2f}")

    plot_trajectory(traj_func, duration, "Trajectory with Negative Displacement")


def example_4_duration_constrained() -> None:
    """Demonstrates a trajectory with fixed duration."""
    print("\nExample 4: Duration-Constrained Trajectory")
    print("----------------------------------------")

    params = TrajectoryParams(
        q0=0.0, q1=10.0, v0=0.0, v1=0.0, amax=5.0, duration=4.0  # Fixed duration of 4 seconds
    )

    traj_func, duration = TrapezoidalTrajectory.generate_trajectory(params)

    print("Specified duration: 4.00 seconds")
    print(f"Actual duration: {duration:.2f} seconds")

    plot_trajectory(traj_func, duration, "Duration-Constrained Trajectory (T=4s)")


def example_5_triangular_profile() -> None:
    """Demonstrates a triangular profile for short displacements."""
    print("\nExample 5: Triangular Profile for Short Displacement")
    print("-------------------------------------------------")

    params = TrajectoryParams(
        q0=0.0,
        q1=2.0,  # Short displacement that doesn't reach vmax
        v0=0.0,
        v1=0.0,
        amax=3.0,
        vmax=5.0,
    )

    traj_func, duration = TrapezoidalTrajectory.generate_trajectory(params)

    print(f"Duration: {duration:.2f} seconds")

    # Sample velocity at multiple points to show it never reaches vmax
    times = np.linspace(0, duration, 10)
    max_vel = max(traj_func(t)[1] for t in times)

    print(f"Maximum velocity: {max_vel:.2f} (less than vmax={5.0})")

    plot_trajectory(traj_func, duration, "Triangular Profile for Short Displacement")


def example_6_asymmetric_profile() -> None:
    """Demonstrates an asymmetric acceleration and deceleration profile."""
    print("\nExample 6: Asymmetric Acceleration and Deceleration")
    print("------------------------------------------------")

    params = TrajectoryParams(
        q0=0.0, q1=15.0, v0=3.0, v1=0.0, amax=4.0, vmax=6.0  # Start already moving  # End at rest
    )

    traj_func, duration = TrapezoidalTrajectory.generate_trajectory(params)

    print(f"Duration: {duration:.2f} seconds")
    print(f"Initial velocity: {traj_func(0)[1]:.2f}")
    print(f"Final velocity: {traj_func(duration)[1]:.2f}")

    plot_trajectory(traj_func, duration, "Asymmetric Acceleration and Deceleration")


def example_7_multi_point_custom_velocities() -> None:
    """Demonstrates a multi-point trajectory with custom intermediate velocities."""
    print("\nExample 7: Multi-Point Trajectory with Custom Velocities")
    print("-----------------------------------------------------")

    points = [0.0, 8.0, 12.0, 5.0, 10.0]
    inter_velocities = [3.0, 0.0, -2.0]  # Velocities at intermediate waypoints

    params = InterpolationParams(
        points=points, v0=0.0, vn=0.0, inter_velocities=inter_velocities, amax=4.0, vmax=5.0
    )

    traj_func, duration = TrapezoidalTrajectory.interpolate_waypoints(params)

    print(f"Waypoints: {points}")
    print(f"Intermediate velocities: {inter_velocities}")
    print(f"Total duration: {duration:.2f} seconds")

    plot_trajectory(traj_func, duration, "Trajectory with Custom Intermediate Velocities")


def example_8_time_constrained_multi_point() -> None:
    """Demonstrates a multi-point trajectory with specific timing constraints."""
    print("\nExample 8: Time-Constrained Multi-Point Trajectory")
    print("------------------------------------------------")

    points = [0.0, 5.0, 10.0, 7.0, 15.0]
    times = [0.0, 2.0, 4.0, 7.0, 10.0]  # Specific times to reach each point

    params = InterpolationParams(points=points, v0=0.0, vn=0.0, times=times, amax=15.0)

    traj_func, duration = TrapezoidalTrajectory.interpolate_waypoints(params)

    print(f"Waypoints: {points}")
    print(f"Time schedule: {times}")
    print(f"Total duration: {duration:.2f} seconds")

    # Verify the trajectory passes through the points at the specified times
    print("\nVerifying position at specified times:")
    for i, t in enumerate(times):
        pos = traj_func(t)[0]
        print(f"At t={t:.1f}s: Expected={points[i]:.1f}, Actual={pos:.1f}")

    plot_trajectory(traj_func, duration, "Time-Constrained Multi-Point Trajectory")


def example_9_oscillating_trajectory() -> None:
    """Demonstrates an oscillating trajectory with sign changes."""
    print("\nExample 9: Oscillating Trajectory with Sign Changes")
    print("------------------------------------------------")

    oscillating_points = [0.0, 5.0, -3.0, 8.0, -2.0, 4.0, 0.0]

    params = InterpolationParams(points=oscillating_points, v0=0.0, vn=0.0, amax=8.0, vmax=6.0)

    traj_func, duration = TrapezoidalTrajectory.interpolate_waypoints(params)

    print(f"Oscillating waypoints: {oscillating_points}")
    print(f"Total duration: {duration:.2f} seconds")

    plot_trajectory(traj_func, duration, "Oscillating Trajectory with Sign Changes")


def example_10_complex_velocity_profile() -> None:
    """Demonstrates a complex trajectory with custom velocity profiles."""
    print("\nExample 10: Complex Trajectory with Custom Velocity Profiles")
    print("--------------------------------------------------------")

    complex_points = [0.0, 10.0, 15.0, 5.0, 8.0, 3.0]
    complex_velocities = [4.0, 0.0, -5.0, 2.0]

    params = InterpolationParams(
        points=complex_points,
        v0=0.0,
        vn=0.0,
        inter_velocities=complex_velocities,
        amax=6.0,
        vmax=8.0,
    )

    traj_func, duration = TrapezoidalTrajectory.interpolate_waypoints(params)

    print(f"Complex waypoints: {complex_points}")
    print(f"Complex velocities: [0.0, {', '.join(str(v) for v in complex_velocities)}, 0.0]")
    print(f"Total duration: {duration:.2f} seconds")

    plot_trajectory(traj_func, duration, "Complex Trajectory with Custom Velocity Profiles")


if __name__ == "__main__":
    print("TrapezoidalTrajectory Class - Usage Examples")
    print("===========================================\n")

    # Run all examples
    example_1_basic_trajectory()
    example_2_nonzero_velocities()
    example_3_negative_displacement()
    example_4_duration_constrained()
    example_5_triangular_profile()
    example_6_asymmetric_profile()
    example_7_multi_point_custom_velocities()
    example_8_time_constrained_multi_point()
    example_9_oscillating_trajectory()
    example_10_complex_velocity_profile()
