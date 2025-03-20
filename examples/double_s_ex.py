"""
Example demonstrating the usage of the DoubleSTrajectory class for motion planning.

This file showcases various capabilities of the double-s trajectory generator, including:
1. Standard trajectories with bounded jerk, acceleration, and velocity
2. Velocity matching with equal positions
3. Negative displacement trajectories
4. Asymmetric velocities (non-zero initial and final velocities)
5. Visualization of position, velocity, acceleration, and jerk profiles

Note: This example follows the project's coding standards with proper type annotations
and variable naming conventions.
"""

import matplotlib.pyplot as plt
import numpy as np

from interpolatepy.double_s import DoubleSTrajectory
from interpolatepy.double_s import StateParams
from interpolatepy.double_s import TrajectoryBounds


def example_standard_trajectory() -> None:
    """Demonstrate a standard double-s trajectory with different positions."""
    print("\nStandard Trajectory Example")
    print("---------------------------")

    # Example parameters
    q_0 = 0.0  # Start position
    q_1 = 30.0  # End position
    v_0 = 0.0  # Start velocity
    v_1 = 0.0  # End velocity

    # Create state parameters and trajectory bounds
    state = StateParams(q_0=q_0, q_1=q_1, v_0=v_0, v_1=v_1)
    bounds = TrajectoryBounds(v_bound=20.0, a_bound=60.0, j_bound=120.0)

    # Create the trajectory
    trajectory = DoubleSTrajectory(state, bounds)

    # Get the total duration
    t_duration = trajectory.get_duration()
    print(f"Trajectory duration: {t_duration:.3f} seconds")

    # Create time points for evaluation
    t_sample = 0.001  # 1ms sample time
    time_array = np.arange(0, t_duration + t_sample, t_sample)

    # Evaluate trajectory at those time points
    positions, velocities, accelerations, jerks = trajectory.evaluate(time_array)

    # Get phase durations
    phase_durations = trajectory.get_phase_durations()
    t_accel = phase_durations["acceleration"]
    t_const = phase_durations["constant_velocity"]

    # Plot results
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Standard Double-S Trajectory")

    # Position plot
    ax1.plot(time_array, positions, "b-", linewidth=2)
    ax1.set_ylabel("Position")
    ax1.grid(True)

    # Add phase markers
    ax1.axvline(x=t_accel, color="r", linestyle="--", alpha=0.5)
    ax1.axvline(x=t_accel + t_const, color="r", linestyle="--", alpha=0.5)

    # Velocity plot
    ax2.plot(time_array, velocities, "g-", linewidth=2)
    ax2.set_ylabel("Velocity")
    ax2.grid(True)

    # Add phase markers
    ax2.axvline(x=t_accel, color="r", linestyle="--", alpha=0.5)
    ax2.axvline(x=t_accel + t_const, color="r", linestyle="--", alpha=0.5)

    # Acceleration plot
    ax3.plot(time_array, accelerations, "r-", linewidth=2)
    ax3.set_ylabel("Acceleration")
    ax3.grid(True)

    # Add phase markers
    ax3.axvline(x=t_accel, color="r", linestyle="--", alpha=0.5)
    ax3.axvline(x=t_accel + t_const, color="r", linestyle="--", alpha=0.5)

    # Jerk plot
    ax4.plot(time_array, jerks, "m-", linewidth=2)
    ax4.set_ylabel("Jerk")
    ax4.set_xlabel("Time (s)")
    ax4.grid(True)

    # Add phase markers
    ax4.axvline(x=t_accel, color="r", linestyle="--", alpha=0.5)
    ax4.axvline(x=t_accel + t_const, color="r", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

    # Handle index access with proper type checking to satisfy mypy
    if isinstance(positions, np.ndarray) and len(positions) > 0:
        final_pos = positions[-1]
        final_vel = velocities[-1]  # type: ignore
        print(f"Final position: {final_pos:.3f}, Final velocity: {final_vel:.3f}")
    else:
        print("Trajectory evaluation returned empty result")
    print(f"Phase durations: {phase_durations}")


def example_velocity_matching() -> None:
    """Demonstrate matching velocities when positions are the same."""
    print("\nVelocity Matching Example")
    print("------------------------")

    # Example parameters
    q_0 = q_1 = 10.0  # Same start and end position
    v_0 = 0.0  # Start velocity
    v_1 = 5.0  # End velocity (different)

    # Create state parameters and trajectory bounds
    state = StateParams(q_0=q_0, q_1=q_1, v_0=v_0, v_1=v_1)
    bounds = TrajectoryBounds(v_bound=20.0, a_bound=60.0, j_bound=120.0)

    # Create the trajectory
    trajectory = DoubleSTrajectory(state, bounds)

    # Get the total duration
    t_duration = trajectory.get_duration()
    print(f"Trajectory duration: {t_duration:.3f} seconds")

    # Create time points for evaluation
    t_sample = 0.001  # 1ms sample time
    time_array = np.arange(0, t_duration + t_sample, t_sample)

    # Evaluate trajectory at those time points
    positions, velocities, accelerations, jerks = trajectory.evaluate(time_array)

    # Plot results
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Velocity Matching (Equal Positions)")

    # Position plot
    ax1.plot(time_array, positions, "b-", linewidth=2)
    ax1.set_ylabel("Position")
    ax1.grid(True)

    # Velocity plot
    ax2.plot(time_array, velocities, "g-", linewidth=2)
    ax2.set_ylabel("Velocity")
    ax2.grid(True)

    # Acceleration plot
    ax3.plot(time_array, accelerations, "r-", linewidth=2)
    ax3.set_ylabel("Acceleration")
    ax3.grid(True)

    # Jerk plot
    ax4.plot(time_array, jerks, "m-", linewidth=2)
    ax4.set_ylabel("Jerk")
    ax4.set_xlabel("Time (s)")
    ax4.grid(True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

    # Handle index access with proper type checking
    if isinstance(positions, np.ndarray) and len(positions) > 0:
        final_pos = positions[-1]
        final_vel = velocities[-1]  # type: ignore
        print(f"Final position: {final_pos:.3f}, Final velocity: {final_vel:.3f}")
    else:
        print("Trajectory evaluation returned empty result")


def example_negative_displacement() -> None:
    """Demonstrate a trajectory with negative displacement."""
    print("\nNegative Displacement Example")
    print("----------------------------")

    # Example parameters
    q_0 = 0.0  # Start position
    q_1 = -30.0  # End position (negative displacement)
    v_0 = 0.0  # Start velocity
    v_1 = 0.0  # End velocity

    # Create state parameters and trajectory bounds
    state = StateParams(q_0=q_0, q_1=q_1, v_0=v_0, v_1=v_1)
    bounds = TrajectoryBounds(v_bound=20.0, a_bound=60.0, j_bound=120.0)

    # Create the trajectory
    trajectory = DoubleSTrajectory(state, bounds)

    # Get the total duration
    t_duration = trajectory.get_duration()
    print(f"Trajectory duration: {t_duration:.3f} seconds")

    # Create time points for evaluation
    t_sample = 0.001  # 1ms sample time
    time_array = np.arange(0, t_duration + t_sample, t_sample)

    # Evaluate trajectory at those time points
    positions, velocities, accelerations, jerks = trajectory.evaluate(time_array)

    # Get phase durations
    phase_durations = trajectory.get_phase_durations()
    t_accel = phase_durations["acceleration"]
    t_const = phase_durations["constant_velocity"]

    # Plot results
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Negative Displacement Trajectory")

    # Position plot
    ax1.plot(time_array, positions, "b-", linewidth=2)
    ax1.set_ylabel("Position")
    ax1.grid(True)

    # Add phase markers
    ax1.axvline(x=t_accel, color="r", linestyle="--", alpha=0.5)
    ax1.axvline(x=t_accel + t_const, color="r", linestyle="--", alpha=0.5)

    # Velocity plot
    ax2.plot(time_array, velocities, "g-", linewidth=2)
    ax2.set_ylabel("Velocity")
    ax2.grid(True)

    # Add phase markers
    ax2.axvline(x=t_accel, color="r", linestyle="--", alpha=0.5)
    ax2.axvline(x=t_accel + t_const, color="r", linestyle="--", alpha=0.5)

    # Acceleration plot
    ax3.plot(time_array, accelerations, "r-", linewidth=2)
    ax3.set_ylabel("Acceleration")
    ax3.grid(True)

    # Add phase markers
    ax3.axvline(x=t_accel, color="r", linestyle="--", alpha=0.5)
    ax3.axvline(x=t_accel + t_const, color="r", linestyle="--", alpha=0.5)

    # Jerk plot
    ax4.plot(time_array, jerks, "m-", linewidth=2)
    ax4.set_ylabel("Jerk")
    ax4.set_xlabel("Time (s)")
    ax4.grid(True)

    # Add phase markers
    ax4.axvline(x=t_accel, color="r", linestyle="--", alpha=0.5)
    ax4.axvline(x=t_accel + t_const, color="r", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

    # Handle index access with proper type checking
    if isinstance(positions, np.ndarray) and len(positions) > 0:
        final_pos = positions[-1]
        final_vel = velocities[-1]  # type: ignore
        print(f"Final position: {final_pos:.3f}, Final velocity: {final_vel:.3f}")
    else:
        print("Trajectory evaluation returned empty result")


def example_asymmetric_velocities() -> None:
    """Demonstrate a trajectory with different initial and final velocities."""
    print("\nAsymmetric Velocities Example")
    print("---------------------------")

    # Example parameters
    q_0 = 0.0  # Start position
    q_1 = 50.0  # End position
    v_0 = 10.0  # Start velocity (non-zero)
    v_1 = -5.0  # End velocity (non-zero and different from start)

    # Create state parameters and trajectory bounds
    state = StateParams(q_0=q_0, q_1=q_1, v_0=v_0, v_1=v_1)
    bounds = TrajectoryBounds(v_bound=20.0, a_bound=60.0, j_bound=120.0)

    # Create the trajectory
    trajectory = DoubleSTrajectory(state, bounds)

    # Get the total duration
    t_duration = trajectory.get_duration()
    print(f"Trajectory duration: {t_duration:.3f} seconds")

    # Create time points for evaluation
    t_sample = 0.001  # 1ms sample time
    time_array = np.arange(0, t_duration + t_sample, t_sample)

    # Evaluate trajectory at those time points
    positions, velocities, accelerations, jerks = trajectory.evaluate(time_array)

    # Get phase durations
    phase_durations = trajectory.get_phase_durations()
    t_accel = phase_durations["acceleration"]
    t_const = phase_durations["constant_velocity"]

    # Plot results
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Asymmetric Velocities Trajectory")

    # Position plot
    ax1.plot(time_array, positions, "b-", linewidth=2)
    ax1.set_ylabel("Position")
    ax1.grid(True)

    # Add phase markers
    ax1.axvline(x=t_accel, color="r", linestyle="--", alpha=0.5)
    ax1.axvline(x=t_accel + t_const, color="r", linestyle="--", alpha=0.5)

    # Velocity plot
    ax2.plot(time_array, velocities, "g-", linewidth=2)
    ax2.set_ylabel("Velocity")
    ax2.grid(True)

    # Add phase markers
    ax2.axvline(x=t_accel, color="r", linestyle="--", alpha=0.5)
    ax2.axvline(x=t_accel + t_const, color="r", linestyle="--", alpha=0.5)

    # Acceleration plot
    ax3.plot(time_array, accelerations, "r-", linewidth=2)
    ax3.set_ylabel("Acceleration")
    ax3.grid(True)

    # Add phase markers
    ax3.axvline(x=t_accel, color="r", linestyle="--", alpha=0.5)
    ax3.axvline(x=t_accel + t_const, color="r", linestyle="--", alpha=0.5)

    # Jerk plot
    ax4.plot(time_array, jerks, "m-", linewidth=2)
    ax4.set_ylabel("Jerk")
    ax4.set_xlabel("Time (s)")
    ax4.grid(True)

    # Add phase markers
    ax4.axvline(x=t_accel, color="r", linestyle="--", alpha=0.5)
    ax4.axvline(x=t_accel + t_const, color="r", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

    # Handle index access with proper type checking
    if isinstance(positions, np.ndarray) and len(positions) > 0:
        final_pos = positions[-1]
        final_vel = velocities[-1]  # type: ignore
        print(f"Final position: {final_pos:.3f}, Final velocity: {final_vel:.3f}")
    else:
        print("Trajectory evaluation returned empty result")


def example_factory_method() -> None:
    """Demonstrate using the static factory method to create a trajectory function."""
    print("\nFactory Method Example")
    print("---------------------")

    # Example parameters
    q_0 = 0.0  # Start position
    q_1 = 20.0  # End position
    v_0 = 0.0  # Start velocity
    v_1 = 0.0  # End velocity

    # Create state parameters and trajectory bounds
    state = StateParams(q_0=q_0, q_1=q_1, v_0=v_0, v_1=v_1)
    bounds = TrajectoryBounds(v_bound=15.0, a_bound=45.0, j_bound=90.0)

    # Use the factory method to create a trajectory function
    traj_func, duration = DoubleSTrajectory.create_trajectory(state, bounds)
    print(f"Trajectory duration: {duration:.3f} seconds")

    # Create time points for evaluation
    t_sample = 0.001  # 1ms sample time
    time_array = np.arange(0, duration + t_sample, t_sample)

    # Evaluate trajectory at those time points using the function
    positions, velocities, accelerations, jerks = traj_func(time_array)

    # Plot results
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Trajectory Using Factory Method")

    # Position plot
    ax1.plot(time_array, positions, "b-", linewidth=2)
    ax1.set_ylabel("Position")
    ax1.grid(True)

    # Velocity plot
    ax2.plot(time_array, velocities, "g-", linewidth=2)
    ax2.set_ylabel("Velocity")
    ax2.grid(True)

    # Acceleration plot
    ax3.plot(time_array, accelerations, "r-", linewidth=2)
    ax3.set_ylabel("Acceleration")
    ax3.grid(True)

    # Jerk plot
    ax4.plot(time_array, jerks, "m-", linewidth=2)
    ax4.set_ylabel("Jerk")
    ax4.set_xlabel("Time (s)")
    ax4.grid(True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

    # Handle index access with proper type checking
    if isinstance(positions, np.ndarray) and len(positions) > 0:
        final_pos = positions[-1]
        final_vel = velocities[-1]  # type: ignore
        print(f"Final position: {final_pos:.3f}, Final velocity: {final_vel:.3f}")
    else:
        print("Trajectory evaluation returned empty result")


def main() -> None:
    """Run all examples."""
    print("Double-S Trajectory Examples")
    print("===========================")

    try:
        example_standard_trajectory()
        example_velocity_matching()
        example_negative_displacement()
        example_asymmetric_velocities()
        example_factory_method()  # Added example for the factory method
        print("\nAll examples completed successfully!")
    except (ValueError, TypeError, RuntimeError) as e:
        # Catch specific exceptions that might occur during trajectory calculation
        print(f"Error running examples: {e}")


if __name__ == "__main__":
    main()
