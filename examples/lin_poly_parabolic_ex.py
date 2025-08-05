from collections.abc import Callable
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from interpolatepy.lin_poly_parabolic import ParabolicBlendTrajectory


def plot_trajectory_with_waypoints(
    traj: ParabolicBlendTrajectory,
    q: Sequence[float],
    t: Sequence[float],
) -> None:
    """Plot the trajectory along with waypoints highlighted.

    The function generates position, velocity, and acceleration profiles
    from the provided trajectory and displays them in three subplots.
    A dashed linear interpolation between waypoints is also plotted.

    Parameters
    ----------
    traj : ParabolicBlendTrajectory
        Trajectory object created with waypoints and blend durations.
    q : Sequence[float]
        List or array of position waypoints.
    t : Sequence[float]
        List or array of times corresponding to each waypoint.

    Returns
    -------
    None
        Displays the plot directly.
    """
    traj_func, duration = traj.generate()
    times = np.arange(0.0, duration + traj.dt, traj.dt)

    # Evaluate trajectory at each time point
    positions = np.empty_like(times)
    velocities = np.empty_like(times)
    accelerations = np.empty_like(times)

    for i, time in enumerate(times):
        positions[i], velocities[i], accelerations[i] = traj_func(time)

    # Adjust waypoint times to align with trajectory time scale
    adjusted_t = [time + traj.dt_blend[0] / 2 for time in t]

    # Create the figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Plot position trajectory with waypoints
    ax1.plot(times, positions, label="Position Trajectory")
    linear_positions = np.interp(times, adjusted_t, q)
    ax1.plot(times, linear_positions, "--", label="Linear Interpolation")
    ax1.scatter(adjusted_t, q, color="red", s=50, zorder=5, label="Via Points")
    ax1.set_ylabel("Position")
    ax1.legend()
    ax1.grid(True)

    # Plot velocity profile
    ax2.plot(times, velocities, label="Velocity")
    ax2.set_ylabel("Velocity")
    ax2.axhline(0, linestyle="--", alpha=0.3)
    ax2.legend()
    ax2.grid(True)

    # Plot acceleration profile
    ax3.plot(times, accelerations, label="Acceleration")
    ax3.set_ylabel("Acceleration")
    ax3.set_xlabel("Time [s]")
    ax3.axhline(0, linestyle="--", alpha=0.3)
    ax3.legend()
    ax3.grid(True)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Define waypoints and blend durations
    q: list[float] = [0, 2 * np.pi, np.pi / 2, np.pi]
    t: list[float] = [0, 2, 3, 5]
    dt_blend: np.ndarray = np.full(len(t), 0.6)

    print("Creating parabolic blend trajectory with:")
    print(f"  Positions: {q}")
    print(f"  Times: {t}")
    print(f"  Blend durations: {dt_blend}")

    traj: ParabolicBlendTrajectory = ParabolicBlendTrajectory(q, t, dt_blend)

    # Option 1: Use the built-in plot method
    print("\nPlotting trajectory using built-in plot method...")
    traj.plot()
    plt.show()

    # Option 2: Custom plotting with waypoints highlighted
    print("\nPlotting trajectory with highlighted waypoints...")
    plot_trajectory_with_waypoints(traj, q, t)

    # Option 3: Direct usage of the trajectory function
    print("\nDirect usage of trajectory function:")
    traj_func: Callable[[float], tuple[float, float, float]]
    traj_func, duration = traj.generate()

    # Evaluate at specific times
    evaluation_times: list[float] = [0.5, 2.1, 3.5, 4.8]
    print(f"\nTotal trajectory duration: {duration:.2f} seconds")
    print("\nEvaluating trajectory at specific time points:")

    for time_point in evaluation_times:
        position, velocity, acceleration = traj_func(time_point)
        print(
            f"At t={time_point:.2f}s: position={position:.4f}, velocity={velocity:.4f}, acceleration={acceleration:.4f}"
        )

    # Demonstrate out-of-bounds handling
    print("\nDemonstrating out-of-bounds handling:")
    out_of_bounds_time: float = duration + 1.0
    position, velocity, acceleration = traj_func(out_of_bounds_time)
    print(
        f"At t={out_of_bounds_time:.2f}s (beyond duration): position={position:.4f}, velocity={velocity:.4f}, acceleration={acceleration:.4f}"  # noqa: E501
    )
