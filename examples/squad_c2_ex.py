"""
Simple SQUAD vs SQUAD C2 Quaternion Interpolation Example

This example demonstrates the basic differences between traditional SQUAD
and SQUAD C2 interpolation methods with a simple trajectory.

Key Points:
- Traditional SQUAD: Classic spherical quadrangle interpolation
- SQUAD C2: Advanced method with C² continuity and zero-clamped boundaries
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np

from interpolatepy.quat_core import Quaternion
from interpolatepy.quat_spline import QuaternionSpline
from interpolatepy.squad_c2 import SquadC2
from interpolatepy.quat_visualization import QuaternionTrajectoryVisualizer

if TYPE_CHECKING:
    from numpy.typing import NDArray


def create_simple_trajectory() -> tuple[list[float], list[Quaternion]]:
    """Create a simple quaternion trajectory for demonstration."""
    time_points = [0.0, 1.0, 2.0, 3.0, 4.0]

    quaternions = [
        Quaternion.identity(),
        Quaternion.from_euler_angles(0.3, 0.2, 0.0),
        Quaternion.from_euler_angles(0.6, 0.5, 0.4),
        Quaternion.from_euler_angles(0.2, 0.8, 0.7),
        Quaternion.from_euler_angles(0.0, 0.4, 1.0),
    ]

    return time_points, quaternions


def plot_individual_methods(  # noqa: PLR0913
    squad_trajectory: list[Quaternion],
    squad_c2_trajectory: list[Quaternion],
    slerp_trajectory: list[Quaternion],
    eval_times: NDArray[np.floating],
    visualizer: QuaternionTrajectoryVisualizer,
    waypoints: list[Quaternion],
    waypoint_times: list[float]
) -> None:
    """Plot individual method analysis with separate plots for each interpolation method.

    Args:
        squad_trajectory: SQUAD interpolated quaternions
        squad_c2_trajectory: SQUAD C2 interpolated quaternions
        slerp_trajectory: SLERP interpolated quaternions
        eval_times: Time points for evaluation
        visualizer: Quaternion trajectory visualizer
        waypoints: Original waypoint quaternions
        waypoint_times: Time points for waypoints
    """
    # Project trajectories to 3D space
    projected_squad = visualizer.project_trajectory(squad_trajectory)
    projected_squad_c2 = visualizer.project_trajectory(squad_c2_trajectory)
    projected_slerp = visualizer.project_trajectory(slerp_trajectory)
    projected_waypoints = visualizer.project_trajectory(waypoints)

    # Compute velocity magnitudes
    eval_times_list = eval_times.tolist()
    squad_times, squad_velocities = visualizer.compute_velocity_magnitudes(
        squad_trajectory, eval_times_list
    )
    squad_c2_times, squad_c2_velocities = visualizer.compute_velocity_magnitudes(
        squad_c2_trajectory, eval_times_list
    )
    slerp_times, slerp_velocities = visualizer.compute_velocity_magnitudes(
        slerp_trajectory, eval_times_list
    )

    # Create figure for individual method plots
    fig = plt.figure(figsize=(15, 10))

    # SQUAD plot with waypoints
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    if len(projected_squad) > 0:
        ax1.plot(projected_squad[:, 0], projected_squad[:, 1], projected_squad[:, 2],
                color="red", linewidth=3, alpha=0.8, label="SQUAD")
    if len(projected_waypoints) > 0:
        ax1.scatter(projected_waypoints[:, 0], projected_waypoints[:, 1], projected_waypoints[:, 2],
                   color="black", s=60, marker="D", alpha=0.9, edgecolors="white",
                   linewidth=1, label="Waypoints", zorder=10)
    ax1.set_title("SQUAD with Waypoints")
    ax1.set_xlabel("MRP X")
    ax1.set_ylabel("MRP Y")
    ax1.set_zlabel("MRP Z")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # SQUAD C2 plot with waypoints
    ax2 = fig.add_subplot(2, 3, 2, projection="3d")
    if len(projected_squad_c2) > 0:
        ax2.plot(projected_squad_c2[:, 0], projected_squad_c2[:, 1], projected_squad_c2[:, 2],
                color="blue", linewidth=3, alpha=0.8, label="SQUAD C2")
    if len(projected_waypoints) > 0:
        ax2.scatter(projected_waypoints[:, 0], projected_waypoints[:, 1], projected_waypoints[:, 2],
                   color="black", s=60, marker="D", alpha=0.9, edgecolors="white",
                   linewidth=1, label="Waypoints", zorder=10)
    ax2.set_title("SQUAD C2 with Waypoints")
    ax2.set_xlabel("MRP X")
    ax2.set_ylabel("MRP Y")
    ax2.set_zlabel("MRP Z")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # SLERP plot with waypoints
    ax3 = fig.add_subplot(2, 3, 3, projection="3d")
    if len(projected_slerp) > 0:
        ax3.plot(projected_slerp[:, 0], projected_slerp[:, 1], projected_slerp[:, 2],
                color="green", linewidth=3, alpha=0.8, label="SLERP")
    if len(projected_waypoints) > 0:
        ax3.scatter(projected_waypoints[:, 0], projected_waypoints[:, 1], projected_waypoints[:, 2],
                   color="black", s=60, marker="D", alpha=0.9, edgecolors="white",
                   linewidth=1, label="Waypoints", zorder=10)
    ax3.set_title("SLERP with Waypoints")
    ax3.set_xlabel("MRP X")
    ax3.set_ylabel("MRP Y")
    ax3.set_zlabel("MRP Z")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Individual velocity plots
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(squad_times, squad_velocities, color="red", linewidth=2.5, alpha=0.8)
    # Mark waypoint times
    for t in waypoint_times:
        ax4.axvline(x=t, color="black", linestyle="--", alpha=0.6)
    ax4.set_title("SQUAD Velocity")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Angular Velocity (rad/s)")
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(squad_c2_times, squad_c2_velocities, color="blue", linewidth=2.5, alpha=0.8)
    # Mark waypoint times
    for t in waypoint_times:
        ax5.axvline(x=t, color="black", linestyle="--", alpha=0.6)
    ax5.set_title("SQUAD C2 Velocity")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Angular Velocity (rad/s)")
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(slerp_times, slerp_velocities, color="green", linewidth=2.5, alpha=0.8)
    # Mark waypoint times
    for t in waypoint_times:
        ax6.axvline(x=t, color="black", linestyle="--", alpha=0.6)
    ax6.set_title("SLERP Velocity")
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("Angular Velocity (rad/s)")
    ax6.grid(True, alpha=0.3)

    fig.suptitle("Individual Method Analysis with Waypoints", fontsize=16)
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Main demonstration function."""
    print("Individual Method Analysis")
    print("=========================")

    # Create trajectory
    time_points, quaternions = create_simple_trajectory()
    print(f"Created trajectory with {len(quaternions)} waypoints")

    # Create interpolators
    squad_spline = QuaternionSpline(time_points, quaternions, Quaternion.SQUAD)
    squad_c2 = SquadC2(time_points, quaternions)

    # Generate trajectories
    eval_times = np.linspace(time_points[0], time_points[-1], 100)

    # Traditional SQUAD trajectory
    squad_trajectory = []
    for t in eval_times:
        q, status = squad_spline.interpolate_squad(t)
        if status == 0:
            squad_trajectory.append(q)
        else:
            # Fallback to SLERP for boundary segments
            q_slerp, _ = squad_spline.interpolate_slerp(t)
            squad_trajectory.append(q_slerp)

    # SQUAD C2 trajectory
    squad_c2_trajectory = [squad_c2.evaluate(t) for t in eval_times]

    # SLERP trajectory for comparison
    slerp_trajectory = []
    for t in eval_times:
        q_slerp, _ = squad_spline.interpolate_slerp(t)
        slerp_trajectory.append(q_slerp)

    # Create visualization
    visualizer = QuaternionTrajectoryVisualizer()

    # Plot individual methods with waypoints
    plot_individual_methods(
        squad_trajectory,
        squad_c2_trajectory,
        slerp_trajectory,
        eval_times,
        visualizer,
        waypoints=quaternions,
        waypoint_times=time_points
    )

    # Print summary
    print("\nKey Differences:")
    print("• SQUAD C2 has zero-clamped boundaries (zero velocity at start/end)")
    print("• SQUAD C2 provides C² continuity (smooth acceleration)")
    print("• Traditional SQUAD uses cubic interpolation, SQUAD C2 uses quintic")


if __name__ == "__main__":
    main()
