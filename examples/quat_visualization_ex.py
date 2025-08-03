"""
SLERP vs SQUAD Quaternion Interpolation Comparison

This example demonstrates the differences between SLERP (Spherical Linear Interpolation)
and SQUAD (Spherical Cubic Interpolation) methods for quaternion trajectory planning.

Key differences:
- SLERP: Linear interpolation between quaternions (C0 continuous)
- SQUAD: Cubic interpolation with smoother transitions (C1 continuous)
"""

import matplotlib.pyplot as plt
import numpy as np
from interpolatepy.quat_core import Quaternion
from interpolatepy.quat_spline import QuaternionSpline
from interpolatepy.quat_visualization import QuaternionTrajectoryVisualizer


def create_waypoint_trajectory() -> tuple[list[float], list[Quaternion]]:
    """Create a quaternion trajectory with several waypoints for comparison."""
    # Define time points with more waypoints for smoother trajectory
    time_points = [0.0, 0.8, 2.4, 4.0, 5.6, 6.4, 8.0]

    # Create quaternions with gradual, smooth rotation sequences
    quaternions = [
        Quaternion.identity(),  # Start at identity
        Quaternion.from_euler_angles(0.15, 0.05, 0.0),
        Quaternion.from_euler_angles(0.3, 0.25, 0.2),
        Quaternion.from_euler_angles(0.2, 0.6, 0.5),
        Quaternion.from_euler_angles(0.05, 0.5, 0.85),
        Quaternion.from_euler_angles(0.0, 0.25, 1.0),
        Quaternion.from_euler_angles(0.0, 0.0, 1.05),
    ]

    return time_points, quaternions


def generate_interpolated_trajectories(
    time_points: list[float], quaternions: list[Quaternion], num_samples: int = 100
) -> tuple[np.ndarray, list[Quaternion], list[Quaternion]]:
    """Generate interpolated trajectories using both SLERP and SQUAD methods."""
    # Create spline objects
    slerp_spline = QuaternionSpline(time_points, quaternions, Quaternion.SLERP)
    squad_spline = QuaternionSpline(time_points, quaternions, Quaternion.SQUAD)

    # Generate evaluation times
    t_min, t_max = time_points[0], time_points[-1]
    eval_times = np.linspace(t_min, t_max, num_samples)

    # Interpolate using both methods
    slerp_trajectory = []
    squad_trajectory = []

    for t in eval_times:
        # SLERP interpolation
        q_slerp, _ = slerp_spline.interpolate_at_time(t)
        slerp_trajectory.append(q_slerp)

        # SQUAD interpolation
        q_squad, _ = squad_spline.interpolate_at_time(t)
        squad_trajectory.append(q_squad)

    return eval_times, slerp_trajectory, squad_trajectory


def plot_quaternion_components_comparison(
    eval_times: np.ndarray,
    slerp_traj: list[Quaternion],
    squad_traj: list[Quaternion],
    waypoint_times: list[float],
    waypoints: list[Quaternion],
) -> plt.Figure:
    """Create comparison plots of quaternion components over time."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("SLERP vs SQUAD: Quaternion Components Comparison", fontsize=16)

    # Extract components
    components = ["w", "x", "y", "z"]
    colors_slerp = ["red", "green", "blue", "orange"]
    colors_squad = ["darkred", "darkgreen", "darkblue", "darkorange"]

    for i, comp in enumerate(components):
        ax = axes[i // 2, i % 2]

        # Extract component values
        slerp_vals = [getattr(q, comp) for q in slerp_traj]
        squad_vals = [getattr(q, comp) for q in squad_traj]
        waypoint_vals = [getattr(q, comp) for q in waypoints]

        # Plot trajectories
        ax.plot(
            eval_times, slerp_vals, colors_slerp[i], linewidth=2, label=f"SLERP {comp}", alpha=0.8
        )
        ax.plot(
            eval_times,
            squad_vals,
            colors_squad[i],
            linewidth=2,
            label=f"SQUAD {comp}",
            linestyle="--",
            alpha=0.8,
        )

        # Plot waypoints
        ax.scatter(waypoint_times, waypoint_vals, color="black", s=60, zorder=5, label="Waypoints")

        ax.set_xlabel("Time")
        ax.set_ylabel(f"Quaternion {comp.upper()}")
        ax.set_title(f"Component {comp.upper()}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_3d_trajectory_comparison(
    slerp_traj: list[Quaternion], squad_traj: list[Quaternion]
) -> plt.Figure:
    """Create 3D visualization using stereographic projection."""
    visualizer = QuaternionTrajectoryVisualizer()

    fig = plt.figure(figsize=(15, 6))

    # SLERP trajectory
    ax1 = fig.add_subplot(121, projection="3d")
    slerp_points = visualizer.project_trajectory(slerp_traj)
    if len(slerp_points) > 0:
        colors = plt.colormaps["viridis"](np.linspace(0, 1, len(slerp_points)))
        ax1.plot(
            slerp_points[:, 0],
            slerp_points[:, 1],
            slerp_points[:, 2],
            "b-",
            linewidth=2,
            alpha=0.7,
            label="SLERP",
        )
        ax1.scatter(
            slerp_points[:, 0], slerp_points[:, 1], slerp_points[:, 2], c=colors, s=20, alpha=0.8
        )
    ax1.set_xlabel("MRP X")
    ax1.set_ylabel("MRP Y")
    ax1.set_zlabel("MRP Z")
    ax1.set_title("SLERP Trajectory\n(Stereographic Projection)")

    # SQUAD trajectory
    ax2 = fig.add_subplot(122, projection="3d")
    squad_points = visualizer.project_trajectory(squad_traj)
    if len(squad_points) > 0:
        colors = plt.colormaps["viridis"](np.linspace(0, 1, len(squad_points)))
        ax2.plot(
            squad_points[:, 0],
            squad_points[:, 1],
            squad_points[:, 2],
            "r--",
            linewidth=2,
            alpha=0.7,
            label="SQUAD",
        )
        ax2.scatter(
            squad_points[:, 0], squad_points[:, 1], squad_points[:, 2], c=colors, s=20, alpha=0.8
        )
    ax2.set_xlabel("MRP X")
    ax2.set_ylabel("MRP Y")
    ax2.set_zlabel("MRP Z")
    ax2.set_title("SQUAD Trajectory\n(Stereographic Projection)")

    plt.tight_layout()
    return fig


def demo_simple_3d_plot() -> None:
    """Demonstrate simple 3D trajectory plotting."""
    print("\n--- Simple 3D Trajectory Plot ---")

    # Create a simple trajectory
    _time_points, quaternions = create_waypoint_trajectory()
    visualizer = QuaternionTrajectoryVisualizer()

    # Plot simple 3D trajectory
    visualizer.plot_3d_trajectory(
        quaternions,
        title="Simple 3D Quaternion Trajectory",
        color="purple",
        line_width=2.5,
        point_size=30
    )

    print(f"Created simple 3D plot with {len(quaternions)} quaternions")
    print("Shows stereographic projection of quaternion trajectory in 3D space")


def demo_velocity_plot() -> None:
    """Demonstrate velocity magnitude plotting using custom formula."""
    print("\n--- Velocity Magnitude Plot ---")

    time_points, quaternions = create_waypoint_trajectory()
    visualizer = QuaternionTrajectoryVisualizer()

    # Plot velocity using custom formula: V(qi) = [||qi - qi-1|| + ||qi - qi+1||] / 2
    visualizer.plot_angular_velocity(
        quaternions,
        time_points=time_points,
        title="Quaternion Velocity Analysis",
        color="orange",
        line_width=2.5
    )

    print(f"Created velocity plot for {len(quaternions)} quaternions")
    print("Uses formula: V(qi) = [||qi - qi-1|| + ||qi - qi+1||] / 2")

    # Also demonstrate without time points (using indices)
    visualizer.plot_angular_velocity(
        quaternions[:5],  # Use fewer points for cleaner index-based plot
        title="Velocity by Index (First 5 Points)",
        color="green"
    )
    print("Also created index-based velocity plot")


def demo_combined_plot() -> None:
    """Demonstrate combined 3D trajectory and velocity plot."""
    print("\n--- Combined Trajectory + Velocity Plot ---")

    time_points, quaternions = create_waypoint_trajectory()
    visualizer = QuaternionTrajectoryVisualizer()

    # Create combined plot
    visualizer.plot_trajectory_with_velocity(
        quaternions,
        time_points=time_points,
        title="Complete Quaternion Analysis: 3D Trajectory + Velocity",
        trajectory_color="navy",
        velocity_color="crimson"
    )

    print("Created combined plot showing both 3D trajectory and velocity analysis")
    print("Left: 3D stereographic projection | Right: Velocity magnitude over time")


def main() -> None:
    """Main comparison function."""
    print("SLERP vs SQUAD Quaternion Interpolation Comparison")
    print("=" * 55)

    time_points, quaternions = create_waypoint_trajectory()

    print(f"Created trajectory with {len(quaternions)} waypoints")
    print(f"Time range: {time_points[0]:.1f} to {time_points[-1]:.1f} seconds")

    # Generate interpolated trajectories with high resolution
    eval_times, slerp_traj, squad_traj = generate_interpolated_trajectories(
        time_points, quaternions, num_samples=800
    )
    print(f"Generated {len(eval_times)} interpolated points for each method")

    # Create visualizations
    print("\nGenerating comparison plots...")

    # Quaternion components comparison
    plot_quaternion_components_comparison(
        eval_times, slerp_traj, squad_traj, time_points, quaternions
    )

    # 3D trajectory comparison
    plot_3d_trajectory_comparison(slerp_traj, squad_traj)

    print("\nDisplaying SLERP vs SQUAD comparison plots...")
    plt.show()

    # Demonstrate new simple plotting methods
    print("\n" + "=" * 60)
    print("NEW SIMPLE PLOTTING METHODS DEMONSTRATION")
    print("=" * 60)

    # Simple 3D trajectory plot
    demo_simple_3d_plot()

    # Velocity magnitude plot with custom formula
    demo_velocity_plot()

    # Combined trajectory + velocity plot
    demo_combined_plot()

    print("\nDisplaying new simple plotting examples...")
    plt.show()


if __name__ == "__main__":
    main()
