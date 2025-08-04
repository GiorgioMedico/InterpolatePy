"""
Logarithmic Quaternion B-spline Interpolation Examples

This example demonstrates the LogQuaternionBSpline class for smooth quaternion
trajectory generation using logarithmic quaternion representation with B-spline
interpolation, combined with visualization using QuaternionTrajectoryVisualizer.

Key Features Demonstrated:
- Logarithmic quaternion space interpolation for smooth trajectories
- Different B-spline degrees (cubic, quartic, quintic)
- Boundary condition constraints (velocity, acceleration)
- 3D trajectory visualization using stereographic projection
- Angular velocity and acceleration analysis
- Comparison with traditional SLERP/SQUAD methods

Mathematical Background:
- Works in 3D logarithmic quaternion space (vector part of log(q))
- Uses B-spline interpolation for smooth, continuously differentiable trajectories
- Transforms back to unit quaternions using exponential map
- Provides C² continuity (or higher depending on degree)
"""

import matplotlib.pyplot as plt
import numpy as np

from interpolatepy.log_quat import LogQuaternionBSpline
from interpolatepy.quat_core import Quaternion
from interpolatepy.quat_spline import QuaternionSpline
from interpolatepy.quat_visualization import QuaternionTrajectoryVisualizer


def create_basic_trajectory() -> tuple[list[float], list[Quaternion]]:
    """Create a basic quaternion trajectory for demonstration."""
    time_points = [0.0, 1.0, 2.5, 4.0, 5.5]

    quaternions = [
        Quaternion.identity(),
        Quaternion.from_euler_angles(0.2, 0.1, 0.0),
        Quaternion.from_euler_angles(0.5, 0.4, 0.3),
        Quaternion.from_euler_angles(0.3, 0.8, 0.6),
        Quaternion.from_euler_angles(0.0, 0.5, 1.0),
    ]

    return time_points, quaternions


def create_complex_trajectory() -> tuple[list[float], list[Quaternion]]:
    """Create a more complex quaternion trajectory for advanced examples."""
    time_points = [0.0, 0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 6.0]

    quaternions = [
        Quaternion.identity(),
        Quaternion.from_euler_angles(0.15, 0.05, 0.0),
        Quaternion.from_euler_angles(0.3, 0.25, 0.2),
        Quaternion.from_euler_angles(0.45, 0.6, 0.4),
        Quaternion.from_euler_angles(0.2, 0.8, 0.7),
        Quaternion.from_euler_angles(0.05, 0.6, 0.9),
        Quaternion.from_euler_angles(0.0, 0.3, 1.1),
        Quaternion.from_euler_angles(0.0, 0.0, 1.2),
    ]

    return time_points, quaternions


def demo_basic_interpolation() -> None:
    """Demonstrate basic logarithmic quaternion B-spline interpolation."""
    print("\n--- Basic Logarithmic Quaternion B-spline Interpolation ---")

    time_points, quaternions = create_basic_trajectory()

    # Create cubic B-spline interpolator
    log_spline = LogQuaternionBSpline(time_points, quaternions, degree=3)

    print(f"Created LogQuaternionBSpline with {len(quaternions)} waypoints")
    print(f"Time range: {time_points[0]:.1f} to {time_points[-1]:.1f} seconds")
    print(f"B-spline degree: {log_spline.degree}")

    # Generate trajectory
    _eval_times, trajectory = log_spline.generate_trajectory(num_points=100)

    # Visualize
    visualizer = QuaternionTrajectoryVisualizer()

    visualizer.plot_3d_trajectory(
        trajectory,
        title="Basic Logarithmic Quaternion B-spline Trajectory",
        color="blue",
        line_width=2.5,
        point_size=25
    )

    print("Created 3D visualization of basic log quaternion B-spline trajectory")


def demo_comparison_with_traditional_methods() -> None:
    """Compare logarithmic B-spline with traditional SLERP/SQUAD methods."""
    print("\n--- Comparison with Traditional Methods ---")

    time_points, quaternions = create_basic_trajectory()

    # Logarithmic B-spline
    log_spline = LogQuaternionBSpline(time_points, quaternions, degree=3)

    # Traditional methods
    slerp_spline = QuaternionSpline(time_points, quaternions, Quaternion.SLERP)
    squad_spline = QuaternionSpline(time_points, quaternions, Quaternion.SQUAD)

    # Generate trajectories
    eval_times = np.linspace(time_points[0], time_points[-1], 100)

    log_trajectory = [log_spline.evaluate(t) for t in eval_times]
    slerp_trajectory = [slerp_spline.interpolate_at_time(t)[0] for t in eval_times]
    squad_trajectory = [squad_spline.interpolate_at_time(t)[0] for t in eval_times]

    # Create comparison visualization
    fig = plt.figure(figsize=(18, 12))

    visualizer = QuaternionTrajectoryVisualizer()
    methods = ["Logarithmic B-spline", "SLERP", "SQUAD"]
    trajectories = [log_trajectory, slerp_trajectory, squad_trajectory]
    colors = ["blue", "red", "green"]

    # 3D trajectory comparison
    for i, (method, trajectory, color) in enumerate(zip(methods, trajectories, colors)):
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")

        projected_points = visualizer.project_trajectory(trajectory)
        if len(projected_points) > 0:
            ax.plot(projected_points[:, 0], projected_points[:, 1], projected_points[:, 2],
                   color=color, linewidth=2.5, alpha=0.8, label=method)

            # Color points by progression
            point_colors = plt.colormaps["viridis"](np.linspace(0, 1, len(projected_points)))
            ax.scatter(projected_points[:, 0], projected_points[:, 1], projected_points[:, 2],
                      c=point_colors, s=15, alpha=0.6)

            # Mark waypoints
            ax.scatter(*projected_points[0], color="green", s=50, marker="o")
            ax.scatter(*projected_points[-1], color="red", s=50, marker="s")

        ax.set_xlabel("MRP X")
        ax.set_ylabel("MRP Y")
        ax.set_zlabel("MRP Z")
        ax.set_title(f"{method}\n3D Trajectory")
        ax.grid(True, alpha=0.3)

    # Velocity magnitude comparison
    for i, (method, trajectory, color) in enumerate(zip(methods, trajectories, colors)):
        ax = fig.add_subplot(2, 3, i + 4)

        times, velocities = visualizer.compute_velocity_magnitudes(trajectory, eval_times)

        ax.plot(times, velocities, color=color, linewidth=2.5, alpha=0.8, label=method)
        ax.fill_between(times, velocities, alpha=0.3, color=color)

        # Mark peaks
        min_points_for_peak = 2
        if len(velocities) > min_points_for_peak:
            max_idx = np.argmax(velocities)
            ax.scatter(times[max_idx], velocities[max_idx],
                      color=color, s=50, zorder=5, marker="^",
                      label=f"Max: {velocities[max_idx]:.3f}")

        ax.set_xlabel("Time")
        ax.set_ylabel("Velocity Magnitude")
        ax.set_title(f"{method}\nVelocity Profile")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Logarithmic B-spline vs Traditional Methods Comparison", fontsize=16)
    plt.tight_layout()

    print("Created comprehensive comparison between log B-spline and traditional methods")
    print("Log B-spline provides smooth C² continuous trajectories in 3D log space")


def main() -> None:
    """Main demonstration function."""
    print("Logarithmic Quaternion B-spline Interpolation Examples")
    print("=" * 60)
    print()
    print("This example demonstrates the LogQuaternionBSpline class for smooth")
    print("quaternion trajectory generation using logarithmic quaternion representation")
    print("with B-spline interpolation and visualization capabilities.")
    print()

    # Basic interpolation
    demo_basic_interpolation()

    # Comparison with traditional methods
    demo_comparison_with_traditional_methods()

    print("\n" + "=" * 60)
    print("All demonstrations completed successfully!")
    print("Key advantages of Logarithmic Quaternion B-splines:")
    print("  • Works in 3D logarithmic space (simpler than 4D quaternion space)")
    print("  • Provides smooth C² continuous trajectories (or higher)")
    print("  • Supports flexible boundary conditions (velocity, acceleration)")
    print("  • Avoids quaternion interpolation artifacts")
    print("  • Ideal for robotics and animation applications")
    print()
    print("Displaying all visualization plots...")
    plt.show()


if __name__ == "__main__":
    main()
