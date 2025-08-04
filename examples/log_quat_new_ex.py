"""
Logarithmic Quaternion Interpolation (LQI) Examples

This example demonstrates the LogQuaternionInterpolation class (LQI method)
for smooth quaternion trajectory generation using logarithmic quaternion
representation with B-spline interpolation.

Note: This example has been updated to use LogQuaternionInterpolation instead
of the deprecated LogQuaternionBSpline class.

Key Features Demonstrated:
- Logarithmic quaternion space interpolation for smooth trajectories
- Algorithm 1 from Parker et al. (2023) for continuous axis-angle recovery
- Different B-spline degrees (cubic, quartic, quintic)
- Boundary condition constraints (velocity, acceleration)
- 3D trajectory visualization using stereographic projection
- Angular velocity and acceleration analysis
- Comparison with traditional SLERP/SQUAD methods

Mathematical Background:
- Uses axis-angle representation r = θ*n̂ for interpolation
- Handles quaternion double-cover and axis-angle discontinuities
- Uses B-spline interpolation for smooth, continuously differentiable trajectories
- Provides C² continuity (or higher depending on degree)
"""

import matplotlib.pyplot as plt
import numpy as np

# Using LogQuaternionInterpolation instead of deprecated LogQuaternionBSpline
from interpolatepy.log_quat import LogQuaternionInterpolation, ModifiedLogQuaternionInterpolation
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
    """Demonstrate basic logarithmic quaternion interpolation using LQI method."""
    print("\n--- Basic Logarithmic Quaternion Interpolation (LQI) ---")

    time_points, quaternions = create_basic_trajectory()

    # Create cubic B-spline interpolator using LogQuaternionInterpolation (LQI method)
    log_spline = LogQuaternionInterpolation(time_points, quaternions, degree=3)

    print(f"Created LogQuaternionInterpolation with {len(quaternions)} waypoints")
    print(f"Time range: {time_points[0]:.1f} to {time_points[-1]:.1f} seconds")
    print(f"B-spline degree: {log_spline.degree}")

    # Generate trajectory
    _eval_times, trajectory = log_spline.generate_trajectory(num_points=100)

    # Visualize
    visualizer = QuaternionTrajectoryVisualizer()

    visualizer.plot_3d_trajectory(
        trajectory,
        title="Basic Logarithmic Quaternion Interpolation (LQI) Trajectory",
        color="blue",
        line_width=2.5,
        point_size=25,
    )

    print("Created 3D visualization of basic log quaternion interpolation trajectory")


def demo_comparison_with_traditional_methods() -> None:
    """Compare logarithmic quaternion interpolation methods (LQI, mLQI) with traditional SLERP/SQUAD methods."""
    print("\n--- Comparison with Traditional and Logarithmic Methods ---")

    time_points, quaternions = create_basic_trajectory()

    # Logarithmic B-spline methods
    log_spline = LogQuaternionInterpolation(time_points, quaternions, degree=3)
    mod_log_spline = ModifiedLogQuaternionInterpolation(time_points, quaternions, degree=3)

    # Traditional methods
    slerp_spline = QuaternionSpline(time_points, quaternions, Quaternion.SLERP)
    squad_spline = QuaternionSpline(time_points, quaternions, Quaternion.SQUAD)

    # Generate trajectories
    eval_times = np.linspace(time_points[0], time_points[-1], 100)

    log_trajectory = [log_spline.evaluate(t) for t in eval_times]
    mod_log_trajectory = [mod_log_spline.evaluate(t) for t in eval_times]
    slerp_trajectory = [slerp_spline.interpolate_at_time(t)[0] for t in eval_times]
    squad_trajectory = [squad_spline.interpolate_at_time(t)[0] for t in eval_times]

    # Create comparison visualization
    fig = plt.figure(figsize=(24, 12))

    visualizer = QuaternionTrajectoryVisualizer()
    methods = ["Logarithmic Interpolation (LQI)", "Modified LQI (mLQI)", "SLERP", "SQUAD"]
    trajectories = [log_trajectory, mod_log_trajectory, slerp_trajectory, squad_trajectory]
    colors = ["blue", "purple", "red", "green"]

    # 3D trajectory comparison
    for i, (method, trajectory, color) in enumerate(zip(methods, trajectories, colors)):
        ax = fig.add_subplot(2, 4, i + 1, projection="3d")

        projected_points = visualizer.project_trajectory(trajectory)
        if len(projected_points) > 0:
            ax.plot(
                projected_points[:, 0],
                projected_points[:, 1],
                projected_points[:, 2],
                color=color,
                linewidth=2.5,
                alpha=0.8,
                label=method,
            )

            # Color points by progression
            point_colors = plt.colormaps["viridis"](np.linspace(0, 1, len(projected_points)))
            ax.scatter(
                projected_points[:, 0],
                projected_points[:, 1],
                projected_points[:, 2],
                c=point_colors,
                s=15,
                alpha=0.6,
            )

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
        ax = fig.add_subplot(2, 4, i + 5)

        times, velocities = visualizer.compute_velocity_magnitudes(trajectory, eval_times)

        ax.plot(times, velocities, color=color, linewidth=2.5, alpha=0.8, label=method)
        ax.fill_between(times, velocities, alpha=0.3, color=color)

        # Mark peaks
        min_points_for_peak = 2
        if len(velocities) > min_points_for_peak:
            max_idx = np.argmax(velocities)
            ax.scatter(
                times[max_idx],
                velocities[max_idx],
                color=color,
                s=50,
                zorder=5,
                marker="^",
                label=f"Max: {velocities[max_idx]:.3f}",
            )

        ax.set_xlabel("Time")
        ax.set_ylabel("Velocity Magnitude")
        ax.set_title(f"{method}\nVelocity Profile")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Logarithmic Quaternion Interpolation Methods (LQI, mLQI) vs Traditional Methods Comparison", fontsize=16
    )
    plt.tight_layout()

    print("Created comprehensive comparison between LQI, mLQI and traditional methods")
    print("LQI and mLQI provide smooth C² continuous trajectories using axis-angle representation")
    print("mLQI decouples angle and axis components for improved numerical stability")


def demo_lqi_vs_mlqi_comparison() -> None:
    """Detailed comparison between LQI and mLQI methods."""
    print("\n--- Detailed LQI vs Modified LQI (mLQI) Comparison ---")

    time_points, quaternions = create_complex_trajectory()

    # Create both logarithmic interpolators
    lqi_spline = LogQuaternionInterpolation(time_points, quaternions, degree=3)
    mlqi_spline = ModifiedLogQuaternionInterpolation(time_points, quaternions, degree=3)

    print(f"Comparing LQI vs mLQI with {len(quaternions)} waypoints")
    print("LQI: Standard axis-angle representation (θ*n̂)")
    print("mLQI: Decoupled representation (θ, X, Y, Z) with X²+Y²+Z²=1")

    # Generate trajectories
    eval_times = np.linspace(time_points[0], time_points[-1], 150)
    lqi_trajectory = [lqi_spline.evaluate(t) for t in eval_times]
    mlqi_trajectory = [mlqi_spline.evaluate(t) for t in eval_times]

    # Create detailed comparison visualization
    fig = plt.figure(figsize=(20, 15))
    visualizer = QuaternionTrajectoryVisualizer()

    # 3D trajectory comparison
    ax1 = fig.add_subplot(3, 2, 1, projection="3d")
    projected_lqi = visualizer.project_trajectory(lqi_trajectory)
    if len(projected_lqi) > 0:
        ax1.plot(projected_lqi[:, 0], projected_lqi[:, 1], projected_lqi[:, 2], 
                color="blue", linewidth=3, alpha=0.8, label="LQI")
        # Color points by progression
        point_colors = plt.colormaps["viridis"](np.linspace(0, 1, len(projected_lqi)))
        ax1.scatter(projected_lqi[:, 0], projected_lqi[:, 1], projected_lqi[:, 2], 
                   c=point_colors, s=20, alpha=0.7)
    ax1.set_title("LQI Method\n3D Trajectory", fontsize=14)
    ax1.set_xlabel("MRP X")
    ax1.set_ylabel("MRP Y")
    ax1.set_zlabel("MRP Z")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(3, 2, 2, projection="3d")
    projected_mlqi = visualizer.project_trajectory(mlqi_trajectory)
    if len(projected_mlqi) > 0:
        ax2.plot(projected_mlqi[:, 0], projected_mlqi[:, 1], projected_mlqi[:, 2], 
                color="purple", linewidth=3, alpha=0.8, label="mLQI")
        # Color points by progression
        point_colors = plt.colormaps["plasma"](np.linspace(0, 1, len(projected_mlqi)))
        ax2.scatter(projected_mlqi[:, 0], projected_mlqi[:, 1], projected_mlqi[:, 2], 
                   c=point_colors, s=20, alpha=0.7)
    ax2.set_title("Modified LQI Method\n3D Trajectory", fontsize=14)
    ax2.set_xlabel("MRP X")
    ax2.set_ylabel("MRP Y")
    ax2.set_zlabel("MRP Z")
    ax2.grid(True, alpha=0.3)

    # Velocity magnitude comparison
    ax3 = fig.add_subplot(3, 2, 3)
    lqi_times, lqi_velocities = visualizer.compute_velocity_magnitudes(lqi_trajectory, eval_times)
    mlqi_times, mlqi_velocities = visualizer.compute_velocity_magnitudes(mlqi_trajectory, eval_times)

    ax3.plot(lqi_times, lqi_velocities, color="blue", linewidth=2.5, alpha=0.8, label="LQI")
    ax3.plot(mlqi_times, mlqi_velocities, color="purple", linewidth=2.5, alpha=0.8, label="mLQI")
    ax3.fill_between(lqi_times, lqi_velocities, alpha=0.2, color="blue")
    ax3.fill_between(mlqi_times, mlqi_velocities, alpha=0.2, color="purple")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Velocity Magnitude")
    ax3.set_title("Angular Velocity Comparison")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Acceleration magnitude comparison
    ax4 = fig.add_subplot(3, 2, 4)
    # Compute acceleration magnitudes (simplified numerical differentiation)
    lqi_accel = np.gradient(lqi_velocities, lqi_times)
    mlqi_accel = np.gradient(mlqi_velocities, mlqi_times)

    ax4.plot(lqi_times, np.abs(lqi_accel), color="blue", linewidth=2.5, alpha=0.8, label="LQI")
    ax4.plot(mlqi_times, np.abs(mlqi_accel), color="purple", linewidth=2.5, alpha=0.8, label="mLQI")
    ax4.fill_between(lqi_times, np.abs(lqi_accel), alpha=0.2, color="blue")
    ax4.fill_between(mlqi_times, np.abs(mlqi_accel), alpha=0.2, color="purple")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Acceleration Magnitude")
    ax4.set_title("Angular Acceleration Comparison")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Combined trajectory overlay
    ax5 = fig.add_subplot(3, 2, 5, projection="3d")
    if len(projected_lqi) > 0 and len(projected_mlqi) > 0:
        ax5.plot(projected_lqi[:, 0], projected_lqi[:, 1], projected_lqi[:, 2], 
                color="blue", linewidth=2.5, alpha=0.7, label="LQI")
        ax5.plot(projected_mlqi[:, 0], projected_mlqi[:, 1], projected_mlqi[:, 2], 
                color="purple", linewidth=2.5, alpha=0.7, label="mLQI")
        ax5.legend()
    ax5.set_title("Overlay Comparison\nLQI vs mLQI", fontsize=14)
    ax5.set_xlabel("MRP X")
    ax5.set_ylabel("MRP Y")
    ax5.set_zlabel("MRP Z")
    ax5.grid(True, alpha=0.3)

    # Performance metrics comparison
    ax6 = fig.add_subplot(3, 2, 6)
    
    # Compute smoothness metrics (velocity variance as proxy)
    lqi_smoothness = np.var(lqi_velocities)
    mlqi_smoothness = np.var(mlqi_velocities)
    
    # Compute maximum velocities
    lqi_max_vel = np.max(lqi_velocities)
    mlqi_max_vel = np.max(mlqi_velocities)
    
    # Compute trajectory length (approximation)
    lqi_length = np.sum(np.linalg.norm(np.diff(projected_lqi, axis=0), axis=1)) if len(projected_lqi) > 1 else 0
    mlqi_length = np.sum(np.linalg.norm(np.diff(projected_mlqi, axis=0), axis=1)) if len(projected_mlqi) > 1 else 0

    metrics = ['Velocity\nVariance', 'Max\nVelocity', 'Trajectory\nLength']
    lqi_values = [lqi_smoothness, lqi_max_vel, lqi_length]
    mlqi_values = [mlqi_smoothness, mlqi_max_vel, mlqi_length]

    x = np.arange(len(metrics))
    width = 0.35

    ax6.bar(x - width/2, lqi_values, width, label='LQI', color='blue', alpha=0.7)
    ax6.bar(x + width/2, mlqi_values, width, label='mLQI', color='purple', alpha=0.7)

    ax6.set_xlabel('Metrics')
    ax6.set_ylabel('Values')
    ax6.set_title('Performance Metrics Comparison')
    ax6.set_xticks(x)
    ax6.set_xticklabels(metrics)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle("Detailed Comparison: LQI vs Modified LQI (mLQI)", fontsize=18)
    plt.tight_layout()

    print("Created detailed comparison between LQI and mLQI methods")
    print(f"LQI velocity variance: {lqi_smoothness:.6f}")
    print(f"mLQI velocity variance: {mlqi_smoothness:.6f}")
    print(f"Trajectory length difference: {abs(lqi_length - mlqi_length):.6f}")


def main() -> None:
    """Main demonstration function."""
    print("Logarithmic Quaternion Interpolation (LQI) Examples")
    print("=" * 60)
    print()
    print("This example demonstrates the LogQuaternionInterpolation class for smooth")
    print("quaternion trajectory generation using the LQI method from Parker et al. (2023)")
    print("with B-spline interpolation and visualization capabilities.")
    print()

    # Basic interpolation
    demo_basic_interpolation()

    # Comparison with traditional methods
    demo_comparison_with_traditional_methods()

    # Detailed LQI vs mLQI comparison
    demo_lqi_vs_mlqi_comparison()

    print("\n" + "=" * 60)
    print("All demonstrations completed successfully!")
    print("Key advantages of Logarithmic Quaternion Interpolation methods:")
    print()
    print("LQI (Standard Method):")
    print("  • Uses axis-angle representation r = θ*n̂ for natural interpolation")
    print("  • Handles quaternion double-cover and axis-angle discontinuities")
    print("  • Provides smooth C² continuous trajectories (or higher)")
    print("  • Direct B-spline interpolation of 3D axis-angle vectors")
    print()
    print("mLQI (Modified Method):")
    print("  • Decouples angle θ from unit vector components (X,Y,Z)")
    print("  • Uses separate B-spline interpolators for better numerical stability")
    print("  • Representation: (θ, X, Y, Z) where X²+Y²+Z²=1")
    print("  • Improved conditioning for complex trajectories")
    print("  • Optional normalized/unnormalized unit vector interpolation")
    print()
    print("Common Features:")
    print("  • Supports flexible boundary conditions (velocity, acceleration)")
    print("  • Based on proven algorithms from Parker et al. (2023)")
    print("  • Ideal for robotics and animation applications")
    print("  • Superior to traditional SLERP/SQUAD for complex trajectories")
    print()
    print("Displaying all visualization plots...")
    plt.show()


if __name__ == "__main__":
    main()
