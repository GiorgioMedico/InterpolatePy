import matplotlib.pyplot as plt
import numpy as np

from interpolatepy.c_s_with_acc1 import CubicSplineWithAcceleration1


# Example usage 0
def simple_example() -> None:
    """Run a simple example from the book."""
    print("Example 0: From book")
    t_points = [0.0, 5.0, 7.0, 8.0, 10.0, 15.0, 18.0]
    q_points = [3.0, -2.0, -5.0, 0.0, 6.0, 12.0, 8.0]

    # Create spline with velocity and acceleration constraints
    spline = CubicSplineWithAcceleration1(
        t_points,
        q_points,
        v0=2.0,
        vn=-3.0,  # Initial and final velocities
        a0=0.0,
        an=0.0,  # Initial and final accelerations
        debug=True,
    )

    # Plot the trajectory
    spline.plot(1000)
    print("\n")


# Example 1: Robot Joint Motion Planning
def robot_joint_example() -> CubicSplineWithAcceleration1:
    """
    Example for planning a smooth joint trajectory for a robot arm
    with specific velocity and acceleration constraints.

    Returns:
        The created cubic spline instance
    """
    print("Example 1: Robot Joint Motion Planning")

    # Define joint angle waypoints (in radians) and corresponding times
    t_points = [0.0, 2.0, 4.0, 7.0, 10.0]
    q_points = [0.0, 1.57, 0.5, -0.5, 0.0]  # Start at 0, move to π/2, then back to 0

    # Create spline with zero initial/final velocity and acceleration
    # This ensures the robot starts and stops smoothly
    spline = CubicSplineWithAcceleration1(
        t_points,
        q_points,
        v0=0.0,
        vn=0.0,  # Zero initial and final velocities
        a0=0.0,
        an=0.0,  # Zero initial and final accelerations
    )

    # Output values at specific times
    print(f"Position at t=3.0: {spline.evaluate(3.0):.4f} rad")
    print(f"Velocity at t=3.0: {spline.evaluate_velocity(3.0):.4f} rad/s")
    print(f"Acceleration at t=3.0: {spline.evaluate_acceleration(3.0):.4f} rad/s²")

    # Plot the trajectory
    spline.plot()

    return spline


# Example 2: Camera Pan Trajectory
def camera_pan_example() -> CubicSplineWithAcceleration1:
    """
    Example for creating a smooth camera pan motion
    with non-zero initial and final velocities.

    Returns:
        The created cubic spline instance
    """
    print("\nExample 2: Camera Pan Trajectory")

    # Define camera angle waypoints (in degrees) with times
    t_points = [0.0, 3.0, 7.0, 12.0, 15.0]
    q_points = [0.0, 45.0, 90.0, 120.0, 180.0]

    # Create spline with specific initial and final velocities
    # Camera starts and ends with some motion for a more natural look
    spline = CubicSplineWithAcceleration1(
        t_points,
        q_points,
        v0=10.0,
        vn=5.0,  # Initial and final velocities (deg/s)
        a0=0.0,
        an=0.0,  # Zero initial and final accelerations
    )

    # Sample the trajectory at regular intervals
    sample_times = np.linspace(0, 15, 16)
    positions = np.asarray(spline.evaluate(sample_times))
    velocities = np.asarray(spline.evaluate_velocity(sample_times))

    # Print the sampled trajectory
    print("Time (s) | Position (deg) | Velocity (deg/s)")
    print("-" * 45)
    for t, p, v in zip(sample_times, positions, velocities):
        print(f"{t:7.1f} | {p:13.2f} | {v:15.2f}")

    # Plot the trajectory
    spline.plot()

    return spline


# Example 3: Drone Height Trajectory with Time Scaling
def drone_height_example() -> tuple[CubicSplineWithAcceleration1, CubicSplineWithAcceleration1]:
    """
    Example for planning a drone's height trajectory
    with acceleration limits and time scaling.

    Returns:
        A tuple containing the original and time-scaled cubic spline instances
    """
    print("\nExample 3: Drone Height Trajectory with Time Scaling")

    # Define height waypoints (in meters) with times
    t_points = [0.0, 5.0, 10.0, 20.0, 25.0]
    q_points = [0.0, 10.0, 15.0, 5.0, 2.0]

    # Create spline with limited initial and final accelerations
    spline = CubicSplineWithAcceleration1(
        t_points,
        q_points,
        v0=0.0,
        vn=0.0,  # Zero initial and final velocities
        a0=0.5,
        an=-0.5,  # Limited initial and final accelerations (m/s²)
    )

    # Time scaling factor (to slow down or speed up the trajectory)
    time_scale = 1.5  # Slow down by 50%

    # Generate evaluation points with original and scaled timing
    t_eval_original = np.linspace(t_points[0], t_points[-1], 100)
    t_eval_scaled = np.linspace(t_points[0], t_points[-1] * time_scale, 100)

    # Evaluate trajectories
    q_original = spline.evaluate(t_eval_original)

    # Create a new spline with scaled timing
    t_points_scaled = [t * time_scale for t in t_points]
    spline_scaled = CubicSplineWithAcceleration1(
        t_points_scaled,
        q_points,
        v0=0.0,
        vn=0.0,
        a0=0.5,
        an=-0.5,
    )

    # Create plots to compare original and time-scaled trajectories
    _fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    # Position plot
    ax1.plot(t_eval_original, q_original, "b-", linewidth=2, label="Original")
    ax1.plot(
        t_eval_scaled,
        spline_scaled.evaluate(t_eval_scaled),
        "r--",
        linewidth=2,
        label="Time-scaled",
    )
    ax1.plot(t_points, q_points, "ko", markersize=6, label="Waypoints")
    ax1.set_ylabel("Height (m)")
    ax1.grid(True)
    ax1.legend()
    ax1.set_title("Drone Height Trajectory - Original vs. Time-scaled")

    # Velocity plot
    ax2.plot(t_eval_original, spline.evaluate_velocity(t_eval_original), "b-", linewidth=2)
    ax2.plot(t_eval_scaled, spline_scaled.evaluate_velocity(t_eval_scaled), "r--", linewidth=2)
    ax2.set_ylabel("Velocity (m/s)")
    ax2.grid(True)

    # Acceleration plot
    ax3.plot(t_eval_original, spline.evaluate_acceleration(t_eval_original), "b-", linewidth=2)
    ax3.plot(t_eval_scaled, spline_scaled.evaluate_acceleration(t_eval_scaled), "r--", linewidth=2)
    ax3.set_ylabel("Acceleration (m/s²)")
    ax3.set_xlabel("Time (s)")
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

    print(f"Original duration: {t_points[-1]:.1f}s")
    print(f"Scaled duration: {t_points_scaled[-1]:.1f}s")

    # Fix long lines by splitting them
    max_acc_original = np.max(np.abs(spline.evaluate_acceleration(t_eval_original)))
    print(f"Max acceleration (original): {max_acc_original:.2f} m/s²")

    max_acc_scaled = np.max(np.abs(spline_scaled.evaluate_acceleration(t_eval_scaled)))
    print(f"Max acceleration (scaled): {max_acc_scaled:.2f} m/s²")

    return spline, spline_scaled


# Example 4: Multi-dimensional Trajectory (3D Path)
def multi_dimensional_example() -> (
    tuple[CubicSplineWithAcceleration1, CubicSplineWithAcceleration1, CubicSplineWithAcceleration1]
):
    """
    Example for creating a 3D trajectory using three independent splines
    for x, y, and z coordinates.

    Returns:
        A tuple containing the x, y, and z cubic spline instances
    """
    print("\nExample 4: Multi-dimensional Trajectory (3D Path)")

    # Define time points for the trajectory
    t_points = [0.0, 2.0, 5.0, 8.0, 10.0, 15.0]

    # Define waypoints for each dimension
    x_points = [0.0, 2.0, 5.0, 8.0, 6.0, 10.0]
    y_points = [0.0, 3.0, 8.0, 4.0, 2.0, 5.0]
    z_points = [0.0, 1.0, 2.0, 4.0, 3.0, 1.0]

    # Create a spline for each dimension
    spline_x = CubicSplineWithAcceleration1(t_points, x_points, v0=0.0, vn=0.0, a0=0.0, an=0.0)

    spline_y = CubicSplineWithAcceleration1(t_points, y_points, v0=0.0, vn=0.0, a0=0.0, an=0.0)

    spline_z = CubicSplineWithAcceleration1(t_points, z_points, v0=0.0, vn=0.0, a0=0.0, an=0.0)

    # Generate points for smooth trajectory visualization
    t_eval = np.linspace(t_points[0], t_points[-1], 200)

    # Evaluate each dimension
    x_traj = spline_x.evaluate(t_eval)
    y_traj = spline_y.evaluate(t_eval)
    z_traj = spline_z.evaluate(t_eval)

    # Calculate speeds
    vx = spline_x.evaluate_velocity(t_eval)
    vy = spline_y.evaluate_velocity(t_eval)
    vz = spline_z.evaluate_velocity(t_eval)
    speed = np.sqrt(vx**2 + vy**2 + vz**2)

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the trajectory with color based on speed
    points = ax.scatter(x_traj, y_traj, z_traj, c=speed, cmap="viridis", s=2, alpha=0.8)
    fig.colorbar(points, ax=ax, label="Speed")

    # Plot the waypoints
    ax.scatter(x_points, y_points, z_points, color="red", s=50, marker="o", label="Waypoints")

    # Connect waypoints with straight lines for comparison
    ax.plot(x_points, y_points, z_points, "r--", alpha=0.5, linewidth=1)

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Trajectory with Speed Coloring")
    ax.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

    # Print some statistics and fix the long line by splitting the calculation
    path_segments = np.sqrt(np.diff(x_traj) ** 2 + np.diff(y_traj) ** 2 + np.diff(z_traj) ** 2)
    total_path_length = np.sum(path_segments)
    print(f"Total path length: {total_path_length:.2f} units")

    print(f"Maximum speed: {np.max(speed):.2f} units/s")
    print(f"Average speed: {np.mean(speed):.2f} units/s")

    return spline_x, spline_y, spline_z


# Example 5: Comparing Boundary Conditions
def compare_boundary_conditions() -> tuple[
    CubicSplineWithAcceleration1,
    CubicSplineWithAcceleration1,
    CubicSplineWithAcceleration1,
    CubicSplineWithAcceleration1,
]:
    """
    Example comparing different boundary conditions for the same waypoints.

    Returns:
        A tuple containing the four cubic spline instances with different boundary conditions
    """
    print("\nExample 5: Comparing Different Boundary Conditions")

    # Define waypoints
    t_points = [0.0, 4.0, 8.0, 12.0, 16.0]
    q_points = [0.0, 10.0, 5.0, 15.0, 10.0]

    # Create splines with different boundary conditions
    spline1 = CubicSplineWithAcceleration1(
        t_points,
        q_points,
        v0=0.0,
        vn=0.0,  # Zero velocity
        a0=0.0,
        an=0.0,  # Zero acceleration
    )

    spline2 = CubicSplineWithAcceleration1(
        t_points,
        q_points,
        v0=2.0,
        vn=-2.0,  # Non-zero velocity
        a0=0.0,
        an=0.0,  # Zero acceleration
    )

    spline3 = CubicSplineWithAcceleration1(
        t_points,
        q_points,
        v0=0.0,
        vn=0.0,  # Zero velocity
        a0=1.0,
        an=-1.0,  # Non-zero acceleration
    )

    spline4 = CubicSplineWithAcceleration1(
        t_points,
        q_points,
        v0=2.0,
        vn=-2.0,  # Non-zero velocity
        a0=1.0,
        an=-1.0,  # Non-zero acceleration
    )

    # Generate evaluation points
    t_eval = np.linspace(t_points[0], t_points[-1], 1000)

    # Create a figure for comparison
    _fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Position plot
    ax1.plot(t_eval, spline1.evaluate(t_eval), "b-", linewidth=2, label="v₀=vₙ=0, a₀=aₙ=0")
    ax1.plot(t_eval, spline2.evaluate(t_eval), "g-", linewidth=2, label="v₀=2, vₙ=-2, a₀=aₙ=0")
    ax1.plot(t_eval, spline3.evaluate(t_eval), "r-", linewidth=2, label="v₀=vₙ=0, a₀=1, aₙ=-1")
    ax1.plot(t_eval, spline4.evaluate(t_eval), "m-", linewidth=2, label="v₀=2, vₙ=-2, a₀=1, aₙ=-1")
    ax1.plot(t_points, q_points, "ko", markersize=8, label="Waypoints")
    ax1.set_ylabel("Position")
    ax1.grid(True)
    ax1.legend(loc="upper right")
    ax1.set_title("Effect of Different Boundary Conditions")

    # Velocity plot
    ax2.plot(t_eval, spline1.evaluate_velocity(t_eval), "b-", linewidth=2)
    ax2.plot(t_eval, spline2.evaluate_velocity(t_eval), "g-", linewidth=2)
    ax2.plot(t_eval, spline3.evaluate_velocity(t_eval), "r-", linewidth=2)
    ax2.plot(t_eval, spline4.evaluate_velocity(t_eval), "m-", linewidth=2)
    ax2.plot([t_points[0]], [0.0], "bo", markersize=6)
    ax2.plot([t_points[0]], [2.0], "go", markersize=6)
    ax2.plot([t_points[0]], [0.0], "ro", markersize=6)
    ax2.plot([t_points[0]], [2.0], "mo", markersize=6)
    ax2.plot([t_points[-1]], [0.0], "bo", markersize=6)
    ax2.plot([t_points[-1]], [-2.0], "go", markersize=6)
    ax2.plot([t_points[-1]], [0.0], "ro", markersize=6)
    ax2.plot([t_points[-1]], [-2.0], "mo", markersize=6)
    ax2.set_ylabel("Velocity")
    ax2.grid(True)

    # Acceleration plot
    ax3.plot(t_eval, spline1.evaluate_acceleration(t_eval), "b-", linewidth=2)
    ax3.plot(t_eval, spline2.evaluate_acceleration(t_eval), "g-", linewidth=2)
    ax3.plot(t_eval, spline3.evaluate_acceleration(t_eval), "r-", linewidth=2)
    ax3.plot(t_eval, spline4.evaluate_acceleration(t_eval), "m-", linewidth=2)
    ax3.plot([t_points[0]], [0.0], "bo", markersize=6)
    ax3.plot([t_points[0]], [0.0], "go", markersize=6)
    ax3.plot([t_points[0]], [1.0], "ro", markersize=6)
    ax3.plot([t_points[0]], [1.0], "mo", markersize=6)
    ax3.plot([t_points[-1]], [0.0], "bo", markersize=6)
    ax3.plot([t_points[-1]], [0.0], "go", markersize=6)
    ax3.plot([t_points[-1]], [-1.0], "ro", markersize=6)
    ax3.plot([t_points[-1]], [-1.0], "mo", markersize=6)
    ax3.set_ylabel("Acceleration")
    ax3.set_xlabel("Time")
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

    # Calculate smoothness metrics for each spline
    # Using the integral of squared jerk (third derivative) as a smoothness metric
    def calculate_jerk(spline: CubicSplineWithAcceleration1, t: np.ndarray) -> np.ndarray:
        dt = t[1] - t[0]
        acc = spline.evaluate_acceleration(t)
        # Return directly without creating a temporary variable
        return np.diff(acc) / dt

    jerks = [
        calculate_jerk(spline1, t_eval),
        calculate_jerk(spline2, t_eval),
        calculate_jerk(spline3, t_eval),
        calculate_jerk(spline4, t_eval),
    ]

    jerk_metrics = [np.sum(jerk**2) for jerk in jerks]

    print("Smoothness comparison (smaller is better):")
    print(f"Spline 1 (v₀=vₙ=0, a₀=aₙ=0): {jerk_metrics[0]:.2f}")
    print(f"Spline 2 (v₀=2, vₙ=-2, a₀=aₙ=0): {jerk_metrics[1]:.2f}")
    print(f"Spline 3 (v₀=vₙ=0, a₀=1, aₙ=-1): {jerk_metrics[2]:.2f}")
    print(f"Spline 4 (v₀=2, vₙ=-2, a₀=1, aₙ=-1): {jerk_metrics[3]:.2f}")

    return spline1, spline2, spline3, spline4


# Run all examples
if __name__ == "__main__":
    simple_example()
    robot_joint_example()
    camera_pan_example()
    drone_height_example()
    multi_dimensional_example()
    compare_boundary_conditions()
