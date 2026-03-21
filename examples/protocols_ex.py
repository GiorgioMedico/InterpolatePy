"""
Example demonstrating protocol-based generic functions in InterpolatePy.

Protocols (PEP 544 structural typing) allow writing algorithm-agnostic code.
A function annotated with a protocol type accepts ANY class that has matching
method signatures — no inheritance required.

This file showcases four protocol families:
1. ScalarTrajectory: CubicSpline vs DoubleSTrajectory
2. CurveEvaluator: BSplineInterpolator vs BSpline
3. GeometricPath: LinearPath vs CircularPath
4. QuaternionTrajectory: SquadC2 vs QuaternionSpline
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from interpolatepy import (
    # Protocols
    CurveEvaluator,
    GeometricPath,
    QuaternionTrajectory,
    ScalarTrajectory,
    # Concrete classes
    BSpline,
    BSplineInterpolator,
    CircularPath,
    CubicSpline,
    DoubleSTrajectory,
    LinearPath,
    Quaternion,
    QuaternionSpline,
    SquadC2,
    StateParams,
    TrajectoryBounds,
)


# ---------------------------------------------------------------------------
# 1. ScalarTrajectory — generic sampling across spline & motion-profile
# ---------------------------------------------------------------------------


def sample_scalar_trajectory(
    traj: ScalarTrajectory,
    t_start: float,
    t_end: float,
    num_points: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample position, velocity, and acceleration from any ScalarTrajectory.

    Parameters
    ----------
    traj : ScalarTrajectory
        Any object conforming to the ScalarTrajectory protocol.
    t_start : float
        Start time.
    t_end : float
        End time.
    num_points : int
        Number of sample points.

    Returns
    -------
    tuple of np.ndarray
        (time, position, velocity, acceleration) arrays.
    """
    t = np.linspace(t_start, t_end, num_points)
    pos = np.array([traj.evaluate(ti) for ti in t])
    vel = np.array([traj.evaluate_velocity(ti) for ti in t])
    acc = np.array([traj.evaluate_acceleration(ti) for ti in t])
    return t, pos, vel, acc


def example_scalar_trajectory() -> None:
    """Compare CubicSpline and DoubleSTrajectory through the same generic function."""
    print("\n1. ScalarTrajectory Protocol")
    print("=" * 40)

    # --- CubicSpline ---
    t_points = [0.0, 1.0, 2.0, 3.0, 4.0]
    q_points = [0.0, 5.0, 3.0, 8.0, 10.0]
    spline = CubicSpline(t_points, q_points)
    print(f"  CubicSpline  conforms: {isinstance(spline, ScalarTrajectory)}")

    # --- DoubleSTrajectory ---
    state = StateParams(q_0=0.0, q_1=10.0, v_0=0.0, v_1=0.0)
    bounds = TrajectoryBounds(v_bound=8.0, a_bound=20.0, j_bound=60.0)
    double_s = DoubleSTrajectory(state, bounds)
    duration = double_s.get_duration()
    print(f"  DoubleSTrajectory conforms: {isinstance(double_s, ScalarTrajectory)}")

    # Sample both through the *same* generic function
    t_cs, pos_cs, vel_cs, acc_cs = sample_scalar_trajectory(spline, 0.0, 4.0)
    t_ds, pos_ds, vel_ds, acc_ds = sample_scalar_trajectory(double_s, 0.0, duration)

    # Plot side-by-side
    fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex="col")
    fig.suptitle("ScalarTrajectory Protocol — same function, two algorithms")

    labels = ["Position", "Velocity", "Acceleration"]
    cs_data = [pos_cs, vel_cs, acc_cs]
    ds_data = [pos_ds, vel_ds, acc_ds]

    for row, (label, cs_vals, ds_vals) in enumerate(zip(labels, cs_data, ds_data)):
        axes[row, 0].plot(t_cs, cs_vals, "b-", linewidth=2)
        axes[row, 0].set_ylabel(label)
        axes[row, 0].grid(True)
        if row == 0:
            axes[row, 0].set_title("CubicSpline")

        axes[row, 1].plot(t_ds, ds_vals, "r-", linewidth=2)
        axes[row, 1].grid(True)
        if row == 0:
            axes[row, 1].set_title("DoubleSTrajectory")

    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 2. CurveEvaluator — unified derivative evaluation across B-spline variants
# ---------------------------------------------------------------------------


def evaluate_curve(
    curve: CurveEvaluator,
    u_start: float,
    u_end: float,
    num_points: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a parametric curve and its first derivative.

    Parameters
    ----------
    curve : CurveEvaluator
        Any object conforming to the CurveEvaluator protocol.
    u_start : float
        Start parameter value.
    u_end : float
        End parameter value.
    num_points : int
        Number of sample points.

    Returns
    -------
    tuple of np.ndarray
        (parameter, positions, derivatives) arrays.
    """
    u = np.linspace(u_start, u_end, num_points)
    positions = np.array([curve.evaluate(ui) for ui in u])
    derivatives = np.array([curve.evaluate_derivative(ui, order=1) for ui in u])
    return u, positions, derivatives


def example_curve_evaluator() -> None:
    """Compare BSplineInterpolator and BSpline through the same generic function."""
    print("\n2. CurveEvaluator Protocol")
    print("=" * 40)

    # --- BSplineInterpolator (interpolates through points) ---
    points_2d = np.array([
        [0.0, 0.0],
        [1.0, 2.0],
        [3.0, 3.0],
        [5.0, 1.0],
        [6.0, 4.0],
    ])
    interpolator = BSplineInterpolator(degree=3, points=points_2d)
    print(f"  BSplineInterpolator conforms: {isinstance(interpolator, CurveEvaluator)}")

    # --- BSpline (approximation with control polygon) ---
    control_pts = np.array([
        [0.0, 0.0],
        [1.0, 3.0],
        [2.5, 4.0],
        [4.0, 2.0],
        [5.5, 3.5],
        [6.0, 1.0],
    ])
    n_ctrl = len(control_pts)
    degree = 3
    n_knots = n_ctrl + degree + 1
    knots = np.concatenate([
        np.zeros(degree + 1),
        np.linspace(0, 1, n_knots - 2 * (degree + 1) + 2)[1:-1],
        np.ones(degree + 1),
    ])
    bspline = BSpline(degree=degree, knots=knots, control_points=control_pts)
    print(f"  BSpline conforms: {isinstance(bspline, CurveEvaluator)}")

    # Sample both through the *same* generic function
    _, pos_interp, der_interp = evaluate_curve(interpolator, interpolator.u_min, interpolator.u_max)
    _, pos_bs, der_bs = evaluate_curve(bspline, bspline.u_min, bspline.u_max)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CurveEvaluator Protocol — same function, two B-spline variants")

    # BSplineInterpolator
    axes[0].plot(pos_interp[:, 0], pos_interp[:, 1], "b-", linewidth=2, label="Curve")
    axes[0].plot(points_2d[:, 0], points_2d[:, 1], "ko", markersize=7, label="Data points")
    # Tangent arrows at a few sample locations
    step = len(pos_interp) // 8
    scale = 0.3
    for i in range(0, len(pos_interp), step):
        axes[0].annotate(
            "",
            xy=(pos_interp[i, 0] + der_interp[i, 0] * scale, pos_interp[i, 1] + der_interp[i, 1] * scale),
            xytext=(pos_interp[i, 0], pos_interp[i, 1]),
            arrowprops={"arrowstyle": "->", "color": "green", "lw": 1.5},
        )
    axes[0].set_title("BSplineInterpolator")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_aspect("equal", adjustable="datalim")

    # BSpline
    axes[1].plot(pos_bs[:, 0], pos_bs[:, 1], "r-", linewidth=2, label="Curve")
    axes[1].plot(control_pts[:, 0], control_pts[:, 1], "ks--", markersize=6, alpha=0.5, label="Control polygon")
    step = len(pos_bs) // 8
    for i in range(0, len(pos_bs), step):
        axes[1].annotate(
            "",
            xy=(pos_bs[i, 0] + der_bs[i, 0] * scale, pos_bs[i, 1] + der_bs[i, 1] * scale),
            xytext=(pos_bs[i, 0], pos_bs[i, 1]),
            arrowprops={"arrowstyle": "->", "color": "green", "lw": 1.5},
        )
    axes[1].set_title("BSpline")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_aspect("equal", adjustable="datalim")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 3. GeometricPath — path-agnostic 3D visualization with velocity arrows
# ---------------------------------------------------------------------------


def trace_geometric_path(
    path: GeometricPath,
    s_start: float,
    s_end: float,
    num_points: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trace position and velocity vectors along any GeometricPath.

    Parameters
    ----------
    path : GeometricPath
        Any object conforming to the GeometricPath protocol.
    s_start : float
        Start arc-length parameter.
    s_end : float
        End arc-length parameter.
    num_points : int
        Number of sample points.

    Returns
    -------
    tuple of np.ndarray
        (positions, velocities, accelerations) arrays of shape (N, 3).
    """
    s_vals = np.linspace(s_start, s_end, num_points)
    positions = np.array([path.position(si) for si in s_vals])
    velocities = np.array([path.velocity(si) for si in s_vals])
    accelerations = np.array([path.acceleration(si) for si in s_vals])
    return positions, velocities, accelerations


def example_geometric_path() -> None:
    """Compare LinearPath and CircularPath through the same generic function."""
    print("\n3. GeometricPath Protocol")
    print("=" * 40)

    # --- LinearPath ---
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([5.0, 3.0, 2.0])
    linear = LinearPath(start, end)
    print(f"  LinearPath   conforms: {isinstance(linear, GeometricPath)}")

    # --- CircularPath ---
    axis = np.array([0.0, 0.0, 1.0])
    center = np.array([0.0, 0.0, 0.0])
    point_on_circle = np.array([3.0, 0.0, 0.0])
    circular = CircularPath(axis, center, point_on_circle)
    arc_length = np.pi * circular.radius  # half circle
    print(f"  CircularPath conforms: {isinstance(circular, GeometricPath)}")

    # Sample both through the *same* generic function
    pos_lin, vel_lin, _ = trace_geometric_path(linear, 0.0, linear.length)
    pos_cir, vel_cir, acc_cir = trace_geometric_path(circular, 0.0, float(arc_length))

    # 3D plot with velocity quiver arrows
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle("GeometricPath Protocol — same function, two path types")

    # LinearPath
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot(pos_lin[:, 0], pos_lin[:, 1], pos_lin[:, 2], "b-", linewidth=2, label="Path")
    step = max(1, len(pos_lin) // 10)
    ax1.quiver(
        pos_lin[::step, 0], pos_lin[::step, 1], pos_lin[::step, 2],
        vel_lin[::step, 0], vel_lin[::step, 1], vel_lin[::step, 2],
        length=0.5, color="green", arrow_length_ratio=0.3, label="Velocity",
    )
    ax1.set_title("LinearPath")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.legend()

    # CircularPath
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot(pos_cir[:, 0], pos_cir[:, 1], pos_cir[:, 2], "r-", linewidth=2, label="Path")
    step = max(1, len(pos_cir) // 12)
    ax2.quiver(
        pos_cir[::step, 0], pos_cir[::step, 1], pos_cir[::step, 2],
        vel_cir[::step, 0], vel_cir[::step, 1], vel_cir[::step, 2],
        length=0.8, color="green", arrow_length_ratio=0.3, label="Velocity",
    )
    ax2.quiver(
        pos_cir[::step, 0], pos_cir[::step, 1], pos_cir[::step, 2],
        acc_cir[::step, 0], acc_cir[::step, 1], acc_cir[::step, 2],
        length=0.8, color="orange", arrow_length_ratio=0.3, label="Acceleration",
    )
    ax2.set_title("CircularPath")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.legend()

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 4. QuaternionTrajectory — rotation algorithm interchangeability
# ---------------------------------------------------------------------------


def sample_rotation(
    traj: QuaternionTrajectory,
    t_start: float,
    t_end: float,
    num_points: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract Euler angles and angular velocity magnitude from any QuaternionTrajectory.

    Parameters
    ----------
    traj : QuaternionTrajectory
        Any object conforming to the QuaternionTrajectory protocol.
    t_start : float
        Start time.
    t_end : float
        End time.
    num_points : int
        Number of sample points.

    Returns
    -------
    tuple of np.ndarray
        (time, euler_angles (N,3), angular_velocity_magnitude (N,)) arrays.
    """
    t = np.linspace(t_start, t_end, num_points)
    euler = np.zeros((num_points, 3))
    omega_mag = np.zeros(num_points)

    for i, ti in enumerate(t):
        q = traj.evaluate(ti)
        euler[i] = q.to_euler_angles()
        omega = traj.evaluate_velocity(ti)
        omega_mag[i] = float(np.linalg.norm(omega))

    return t, euler, omega_mag


def example_quaternion_trajectory() -> None:
    """Compare SquadC2 and QuaternionSpline through the same generic function."""
    print("\n4. QuaternionTrajectory Protocol")
    print("=" * 40)

    # Shared waypoints: identity → 90° about Z → 90° about X → back to identity
    q1 = Quaternion.identity()
    q2 = Quaternion.from_euler_angles(0.0, 0.0, np.pi / 2)
    q3 = Quaternion.from_euler_angles(np.pi / 2, 0.0, np.pi / 2)
    q4 = Quaternion.identity()
    times = [0.0, 1.0, 2.0, 3.0]
    quats = [q1, q2, q3, q4]

    # --- SquadC2 ---
    squad = SquadC2(time_points=times, quaternions=quats)
    print(f"  SquadC2          conforms: {isinstance(squad, QuaternionTrajectory)}")

    # --- QuaternionSpline (SQUAD method) ---
    qspline = QuaternionSpline(time_points=times, quaternions=quats, interpolation_method=Quaternion.SQUAD)
    print(f"  QuaternionSpline conforms: {isinstance(qspline, QuaternionTrajectory)}")

    # Sample both through the *same* generic function
    t_sq, euler_sq, omega_sq = sample_rotation(squad, 0.0, 3.0)
    t_qs, euler_qs, omega_qs = sample_rotation(qspline, 0.0, 3.0)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex="col")
    fig.suptitle("QuaternionTrajectory Protocol — same function, two rotation algorithms")

    angle_labels = ["Roll", "Pitch", "Yaw"]
    colors = ["r", "g", "b"]

    # SquadC2 — Euler angles
    for j in range(3):
        axes[0, 0].plot(t_sq, np.degrees(euler_sq[:, j]), color=colors[j], linewidth=2, label=angle_labels[j])
    axes[0, 0].set_ylabel("Angle (deg)")
    axes[0, 0].set_title("SquadC2 — Euler Angles")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # SquadC2 — Angular velocity magnitude
    axes[1, 0].plot(t_sq, omega_sq, "m-", linewidth=2)
    axes[1, 0].set_ylabel("||ω|| (rad/s)")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_title("SquadC2 — Angular Velocity")
    axes[1, 0].grid(True)

    # QuaternionSpline — Euler angles
    for j in range(3):
        axes[0, 1].plot(t_qs, np.degrees(euler_qs[:, j]), color=colors[j], linewidth=2, label=angle_labels[j])
    axes[0, 1].set_ylabel("Angle (deg)")
    axes[0, 1].set_title("QuaternionSpline — Euler Angles")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # QuaternionSpline — Angular velocity magnitude
    axes[1, 1].plot(t_qs, omega_qs, "m-", linewidth=2)
    axes[1, 1].set_ylabel("||ω|| (rad/s)")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_title("QuaternionSpline — Angular Velocity")
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all protocol demonstration examples."""
    print("Protocol-Based Generic Functions — InterpolatePy")
    print("=" * 50)

    try:
        example_scalar_trajectory()
        example_curve_evaluator()
        example_geometric_path()
        example_quaternion_trajectory()
        print("\nAll protocol examples completed successfully!")
    except (ValueError, TypeError, RuntimeError) as e:
        print(f"Error running examples: {e}")


if __name__ == "__main__":
    main()
