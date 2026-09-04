"""
Modified Logarithmic Quaternion Interpolation (mLQI) along a cylindrical
helix oriented with the Frenet frame.

Pipeline:
1. Sample a cylindrical helix and compute its Frenet frame at each waypoint.
2. Convert each Frenet rotation matrix to a unit quaternion.
3. Feed the (time, quaternion) waypoints to ModifiedLogQuaternionInterpolation.
4. Evaluate the position helix and the interpolated orientation at a dense
   grid of times and visualize the tool frames side by side with the
   ground-truth analytical Frenet frames.
"""

from __future__ import annotations

from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from interpolatepy import (
    compute_trajectory_frames,
    helicoidal_trajectory_with_derivatives,
    plot_frames,
)
from interpolatepy import ModifiedLogQuaternionInterpolation
from interpolatepy import Quaternion
from interpolatepy.quat_visualization import QuaternionTrajectoryVisualizer


def angular_error_deg(frame_truth: np.ndarray, frame_est: np.ndarray) -> float:
    """Geodesic angle between two rotation matrices, in degrees."""
    r = frame_truth.T @ frame_est
    cos_angle = np.clip(0.5 * (np.trace(r) - 1.0), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def main() -> None:
    print("mLQI on a Cylindrical Helix with Frenet Frame Waypoints")
    print("=" * 60)

    r = 1.0           # helix radius
    d = 0.3           # vertical rise per radian
    u_min = 0.5
    u_max = 6.0 * np.pi
    n_waypoints = 1000
    n_dense = 240

    helix_func = partial(helicoidal_trajectory_with_derivatives, r=r, d=d)

    # Sample waypoints + their Frenet frames.
    u_waypoints = np.linspace(u_min, u_max, n_waypoints)
    wp_positions, wp_frames = compute_trajectory_frames(helix_func, u_waypoints)
    quaternions = [Quaternion.from_rotation_matrix(f).unit() for f in wp_frames]

    print(f"Sampled {n_waypoints} Frenet waypoints in u ∈ [{u_min:.2f}, {u_max:.2f}]")
    print(f"Time/parameter range used for mLQI: [{u_waypoints[0]:.3f}, {u_waypoints[-1]:.3f}]")

    mlqi = ModifiedLogQuaternionInterpolation(
        time_points=u_waypoints,
        quaternions=quaternions,
        degree=3,
        normalize_axis=True,
    )

    # Dense evaluation of position, orientation, internal (theta, xyz) state,
    # and physical kinematics — all in a single pass over u_dense so we don't
    # walk the underlying B-splines three times.
    u_dense = np.linspace(u_min, u_max, n_dense)
    dense_positions = np.zeros((n_dense, 3))
    dense_frames_mlqi = np.zeros((n_dense, 3, 3))
    dense_quaternions_mlqi: list[Quaternion] = []
    theta_dense = np.zeros(n_dense)
    xyz_dense = np.zeros((n_dense, 3))
    omega = np.zeros((n_dense, 3))
    alpha = np.zeros((n_dense, 3))

    for i, u in enumerate(u_dense):
        dense_positions[i], _, _ = helix_func(u)
        q = mlqi.evaluate(u)
        dense_frames_mlqi[i] = q.to_rotation_matrix()
        dense_quaternions_mlqi.append(q)
        theta_dense[i] = mlqi.theta_interpolator.evaluate(u)[0]
        xyz_dense[i] = mlqi.xyz_interpolator.evaluate(u)
        omega[i], alpha[i] = mlqi.get_physical_kinematics(u)

    _, dense_frames_truth = compute_trajectory_frames(helix_func, u_dense)

    errors = np.array([
        angular_error_deg(dense_frames_truth[i], dense_frames_mlqi[i])
        for i in range(n_dense)
    ])
    print(f"Mean angular error vs Frenet truth: {errors.mean():.3f} deg")
    print(f"Max  angular error vs Frenet truth: {errors.max():.3f} deg")

    # Cap the number of waypoint markers actually drawn so plots stay readable
    # when n_waypoints is large. The interpolator still uses all of them.
    max_shown_waypoints = 30
    if n_waypoints > max_shown_waypoints:
        show_idx = np.linspace(0, n_waypoints - 1, max_shown_waypoints, dtype=int)
    else:
        show_idx = np.arange(n_waypoints)
    wp_positions_shown = wp_positions[show_idx]
    times_shown = u_waypoints[show_idx]
    quaternions_shown = [quaternions[i] for i in show_idx]
    theta_wp = np.array([mlqi.theta_interpolator.evaluate(u)[0] for u in times_shown])
    xyz_wp = np.array([mlqi.xyz_interpolator.evaluate(u) for u in times_shown])

    fig = plt.figure(figsize=(16, 7))

    ax_left = fig.add_subplot(121, projection="3d")
    plot_frames(ax_left, dense_positions, dense_frames_mlqi, scale=0.6, skip=12)
    ax_left.scatter(
        wp_positions_shown[:, 0],
        wp_positions_shown[:, 1],
        wp_positions_shown[:, 2],
        color="magenta",
        s=40,
        depthshade=False,
        label=f"mLQI waypoints ({len(show_idx)} of {n_waypoints} shown)",
    )
    ax_left.set_title("Helix + mLQI-interpolated frames")
    ax_left.set_xlabel("x")
    ax_left.set_ylabel("y")
    ax_left.set_zlabel("z")
    ax_left.legend(loc="upper left")

    ax_right = fig.add_subplot(122, projection="3d")
    plot_frames(ax_right, dense_positions, dense_frames_truth, scale=0.6, skip=12)
    ax_right.set_title("Analytical Frenet frames (reference)")
    ax_right.set_xlabel("x")
    ax_right.set_ylabel("y")
    ax_right.set_zlabel("z")

    for ax in (ax_left, ax_right):
        ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()

    visualizer = QuaternionTrajectoryVisualizer()
    visualizer.plot_3d_trajectory(
        dense_quaternions_mlqi,
        waypoints=quaternions_shown,
        waypoint_times=list(times_shown),
        title="mLQI quaternion trajectory (stereographic MRP projection)",
        color="purple",
        line_width=2.5,
        point_size=15,
        waypoint_color="magenta",
        show_waypoint_labels=False,
        figsize=(10, 8),
    )

    _fig2, ax_err = plt.subplots(figsize=(10, 3.5))
    ax_err.plot(u_dense, errors, color="purple", linewidth=2.0)
    ax_err.fill_between(u_dense, errors, alpha=0.25, color="purple")
    ax_err.set_xlabel("Parameter u")
    ax_err.set_ylabel("Angular error [deg]")
    ax_err.set_title("Geodesic deviation: mLQI orientation vs Frenet truth")
    ax_err.grid(True, alpha=0.3)
    plt.tight_layout()

    _fig3, axes3 = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    component_labels = [r"$\theta$ [rad]", "X", "Y", "Z"]
    component_data = [theta_dense, xyz_dense[:, 0], xyz_dense[:, 1], xyz_dense[:, 2]]
    waypoint_data = [theta_wp, xyz_wp[:, 0], xyz_wp[:, 1], xyz_wp[:, 2]]
    component_colors = ["tab:orange", "tab:red", "tab:green", "tab:blue"]

    for ax, label, curve, wp_vals, c in zip(
        axes3, component_labels, component_data, waypoint_data, component_colors
    ):
        ax.plot(u_dense, curve, color=c, linewidth=2.0, label="mLQI")
        ax.scatter(times_shown, wp_vals, color="magenta", s=25, zorder=5, label="waypoints")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes3[-1].set_xlabel("Parameter u")
    axes3[0].set_title(
        r"mLQI internal state: angle $\theta(u)$ and unit-axis components $(X, Y, Z)(u)$"
    )
    plt.tight_layout()

    omega_norm = np.linalg.norm(omega, axis=1)
    alpha_norm = np.linalg.norm(alpha, axis=1)

    _fig4, (ax_w, ax_a) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for k, axis_name, c in zip(range(3), ("x", "y", "z"), ("tab:red", "tab:green", "tab:blue")):
        ax_w.plot(u_dense, omega[:, k], color=c, linewidth=1.8, label=rf"$\omega_{axis_name}$")
        ax_a.plot(u_dense, alpha[:, k], color=c, linewidth=1.8, label=rf"$\alpha_{axis_name}$")
    ax_w.plot(u_dense, omega_norm, color="black", linewidth=1.2, linestyle="--", label=r"$\|\omega\|$")
    ax_a.plot(u_dense, alpha_norm, color="black", linewidth=1.2, linestyle="--", label=r"$\|\alpha\|$")

    ax_w.set_ylabel(r"Angular velocity $\omega$ [rad/u]")
    ax_w.set_title("Physical angular velocity and acceleration from mLQI")
    ax_w.legend(loc="upper right", ncol=4, fontsize=9)
    ax_w.grid(True, alpha=0.3)

    ax_a.set_ylabel(r"Angular acceleration $\alpha$ [rad/u$^2$]")
    ax_a.set_xlabel("Parameter u")
    ax_a.legend(loc="upper right", ncol=4, fontsize=9)
    ax_a.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
