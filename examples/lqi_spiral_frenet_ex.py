"""
Logarithmic Quaternion Interpolation (LQI) along a cylindrical helix oriented
with the Frenet frame.

Pipeline:
1. Define a cylindrical helix p(u) = (r*cos(u), r*sin(u), b*u) with analytical
   derivatives so the Frenet frame can be evaluated in closed form. The radius
   is constant, so curvature and torsion are constant along the curve.
2. Sample a sparse set of waypoints along the helix and compute the Frenet
   frame [tangent, normal, binormal] at each waypoint.
3. Convert each Frenet rotation matrix to a unit quaternion.
4. Feed the (time, quaternion) waypoints to LogQuaternionInterpolation to
   obtain a smooth C^2 orientation trajectory. LQI splines the rotation
   vector r(u) = theta(u) * n_hat(u) as a single 3D quantity (in contrast
   with mLQI, which splines theta and (X, Y, Z) on separate channels).
5. Evaluate the position helix and the interpolated orientation at a dense
   set of times and visualize the resulting tool frames side by side with the
   ground-truth Frenet frames.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

from interpolatepy.frenet_frame import compute_trajectory_frames
from interpolatepy.frenet_frame import plot_frames
from interpolatepy.log_quat import LogQuaternionInterpolation
from interpolatepy.quat_core import Quaternion
from interpolatepy.quat_visualization import QuaternionTrajectoryVisualizer


def cylindrical_helix_with_derivatives(
    u: float, r: float = 1.0, b: float = 0.3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cylindrical helix (constant radius) position and first two derivatives.

    p(u)      = ( r*cos(u),    r*sin(u),    b*u )
    dp/du     = (-r*sin(u),    r*cos(u),    b   )
    d2p/du2   = (-r*cos(u),   -r*sin(u),    0   )
    """
    cos_u = np.cos(u)
    sin_u = np.sin(u)

    p = np.array([r * cos_u, r * sin_u, b * u])
    dp_du = np.array([-r * sin_u, r * cos_u, b])
    d2p_du2 = np.array([-r * cos_u, -r * sin_u, 0.0])

    return p, dp_du, d2p_du2


def frame_to_quaternion(frame: np.ndarray) -> Quaternion:
    """
    Convert a 3x3 frame whose columns are [tangent, normal, binormal] into a
    unit quaternion. The frame matrix is itself the rotation matrix from the
    local Frenet basis to the world basis, so we can hand it directly to
    Quaternion.from_rotation_matrix.
    """
    return Quaternion.from_rotation_matrix(frame).unit()


def build_waypoints(
    n_waypoints: int = 10,
    u_min: float = 0.5,
    u_max: float = 6.0 * np.pi,
    r: float = 1.0,
    b: float = 0.3,
) -> tuple[np.ndarray, list[Quaternion], np.ndarray, np.ndarray]:
    """
    Sample the helix at n_waypoints values of u, compute Frenet frames there,
    and return (times, quaternions, waypoint_positions, waypoint_frames).

    Times are taken equal to the parameter u so the orientation evolution
    matches the curve parameterization.
    """
    u_waypoints = np.linspace(u_min, u_max, n_waypoints)

    def helix_func(u: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return cylindrical_helix_with_derivatives(u, r=r, b=b)

    positions, frames = compute_trajectory_frames(helix_func, u_waypoints)

    quaternions = [frame_to_quaternion(frames[i]) for i in range(len(u_waypoints))]

    return u_waypoints, quaternions, positions, frames


def evaluate_interpolated_trajectory(
    lqi: LogQuaternionInterpolation,
    u_dense: np.ndarray,
    r: float,
    b: float,
) -> tuple[np.ndarray, np.ndarray, list[Quaternion]]:
    """
    Evaluate the helix positions, the LQI-interpolated orientation frames,
    and the underlying interpolated quaternions at the dense parameter values.
    """
    dense_positions = np.zeros((len(u_dense), 3))
    dense_frames = np.zeros((len(u_dense), 3, 3))
    dense_quaternions: list[Quaternion] = []

    for i, u in enumerate(u_dense):
        dense_positions[i], _, _ = cylindrical_helix_with_derivatives(u, r=r, b=b)
        q = lqi.evaluate(u)
        dense_frames[i] = q.to_rotation_matrix()
        dense_quaternions.append(q)

    return dense_positions, dense_frames, dense_quaternions


def angular_error_deg(frame_truth: np.ndarray, frame_est: np.ndarray) -> float:
    """
    Geodesic angle (in degrees) between two rotation matrices.
    """
    r = frame_truth.T @ frame_est
    cos_angle = np.clip(0.5 * (np.trace(r) - 1.0), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def main() -> None:
    print("LQI on a Cylindrical Helix with Frenet Frame Waypoints")
    print("=" * 60)

    # Helix parameters
    r = 1.0   # constant radius
    b = 0.3   # vertical rise per radian
    u_min = 0.5
    u_max = 6.0 * np.pi
    n_waypoints = 1000

    # Sample waypoints + their Frenet frames.
    times, quaternions, wp_positions, wp_frames = build_waypoints(
        n_waypoints=n_waypoints,
        u_min=u_min,
        u_max=u_max,
        r=r,
        b=b,
    )

    print(f"Sampled {n_waypoints} Frenet waypoints in u ∈ [{u_min:.2f}, {u_max:.2f}]")
    print(f"Time/parameter range used for LQI: [{times[0]:.3f}, {times[-1]:.3f}]")

    # Build the LQI interpolator over the waypoint orientations.
    lqi = LogQuaternionInterpolation(
        time_points=times,
        quaternions=quaternions,
        degree=3,
    )

    # Dense evaluation of position + interpolated orientation.
    n_dense = 240
    u_dense = np.linspace(u_min, u_max, n_dense)
    dense_positions, dense_frames_lqi, dense_quaternions_lqi = (
        evaluate_interpolated_trajectory(lqi, u_dense, r=r, b=b)
    )

    # Ground-truth Frenet frames on the same dense grid, for comparison.
    def helix_func(u: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return cylindrical_helix_with_derivatives(u, r=r, b=b)

    _, dense_frames_truth = compute_trajectory_frames(helix_func, u_dense)

    # Report worst-case angular deviation between interpolated orientation and
    # the analytical Frenet frame.
    errors = np.array([
        angular_error_deg(dense_frames_truth[i], dense_frames_lqi[i])
        for i in range(n_dense)
    ])
    print(f"Mean angular error vs Frenet truth: {errors.mean():.3f} deg")
    print(f"Max  angular error vs Frenet truth: {errors.max():.3f} deg")

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    # Cap the number of waypoint markers actually drawn so the plots stay
    # readable when n_waypoints is large (e.g. 1000). The interpolator still
    # uses all of them — this only affects the visual overlay.
    max_shown_waypoints = 30
    if n_waypoints > max_shown_waypoints:
        show_idx = np.linspace(0, n_waypoints - 1, max_shown_waypoints, dtype=int)
    else:
        show_idx = np.arange(n_waypoints)
    wp_positions_shown = wp_positions[show_idx]
    times_shown = times[show_idx]
    quaternions_shown = [quaternions[i] for i in show_idx]

    fig = plt.figure(figsize=(16, 7))

    # Left panel: waypoints + interpolated LQI tool frames.
    ax_left = fig.add_subplot(121, projection="3d")
    plot_frames(ax_left, dense_positions, dense_frames_lqi, scale=0.6, skip=12)
    ax_left.scatter(
        wp_positions_shown[:, 0],
        wp_positions_shown[:, 1],
        wp_positions_shown[:, 2],
        color="magenta",
        s=40,
        depthshade=False,
        label=f"LQI waypoints ({len(show_idx)} of {n_waypoints} shown)",
    )
    ax_left.set_title("Helix + LQI-interpolated frames")
    ax_left.set_xlabel("x")
    ax_left.set_ylabel("y")
    ax_left.set_zlabel("z")
    ax_left.legend(loc="upper left")

    # Right panel: ground-truth analytical Frenet frames on the dense grid.
    ax_right = fig.add_subplot(122, projection="3d")
    plot_frames(ax_right, dense_positions, dense_frames_truth, scale=0.6, skip=12)
    ax_right.set_title("Analytical Frenet frames (reference)")
    ax_right.set_xlabel("x")
    ax_right.set_ylabel("y")
    ax_right.set_zlabel("z")

    for ax in (ax_left, ax_right):
        ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()

    # Quaternion trajectory in stereographic (MRP) projection space.
    # This shows the orientation evolution in the *rotation* space rather than
    # in the cartesian position space, with the input waypoints highlighted.
    visualizer = QuaternionTrajectoryVisualizer()
    visualizer.plot_3d_trajectory(
        dense_quaternions_lqi,
        waypoints=quaternions_shown,
        waypoint_times=list(times_shown),
        title="LQI quaternion trajectory (stereographic MRP projection)",
        color="purple",
        line_width=2.5,
        point_size=15,
        waypoint_color="magenta",
        show_waypoint_labels=False,
        figsize=(10, 8),
    )

    # Bottom: angular error along the curve.
    _fig2, ax_err = plt.subplots(figsize=(10, 3.5))
    ax_err.plot(u_dense, errors, color="purple", linewidth=2.0)
    ax_err.fill_between(u_dense, errors, alpha=0.25, color="purple")
    ax_err.set_xlabel("Parameter u")
    ax_err.set_ylabel("Angular error [deg]")
    ax_err.set_title("Geodesic deviation: LQI orientation vs Frenet truth")
    ax_err.grid(True, alpha=0.3)
    plt.tight_layout()

    # ------------------------------------------------------------------
    # Rotation-vector decomposition produced internally by LQI.
    # LQI splines a single 3D vector r(u) = theta(u) * n_hat(u). We plot
    # both the raw components r_x, r_y, r_z and the (theta, n_hat)
    # decomposition recovered from |r| and r/|r|.
    # ------------------------------------------------------------------
    r_dense = np.array([lqi.bspline_interpolator.evaluate(u) for u in u_dense])
    r_wp = np.array([lqi.bspline_interpolator.evaluate(u) for u in times_shown])

    theta_dense = np.linalg.norm(r_dense, axis=1)
    theta_wp = np.linalg.norm(r_wp, axis=1)

    # Unit axis from r / |r|, with a safe fallback near theta ~ 0.
    eps_axis = 1e-12
    safe_theta_dense = np.where(theta_dense > eps_axis, theta_dense, 1.0)
    nhat_dense = r_dense / safe_theta_dense[:, None]
    nhat_dense[theta_dense <= eps_axis] = np.array([1.0, 0.0, 0.0])

    safe_theta_wp = np.where(theta_wp > eps_axis, theta_wp, 1.0)
    nhat_wp = r_wp / safe_theta_wp[:, None]
    nhat_wp[theta_wp <= eps_axis] = np.array([1.0, 0.0, 0.0])

    _fig3, axes3 = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    component_labels = [r"$\theta = \|r\|$ [rad]", r"$\hat n_x$", r"$\hat n_y$", r"$\hat n_z$"]
    component_data = [theta_dense, nhat_dense[:, 0], nhat_dense[:, 1], nhat_dense[:, 2]]
    waypoint_data = [theta_wp, nhat_wp[:, 0], nhat_wp[:, 1], nhat_wp[:, 2]]
    component_colors = ["tab:orange", "tab:red", "tab:green", "tab:blue"]

    for ax, label, curve, wp_vals, c in zip(
        axes3, component_labels, component_data, waypoint_data, component_colors
    ):
        ax.plot(u_dense, curve, color=c, linewidth=2.0, label="LQI")
        ax.scatter(times_shown, wp_vals, color="magenta", s=25, zorder=5, label="waypoints")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes3[-1].set_xlabel("Parameter u")
    axes3[0].set_title(
        r"LQI internal state recovered from $r(u) = \theta(u)\,\hat n(u)$"
    )
    plt.tight_layout()

    # Raw rotation-vector components r_x, r_y, r_z (what LQI actually splines).
    _fig3b, axes3b = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    raw_labels = [r"$r_x$", r"$r_y$", r"$r_z$"]
    raw_colors = ["tab:red", "tab:green", "tab:blue"]
    for ax, label, k, c in zip(axes3b, raw_labels, range(3), raw_colors):
        ax.plot(u_dense, r_dense[:, k], color=c, linewidth=2.0, label="LQI")
        ax.scatter(times_shown, r_wp[:, k], color="magenta", s=25, zorder=5, label="waypoints")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes3b[-1].set_xlabel("Parameter u")
    axes3b[0].set_title(r"LQI spline state: rotation-vector components $r(u)$")
    plt.tight_layout()

    # ------------------------------------------------------------------
    # Physical angular velocity (omega) and acceleration (alpha) in 3D.
    # ------------------------------------------------------------------
    omega = np.zeros((n_dense, 3))
    alpha = np.zeros((n_dense, 3))
    for i, u in enumerate(u_dense):
        omega[i], alpha[i] = lqi.get_physical_kinematics(u)

    omega_norm = np.linalg.norm(omega, axis=1)
    alpha_norm = np.linalg.norm(alpha, axis=1)

    _fig4, (ax_w, ax_a) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    for k, axis_name, c in zip(range(3), ("x", "y", "z"), ("tab:red", "tab:green", "tab:blue")):
        ax_w.plot(u_dense, omega[:, k], color=c, linewidth=1.8, label=rf"$\omega_{axis_name}$")
        ax_a.plot(u_dense, alpha[:, k], color=c, linewidth=1.8, label=rf"$\alpha_{axis_name}$")

    ax_w.plot(u_dense, omega_norm, color="black", linewidth=1.2, linestyle="--", label=r"$\|\omega\|$")
    ax_a.plot(u_dense, alpha_norm, color="black", linewidth=1.2, linestyle="--", label=r"$\|\alpha\|$")

    ax_w.set_ylabel(r"Angular velocity $\omega$ [rad/u]")
    ax_w.set_title("Physical angular velocity and acceleration from LQI")
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
