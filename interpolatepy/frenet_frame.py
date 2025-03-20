"""
Core functions for computing and visualizing Frenet frames along parametric curves.
"""

from collections.abc import Callable

import numpy as np
from matplotlib.axes import Axes


EPS = 1e-10


def compute_trajectory_frames(
    position_func: Callable[[float], tuple[np.ndarray, np.ndarray, np.ndarray]],
    u_values: np.ndarray,
    tool_orientation: float | tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Frenet frames along a parametric curve and optionally apply tool orientation.

    Parameters
    ----------
    position_func : callable
        A function that takes a parameter u and returns:
        - position p(u)
        - first derivative dp/du
        - second derivative d2p/du2
    u_values : ndarray
        Parameter values at which to compute the frames.
    tool_orientation : float or tuple, optional
        If None: Returns the Frenet frames without modification.
        If float: Angle in radians to rotate around the binormal axis (legacy mode).
        If tuple: (roll, pitch, yaw) angles in radians representing RPY rotations.

    Returns
    -------
    points : ndarray
        Points on the curve with shape (len(u_values), 3).
    frames : ndarray
        Frames at each point [et, en, eb] with shape (len(u_values), 3, 3),
        either Frenet frames or tool frames.
    """
    n_points = len(u_values)
    points = np.zeros((n_points, 3))
    frenet_frames = np.zeros((n_points, 3, 3))

    for i, u in enumerate(u_values):
        # Get position and derivatives from the provided function
        p, dp_du, d2p_du2 = position_func(u)

        # Store the position
        points[i] = p

        # Compute tangent vector (normalized first derivative)
        dp_du_norm = np.linalg.norm(dp_du)
        et = dp_du / dp_du_norm

        # Compute the derivative of tangent vector with respect to u
        # When parameter is not arc length, this formula applies (from footnote 11)
        det_du = d2p_du2 / dp_du_norm - (dp_du * np.dot(dp_du, d2p_du2)) / (dp_du_norm**3)

        # Compute normal vector (normalized derivative of tangent)
        det_du_norm = np.linalg.norm(det_du)

        if det_du_norm < EPS:
            # Handle case of zero or near-zero curvature by selecting any perpendicular vector
            if abs(et[0]) > abs(et[1]):
                en = np.array([et[2], 0, -et[0]])
                en /= np.linalg.norm(en)
            else:
                en = np.array([0, et[2], -et[1]])
                en /= np.linalg.norm(en)
        else:
            en = det_du / det_du_norm

        # Compute binormal vector (cross product) to complete the right-handed frame
        eb = np.cross(et, en)

        # Store the Frenet frame
        frenet_frames[i, :, 0] = et
        frenet_frames[i, :, 1] = en
        frenet_frames[i, :, 2] = eb

    # If tool orientation is specified, apply it to the Frenet frames
    if tool_orientation is not None:
        # Create final frames array
        tool_frames = np.zeros_like(frenet_frames)

        # Generate the appropriate rotation matrix based on the input type
        if isinstance(tool_orientation, int | float):
            # Legacy mode: simple rotation about binormal axis
            alpha = tool_orientation
            r_tool = np.array(
                [[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]]
            )
        else:
            # RPY angles (roll, pitch, yaw) in XYZ order
            roll, pitch, yaw = tool_orientation

            # X rotation (roll)
            r_x = np.array(
                [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
            )

            # Y rotation (pitch)
            r_y = np.array(
                [[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]]
            )

            # Z rotation (yaw)
            r_z = np.array(
                [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
            )

            # Combined rotation matrix in XYZ order
            r_tool = np.dot(r_z, np.dot(r_y, r_x))

        # Apply rotation to each Frenet frame
        for i in range(n_points):
            tool_frames[i] = np.dot(r_tool, frenet_frames[i])

        return points, tool_frames
    return points, frenet_frames


def helicoidal_trajectory_with_derivatives(
    u: float, r: float = 2.0, d: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Helicoidal trajectory function returning position and derivatives.
    This implements equation 8.7 from the textbook.

    Parameters
    ----------
    u : float
        Parameter value.
    r : float, optional
        Radius of the helix, by default 2.0.
    d : float, optional
        Pitch parameter, by default 0.5.

    Returns
    -------
    p : ndarray
        Position vector.
    dp_du : ndarray
        First derivative vector.
    d2p_du2 : ndarray
        Second derivative vector.
    """
    # Position (equation 8.7)
    p = np.array([r * np.cos(u), r * np.sin(u), d * u])

    # First derivative dp/du
    dp_du = np.array([-r * np.sin(u), r * np.cos(u), d])

    # Second derivative d2p/du2
    d2p_du2 = np.array([-r * np.cos(u), -r * np.sin(u), 0])

    return p, dp_du, d2p_du2


def circular_trajectory_with_derivatives(
    u: float, r: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Circular trajectory function returning position and derivatives.

    Parameters
    ----------
    u : float
        Parameter value.
    r : float, optional
        Radius of the circle, by default 2.0.

    Returns
    -------
    p : ndarray
        Position vector.
    dp_du : ndarray
        First derivative vector.
    d2p_du2 : ndarray
        Second derivative vector.
    """
    # Position
    p = np.array([r * np.cos(u), r * np.sin(u), 0])

    # First derivative dp/du
    dp_du = np.array([-r * np.sin(u), r * np.cos(u), 0])

    # Second derivative d2p/du2
    d2p_du2 = np.array([-r * np.cos(u), -r * np.sin(u), 0])

    return p, dp_du, d2p_du2


def plot_frames(  # noqa: PLR0913, PLR0917
    ax: Axes,
    points: np.ndarray,
    frames: np.ndarray,
    scale: float = 0.5,
    skip: int = 5,
    colors: list[str] | None = None,
) -> None:
    """
    Plot the trajectory and frames.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The 3D axes to plot on.
    points : ndarray
        Points on the trajectory with shape (n, 3).
    frames : ndarray
        Frames at each point [et, en, eb] with shape (n, 3, 3).
    scale : float, optional
        Scale factor for the frame vectors, by default 0.5.
    skip : int, optional
        Number of frames to skip between plotted frames, by default 5.
    colors : list of str, optional
        Colors for the three vectors, by default ['r', 'g', 'b'].
    """
    # Plot the trajectory
    if colors is None:
        colors = ["r", "g", "b"]
    ax.plot(points[:, 0], points[:, 1], points[:, 2], "k-")

    # Plot selected frames
    for i in range(0, len(points), skip):
        p = points[i]

        # Plot the three vectors of the frame
        for j in range(3):
            ax.quiver(
                p[0],
                p[1],
                p[2],
                frames[i, 0, j],
                frames[i, 1, j],
                frames[i, 2, j],
                color=colors[j],
                length=scale,
                normalize=True,
            )
