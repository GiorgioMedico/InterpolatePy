"""
B-spline curve interpolation through specified points.

This module implements exact B-spline interpolation where the curve passes through
all specified data points. The interpolation constructs smooth curves with precise
control over continuity and boundary conditions.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from mpl_toolkits.mplot3d import Axes3D

from .core import BSpline
from ._interpolation_system import compute_control_points

CUBIC_DEGREE = 3
QUARTIC_DEGREE = 4
QUINTIC_DEGREE = 5
VALID_DEGREES = {CUBIC_DEGREE, QUARTIC_DEGREE, QUINTIC_DEGREE}
MAX_POINTS_FOR_LABELS = 10
TWO_DIMENSIONAL = 2
THREE_DIMENSIONAL = 3


class BSplineInterpolator(BSpline):
    """A B-spline that interpolates a set of points with specified degrees of continuity.

    This class inherits from BSpline and computes the knot vector and control points
    required to interpolate given data points at specified times, while maintaining
    desired continuity constraints.

    The degree-specific knot and boundary-row construction supports:
    - Cubic splines (degree 3) with C² continuity
    - Quartic splines (degree 4) with C³ continuity (continuous jerk)
    - Quintic splines (degree 5) with C⁴ continuity (continuous snap)

    Points can be of any dimension, including 2D and 3D.
    """

    def __init__(  # noqa: PLR0913
        self,
        degree: int,
        points: list | np.ndarray,
        times: list | np.ndarray | None = None,
        initial_velocity: list | np.ndarray | None = None,
        final_velocity: list | np.ndarray | None = None,
        initial_acceleration: list | np.ndarray | None = None,
        final_acceleration: list | np.ndarray | None = None,
        cyclic: bool = False,
    ) -> None:
        """Initialize a B-spline interpolator.

        Parameters
        ----------
        degree : int
            The degree of the B-spline (3, 4, or 5).
        points : list or numpy.ndarray
            The points to be interpolated.
        times : list or numpy.ndarray or None, optional
            The time instants for each point. If None, uses uniform spacing.
        initial_velocity : list or numpy.ndarray or None, optional
            Initial velocity constraint.
        final_velocity : list or numpy.ndarray or None, optional
            Final velocity constraint.
        initial_acceleration : list or numpy.ndarray or None, optional
            Initial acceleration constraint.
        final_acceleration : list or numpy.ndarray or None, optional
            Final acceleration constraint.
        cyclic : bool, default=False
            Whether to use cyclic (periodic) conditions.

        Raises
        ------
        ValueError
            If the degree is not 3, 4, or 5.
        ValueError
            If there are not enough points for the specified degree.
        """
        # Validate inputs
        if degree not in VALID_DEGREES:
            raise ValueError(f"Degree must be 3, 4, or 5, got {degree}")

        # Convert inputs to numpy arrays
        if not isinstance(points, np.ndarray):
            points = np.array(points, dtype=np.float64)

        # Ensure points are 2D
        if points.ndim == 1:
            # For 1D points, reshape to column vector
            points = points.reshape(-1, 1)

        # Validate number of points relative to degree
        num_points = len(points)
        # The system has n+1 interpolation rows plus p-1 (odd p) or p (even p)
        # boundary rows against the same number of control points, so it stays
        # square and full rank down to two points.
        min_points = 2
        if num_points < min_points:
            raise ValueError(
                f"Not enough points for degree {degree} B-spline interpolation. "
                f"Need at least {min_points} points, but got {num_points}. "
                f"Either reduce the degree or provide more points."
            )

        # Set up time sequence if not provided
        if times is None:
            times = np.arange(len(points), dtype=np.float64)
        elif not isinstance(times, np.ndarray):
            times = np.array(times, dtype=np.float64)

        # Store attributes used for interpolation
        self.interp_points = points.copy()
        self.times = times
        self.initial_velocity = initial_velocity
        self.final_velocity = final_velocity
        self.initial_acceleration = initial_acceleration
        self.final_acceleration = final_acceleration
        self.cyclic = cyclic

        # Compute knots and control points for interpolation
        knots = self._create_knot_vector(degree, points, times)

        # Create a temporary BSpline to use its methods for basis function calculations
        # Use a single control point since we only need it for basis functions
        temp_control_points = np.zeros((len(knots) - degree - 1, 1))
        self.temp_spline = BSpline(degree, knots, temp_control_points)

        # Compute control points using the temporary BSpline
        control_points = self._compute_control_points(degree, points, times)

        # Initialize the base BSpline class with computed values
        super().__init__(degree, knots, control_points)

    @staticmethod
    def _create_knot_vector(degree: int, points: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Create the knot vector based on the degree.

        - For odd degrees (3, 5): knots at interpolation points (eq. 4.42)
        - For even degrees (4): knots at midpoints (eq. 4.43)

        Parameters
        ----------
        degree : int
            The degree of the B-spline.
        points : numpy.ndarray
            The points to be interpolated.
        times : numpy.ndarray
            The time instants for each point.

        Returns
        -------
        numpy.ndarray
            The computed knot vector.
        """
        n = len(points) - 1  # n segments (n+1 points)
        p = degree

        if p % 2 == 1:  # Odd degree (3, 5): knots at points
            # Odd degrees place interior knots at sample parameters.
            # u = [t0, ..., t0, t1, ..., tn-1, tn, ..., tn]
            #      p+1 times         p+1 times

            knots = np.zeros(n + 2 * p + 1)  # Total knots: n + 2p + 1

            # Set first p+1 knots to t0
            knots[: p + 1] = times[0]

            # Set internal knots to interpolation points
            knots[p + 1 : p + 1 + n - 1] = times[1:-1]

            # Set last p+1 knots to tn
            knots[p + n :] = times[-1]
        else:  # Even degree (4): knots at midpoints
            # Even degrees place interior knots at parameter midpoints.
            # u = [t0, ..., t0, (t0+t1)/2, ..., (tn-1+tn)/2, tn, ..., tn]
            #      p+1 times                  p+1 times

            knots = np.zeros(n + 2 * p + 2)  # Total knots: n + 2p + 2

            # Set first p+1 knots to t0
            knots[: p + 1] = times[0]

            # Set internal knots to midpoints between interpolation points
            for i in range(n):
                knots[p + 1 + i] = (times[i] + times[i + 1]) / 2.0

            # Set last p+1 knots to tn
            knots[p + 1 + n :] = times[-1]

        return knots

    def _compute_control_points(
        self, degree: int, points: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        """Build and solve the interpolation system for its control points."""
        return compute_control_points(self, degree, points, times)

    def plot_with_points(
        self,
        num_points: int = 100,
        show_control_polygon: bool = True,
        ax: Axes | None = None,
    ) -> Axes:
        """Plot the 2D B-spline curve along with the interpolation points.

        Parameters
        ----------
        num_points : int, default=100
            Number of points to generate for the curve.
        show_control_polygon : bool, default=True
            Whether to show the control polygon.
        ax : matplotlib.axes.Axes or None, optional
            Optional matplotlib axis to use.

        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axis object.

        Raises
        ------
        ValueError
            If points are not 2D.
        """
        if self.interp_points.shape[1] != TWO_DIMENSIONAL:
            raise ValueError(f"Points must be 2D for this plot, got {self.interp_points.shape[1]}D")

        # Plot the B-spline using the parent class method
        ax = self.plot_2d(num_points=num_points, show_control_polygon=show_control_polygon, ax=ax)

        # Add interpolation points
        ax.plot(
            self.interp_points[:, 0],
            self.interp_points[:, 1],
            "go",
            markersize=8,
            label="Interpolation points",
        )

        # Add time labels if not too many points
        if len(self.interp_points) <= MAX_POINTS_FOR_LABELS:
            for i, (x, y) in enumerate(self.interp_points):
                ax.text(x, y + 0.1, f"t={self.times[i]:.1f}", horizontalalignment="center")

        ax.legend()
        return ax

    def plot_with_points_3d(
        self,
        num_points: int = 100,
        show_control_polygon: bool = True,
        ax: Axes3D | None = None,
    ) -> Axes3D:
        """Plot the 3D B-spline curve along with the interpolation points.

        Parameters
        ----------
        num_points : int, default=100
            Number of points to generate for the curve.
        show_control_polygon : bool, default=True
            Whether to show the control polygon.
        ax : matplotlib.axes.Axes or None, optional
            Optional matplotlib 3D axis to use.

        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib 3D axis object.

        Raises
        ------
        ValueError
            If points are not 3D.
        """
        if self.interp_points.shape[1] != THREE_DIMENSIONAL:
            raise ValueError(f"Points must be 3D for this plot, got {self.interp_points.shape[1]}D")

        # Plot the B-spline using the parent class method
        ax = self.plot_3d(num_points=num_points, show_control_polygon=show_control_polygon, ax=ax)

        # Add interpolation points
        ax.scatter(
            self.interp_points[:, 0],
            self.interp_points[:, 1],
            self.interp_points[:, 2],  # pyright: ignore[reportArgumentType]
            color="g",
            s=64,
            label="Interpolation points",
        )

        # Add time labels if not too many points
        if len(self.interp_points) <= MAX_POINTS_FOR_LABELS:
            for i, (x, y, z) in enumerate(self.interp_points):
                ax.text(x, y, z, f"t={self.times[i]:.1f}")

        ax.legend()
        return ax
