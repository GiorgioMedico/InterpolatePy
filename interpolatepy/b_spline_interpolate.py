import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve

from interpolatepy.b_spline import BSpline


CUBIC_DEGREE = 3
QUARTIC_DEGREE = 4
QUINTIC_DEGREE = 5
VALID_DEGREES = {CUBIC_DEGREE, QUARTIC_DEGREE, QUINTIC_DEGREE}
MAX_POINTS_FOR_LABELS = 10
TWO_DIMENSIONAL = 2
THREE_DIMENSIONAL = 3
EPS_P = 1e12
EPS_N = 1e-10


class BSplineInterpolator(BSpline):
    """A B-spline that interpolates a set of points with specified degrees of continuity.

    This class inherits from BSpline and computes the knot vector and control points
    required to interpolate given data points at specified times, while maintaining
    desired continuity constraints.

    The implementation follows section 4.5 of the document, supporting:
    - Cubic splines (degree 3) with C² continuity
    - Quartic splines (degree 4) with C³ continuity (continuous jerk)
    - Quintic splines (degree 5) with C⁴ continuity (continuous snap)

    Points can be of any dimension, including 2D and 3D.
    """

    def __init__(  # noqa: PLR0913, PLR0917
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
        min_points = degree + 1
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
            # Using equation 4.42 from the document
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
            # Using equation 4.43 from the document
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
        """Compute the control points by solving the linear system.

        The system includes:
        - Interpolation conditions (curve passes through each point)
        - Boundary conditions (velocity/acceleration or cyclic conditions)

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
            The computed control points.

        Raises
        ------
        ValueError
            If the linear system cannot be solved or is ill-conditioned.
        """
        n = len(points) - 1  # Number of segments
        p = degree

        # Determine number of control points based on degree
        num_control_points = (n + 1) + p - 1 if p % 2 == 1 else (n + 1) + p

        # Determine number of additional conditions needed
        num_additional = p if p % 2 == 0 else p - 1

        # Create the linear system: A * P = b
        a_matrix = np.zeros((n + 1 + num_additional, num_control_points))
        b = np.zeros((n + 1 + num_additional, points.shape[1]))

        # Fill the interpolation conditions (points must lie on the curve)
        for i in range(n + 1):
            t = times[i]

            # Find which basis functions are non-zero at this point
            span = self.temp_spline.find_knot_span(t)
            basis_values = self.temp_spline.basis_functions(t, span)

            for j in range(p + 1):
                col = span - p + j
                if 0 <= col < num_control_points:
                    a_matrix[i, col] = basis_values[j]

            # The right side is the point to interpolate
            b[i] = points[i]

        # Add boundary conditions
        row = n + 1  # Start adding boundary conditions after interpolation

        if self.cyclic:
            # Add cyclic conditions: derivatives at start = derivatives at end
            for k in range(1, num_additional + 1):
                t0 = times[0]
                tn = times[-1]

                span0 = self.temp_spline.find_knot_span(t0)
                spann = self.temp_spline.find_knot_span(tn)

                # Get derivative basis functions
                ders0 = self.temp_spline.basis_function_derivatives(t0, span0, k)
                dersn = self.temp_spline.basis_function_derivatives(tn, spann, k)

                # Fill the derivative constraint: s^(k)(t0) - s^(k)(tn) = 0
                for j in range(p + 1):
                    col0 = span0 - p + j
                    if 0 <= col0 < num_control_points:
                        a_matrix[row, col0] = ders0[k, j]

                    coln = spann - p + j
                    if 0 <= coln < num_control_points:
                        a_matrix[row, coln] = -dersn[k, j]

                # Right side is zero for cyclic conditions
                # b[row] already initialized to zero

                row += 1
                if row >= n + 1 + num_additional:
                    break
        else:
            # Add velocity constraints if provided
            if self.initial_velocity is not None and row < n + 1 + num_additional:
                t = times[0]
                span = self.temp_spline.find_knot_span(t)
                ders = self.temp_spline.basis_function_derivatives(t, span, 1)

                for j in range(p + 1):
                    col = span - p + j
                    if 0 <= col < num_control_points:
                        a_matrix[row, col] = ders[1, j]

                b[row] = self.initial_velocity
                row += 1

            if self.final_velocity is not None and row < n + 1 + num_additional:
                t = times[-1]
                span = self.temp_spline.find_knot_span(t)
                ders = self.temp_spline.basis_function_derivatives(t, span, 1)

                for j in range(p + 1):
                    col = span - p + j
                    if 0 <= col < num_control_points:
                        a_matrix[row, col] = ders[1, j]

                b[row] = self.final_velocity
                row += 1

            # Add acceleration constraints if provided
            if self.initial_acceleration is not None and row < n + 1 + num_additional:
                t = times[0]
                span = self.temp_spline.find_knot_span(t)
                ders = self.temp_spline.basis_function_derivatives(t, span, 2)

                for j in range(p + 1):
                    col = span - p + j
                    if 0 <= col < num_control_points:
                        a_matrix[row, col] = ders[2, j]

                b[row] = self.initial_acceleration
                row += 1

            if self.final_acceleration is not None and row < n + 1 + num_additional:
                t = times[-1]
                span = self.temp_spline.find_knot_span(t)
                ders = self.temp_spline.basis_function_derivatives(t, span, 2)

                for j in range(p + 1):
                    col = span - p + j
                    if 0 <= col < num_control_points:
                        a_matrix[row, col] = ders[2, j]

                b[row] = self.final_acceleration
                row += 1

            # If we still need more constraints, add natural spline conditions
            # (zero second derivatives at endpoints)
            while row < n + 1 + num_additional:
                # For cubic splines, use zero second derivatives
                # For higher degree, can use higher derivatives
                deriv_order = min(p - 1, 2)

                # Alternate between initial and final endpoints
                t = times[0] if row % 2 == 0 else times[-1]

                span = self.temp_spline.find_knot_span(t)
                ders = self.temp_spline.basis_function_derivatives(t, span, deriv_order)

                for j in range(p + 1):
                    col = span - p + j
                    if 0 <= col < num_control_points:
                        a_matrix[row, col] = ders[deriv_order, j]

                # Right side is zero (natural spline condition)
                # b[row] already initialized to zero

                row += 1

        # Check if the system is well-posed
        if np.linalg.matrix_rank(a_matrix) < min(a_matrix.shape):
            raise ValueError(
                "Linear system is rank-deficient. This typically occurs when there "
                "are too few points for the specified degree and constraints. "
                "Add more points or reduce the polynomial degree."
            )

        # Solve the linear system for each coordinate
        try:
            # Check if the system is well-conditioned
            condition_number = np.linalg.cond(a_matrix)
            if condition_number > EPS_P:
                print(
                    f"Warning: The linear system is ill-conditioned "
                    f"(condition number: {condition_number:.2e})"
                )
                print("This may lead to numerical inaccuracies in the spline interpolation.")
                print(
                    "Consider adding more points, using a lower degree, "
                    "or adjusting the time distribution."
                )

                # Add a small regularization term for stability
                if condition_number > EPS_P:
                    epsilon = EPS_N
                    a_matrix += epsilon * np.eye(a_matrix.shape[0], a_matrix.shape[1])
                    print(
                        f"Adding regularization (epsilon={epsilon}) to improve numerical stability."
                    )

            # Solve for control points
            control_points = np.zeros((num_control_points, points.shape[1]))
            for dim in range(points.shape[1]):
                control_points[:, dim] = solve(a_matrix, b[:, dim])

            return control_points  # noqa: TRY300

        except np.linalg.LinAlgError as e:
            recommended_points = degree + 2  # Safe minimum
            current_points = len(points)

            error_msg = f"Failed to solve for control points: {e}\n"
            error_msg += "This is likely due to an ill-posed interpolation problem.\n"
            error_msg += (
                f"For degree {degree} B-splines, you should have at least "
                f"{recommended_points} points "
            )
            error_msg += f"(you provided {current_points}).\n"

            if self.cyclic:
                error_msg += "When using cyclic conditions, you may need even more points.\n"

            if (
                self.initial_velocity is not None
                or self.final_velocity is not None
                or self.initial_acceleration is not None
                or self.final_acceleration is not None
            ):
                error_msg += (
                    "When specifying velocity or acceleration constraints, "
                    "you may need more points.\n"
                )

            if degree in {QUARTIC_DEGREE, QUINTIC_DEGREE}:
                error_msg += (
                    f"Consider using a lower degree (e.g., degree=3) "
                    f"with {current_points} points.\n"
                )

            raise ValueError(error_msg) from e

    def plot_with_points(
        self, num_points: int = 100, show_control_polygon: bool = True, ax: plt.Axes | None = None
    ) -> plt.Axes:
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
        self, num_points: int = 100, show_control_polygon: bool = True, ax: plt.Axes | None = None
    ) -> plt.Axes:
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
            self.interp_points[:, 2],
            color="g",
            s=64,
            label="Interpolation points",
        )

        # Add time labels if not too many points
        if len(self.interp_points) <= MAX_POINTS_FOR_LABELS:
            for i, (x, y, z) in enumerate(self.interp_points):
                ax.text(x, y, z, f"t={self.times[i]:.1f}", horizontalalignment="center")

        ax.legend()
        return ax
