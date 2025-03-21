import numpy as np

from interpolatepy.b_spline import BSpline
from interpolatepy.tridiagonal_inv import solve_tridiagonal


class CubicBSplineInterpolation(BSpline):
    """
    A class for cubic B-spline interpolation of a set of points.

    This class implements a global interpolation algorithm for creating a cubic
    B-spline curve that passes through all specified points with C² continuity.
    It inherits from the BSpline class and extends it with interpolation capabilities.

    Parameters
    ----------
    points : array_like
        The points to interpolate. Should be a 2D array of shape (n, d) where n is
        the number of points and d is the dimension, or a 1D array for single-dimensional points.
    v0 : array_like, optional
        Initial endpoint derivative vector. If None and auto_derivatives is True,
        it will be calculated as (q₁-q₀)/(ū₁-ū₀). If None and auto_derivatives
        is False, zero derivative is used. Default is None.
    vn : array_like, optional
        Final endpoint derivative vector. If None and auto_derivatives is True,
        it will be calculated as (qₙ-qₙ₋₁)/(ūₙ-ūₙ₋₁). If None and auto_derivatives
        is False, zero derivative is used. Default is None.
    method : {'equally_spaced', 'chord_length', 'centripetal'}, optional
        Method for calculating the parameters ūₖ. Default is 'chord_length'.
    auto_derivatives : bool, optional
        Whether to automatically calculate derivatives when not provided.
        Default is False.

    Attributes
    ----------
    interpolation_points : ndarray
        The points used for interpolation.
    n_interpolation_points : int
        The number of interpolation points.
    u_bars : ndarray
        The parameters ūₖ calculated using the specified method.
    v0 : ndarray
        The initial endpoint derivative vector.
    vn : ndarray
        The final endpoint derivative vector.

    Notes
    -----
    The implementation follows the global interpolation algorithm described in
    Section 8.4.2 of "The NURBS Book" by Piegl and Tiller.

    See Also
    --------
    BSpline : Parent class for basic B-spline functionality.
    """

    # Define constants
    PARAM_DIFF_THRESHOLD = 1e-10
    MIN_POINTS_FOR_TRIDIAGONAL = 2

    def __init__(
        self,
        points: list | np.ndarray,
        v0: list | np.ndarray | None = None,
        vn: list | np.ndarray | None = None,
        method: str = "chord_length",
        auto_derivatives: bool = False,
    ) -> None:
        """
        Initialize a cubic B-spline interpolation of a set of points.

        Parameters
        ----------
        points : array_like
            The points to interpolate. Should be a 2D array of shape (n, d) where n is
            the number of points and d is the dimension, or a 1D array for single-dimensional points
        v0 : array_like, optional
            Initial endpoint derivative vector. If None and auto_derivatives is True,
            it will be calculated as (q₁-q₀)/(ū₁-ū₀). If None and auto_derivatives
            is False, zero derivative is used. Default is None.
        vn : array_like, optional
            Final endpoint derivative vector. If None and auto_derivatives is True,
            it will be calculated as (qₙ-qₙ₋₁)/(ūₙ-ūₙ₋₁). If None and auto_derivatives
            is False, zero derivative is used. Default is None.
        method : {'equally_spaced', 'chord_length', 'centripetal'}, optional
            Method for calculating the parameters ūₖ. Default is 'chord_length'.
        auto_derivatives : bool, optional
            Whether to automatically calculate derivatives when not provided.
            Default is False.

        Returns
        -------
        None

        Notes
        -----
        This initializer preprocesses the input points and endpoint derivatives,
        calculates the parameters ūₖ, the knot vector, and the control points,
        then initializes the parent BSpline class with the computed values.
        """
        # Convert points to numpy array and ensure correct format
        if not isinstance(points, np.ndarray):
            points = np.array(points, dtype=np.float64)
        else:
            points = points.astype(np.float64)

        # Get the number of points and the dimension
        n_points = len(points)
        if points.ndim == 1:
            dimension = 1
            # Reshape for a single point
            points = points.reshape(-1, 1)
        else:
            dimension = points.shape[1]

        # Store the interpolation-specific properties
        self.interpolation_points = points
        self.n_interpolation_points = n_points

        # Calculate the parameters ūₖ
        self.u_bars = self._calculate_parameters(method)

        # Process endpoint derivatives
        n = self.n_interpolation_points - 1  # Index of the last point

        # If v0 is None, calculate it or set to zero vector
        if v0 is None:
            if auto_derivatives and n > 0:
                # Calculate v0 = (q₁ - q₀) / (ū₁ - ū₀)
                u_diff = self.u_bars[1] - self.u_bars[0]
                if abs(u_diff) > self.PARAM_DIFF_THRESHOLD:  # Avoid division by zero
                    self.v0 = (points[1] - points[0]) / u_diff
                else:
                    self.v0 = np.zeros(dimension, dtype=np.float64)
            else:
                self.v0 = np.zeros(dimension, dtype=np.float64)
        else:
            # Convert to numpy array if it's not already
            if not isinstance(v0, np.ndarray):
                v0 = np.array(v0, dtype=np.float64)
            else:
                v0 = v0.astype(np.float64)

            # Ensure correct shape
            if v0.ndim == 0:  # scalar
                self.v0 = np.zeros(dimension, dtype=np.float64)
            elif v0.ndim == 1 and len(v0) == dimension:
                self.v0 = v0
            else:
                raise ValueError(f"v0 must be a vector of dimension {dimension}")

        # If vn is None, calculate it or set to zero vector
        if vn is None:
            if auto_derivatives and n > 0:
                # Calculate vn = (qₙ - qₙ₋₁) / (ūₙ - ūₙ₋₁)
                u_diff = self.u_bars[n] - self.u_bars[n - 1]
                if abs(u_diff) > self.PARAM_DIFF_THRESHOLD:  # Avoid division by zero
                    self.vn = (points[n] - points[n - 1]) / u_diff
                else:
                    self.vn = np.zeros(dimension, dtype=np.float64)
            else:
                self.vn = np.zeros(dimension, dtype=np.float64)
        else:
            # Convert to numpy array if it's not already
            if not isinstance(vn, np.ndarray):
                vn = np.array(vn, dtype=np.float64)
            else:
                vn = vn.astype(np.float64)

            # Ensure correct shape
            if vn.ndim == 0:  # scalar
                self.vn = np.zeros(dimension, dtype=np.float64)
            elif vn.ndim == 1 and len(vn) == dimension:
                self.vn = vn
            else:
                raise ValueError(f"vn must be a vector of dimension {dimension}")

        # Calculate the knot vector
        knots = self._calculate_knot_vector()

        # Calculate the control points
        control_points = self._calculate_control_points(knots)

        # Initialize the parent BSpline with cubic degree (3)
        super().__init__(3, knots, control_points)

    # The rest of the class implementation remains unchanged
    def _calculate_parameters(self, method: str) -> np.ndarray:
        """
        Calculate the parameters ūₖ for each point.

        Parameters
        ----------
        method : {'equally_spaced', 'chord_length', 'centripetal'}
            Method for calculating the parameters:
            - 'equally_spaced': Parameters are evenly spaced between 0 and 1.
            - 'chord_length': Parameters are proportional to the cumulative chord length.
            - 'centripetal': Parameters are proportional to the cumulative chord length
              raised to the power of mu (0.5).

        Returns
        -------
        ndarray
            The parameters ūₖ with shape (n,) where n is the number of interpolation points.

        Notes
        -----
        The endpoints are always set to ū₀ = 0 and ūₙ = 1.

        For 'chord_length' method, the parameter spacing is proportional to the distance
        between interpolation points.

        For 'centripetal' method, a value of mu = 0.5 is used as recommended in the literature
        for better shape preservation with non-uniform data.

        Raises
        ------
        ValueError
            If an unknown method is provided.
        """
        n = self.n_interpolation_points - 1  # Index of the last point

        # Initialize the parameters
        u_bars = np.zeros(self.n_interpolation_points, dtype=np.float64)

        # Set the endpoints (equation 8.12)
        u_bars[0] = 0.0
        u_bars[n] = 1.0

        if method == "equally_spaced":
            # Equally spaced parameters (equation 8.12)
            for k in range(1, n):
                u_bars[k] = k / n

        elif method == "chord_length":
            # Chord length distribution (equation 8.13)
            # Calculate total chord length
            total_length = 0.0
            for k in range(1, n + 1):
                total_length += np.linalg.norm(
                    self.interpolation_points[k] - self.interpolation_points[k - 1]
                )

            # Calculate parameters
            for k in range(1, n):
                u_bars[k] = (
                    u_bars[k - 1]
                    + np.linalg.norm(
                        self.interpolation_points[k] - self.interpolation_points[k - 1]
                    )
                    / total_length
                )

        elif method == "centripetal":
            # Centripetal distribution (equation 8.14)
            mu = 0.5  # As recommended in the document

            # Calculate total "centripetal" length
            total_length = 0.0
            for k in range(1, n + 1):
                total_length += (
                    np.linalg.norm(self.interpolation_points[k] - self.interpolation_points[k - 1])
                    ** mu
                )

            # Calculate parameters
            for k in range(1, n):
                u_bars[k] = (
                    u_bars[k - 1]
                    + np.linalg.norm(
                        self.interpolation_points[k] - self.interpolation_points[k - 1]
                    )
                    ** mu
                    / total_length
                )

        else:
            raise ValueError(
                f"Unknown method: {method}. Options are 'equally_spaced', 'chord_length', "
                f"or 'centripetal'."
            )

        return u_bars

    def _calculate_knot_vector(self) -> np.ndarray:
        """
        Calculate the knot vector based on the parameters ūₖ.

        Returns
        -------
        ndarray
            The knot vector with shape (n+7,) where n is the number of interpolation
            points minus 1. The first 3 knots are equal to ū₀, the last 3 knots
            are equal to ūₙ, and the middle knots are set to the parameters ūⱼ for
            j = 0, ..., n.

        Notes
        -----
        This follows equation (8.15) from "The NURBS Book":
        - t₀ = t₁ = t₂ = ū₀
        - tⱼ₊₃ = ūⱼ for j = 0,...,n
        - tₙ₊₄ = tₙ₊₅ = tₙ₊₆ = ūₙ

        This knot vector ensures that the cubic B-spline curve passes through the
        interpolation points.
        """
        n = self.n_interpolation_points - 1  # Index of the last point

        # Create the knot vector with n+7 elements (as per equation 8.15)
        knots = np.zeros(n + 7, dtype=np.float64)

        # Set the first 3 knots to ū₀ (equation 8.15)
        knots[0:3] = self.u_bars[0]

        # Set the last 3 knots to ūₙ (equation 8.15)
        knots[-3:] = self.u_bars[n]

        # Set the middle knots to ūⱼ for j = 0, ..., n (equation 8.15)
        for j in range(n + 1):
            knots[j + 3] = self.u_bars[j]

        return knots

    def _calculate_control_points(self, knots: np.ndarray) -> np.ndarray:
        """
        Calculate the control points by solving a system of equations.

        Parameters
        ----------
        knots : ndarray
            The knot vector with shape (n+7,) where n is the number of interpolation
            points minus 1.

        Returns
        -------
        ndarray
            The control points with shape (n+3, d) where n is the number of
            interpolation points minus 1 and d is the dimension.

        Notes
        -----
        This follows the algorithm from "The NURBS Book" section 8.4.2:

        First, the first and last two control points are calculated directly:
        - p₀ = q₀ (first interpolation point)
        - p₁ = q₀ + (t₄/3) * v₀ (using first derivative)
        - pₙ₊₁ = qₙ - ((1-tₙ₊₂)/3) * vₙ (using last derivative)
        - pₙ₊₂ = qₙ (last interpolation point)

        Then, for the remaining control points p₂, p₃, ..., pₙ, a tridiagonal system
        is solved based on the requirement that the curve passes through all
        interpolation points.

        If there are only two interpolation points, only the directly calculated
        control points are needed.
        """
        n = self.n_interpolation_points - 1  # Index of the last point
        dimension = self.interpolation_points.shape[1]

        # Initialize the control points array with n+3 elements (p₀ to pₙ₊₂)
        control_points = np.zeros((n + 3, dimension), dtype=np.float64)

        # Calculate p₀, p₁, pₙ₊₁, and pₙ₊₂ directly from equation (8.16)
        control_points[0] = self.interpolation_points[0]

        # Calculate p₁ using v0 (scaled by the knot spacing)
        control_points[1] = self.interpolation_points[0] + (knots[4] / 3.0) * self.v0

        # Calculate pₙ₊₁ using vn (scaled by the knot spacing)
        control_points[n + 1] = (
            self.interpolation_points[n] - ((1.0 - knots[n + 2]) / 3.0) * self.vn
        )

        control_points[n + 2] = self.interpolation_points[n]

        # If there are only two points to interpolate, we're done
        if n < self.MIN_POINTS_FOR_TRIDIAGONAL:
            return control_points

        # For more than two points, solve the tridiagonal system for the remaining control points
        lower_diagonal = np.zeros(n - 1, dtype=np.float64)
        main_diagonal = np.zeros(n - 1, dtype=np.float64)
        upper_diagonal = np.zeros(n - 1, dtype=np.float64)
        right_hand_side = np.zeros((n - 1, dimension), dtype=np.float64)

        # Create a temporary BSpline for calculating basis functions
        temp_control = np.zeros((n + 3, dimension), dtype=np.float64)
        temp_bs = BSpline(3, knots, temp_control)

        # Fill the tridiagonal matrix and right-hand side
        for i in range(n - 1):
            k = i + 1  # Point index (from 1 to n-1)

            # The parameter ūₖ
            u_bar = self.u_bars[k]

            # Find the knot span for ūₖ
            span = temp_bs.find_knot_span(u_bar)

            # Calculate the basis functions at ūₖ
            basis_vals = temp_bs.basis_functions(u_bar, span)

            # The basis function values we need
            b3_k = basis_vals[0]
            b3_k1 = basis_vals[1]
            b3_k2 = basis_vals[2]

            # Fill the tridiagonal matrix
            if k == 1:
                # First row
                main_diagonal[0] = b3_k1
                upper_diagonal[0] = b3_k2
                right_hand_side[0] = self.interpolation_points[k] - b3_k * control_points[1]
            elif k == n - 1:
                # Last row
                lower_diagonal[k - 2] = b3_k
                main_diagonal[k - 1] = b3_k1
                right_hand_side[k - 1] = (
                    self.interpolation_points[k] - b3_k2 * control_points[n + 1]
                )
            else:
                # Middle rows
                lower_diagonal[k - 2] = b3_k
                main_diagonal[k - 1] = b3_k1
                upper_diagonal[k - 1] = b3_k2
                right_hand_side[k - 1] = self.interpolation_points[k]

        # Solve the tridiagonal system for each dimension
        for d in range(dimension):
            # Extract the right-hand side for this dimension
            rhs = right_hand_side[:, d]

            # Adjust the lower_diagonal for the tridiagonal solver
            l_diag = np.zeros(n - 1, dtype=np.float64)
            if n - 1 > 1:
                l_diag[1:] = lower_diagonal[:-1]

            # Solve the system
            solution = solve_tridiagonal(l_diag, main_diagonal, upper_diagonal, rhs)

            # Set the control points
            control_points[2 : n + 1, d] = solution

        return control_points
