import numpy as np

from interpolatepy.b_spline import BSpline


class ApproximationBSpline(BSpline):
    """A class for B-spline curve approximation of a set of points.

    Inherits from BSpline class.

    The approximation follows the theory described in Section 8.5 of the reference:
    - The end points are exactly interpolated
    - The internal points are approximated in the least squares sense
    - Degree 3 (cubic) is typically used to ensure C2 continuity

    Attributes
    ----------
    original_points : np.ndarray
        The original points being approximated.
    original_parameters : np.ndarray
        The parameter values corresponding to original points.
    """

    def __init__(  # noqa: PLR0913
        self,
        points: list | np.ndarray,
        num_control_points: int,
        *,  # Make remaining parameters keyword-only
        degree: int = 3,
        weights: list | np.ndarray | None = None,
        method: str = "chord_length",
        debug: bool = False,
    ) -> None:
        """Initialize an approximation B-spline.

        Parameters
        ----------
        points : list or np.ndarray
            The points to approximate.
        num_control_points : int
            The number of control points to use.
        degree : int, default=3
            The degree of the B-spline. Defaults to 3 for cubic.
        weights : list or np.ndarray or None, default=None
            Weights for points in approximation. If None, uniform weighting is used.
        method : str, default="chord_length"
            Method for parameter calculation. Options are 'equally_spaced',
            'chord_length', or 'centripetal'.
        debug : bool, default=False
            Whether to print debug information.

        Raises
        ------
        ValueError
            If inputs do not satisfy approximation requirements.
        """
        # Validate inputs
        if degree < 1:
            raise ValueError("Degree must be at least 1")
        if num_control_points <= degree:
            raise ValueError("Number of control points must be greater than the degree")
        if len(points) <= num_control_points:
            raise ValueError("Number of points must be greater than number of control points")

        # Store debug flag
        self.debug = debug

        if self.debug:
            print("\n" + "=" * 50)
            print("INITIALIZING APPROXIMATION B-SPLINE")
            print(f"Degree: {degree}")
            print(f"Number of control points: {num_control_points}")
            print(f"Number of points to approximate: {len(points)}")
            print(f"Parameterization method: {method}")
            print("=" * 50)

        # Convert points to numpy array if needed
        if not isinstance(points, np.ndarray):
            points = np.array(points, dtype=np.float64)

        # Set default weights if not provided (uniform weights)
        n = len(points) - 1  # Number of points minus 1
        if weights is None:
            weights = np.ones(n - 1)  # Exclude first and last points
        elif not isinstance(weights, np.ndarray):
            weights = np.array(weights, dtype=np.float64)

        # Calculate parameter values for the points
        u_bar = self._compute_parameters(points, method)

        # Calculate knot vector
        knots = self._compute_knots(degree, num_control_points, len(points), u_bar)

        # Calculate control points using least squares approximation
        control_points = self._approximate_control_points(
            points, degree, knots, u_bar, num_control_points, weights
        )

        # Initialize parent class with calculated values
        super().__init__(degree, knots, control_points)

        # Store the original points and parameter values for reference
        self.original_points = points
        self.original_parameters = u_bar

        if self.debug:
            print("\nFINAL RESULTS:")
            print(f"Degree: {degree}")
            print(f"Number of control points: {len(control_points)}")
            print(f"Knot vector: {knots}")
            print("\nControl points:")
            for i, cp in enumerate(control_points):
                print(f"  P{i}: {cp}")

    def _compute_parameters(self, points: np.ndarray, method: str = "chord_length") -> np.ndarray:
        """Calculate parameter values using one of three methods.

        Parameters
        ----------
        points : np.ndarray
            The points to approximate.
        method : str, default="chord_length"
            Method for calculating the parameters. Options are 'equally_spaced',
            'chord_length', or 'centripetal'.

        Returns
        -------
        np.ndarray
            Parameter values for each point, normalized to [0, 1].

        Raises
        ------
        ValueError
            If an unknown method is provided.
        """
        n = len(points) - 1  # Index of the last point

        # Initialize the parameters
        u_bar = np.zeros(n + 1, dtype=np.float64)

        # Set the endpoints
        u_bar[0] = 0.0
        u_bar[n] = 1.0

        if method == "equally_spaced":
            # Equally spaced parameters
            for k in range(1, n):
                u_bar[k] = k / n

        elif method == "chord_length":
            # Chord length distribution
            # Calculate total chord length
            total_length = 0.0
            for k in range(1, n + 1):
                total_length += np.linalg.norm(points[k] - points[k - 1])

            # Calculate parameters
            for k in range(1, n):
                u_bar[k] = u_bar[k - 1] + np.linalg.norm(points[k] - points[k - 1]) / total_length

        elif method == "centripetal":
            # Centripetal distribution
            mu = 0.5  # As recommended in the document

            # Calculate total "centripetal" length
            total_length = 0.0
            for k in range(1, n + 1):
                total_length += np.linalg.norm(points[k] - points[k - 1]) ** mu

            # Calculate parameters
            for k in range(1, n):
                u_bar[k] = (
                    u_bar[k - 1] + np.linalg.norm(points[k] - points[k - 1]) ** mu / total_length
                )

        else:
            raise ValueError(
                f"Unknown method: {method}. Options are 'equally_spaced', "
                f"'chord_length', or 'centripetal'."
            )

        if hasattr(self, "debug") and self.debug:
            print(f"\nPARAMETER VALUES (using '{method}' method):")
            for i, u in enumerate(u_bar):
                print(f"  u_bar[{i}] = {u:.6f}")

        return u_bar

    def _compute_knots(
        self, degree: int, num_control_points: int, num_points: int, u_bar: np.ndarray
    ) -> np.ndarray:
        """Compute knot vector following the algorithm in Section 8.5.1.

        Parameters
        ----------
        degree : int
            The degree of the B-spline.
        num_control_points : int
            The number of control points.
        num_points : int
            The number of points to approximate.
        u_bar : np.ndarray
            Parameter values for the points.

        Returns
        -------
        np.ndarray
            The knot vector.
        """
        # Total number of knots
        num_knots = num_control_points + degree + 1

        if hasattr(self, "debug") and self.debug:
            print("\nKNOT VECTOR CALCULATION:")
            print(f"  Number of knots needed: {num_knots}")

        # Initialize knot vector
        knots = np.zeros(num_knots)

        # Set the first and last knots with multiplicity p+1 to ensure interpolation
        # of end points as specified in the document
        knots[: degree + 1] = u_bar[0]
        knots[-(degree + 1) :] = u_bar[-1]

        # Compute internal knots using the method from Section 8.5.1
        n = num_points - 1  # Number of points minus 1
        m = num_control_points - 1  # Number of control points minus 1

        # Formula from the document: d = (n+1)/(m-p+1)
        d = (n + 1) / (m - degree + 1)

        if hasattr(self, "debug") and self.debug:
            print(f"  d = (n+1)/(m-p+1) = ({n + 1})/({m}-{degree}+1) = {d:.6f}")

        # Compute internal knots using the algorithm from the document
        # For j=1,...,m-p compute:
        # i = floor(j*d)
        # a = j*d - i
        # u_(j+p) = (1-a)ū_(i-1) + aū_i
        for j in range(1, m - degree + 1):
            i = int(j * d)  # floor(j*d)
            alpha = j * d - i  # j*d - floor(j*d)
            knots[j + degree] = (1 - alpha) * u_bar[i - 1] + alpha * u_bar[i]

            if hasattr(self, "debug") and self.debug:
                print(f"  j={j}, i=floor({j}*{d:.6f})={i}, a={alpha:.6f}")
                print(
                    f"  u_{j + degree} = (1-{alpha:.6f})*{u_bar[i - 1]:.6f} + "
                    f"{alpha:.6f}*{u_bar[i]:.6f} = {knots[j + degree]:.6f}"
                )

        if hasattr(self, "debug") and self.debug:
            print("\nFINAL KNOT VECTOR:")
            knot_str = "  ["
            for k in knots:
                knot_str += f"{k:.6f}, "
            knot_str = knot_str[:-2] + "]"
            print(knot_str)

        return knots

    def _approximate_control_points(  # noqa: PLR0917, PLR0913
        self,
        points: np.ndarray,
        degree: int,
        knots: np.ndarray,
        u_bar: np.ndarray,
        num_control_points: int,
        weights: np.ndarray,
    ) -> np.ndarray:
        """Compute control points using least squares approximation.

        Uses the approach described in Section 8.5.

        Parameters
        ----------
        points : np.ndarray
            The points to approximate.
        degree : int
            The degree of the B-spline.
        knots : np.ndarray
            The knot vector.
        u_bar : np.ndarray
            Parameter values for the points.
        num_control_points : int
            The number of control points.
        weights : np.ndarray
            Weights for points in approximation.

        Returns
        -------
        np.ndarray
            The control points.
        """
        n = len(points) - 1  # Number of points minus 1
        m = num_control_points - 1  # Number of control points minus 1

        if hasattr(self, "debug") and self.debug:
            print("\nCONTROL POINTS CALCULATION:")
            print(f"  n = {n} (number of points minus 1)")
            print(f"  m = {m} (number of control points minus 1)")

        # Initialize control points
        control_points = np.zeros((num_control_points, points.shape[1]))

        # First and last control points are fixed to first and last data points
        # as per condition 1 in Section 8.5
        control_points[0] = points[0]
        control_points[m] = points[n]

        if hasattr(self, "debug") and self.debug:
            print("  Fixed control points:")
            print(f"    P_0 = {points[0]}")
            print(f"    P_{m} = {points[n]}")

        # Handle case where we only have 2 control points
        if m <= 1:
            if hasattr(self, "debug") and self.debug:
                print("  Only two control points needed, returning interpolated curve.")
            return control_points

        # Create a temporary B-spline for basis function calculation
        temp_control_points = np.zeros((num_control_points, points.shape[1]))
        temp_bspline = BSpline(degree, knots, temp_control_points)

        # Initialize matrices for internal points
        # B is (n-1) x (m-1) matrix as in equation (8.21)
        b_matrix = np.zeros((n - 1, m - 1))
        # R is (n-1) x d matrix where d is the dimension of points
        r_matrix = np.zeros((n - 1, points.shape[1]))

        if hasattr(self, "debug") and self.debug:
            print(f"  B matrix shape: {b_matrix.shape}")
            print(f"  R matrix shape: {r_matrix.shape}")

        # For each internal parameter value (k=1 to n-1)
        for k in range(1, n):
            u = u_bar[k]

            if hasattr(self, "debug") and self.debug:
                print(f"\n  Processing point {k} at parameter u = {u:.6f}")

            # Calculate all basis function values at parameter u
            # This evaluates all basis functions B_j^p(u_k) for j=0,...,m
            all_basis = np.zeros(m + 1)

            # Find the knot span that contains u
            span = temp_bspline.find_knot_span(u)

            if hasattr(self, "debug") and self.debug:
                print(f"    Knot span for u = {u:.6f} is {span}")

            # Get non-zero basis functions at this parameter
            basis_values = temp_bspline.basis_functions(u, span)

            if hasattr(self, "debug") and self.debug:
                print(f"    Non-zero basis values: {basis_values}")

            # Map the basis functions to the correct indices in all_basis
            # Only p+1 basis functions are non-zero at any parameter value
            for j in range(degree + 1):
                idx = span - degree + j
                if 0 <= idx <= m:
                    all_basis[idx] = basis_values[j]

            if hasattr(self, "debug") and self.debug:
                print("    All basis function values:")
                for j, val in enumerate(all_basis):
                    print(f"      B_{j}^{degree}({u:.6f}) = {val:.6f}")

            # Fill the k-th row of matrix B with values for internal control points (j=1 to m-1)
            # Exactly as in Equation (8.21)
            for j in range(1, m):
                b_matrix[k - 1, j - 1] = all_basis[j]

            # Calculate the k-th row of matrix R
            # R_k = q_k - B_0^p(u_k)q_0 - B_m^p(u_k)q_m
            # This follows directly from the equation after (8.20) in the document
            r_matrix[k - 1] = points[k] - all_basis[0] * points[0] - all_basis[m] * points[n]

            if hasattr(self, "debug") and self.debug:
                print(
                    f"    Row {k - 1} of R matrix = q_{k} - B_0^{degree}({u:.6f})*q_0 - "
                    f"B_{m}^{degree}({u:.6f})*q_{n}"
                )
                print(
                    f"      = {points[k]} - {all_basis[0]:.6f}*{points[0]} - "
                    f"{all_basis[m]:.6f}*{points[n]}"
                )
                print(f"      = {r_matrix[k - 1]}")

        if hasattr(self, "debug") and self.debug:
            print("\n  B matrix:")
            for i in range(b_matrix.shape[0]):
                row = "    ["
                for j in range(b_matrix.shape[1]):
                    row += f"{b_matrix[i, j]:.6f}, "
                row = row[:-2] + "]"
                print(row)

            print("\n  R matrix (first dimension):")
            for i in range(r_matrix.shape[0]):
                row = "    ["
                # Define max columns to display
                max_cols = 3
                for j in range(min(max_cols, r_matrix.shape[1])):
                    row += f"{r_matrix[i, j]:.6f}, "
                if r_matrix.shape[1] > max_cols:
                    row += "..."
                else:
                    row = row[:-2]
                row += "]"
                print(row)

        # Apply weights to minimize the weighted least squares functional in equation (8.19)
        w_matrix = np.diag(weights)

        if hasattr(self, "debug") and self.debug:
            print("\n  Weights:")
            print(f"    {weights}")

        # Compute weighted pseudo-inverse solution according to equation (8.25)
        # B† = (B^T W B)^(-1) B^T W
        btw = np.dot(b_matrix.T, w_matrix)
        btwb = np.dot(btw, b_matrix)
        btwr = np.dot(btw, r_matrix)

        if hasattr(self, "debug") and self.debug:
            print("\n  Calculating pseudo-inverse solution:")
            print(f"    B^T W shape: {btw.shape}")
            print(f"    B^T W B shape: {btwb.shape}")
            print(f"    B^T W R shape: {btwr.shape}")

        # Solve for internal control points: P = B† R
        try:
            # Solve the normal equations for the least squares solution
            internal_control_points = np.linalg.solve(btwb, btwr)

            if hasattr(self, "debug") and self.debug:
                print("    Used np.linalg.solve (direct solution)")
        except np.linalg.LinAlgError:
            # If matrix is singular or poorly conditioned, use pseudo-inverse
            # This provides the least squares solution that minimizes the norm of P
            internal_control_points = np.dot(np.linalg.pinv(btwb), btwr)

            if hasattr(self, "debug") and self.debug:
                print("    Used np.linalg.pinv (matrix was singular or poorly conditioned)")

        # Assign internal control points (p_1 to p_(m-1))
        control_points[1:m] = internal_control_points

        if hasattr(self, "debug") and self.debug:
            print("\n  Calculated internal control points:")
            for i in range(1, m):
                print(f"    P_{i} = {control_points[i]}")

        return control_points

    def calculate_approximation_error(
        self, points: np.ndarray | None = None, u_bar: np.ndarray | None = None
    ) -> float:
        """Calculate the approximation error as the sum of squared distances.

        Computes sum of squared distances between the points and the corresponding
        points on the B-spline.

        Parameters
        ----------
        points : np.ndarray or None, default=None
            The points to compare with the B-spline. If None, the original points
            used for approximation are used.
        u_bar : np.ndarray or None, default=None
            Parameter values for the points. If None, the original parameters are
            used for original points, or computed for new points.

        Returns
        -------
        float
            The sum of squared distances.
        """
        # Use original points and parameters if not provided
        if points is None:
            points = self.original_points
            u_bar = self.original_parameters

        # Calculate the sum of squared distances
        sum_squared_dist = 0.0
        for i, point in enumerate(points):
            # Evaluate the B-spline at the parameter value
            spline_point = self.evaluate(u_bar[i])  # type: ignore

            # Calculate squared distance
            squared_dist = np.sum((point - spline_point) ** 2)
            sum_squared_dist += squared_dist

        return sum_squared_dist

    def refine(
        self, max_error: float = 0.1, max_control_points: int = 100
    ) -> "ApproximationBSpline":
        """Refine the approximation by adding more control points.

        Adds control points until the maximum error is below a threshold or the
        maximum number of control points is reached.

        Parameters
        ----------
        max_error : float, default=0.1
            Maximum acceptable error.
        max_control_points : int, default=100
            Maximum number of control points.

        Returns
        -------
        ApproximationBSpline
            A refined approximation B-spline.
        """
        # Start with the current number of control points
        num_control_points = len(self.control_points)

        # Calculate initial error
        error = self.calculate_approximation_error()

        while error > max_error and num_control_points < max_control_points:
            # Increase the number of control points
            num_control_points += 1

            # Create a new approximation B-spline with more control points
            new_spline = ApproximationBSpline(
                self.original_points, num_control_points, degree=self.degree
            )

            # Calculate the new error
            error = new_spline.calculate_approximation_error()

            # If error is below threshold, return the new spline
            if error <= max_error:
                return new_spline

        # If we've reached max_control_points but error is still above threshold,
        # return the best approximation we have
        return ApproximationBSpline(
            self.original_points, min(num_control_points, max_control_points), degree=self.degree
        )
