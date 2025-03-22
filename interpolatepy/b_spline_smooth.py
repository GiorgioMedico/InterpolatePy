from dataclasses import dataclass

import numpy as np

from interpolatepy.b_spline import BSpline


# Define constant to replace magic number
EPSILON = 1e-10


@dataclass
class BSplineParams:
    """Parameters for initializing a SmoothingCubicBSpline.

    Attributes
    ----------
    mu : float, default=0.5
        Smoothing parameter between 0 and 1, where higher values give more importance to
        fitting the data points exactly.
    weights : array_like or None, default=None
        Weights for each data point.
    v0 : array_like or None, default=None
        Tangent vector at the start point.
    vn : array_like or None, default=None
        Tangent vector at the end point.
    method : str, default="chord_length"
        Method for parameter calculation ('equally_spaced', 'chord_length', or 'centripetal').
    enforce_endpoints : bool, default=False
        Whether to enforce interpolation at the endpoints.
    auto_derivatives : bool, default=False
        Whether to automatically calculate derivatives at endpoints.
    """

    mu: float = 0.5
    weights: list | np.ndarray | None = None
    v0: list | np.ndarray | None = None
    vn: list | np.ndarray | None = None
    method: str = "chord_length"
    enforce_endpoints: bool = False
    auto_derivatives: bool = False


class SmoothingCubicBSpline(BSpline):
    """A class for creating smoothing cubic B-splines that approximate a set of points.

    This class inherits from BSpline and implements the smoothing algorithm
    described in Section 8.7 of the document. It creates a cubic B-spline
    curve that balances between fitting the given points and maintaining smoothness.
    """

    def __init__(
        self,
        points: list | np.ndarray,
        params: BSplineParams | None = None,
    ) -> None:
        """Initialize a smoothing cubic B-spline to approximate a set of points.

        Parameters
        ----------
        points : array_like
            The points to approximate.
        params : BSplineParams, optional
            Configuration parameters. If None, default parameters will be used.
        """
        # Initialize with default parameters if not provided
        if params is None:
            params = BSplineParams()

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

        # Store the points to approximate and related properties
        self.approximation_points = points
        self.n_approximation_points = n_points
        self.dimension = dimension
        self.mu = np.clip(params.mu, 0.0, 1.0)  # Ensure mu is in [0, 1]
        self.lambda_param = (1 - self.mu) / (6 * self.mu) if self.mu > 0 else float("inf")

        # Set weights
        if params.weights is None:
            self.weights = np.ones(n_points, dtype=np.float64)
        else:
            if len(params.weights) != n_points:
                raise ValueError(
                    f"Length of weights ({len(params.weights)}) must match "
                    f"the number of points ({n_points})"
                )
            self.weights = np.array(params.weights, dtype=np.float64)

        # Calculate the parameters ūₖ
        self.u_bars = self._calculate_parameters(params.method)

        # Calculate the knot vector based on ūₖ as per Section 8.7
        self.knots = self._calculate_knot_vector()

        # Process endpoint derivatives for endpoint interpolation case
        n = self.n_approximation_points - 1  # Index of the last point
        self.enforce_endpoints = params.enforce_endpoints
        self.auto_derivatives = params.auto_derivatives

        if params.enforce_endpoints:
            # Process v0
            if params.v0 is None:
                if params.auto_derivatives and n > 0:
                    # Calculate v0 = (q₁ - q₀) / (ū₁ - ū₀)
                    u_diff = self.u_bars[1] - self.u_bars[0]
                    if abs(u_diff) > EPSILON:  # Avoid division by zero
                        self.v0 = (points[1] - points[0]) / u_diff
                    else:
                        self.v0 = np.zeros(dimension, dtype=np.float64)
                else:
                    self.v0 = np.zeros(dimension, dtype=np.float64)
            else:
                # Convert to numpy array if it's not already
                if not isinstance(params.v0, np.ndarray):
                    v0 = np.array(params.v0, dtype=np.float64)
                else:
                    v0 = params.v0.astype(np.float64)

                # Ensure correct shape
                if v0.ndim == 0:  # scalar
                    self.v0 = np.zeros(dimension, dtype=np.float64)
                elif v0.ndim == 1 and len(v0) == dimension:
                    self.v0 = v0
                else:
                    raise ValueError(f"v0 must be a vector of dimension {dimension}")

            # Process vn
            if params.vn is None:
                if params.auto_derivatives and n > 0:
                    # Calculate vn = (qₙ - qₙ₋₁) / (ūₙ - ūₙ₋₁)
                    u_diff = self.u_bars[n] - self.u_bars[n - 1]
                    if abs(u_diff) > EPSILON:  # Avoid division by zero
                        self.vn = (points[n] - points[n - 1]) / u_diff
                    else:
                        self.vn = np.zeros(dimension, dtype=np.float64)
                else:
                    self.vn = np.zeros(dimension, dtype=np.float64)
            else:
                # Convert to numpy array if it's not already
                if not isinstance(params.vn, np.ndarray):
                    vn = np.array(params.vn, dtype=np.float64)
                else:
                    vn = params.vn.astype(np.float64)

                # Ensure correct shape
                if vn.ndim == 0:  # scalar
                    self.vn = np.zeros(dimension, dtype=np.float64)
                elif vn.ndim == 1 and len(vn) == dimension:
                    self.vn = vn
                else:
                    raise ValueError(f"vn must be a vector of dimension {dimension}")

        # First initialize parent BSpline with dummy control points
        # We'll calculate real control points after initialization
        n_control_points = n + 3  # For a cubic smoothing B-spline
        dummy_control_points = np.zeros((n_control_points, dimension), dtype=np.float64)
        super().__init__(3, self.knots, dummy_control_points)

        # Calculate the actual control points according to Section 8.7
        self.control_points = self._calculate_control_points()

    def _calculate_parameters(self, method: str) -> np.ndarray:
        """Calculate the parameters ūₖ for each point using one of three methods.

        Parameters
        ----------
        method : str
            Method for calculating the parameters.
            Options are 'equally_spaced', 'chord_length', or 'centripetal'.

        Returns
        -------
        np.ndarray
            The parameters ūₖ.

        Raises
        ------
        ValueError
            If an unknown method is provided.
        """
        n = self.n_approximation_points - 1  # Index of the last point

        # Initialize the parameters
        u_bars = np.zeros(self.n_approximation_points, dtype=np.float64)

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
                    self.approximation_points[k] - self.approximation_points[k - 1]
                )

            # Calculate parameters
            accumulated_length = 0.0
            for k in range(1, n):
                accumulated_length += np.linalg.norm(
                    self.approximation_points[k] - self.approximation_points[k - 1]
                )
                u_bars[k] = accumulated_length / total_length

        elif method == "centripetal":
            # Centripetal distribution (equation 8.14)
            mu = 0.5  # As recommended in the document

            # Calculate total "centripetal" length
            total_length = 0.0
            for k in range(1, n + 1):
                total_length += (
                    np.linalg.norm(self.approximation_points[k] - self.approximation_points[k - 1])
                    ** mu
                )

            # Calculate parameters
            accumulated_length = 0.0
            for k in range(1, n):
                accumulated_length += (
                    np.linalg.norm(self.approximation_points[k] - self.approximation_points[k - 1])
                    ** mu
                )
                u_bars[k] = accumulated_length / total_length

        else:
            raise ValueError(
                f"Unknown method: {method}. Options are 'equally_spaced', "
                f"'chord_length', or 'centripetal'."
            )

        return u_bars

    def _calculate_knot_vector(self) -> np.ndarray:
        """Calculate the knot vector based on the parameters ūₖ according to Section 8.7.

        As described in the document:
        u0 = ... = u2 = ū0, un+4 = ... = un+6 = ūn, uj+3 = ūj for j = 0, ..., n

        Returns
        -------
        np.ndarray
            The knot vector.
        """
        n = self.n_approximation_points - 1  # Index of the last point

        # Create the knot vector with n+7 elements (as described in Section 8.7)
        knots = np.zeros(n + 7, dtype=np.float64)

        # Set the first 3 knots to ū₀
        knots[0:3] = self.u_bars[0]

        # Set the last 3 knots to ūₙ
        knots[-(3):] = self.u_bars[n]

        # Set the middle knots to ūⱼ for j = 0, ..., n
        for j in range(n + 1):
            knots[j + 3] = self.u_bars[j]

        return knots

    def _construct_b_matrix(self) -> np.ndarray:
        """Construct the B matrix for the smoothing functional.

        B contains the basis function values at each parameter ūₖ.

        Returns
        -------
        np.ndarray
            The B matrix.
        """
        n = self.n_approximation_points - 1  # Index of the last point
        m = n + 2  # Last index of control points (total of m+1 = n+3 points)

        # Construct the B matrix as per Section 8.7
        b_matrix = np.zeros((n + 1, m + 1), dtype=np.float64)

        for k in range(n + 1):
            u = self.u_bars[k]
            try:
                # Find the span and calculate basis functions
                span = self.find_knot_span(u)
                basis_vals = self.basis_functions(u, span)

                # Fill in the B matrix row
                for j in range(self.degree + 1):
                    b_matrix[k, span - self.degree + j] = basis_vals[j]
            except Exception as e:
                print(f"Error calculating basis functions at u={u}: {e}")
                raise

        return b_matrix

    def _construct_a_matrix(self) -> np.ndarray:
        """Construct the A matrix as defined in equation (8.34).

        This matrix is used for the smoothness term in the functional.

        Returns
        -------
        np.ndarray
            The A matrix.
        """
        n = self.n_approximation_points - 1  # Index of the last point
        size = n + 1  # Size of the A matrix for r₀, r₁, ..., rₙ

        a_matrix = np.zeros((size, size), dtype=np.float64)

        # Using the structure from equation (8.34)
        # The diagonal terms 2(ui+3,i+2 + ui+4,i+3) and off-diagonal terms ui+4,i+3
        for i in range(size):
            # Calculate indices for proper knot differences
            idx3 = i + 3  # u_{i+3}
            idx2 = i + 2  # u_{i+2}
            idx4 = i + 4  # u_{i+4}

            # Ensure indices are within bounds
            if idx4 < len(self.knots):
                # Main diagonal term: 2(u_{i+3,i+2} + u_{i+4,i+3})
                ui3_i2 = self.knots[idx3] - self.knots[idx2]
                ui4_i3 = self.knots[idx4] - self.knots[idx3]

                a_matrix[i, i] = 2 * (ui3_i2 + ui4_i3)

                # Upper diagonal term: u_{i+4,i+3}
                if i < size - 1:
                    a_matrix[i, i + 1] = ui4_i3

                # Lower diagonal term: u_{i+3,i+2}
                if i > 0:
                    a_matrix[i, i - 1] = ui3_i2

        return a_matrix

    def _construct_c_matrix(self) -> np.ndarray:
        """Construct the C matrix as defined in equations (8.35) and (8.36).

        This matrix relates the control points p_j to the 2nd derivative control points r_j.

        Returns
        -------
        np.ndarray
            The C matrix.
        """
        n = self.n_approximation_points - 1  # Index of the last point
        size_r = n + 1  # Number of r₀, r₁, ..., rₙ
        size_p = n + 3  # Number of p₀, p₁, ..., pₙ₊₂

        c_matrix = np.zeros((size_r, size_p), dtype=np.float64)

        # Using the coefficient definitions from equation (8.36)
        for k in range(size_r):
            # Calculate the knot indices
            k1 = k + 1  # u_{k+1}
            k2 = k + 2  # u_{k+2}
            k4 = k + 4  # u_{k+4}
            k5 = k + 5  # u_{k+5}

            # Ensure indices are within bounds
            if k4 < len(self.knots) and k5 < len(self.knots):
                # Calculate knot differences
                uk4_k2 = self.knots[k4] - self.knots[k2]
                uk4_k1 = self.knots[k4] - self.knots[k1]
                uk5_k2 = self.knots[k5] - self.knots[k2]

                # Avoid division by zero
                if abs(uk4_k2) > EPSILON and abs(uk4_k1) > EPSILON and abs(uk5_k2) > EPSILON:
                    # c_k,1 coefficient (for p_k)
                    if k < size_p:
                        c_matrix[k, k] = 6 / (uk4_k2 * uk4_k1)

                    # c_k,2 coefficient (for p_k+1)
                    if k + 1 < size_p:
                        c_matrix[k, k + 1] = -6 / uk4_k2 * (1 / uk4_k1 + 1 / uk5_k2)

                    # c_k,3 coefficient (for p_k+2)
                    if k + 2 < size_p:
                        c_matrix[k, k + 2] = 6 / (uk4_k2 * uk5_k2)

        return c_matrix

    def _calculate_control_points(self) -> np.ndarray:
        """Calculate the control points by minimizing the smoothing functional L.

        Minimizes the smoothing functional as described in Section 8.7 of the document.

        Returns
        -------
        np.ndarray
            The control points.
        """
        if self.enforce_endpoints:
            return self._calculate_control_points_with_endpoints()

        n = self.n_approximation_points - 1  # Index of the last point

        # Construct the B matrix
        b_matrix = self._construct_b_matrix()

        # Construct the weight matrix W
        w_matrix = np.diag(self.weights)

        # Construct the A matrix for the smoothness term
        a_matrix = self._construct_a_matrix()

        # Construct the C matrix (relates control points to 2nd derivative)
        c_matrix = self._construct_c_matrix()

        # Calculate CTAC term as in equation (8.33)
        ctac = c_matrix.T @ a_matrix @ c_matrix

        # Solve the linear system to find the control points
        # From equation (8.37): (BTW B + λ CTAC) P = BTW Q
        left_side = b_matrix.T @ w_matrix @ b_matrix + self.lambda_param * ctac
        right_side = b_matrix.T @ w_matrix @ self.approximation_points

        # Solve the system for each dimension
        control_points = np.zeros((n + 3, self.dimension), dtype=np.float64)

        try:
            for d in range(self.dimension):
                control_points[:, d] = np.linalg.solve(left_side, right_side[:, d])
        except np.linalg.LinAlgError as e:
            print(f"Linear algebra error: {e}")
            # Fall back to least squares solution
            for d in range(self.dimension):
                control_points[:, d] = np.linalg.lstsq(left_side, right_side[:, d], rcond=None)[0]
            print("Using least squares solution instead of direct solve")

        return control_points

    def _calculate_control_points_with_endpoints(self) -> np.ndarray:
        """Calculate the control points when enforcing interpolation of endpoints.

        Enforces interpolation of endpoints and their tangent directions,
        as described in Section 8.7.1.

        Returns
        -------
        np.ndarray
            The control points.
        """
        n = self.n_approximation_points - 1  # Index of the last point

        # Initialize control points
        control_points = np.zeros((n + 3, self.dimension), dtype=np.float64)

        # Set p₀ and pₙ₊₂ directly from endpoints (equation at beginning of 8.7.1)
        control_points[0] = self.approximation_points[0]  # p₀ = q₀
        control_points[n + 2] = self.approximation_points[n]  # pₙ₊₂ = qₙ

        # Calculate p₁ and pₙ₊₁ from tangent directions
        # -p₀ + p₁ = (u₄/3) * d₁
        u4 = self.knots[4]
        control_points[1] = control_points[0] + (u4 / 3.0) * self.v0

        # -pₙ₊₁ + pₙ₊₂ = ((1-uₙ₊₂)/3) * dₙ
        un2 = self.knots[n + 2]
        control_points[n + 1] = control_points[n + 2] - ((1.0 - un2) / 3.0) * self.vn

        # If there are only 3 control points (p₀, p₁, p₂), we're done
        if n <= 0:
            return control_points

        # For more control points, solve the reduced system for p₂, ..., pₙ
        # as described in Section 8.7.1

        # Construct the reduced Q vector
        q_reduced = np.zeros((n - 1, self.dimension), dtype=np.float64)
        for k in range(1, n):
            q_reduced[k - 1] = self.approximation_points[k]

        # Construct the reduced B matrix
        b_reduced = np.zeros((n - 1, n - 1), dtype=np.float64)
        for k in range(1, n):
            u = self.u_bars[k]
            try:
                span = self.find_knot_span(u)
                basis_vals = self.basis_functions(u, span)

                # Subtract contribution of p₀ and p₁ from Q
                if span - self.degree <= 0:  # Basis function for p₀ is non-zero
                    q_reduced[k - 1] -= basis_vals[0] * control_points[0]
                if span - self.degree + 1 <= 1:  # Basis function for p₁ is non-zero
                    q_reduced[k - 1] -= basis_vals[1] * control_points[1]

                # Subtract contribution of pₙ₊₁ and pₙ₊₂ from Q
                if span >= n:  # Basis function for pₙ₊₁ is non-zero
                    basis_idx = n + 1 - (span - self.degree)
                    if 0 <= basis_idx < len(basis_vals):
                        q_reduced[k - 1] -= basis_vals[basis_idx] * control_points[n + 1]
                if span >= n + 1:  # Basis function for pₙ₊₂ is non-zero
                    basis_idx = n + 2 - (span - self.degree)
                    if 0 <= basis_idx < len(basis_vals):
                        q_reduced[k - 1] -= basis_vals[basis_idx] * control_points[n + 2]

                # Fill B_reduced for interior control points p₂ to pₙ
                for j in range(2, n + 1):
                    basis_idx = j - (span - self.degree)
                    if 0 <= basis_idx < len(basis_vals):
                        col = j - 2  # Index in B_reduced
                        if 0 <= col < n - 1:
                            b_reduced[k - 1, col] = basis_vals[basis_idx]
            except Exception as e:
                print(f"Error calculating reduced matrices at u={u}: {e}")
                raise

        # Construct the reduced weight matrix W
        w_reduced = np.diag(self.weights[1:n])

        # Construct the A and C matrices for the smoothness term
        # For the reduced case, we need to modify these matrices
        a_matrix = self._construct_a_matrix()
        c_matrix = self._construct_c_matrix()

        # Extract the parts of C that relate to p₂...pₙ
        c_reduced = c_matrix[:, 2 : n + 1]

        # Calculate PZ vector (from Section 8.7.1)
        pz = np.zeros((n + 1, self.dimension))
        # Add terms for p₀ and p₁
        pz[0] = c_matrix[0, 0] * control_points[0] + c_matrix[0, 1] * control_points[1]
        # Add terms for pₙ₊₁ and pₙ₊₂
        pz[n] = (
            c_matrix[n, n + 1] * control_points[n + 1] + c_matrix[n, n + 2] * control_points[n + 2]
        )

        # Calculate the CTAC term for reduced system
        ctac_reduced = c_reduced.T @ a_matrix @ c_reduced

        # Calculate the CTAT*PZ term
        ctat_pz = c_reduced.T @ a_matrix @ pz

        # Solve the linear system for the interior control points
        # (BTW B + λ CTAC) P = BTW Q - λ CTAT*PZ
        left_side = b_reduced.T @ w_reduced @ b_reduced + self.lambda_param * ctac_reduced
        right_side = b_reduced.T @ w_reduced @ q_reduced - self.lambda_param * ctat_pz

        try:
            # Solve the system for each dimension
            for d in range(self.dimension):
                interior_controls = np.linalg.solve(left_side, right_side[:, d])
                control_points[2 : n + 1, d] = interior_controls
        except np.linalg.LinAlgError as e:
            print(f"Linear algebra error in reduced system: {e}")
            # Fall back to least squares solution
            for d in range(self.dimension):
                interior_controls = np.linalg.lstsq(left_side, right_side[:, d], rcond=None)[0]
                control_points[2 : n + 1, d] = interior_controls
            print("Using least squares solution for reduced system")

        return control_points

    def calculate_approximation_error(self) -> np.ndarray:
        """Calculate the approximation error for each point.

        Returns
        -------
        np.ndarray
            Array with error values for each approximation point.
        """
        errors = np.zeros(self.n_approximation_points, dtype=np.float64)

        for k in range(self.n_approximation_points):
            # Evaluate the B-spline at the parameter value
            u = self.u_bars[k]
            point = self.evaluate(u)

            # Calculate the error (Euclidean distance)
            errors[k] = np.linalg.norm(point - self.approximation_points[k])

        return errors

    def calculate_total_error(self) -> float:
        """Calculate the total weighted approximation error.

        Returns
        -------
        float
            The total weighted error.
        """
        total_error = 0.0

        for k in range(self.n_approximation_points):
            # Evaluate the B-spline at the parameter value
            u = self.u_bars[k]
            point = self.evaluate(u)

            # Calculate the squared error weighted by w_k
            diff = point - self.approximation_points[k]
            total_error += self.weights[k] * np.sum(diff**2)

        return total_error

    def calculate_smoothness_measure(self, num_points: int = 100) -> float:
        """Calculate the smoothness measure (integral of squared second derivative).

        Parameters
        ----------
        num_points : int, default=100
            Number of points to use for numerical integration.

        Returns
        -------
        float
            The smoothness measure.
        """
        # Generate parameter values for numerical integration
        u_values = np.linspace(self.u_min, self.u_max, num_points)
        du = (self.u_max - self.u_min) / (num_points - 1)

        # Calculate the second derivative at each parameter value
        smoothness = 0.0
        for u in u_values:
            d2 = self.evaluate_derivative(u, order=2)
            smoothness += np.sum(d2**2) * du

        return smoothness
