import matplotlib.pyplot as plt
import numpy as np


class CubicSmoothingSpline:
    """
    Cubic smoothing spline trajectory planning with control over the smoothness
    versus waypoint accuracy trade-off.

    This class implements cubic smoothing splines for trajectory generation as
    described in section 4.4.5 of the textbook. The algorithm minimizes a weighted
    sum of waypoint error and acceleration magnitude, allowing the user to control
    the trade-off between path smoothness and waypoint accuracy through the
    parameter μ.

    Parameters
    ----------
    t_points : list[float] or array_like
        Time points [t₀, t₁, t₂, ..., tₙ] for the spline knots.
    q_points : list[float] or array_like
        Position points [q₀, q₁, q₂, ..., qₙ] at each time point.
    mu : float, optional
        Trade-off parameter between accuracy (μ=1) and smoothness (μ=0).
        Must be in range (0, 1]. Default is 0.5.
    weights : list[float] or array_like, optional
        Individual point weights [w₀, w₁, ..., wₙ]. Higher values enforce
        closer approximation at corresponding points. Default is None
        (equal weights of 1.0 for all points).
    v0 : float, optional
        Initial velocity constraint at t₀. Default is 0.0.
    vn : float, optional
        Final velocity constraint at tₙ. Default is 0.0.
    debug : bool, optional
        Whether to print debug information. Default is False.

    Attributes
    ----------
    t : ndarray
        Time points array.
    q : ndarray
        Original position points array.
    s : ndarray
        Approximated position points array.
    mu : float
        Smoothing parameter value.
    lambd : float
        Lambda parameter derived from μ: λ = (1-μ)/(6μ).
    omega : ndarray
        Acceleration values at each time point.
    coeffs : ndarray
        Polynomial coefficients for each segment.

    Raises
    ------
    ValueError
        If time and position arrays have different lengths.
        If fewer than 2 points are provided.
        If time points are not strictly increasing.
        If parameter μ is outside the valid range (0, 1].
        If weights array length doesn't match time and position arrays.

    Notes
    -----
    For μ=1, the spline performs exact interpolation.
    For μ approaching 0, the spline becomes increasingly smooth.
    Setting weight to infinity for a point forces exact interpolation at that point.

    References
    ----------
    The implementation follows section 4.4.5 of the robotics textbook
    describing cubic smoothing splines for trajectory generation.
    """

    # Constants to replace magic numbers
    MIN_POINTS_REQUIRED = 2
    HIGH_CONDITION_THRESHOLD = 1e12
    REGULARIZATION_FACTOR = 1e-8

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        t_points: list[float],
        q_points: list[float],
        mu: float = 0.5,
        weights: list[float] | None = None,
        v0: float = 0.0,
        vn: float = 0.0,
        debug: bool = False,
    ) -> None:
        """
        Initialize the cubic smoothing spline.

        Parameters
        ----------
        t_points : list[float] or array_like
            Time points [t₀, t₁, t₂, ..., tₙ] for the spline knots.
        q_points : list[float] or array_like
            Position points [q₀, q₁, q₂, ..., qₙ] at each time point.
        mu : float, optional
            Trade-off parameter between accuracy (μ=1) and smoothness (μ=0).
            Must be in range (0, 1]. Default is 0.5.
        weights : list[float] or array_like, optional
            Individual point weights [w₀, w₁, ..., wₙ]. Higher values enforce
            closer approximation at corresponding points. Default is None
            (equal weights of 1.0 for all points).
        v0 : float, optional
            Initial velocity constraint at t₀. Default is 0.0.
        vn : float, optional
            Final velocity constraint at tₙ. Default is 0.0.
        debug : bool, optional
            Whether to print debug information. Default is False.

        Raises
        ------
        ValueError
            If time and position arrays have different lengths.
            If fewer than 2 points are provided.
            If time points are not strictly increasing.
            If parameter μ is outside the valid range (0, 1].
            If weights array length doesn't match time and position arrays.
        """
        # Validate inputs
        if len(t_points) != len(q_points):
            raise ValueError("Time and position arrays must have the same length")

        if len(t_points) < self.MIN_POINTS_REQUIRED:
            raise ValueError("At least two points are required")

        if not np.all(np.diff(t_points) > 0):
            raise ValueError("Time points must be strictly increasing")

        # Corrected validation for mu - should be (0, 1] not [0, 1]
        if mu <= 0.0 or mu > 1.0:
            raise ValueError("Parameter μ must be in range (0, 1]")

        # Store parameters
        self.t = np.array(t_points, dtype=float)
        self.q = np.array(q_points, dtype=float)
        self.mu = float(mu)
        self.v0 = float(v0)
        self.vn = float(vn)
        self.debug = debug

        # Number of points
        self.n = len(self.t)

        # Compute lambda parameter: λ = (1-μ)/(6μ)
        # Add small epsilon to avoid division by zero if mu is very close to 0
        self.lambd = (1 - self.mu) / (6 * self.mu + 1e-10) if self.mu < 1 else 0

        # Compute time intervals
        self.time_intervals = np.diff(self.t)

        # Initialize weights or use default (equal weights)
        if weights is None:
            # Default: all points have equal weight of 1
            self.w = np.ones(self.n)
        else:
            if len(weights) != self.n:
                raise ValueError(
                    "Weights array must have the same length as time and position arrays"
                )
            self.w = np.array(weights, dtype=float)

        # Handle infinite weights (fixed points)
        self.w_inv = np.zeros(self.n)
        finite_mask = np.isfinite(self.w)
        self.w_inv[finite_mask] = 1.0 / self.w[finite_mask]

        if self.debug:
            print("Smoothing parameter μ:", self.mu)
            print("Lambda λ:", self.lambd)
            print("Weights:", self.w)
            print("Inverse weights:", self.w_inv)

        # Construct the matrices
        self.a_matrix, self.c_matrix = self._construct_matrices()

        # Solve the system to get accelerations
        self.omega = self._solve_system()

        # Compute the approximated positions
        self.s = self._compute_positions()

        # Compute polynomial coefficients
        self.coeffs = self._compute_coefficients()

        if self.debug:
            print("Original points:", self.q)
            print("Approximated points:", self.s)
            print("Accelerations:", self.omega)
            print("Maximum position error:", np.max(np.abs(self.q - self.s)))

    def _construct_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Construct matrices A and C for the linear system.

        Builds the tridiagonal matrix A and the matrix C necessary for solving
        the cubic smoothing spline system of equations.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            a_matrix : ndarray
                Tridiagonal matrix relating accelerations, shape (n, n).
            c_matrix : ndarray
                Matrix relating positions to accelerations, shape (n, n).

        Notes
        -----
        A matrix is constructed according to equation 4.23 in the textbook.
        C matrix is constructed according to equation 4.34 for the smoothing spline.
        """
        n = self.n
        time_intervals = self.time_intervals

        # Construct A matrix (equation 4.23)
        # A is symmetric and tridiagonal
        a_matrix = np.zeros((n, n))

        # Fill main diagonal
        a_matrix[0, 0] = 2 * time_intervals[0]
        a_matrix[n - 1, n - 1] = 2 * time_intervals[n - 2]

        for i in range(1, n - 1):
            a_matrix[i, i] = 2 * (time_intervals[i - 1] + time_intervals[i])

        # Fill upper and lower diagonals
        for i in range(n - 1):
            a_matrix[i, i + 1] = time_intervals[i]
            a_matrix[i + 1, i] = time_intervals[i]  # Symmetric

        # Construct C matrix (equation 4.34) for the smoothing spline
        # This matrix relates positions to accelerations
        c_matrix = np.zeros((n, n))

        # First row with initial velocity constraint
        c_matrix[0, 0] = -6 / time_intervals[0]
        c_matrix[0, 1] = 6 / time_intervals[0]

        # Last row with final velocity constraint
        c_matrix[n - 1, n - 2] = 6 / time_intervals[n - 2]
        c_matrix[n - 1, n - 1] = -6 / time_intervals[n - 2]

        # Interior rows
        for i in range(1, n - 1):
            c_matrix[i, i - 1] = 6 / time_intervals[i - 1]
            c_matrix[i, i] = -(6 / time_intervals[i - 1] + 6 / time_intervals[i])
            c_matrix[i, i + 1] = 6 / time_intervals[i]

        if self.debug:
            print("Matrix A:\n", a_matrix)
            print("Matrix C:\n", c_matrix)

        return a_matrix, c_matrix

    def _solve_system(self) -> np.ndarray:
        """
        Solve the linear system to find the accelerations.

        Solves either the pure interpolation system (μ=1) or the smoothing
        spline system (0<μ<1) to determine the acceleration values at each
        time point.

        Returns
        -------
        np.ndarray
            Vector of accelerations ω at each time point.

        Notes
        -----
        For pure interpolation (μ=1): Aω = c (equation 4.22)
        For smoothing spline (0<μ<1): (A + λCW⁻¹Cᵀ)ω = Cq (equation 4.35)

        If the system is poorly conditioned, a small regularization term
        is added to improve numerical stability.
        """
        n = self.n

        # For pure interpolation (μ = 1): Aω = c (equation 4.22)
        if self.mu == 1.0:
            # Construct the vector c according to equation (4.24)
            c = np.zeros(n)

            # First element - with initial velocity constraint
            c[0] = 6 * ((self.q[1] - self.q[0]) / self.time_intervals[0] - self.v0)

            # Last element - with final velocity constraint
            c[n - 1] = 6 * (self.vn - (self.q[n - 1] - self.q[n - 2]) / self.time_intervals[n - 2])

            # Interior elements
            for i in range(1, n - 1):
                c[i] = 6 * (
                    (self.q[i + 1] - self.q[i]) / self.time_intervals[i]
                    - (self.q[i] - self.q[i - 1]) / self.time_intervals[i - 1]
                )

            if self.debug:
                print("Vector c (pure interpolation):", c)

            # Solve Aω = c using a more robust solver
            # Use solve instead of inv for better numerical stability
            return np.linalg.solve(self.a_matrix, c)

        # For smoothing spline (0 < μ < 1): (A + λCW⁻¹Cᵀ)ω = Cq (equation 4.35)
        # Right hand side: Cq
        rhs = self.c_matrix @ self.q

        # Left hand side: (A + λCW⁻¹Cᵀ)
        # Create w_inv as a diagonal matrix
        w_inv_diag = np.diag(self.w_inv)

        # Compute the matrix product more carefully
        c_w_inv_ct = self.c_matrix @ w_inv_diag @ self.c_matrix.T

        # Make sure c_w_inv_ct is symmetric (could be slightly asymmetric due to numerical issues)
        c_w_inv_ct = (c_w_inv_ct + c_w_inv_ct.T) / 2

        system_matrix = self.a_matrix + self.lambd * c_w_inv_ct

        # Ensure system_matrix is symmetric for better numerical stability
        system_matrix = (system_matrix + system_matrix.T) / 2

        if self.debug:
            print("System matrix (smoothing):\n", system_matrix)
            print("RHS vector:", rhs)
            print("Condition number:", np.linalg.cond(system_matrix))

        # Add small regularization if the matrix is poorly conditioned
        if np.linalg.cond(system_matrix) > self.HIGH_CONDITION_THRESHOLD:
            system_matrix += np.eye(n) * self.REGULARIZATION_FACTOR
            if self.debug:
                print("Added regularization. New condition number:", np.linalg.cond(system_matrix))

        # Solve the system
        try:
            omega = np.linalg.solve(system_matrix, rhs)
        except np.linalg.LinAlgError:
            print("Warning: Linear system is singular or poorly conditioned.")
            # Use least squares to find a solution
            omega, _residuals, _rank, _s = np.linalg.lstsq(system_matrix, rhs, rcond=None)

        return omega

    def _compute_positions(self) -> np.ndarray:
        """
        Compute the approximated positions using equation (4.36).

        For pure interpolation (μ=1), returns exact positions.
        For smoothing spline (0<μ<1), computes approximated positions.

        Returns
        -------
        np.ndarray
            Vector of approximated positions s.

        Notes
        -----
        For pure interpolation (μ=1): s = q (exact fit)
        For smoothing spline (0<μ<1): s = q - λW⁻¹Cᵀω (equation 4.36)
        """
        # For pure interpolation (μ = 1), s = q (exact fit)
        if self.mu == 1.0:
            return self.q.copy()

        # For smoothing spline (0 < μ < 1), s = q - λW⁻¹Cᵀω (equation 4.36)
        # Compute W⁻¹Cᵀω more explicitly to avoid potential issues
        ct_omega = self.c_matrix.T @ self.omega
        adjustment = self.lambd * (self.w_inv * ct_omega)
        s = self.q - adjustment

        if self.debug:
            print(f"Computed s: {s} for mu: {self.mu}")
        return s

    def _compute_coefficients(self) -> np.ndarray:
        """
        Compute polynomial coefficients for each segment.

        For each segment k from 0 to n-2, computes the cubic polynomial
        coefficients that define the spline.

        Returns
        -------
        np.ndarray
            Array of shape (n-1, 4) with coefficients [a₀, a₁, a₂, a₃]
            for each segment.

        Notes
        -----
        For each segment k from 0 to n-2, computes:
        a₀ = s(tₖ)
        a₁ = (s(tₖ₊₁) - s(tₖ))/Tₖ - (Tₖ/6)·(ωₖ₊₁ + 2ωₖ)
        a₂ = ωₖ/2
        a₃ = (ωₖ₊₁ - ωₖ)/(6·Tₖ)

        The resulting polynomial for segment k is:
        p(τ) = a₀ + a₁·τ + a₂·τ² + a₃·τ³
        where τ = t - tₖ is the local time within the segment.
        """
        n_segments = self.n - 1
        coeffs = np.zeros((n_segments, 4))

        for k in range(n_segments):
            # Position coefficient
            coeffs[k, 0] = self.s[k]

            # Velocity coefficient - corrected formula from standard cubic spline theory
            coeffs[k, 1] = (self.s[k + 1] - self.s[k]) / self.time_intervals[k] - (
                self.time_intervals[k] / 6
            ) * (self.omega[k + 1] + 2 * self.omega[k])

            # Acceleration coefficient
            coeffs[k, 2] = self.omega[k] / 2

            # Jerk coefficient
            coeffs[k, 3] = (self.omega[k + 1] - self.omega[k]) / (6 * self.time_intervals[k])

        if self.debug:
            print("Polynomial coefficients:")
            for k in range(n_segments):
                print(f"Segment {k}: {coeffs[k]}")

        return coeffs

    def evaluate(self, t: float | list[float] | np.ndarray) -> float | np.ndarray:
        """
        Evaluate the spline at time t.

        Parameters
        ----------
        t : float or list[float] or np.ndarray
            Time point or array of time points at which to evaluate the spline.

        Returns
        -------
        float or np.ndarray
            Position(s) at the specified time(s). Returns a scalar if input
            is a scalar, otherwise returns an array matching the shape of input.

        Notes
        -----
        For t < t₀ or t > tₙ, the function extrapolates using the first or
        last segment respectively.
        """
        t_array = np.atleast_1d(t)
        result = np.zeros_like(t_array, dtype=float)

        for i, ti in enumerate(t_array):
            # Find segment containing ti
            if ti <= self.t[0]:
                # Improved handling for times before start of trajectory
                # Extrapolate using the first segment
                segment = 0
                tau = ti - self.t[segment]  # Can be negative for extrapolation
            elif ti >= self.t[-1]:
                # Improved handling for times after end of trajectory
                # Extrapolate using the last segment
                segment = self.n - 2
                tau = ti - self.t[segment]  # Will be > T[segment] for extrapolation
            else:
                # Within trajectory - find the correct segment
                segment = np.searchsorted(self.t, ti, side="right") - 1
                tau = ti - self.t[segment]

            # Evaluate polynomial: a₀ + a₁τ + a₂τ² + a₃τ³
            c = self.coeffs[segment]
            result[i] = c[0] + c[1] * tau + c[2] * tau**2 + c[3] * tau**3

        return result[0] if len(t_array) == 1 else result

    def evaluate_velocity(self, t: float | list[float] | np.ndarray) -> float | np.ndarray:
        """
        Evaluate the velocity at time t.

        Parameters
        ----------
        t : float or list[float] or np.ndarray
            Time point or array of time points at which to evaluate velocity.

        Returns
        -------
        float or np.ndarray
            Velocity at the specified time(s). Returns a scalar if input
            is a scalar, otherwise returns an array matching the shape of input.

        Notes
        -----
        Computes the first derivative of the spline at the given time(s).
        For t < t₀ or t > tₙ, the function extrapolates using the first or
        last segment respectively.
        """
        t_array = np.atleast_1d(t)
        result = np.zeros_like(t_array, dtype=float)

        for i, ti in enumerate(t_array):
            # Find segment containing ti
            if ti <= self.t[0]:
                segment = 0
                tau = ti - self.t[segment]  # Corrected extrapolation
            elif ti >= self.t[-1]:
                segment = self.n - 2
                tau = ti - self.t[segment]  # Corrected extrapolation
            else:
                segment = np.searchsorted(self.t, ti, side="right") - 1
                tau = ti - self.t[segment]

            # Evaluate first derivative: a₁ + 2a₂τ + 3a₃τ²
            c = self.coeffs[segment]
            result[i] = c[1] + 2 * c[2] * tau + 3 * c[3] * tau**2

        return result[0] if len(t_array) == 1 else result

    def evaluate_acceleration(self, t: float | list[float] | np.ndarray) -> float | np.ndarray:
        """
        Evaluate the acceleration at time t.

        Parameters
        ----------
        t : float or list[float] or np.ndarray
            Time point or array of time points at which to evaluate acceleration.

        Returns
        -------
        float or np.ndarray
            Acceleration at the specified time(s). Returns a scalar if input
            is a scalar, otherwise returns an array matching the shape of input.

        Notes
        -----
        Computes the second derivative of the spline at the given time(s).
        For t < t₀ or t > tₙ, the function extrapolates using the first or
        last segment respectively.
        """
        t_array = np.atleast_1d(t)
        result = np.zeros_like(t_array, dtype=float)

        for i, ti in enumerate(t_array):
            # Find segment containing ti
            if ti <= self.t[0]:
                segment = 0
                tau = ti - self.t[segment]  # Corrected extrapolation
            elif ti >= self.t[-1]:
                segment = self.n - 2
                tau = ti - self.t[segment]  # Corrected extrapolation
            else:
                segment = np.searchsorted(self.t, ti, side="right") - 1
                tau = ti - self.t[segment]

            # Evaluate second derivative: 2a₂ + 6a₃τ
            c = self.coeffs[segment]
            result[i] = 2 * c[2] + 6 * c[3] * tau

        return result[0] if len(t_array) == 1 else result

    def plot(self, num_points: int = 1000) -> None:
        """
        Plot the spline trajectory with velocity and acceleration profiles.

        Creates a three-panel figure showing position, velocity, and acceleration
        profiles over time.

        Parameters
        ----------
        num_points : int, optional
            Number of points for smooth plotting. Default is 1000.

        Returns
        -------
        None
            Displays the plot using matplotlib's show() function.

        Notes
        -----
        The plot includes:
        - Top panel: position trajectory with original waypoints and approximated points
        - Middle panel: velocity profile
        - Bottom panel: acceleration profile
        """
        # Generate evaluation points
        t_eval = np.linspace(self.t[0], self.t[-1], num_points)

        # Evaluate the spline and its derivatives
        q = self.evaluate(t_eval)
        v = self.evaluate_velocity(t_eval)
        a = self.evaluate_acceleration(t_eval)

        # Create a figure with three subplots
        _fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # Position plot
        ax1.plot(t_eval, q, "b-", linewidth=2, label="Smoothing Spline")
        ax1.plot(self.t, self.q, "ro", markersize=8, label="Original Waypoints")
        ax1.plot(self.t, self.s, "gx", markersize=6, label="Approximated Points")
        ax1.set_ylabel("Position")
        ax1.grid(True)
        ax1.legend()
        ax1.set_title(f"Cubic Smoothing Spline (μ={self.mu:.2f})")

        # Velocity plot
        ax2.plot(t_eval, v, "g-", linewidth=2)
        ax2.set_ylabel("Velocity")
        ax2.grid(True)

        # Acceleration plot
        ax3.plot(t_eval, a, "r-", linewidth=2)
        ax3.set_ylabel("Acceleration")
        ax3.set_xlabel("Time")
        ax3.grid(True)

        plt.tight_layout()
        plt.show()
