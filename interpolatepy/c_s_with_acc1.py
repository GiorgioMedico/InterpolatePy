import matplotlib.pyplot as plt
import numpy as np

from interpolatepy.tridiagonal_inv import solve_tridiagonal


# Constants to replace magic numbers
MIN_POINTS = 2
MIN_SEGMENTS_FOR_UPPER_DIAG = 2
MIN_SEGMENTS_FOR_SECOND_ROW = 3
MIN_SEGMENTS_FOR_UPPER_DIAG_SECOND_ROW = 4
MIN_SEGMENTS_FOR_SECOND_ELEMENT = 4


class CubicSplineWithAcceleration1:
    """
    Cubic spline trajectory planning with both velocity and acceleration constraints
    at endpoints.

    Implements the method described in section 4.4.4 of the paper, which adds
    two extra points in the first and last segments to satisfy the acceleration
    constraints.

    Parameters
    ----------
    t_points : list of float or numpy.ndarray
        Time points [t₀, t₂, t₃, ..., tₙ₋₂, tₙ]
    q_points : list of float or numpy.ndarray
        Position points [q₀, q₂, q₃, ..., qₙ₋₂, qₙ]
    v0 : float, optional
        Initial velocity at t₀. Default is 0.0
    vn : float, optional
        Final velocity at tₙ. Default is 0.0
    a0 : float, optional
        Initial acceleration at t₀. Default is 0.0
    an : float, optional
        Final acceleration at tₙ. Default is 0.0
    debug : bool, optional
        Whether to print debug information. Default is False

    Attributes
    ----------
    t_orig : numpy.ndarray
        Original time points
    q_orig : numpy.ndarray
        Original position points
    v0 : float
        Initial velocity
    vn : float
        Final velocity
    a0 : float
        Initial acceleration
    an : float
        Final acceleration
    debug : bool
        Debug flag
    n_orig : int
        Number of original points
    t : numpy.ndarray
        Time points including extra points
    q : numpy.ndarray
        Position points including extra points
    n : int
        Total number of points including extras
    T : numpy.ndarray
        Time intervals between consecutive points
    omega : numpy.ndarray
        Acceleration values at each point
    coeffs : numpy.ndarray
        Polynomial coefficients for each segment
    original_indices : list
        Indices of original points in expanded arrays

    Notes
    -----
    This implementation adds two extra points (at t₁ and tₙ₋₁) to the trajectory
    to satisfy both velocity and acceleration constraints at the endpoints.
    The extra points are placed at the midpoints of the first and last segments
    of the original trajectory.

    The acceleration (omega) values are computed by solving a tridiagonal system
    that ensures C2 continuity throughout the trajectory.

    Examples
    --------
    >>> import numpy as np
    >>> # Define waypoints
    >>> t_points = [0, 1, 2, 3]
    >>> q_points = [0, 1, 0, 1]
    >>> # Create spline with velocity and acceleration constraints
    >>> spline = CubicSplineWithAcceleration1(t_points, q_points, v0=0.0, vn=0.0, a0=0.0, an=0.0)
    >>> # Evaluate at specific time
    >>> spline.evaluate(1.5)
    >>> # Plot the trajectory
    >>> spline.plot()
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        t_points: list[float],
        q_points: list[float],
        v0: float = 0.0,
        vn: float = 0.0,
        a0: float = 0.0,
        an: float = 0.0,
        debug: bool = False,
    ) -> None:
        """
        Initialize the cubic spline with velocity and acceleration constraints.

        Parameters
        ----------
        t_points : list of float or numpy.ndarray
            Time points [t₀, t₂, t₃, ..., tₙ₋₂, tₙ]
        q_points : list of float or numpy.ndarray
            Position points [q₀, q₂, q₃, ..., qₙ₋₂, qₙ]
        v0 : float, optional
            Initial velocity at t₀. Default is 0.0
        vn : float, optional
            Final velocity at tₙ. Default is 0.0
        a0 : float, optional
            Initial acceleration at t₀. Default is 0.0
        an : float, optional
            Final acceleration at tₙ. Default is 0.0
        debug : bool, optional
            Whether to print debug information. Default is 0.0

        Raises
        ------
        ValueError
            If time and position arrays have different lengths
            If fewer than two points are provided
            If time points are not strictly increasing
        """
        # Validate inputs
        if len(t_points) != len(q_points):
            raise ValueError("Time and position arrays must have the same length")

        if len(t_points) < MIN_POINTS:
            raise ValueError("At least two points are required")

        if not np.all(np.diff(t_points) > 0):
            raise ValueError("Time points must be strictly increasing")

        # Following paper's notation: original points are [q₀, q₂, q₃, ..., qₙ₋₂, qₙ]
        # We will add extra points q₁ and qₙ₋₁
        self.t_orig = np.array(t_points, dtype=float)
        self.q_orig = np.array(q_points, dtype=float)

        # Boundary conditions
        self.v0 = float(v0)  # Initial velocity
        self.vn = float(vn)  # Final velocity
        self.a0 = float(a0)  # Initial acceleration (ω₀)
        self.an = float(an)  # Final acceleration (ωₙ)

        self.debug = debug

        # Number of original points
        self.n_orig = len(self.t_orig)

        # Insert the two extra points
        self.t, self.q = self._add_extra_points()

        # Number of total points including extras
        self.n = len(self.t)

        # Compute time intervals
        self.T = np.diff(self.t)

        if self.debug:
            print("Time interval length: ", self.T)
            print("\n")

        # Solve for the accelerations at all points
        self.omega = self._solve_accelerations()

        # Compute polynomial coefficients for each segment
        self.coeffs = self._compute_coefficients()

        # For plotting, keep track of original point indices
        self.original_indices = self._get_original_indices()

    def _add_extra_points(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Add two extra points at t₁ and tₙ₋₁ to satisfy acceleration constraints.

        Following the paper, the time points are placed at midpoints:
        t₁ = (t₀ + t₂)/2 and tₙ₋₁ = (tₙ₋₂ + tₙ)/2

        Returns
        -------
        tuple of numpy.ndarray
            (t, q): New time and position arrays with extra points

        Notes
        -----
        The q values for the extra points are initially set to zero and
        will be properly computed after solving for the accelerations.
        """
        # Create expanded arrays for time and position
        t_new = np.zeros(self.n_orig + 2)
        q_new = np.zeros(self.n_orig + 2)

        # Set first and last points (same as original)
        t_new[0] = self.t_orig[0]  # t₀
        t_new[-1] = self.t_orig[-1]  # tₙ
        q_new[0] = self.q_orig[0]  # q₀
        q_new[-1] = self.q_orig[-1]  # qₙ

        # Set the original interior points (from q₂ to qₙ₋₂)
        t_new[2:-2] = self.t_orig[1:-1]
        q_new[2:-2] = self.q_orig[1:-1]

        # Add extra points at midpoints as suggested in the paper
        t_new[1] = (self.t_orig[0] + self.t_orig[1]) / 2  # t₁
        t_new[-2] = (self.t_orig[-2] + self.t_orig[-1]) / 2  # tₙ₋₁

        if self.debug:
            print("Original times: ", self.t_orig)
            print("New times with extra points: ", t_new)
            print("\n")

        return t_new, q_new

    def _solve_accelerations(self) -> np.ndarray:
        """
        Solve for the accelerations by setting up and solving the linear system A ω = c.

        Following the paper, we solve for interior accelerations [ω₁, ω₂, ..., ωₙ₋₁],
        with ω₀ = a₀ and ωₙ = aₙ known from boundary conditions.

        Uses the tridiagonal solver for improved efficiency.

        Returns
        -------
        numpy.ndarray
            Array of accelerations [ω₀, ω₁, ..., ωₙ]

        Notes
        -----
        This method sets up and solves the tridiagonal system described in equation (4.28)
        of the paper. It also adjusts the positions of the extra points (q₁ and qₙ₋₁)
        using equations (4.26) and (4.27) after the accelerations are computed.

        The tridiagonal system has special structure for the first and last rows
        to account for the boundary conditions. The system is of size (n-1) x (n-1)
        where n is the number of segments.
        """
        # Number of segments (number of points - 1)
        n_segments = self.n - 1

        # We need to solve for n_segments - 1 interior accelerations
        # The system is (n_segments - 1) x (n_segments - 1)

        # Preparing diagonal arrays
        main_diag = np.zeros(n_segments - 1)
        lower_diag = np.zeros(n_segments - 1)  # First element not used
        upper_diag = np.zeros(n_segments - 1)  # Last element not used

        # First element of main diagonal
        main_diag[0] = 2 * self.T[1] + self.T[0] * (3 + self.T[0] / self.T[1])
        if n_segments > MIN_SEGMENTS_FOR_UPPER_DIAG:
            upper_diag[0] = self.T[1]

        # Last element of main diagonal
        main_diag[-1] = 2 * self.T[-2] + self.T[-1] * (3 + self.T[-1] / self.T[-2])
        if n_segments > MIN_SEGMENTS_FOR_UPPER_DIAG:
            lower_diag[-1] = self.T[-2]

        if n_segments > MIN_SEGMENTS_FOR_SECOND_ROW:
            # Second row (if it exists)
            lower_diag[1] = self.T[1] - (self.T[0] ** 2 / self.T[1])
            main_diag[1] = 2 * (self.T[1] + self.T[2])
            if n_segments > MIN_SEGMENTS_FOR_UPPER_DIAG_SECOND_ROW:
                upper_diag[1] = self.T[2]

            # Second-to-last row (if it exists)
            main_diag[-2] = 2 * (self.T[-3] + self.T[-2])
            lower_diag[-2] = self.T[-3]
            upper_diag[-2] = self.T[-2] - self.T[-1] ** 2 / self.T[-2]

        # Middle rows (if any)
        for i in range(2, n_segments - 3):
            lower_diag[i] = self.T[i]
            main_diag[i] = 2 * (self.T[i] + self.T[i + 1])
            upper_diag[i] = self.T[i + 1]

        # Construct vector c exactly as in equation (4.28)
        c = np.zeros(n_segments - 1)

        # First element
        c[0] = 6 * (
            (self.q[2] - self.q[0]) / self.T[1]
            - self.v0 * (1 + self.T[0] / self.T[1])
            - self.a0 * (1 / 2 + self.T[0] / (3 * self.T[1])) * self.T[0]
        )

        # Last element
        c[-1] = 6 * (
            (self.q[-3] - self.q[-1]) / self.T[-2]
            + self.vn * (1 + self.T[-1] / self.T[-2])
            - self.an * (1 / 2 + self.T[-1] / (3 * self.T[-2])) * self.T[-1]
        )

        if n_segments >= MIN_SEGMENTS_FOR_SECOND_ELEMENT:
            # Second element (if there are at least 4 segments)
            c[1] = 6 * (
                (self.q[3] - self.q[2]) / self.T[2]
                - (self.q[2] - self.q[0]) / self.T[1]
                + self.v0 * self.T[0] / self.T[1]
                + self.a0 * self.T[0] ** 2 / (3 * self.T[1])
            )

            # Second-to-last element (if there are at least 4 segments)
            c[-2] = 6 * (
                (self.q[-1] - self.q[-3]) / self.T[-2]
                - (self.q[-3] - self.q[-4]) / self.T[-3]
                - self.vn * self.T[-1] / self.T[-2]
                + self.an * self.T[-1] ** 2 / (3 * self.T[-2])
            )

            # Middle elements (if there are more than 4 segments)
            for i in range(2, n_segments - 2):
                if i == n_segments - 3:
                    continue  # Skip since we've already handled the second-to-last element
                c[i] = 6 * (
                    (self.q[i + 2] - self.q[i + 1]) / self.T[i + 1]
                    - (self.q[i + 1] - self.q[i]) / self.T[i]
                )

        if self.debug:
            print("Main diagonal:", main_diag)
            print("Lower diagonal:", lower_diag)
            print("Upper diagonal:", upper_diag)
            print("Vector c:", c)
            print("\n")

        # Solve the system using the tridiagonal solver
        interior_omega = solve_tridiagonal(lower_diag, main_diag, upper_diag, c)

        # Complete accelerations vector with boundary values
        omega = np.zeros(self.n)
        omega[0] = self.a0  # ω₀ = a₀
        omega[1:-1] = interior_omega  # ω₁ to ωₙ₋₁
        omega[-1] = self.an  # ωₙ = aₙ

        # Now adjust the positions of extra points using equations (4.26) and (4.27)
        self.q[1] = (
            self.q[0]
            + self.T[0] * self.v0
            + (self.T[0] ** 2 / 3) * self.a0
            + (self.T[0] ** 2 / 6) * omega[1]
        )
        self.q[-2] = (
            self.q[-1]
            - self.T[-1] * self.vn
            + (self.T[-1] ** 2 / 3) * self.an
            + (self.T[-1] ** 2 / 6) * omega[-2]
        )

        if self.debug:
            print("Computed accelerations:", omega)
            print("Adjusted q₁:", self.q[1])
            print("Adjusted qₙ₋₁:", self.q[-2])
            print("\n")

        return omega

    def _compute_coefficients(self) -> np.ndarray:
        """
        Compute the polynomial coefficients for each segment using equation (4.25).

        For each segment k, computes [aₖ₀, aₖ₁, aₖ₂, aₖ₃] where:
            aₖ₀ = qₖ
            aₖ₁ = (qₖ₊₁-qₖ)/Tₖ - (Tₖ/6)(ωₖ₊₁+2ωₖ)
            aₖ₂ = ωₖ/2
            aₖ₃ = (ωₖ₊₁-ωₖ)/(6Tₖ)

        Returns
        -------
        numpy.ndarray
            Array of shape (n_segments, 4) with coefficients for each segment

        Notes
        -----
        These coefficients define the cubic polynomial for each segment:
        q(t) = aₖ₀ + aₖ₁(t-tₖ) + aₖ₂(t-tₖ)² + aₖ₃(t-tₖ)³

        The coefficients are derived to ensure C2 continuity (continuity of position,
        velocity, and acceleration) across segment boundaries.
        """
        n_segments = self.n - 1
        coeffs = np.zeros((n_segments, 4))

        for k in range(n_segments):
            # Equation (4.25):
            # aₖ₀ = qₖ
            coeffs[k, 0] = self.q[k]

            # aₖ₁ = (qₖ₊₁-qₖ)/Tₖ - (Tₖ/6)(ωₖ₊₁+2ωₖ)
            coeffs[k, 1] = (self.q[k + 1] - self.q[k]) / self.T[k] - (self.T[k] / 6) * (
                self.omega[k + 1] + 2 * self.omega[k]
            )

            # aₖ₂ = ωₖ/2
            coeffs[k, 2] = self.omega[k] / 2

            # aₖ₃ = (ωₖ₊₁-ωₖ)/(6Tₖ)
            coeffs[k, 3] = (self.omega[k + 1] - self.omega[k]) / (6 * self.T[k])

        if self.debug:
            print("Polynomial coefficients:")
            for k in range(n_segments):
                print(f"Segment {k}: {coeffs[k]}")

        return coeffs

    def _get_original_indices(self) -> list[int]:
        """
        Get the indices in the expanded array that correspond to original points.

        Returns
        -------
        list of int
            Indices of original points in the expanded arrays

        Notes
        -----
        This is used primarily for visualization to distinguish between
        original waypoints and extra points added to satisfy constraints.
        """
        # First point
        indices = [0]

        # Interior original points - using list.extend instead of append in a loop
        indices.extend([i + 1 for i in range(1, self.n_orig - 1)])  # +1 because we inserted q₁

        # Last point
        indices.append(self.n - 1)

        return indices

    def evaluate(self, t: float | list[float] | np.ndarray) -> float | np.ndarray:
        """
        Evaluate the spline at time t.

        Parameters
        ----------
        t : float or array_like
            Time point or array of time points

        Returns
        -------
        float or numpy.ndarray
            Position(s) at the specified time(s)

        Examples
        --------
        >>> # Evaluate at a single time point
        >>> spline.evaluate(1.5)
        >>> # Evaluate at multiple time points
        >>> spline.evaluate(np.linspace(0, 3, 100))
        """
        t_array = np.atleast_1d(t)
        result = np.zeros_like(t_array, dtype=float)

        for i, ti in enumerate(t_array):
            # Find segment containing ti
            if ti <= self.t[0]:
                # Before start of trajectory
                segment = 0
                tau = 0
            elif ti >= self.t[-1]:
                # After end of trajectory
                segment = self.n - 2
                tau = self.T[segment]
            else:
                # Within trajectory
                segment = np.searchsorted(self.t, ti, side="right") - 1
                tau = ti - self.t[segment]

            # Evaluate polynomial: aₖ₀ + aₖ₁(t-tₖ) + aₖ₂(t-tₖ)² + aₖ₃(t-tₖ)³
            c = self.coeffs[segment]
            result[i] = c[0] + c[1] * tau + c[2] * tau**2 + c[3] * tau**3

        return result[0] if len(result) == 1 else result

    def evaluate_velocity(self, t: float | list[float] | np.ndarray) -> float | np.ndarray:
        """
        Evaluate the velocity at time t.

        Parameters
        ----------
        t : float or array_like
            Time point or array of time points

        Returns
        -------
        float or numpy.ndarray
            Velocity at the specified time(s)

        Examples
        --------
        >>> # Evaluate velocity at a single time point
        >>> spline.evaluate_velocity(1.5)
        >>> # Evaluate velocity at multiple time points
        >>> spline.evaluate_velocity(np.linspace(0, 3, 100))
        """
        t_array = np.atleast_1d(t)
        result = np.zeros_like(t_array, dtype=float)

        for i, ti in enumerate(t_array):
            # Find segment containing ti
            if ti <= self.t[0]:
                segment = 0
                tau = 0
            elif ti >= self.t[-1]:
                segment = self.n - 2
                tau = self.T[segment]
            else:
                segment = np.searchsorted(self.t, ti, side="right") - 1
                tau = ti - self.t[segment]

            # Evaluate derivative: aₖ₁ + 2aₖ₂(t-tₖ) + 3aₖ₃(t-tₖ)²
            c = self.coeffs[segment]
            result[i] = c[1] + 2 * c[2] * tau + 3 * c[3] * tau**2

        return result[0] if len(result) == 1 else result

    def evaluate_acceleration(self, t: float | list[float] | np.ndarray) -> float | np.ndarray:
        """
        Evaluate the acceleration at time t.

        Parameters
        ----------
        t : float or array_like
            Time point or array of time points

        Returns
        -------
        float or numpy.ndarray
            Acceleration at the specified time(s)

        Examples
        --------
        >>> # Evaluate acceleration at a single time point
        >>> spline.evaluate_acceleration(1.5)
        >>> # Evaluate acceleration at multiple time points
        >>> spline.evaluate_acceleration(np.linspace(0, 3, 100))
        """
        t_array = np.atleast_1d(t)
        result = np.zeros_like(t_array, dtype=float)

        for i, ti in enumerate(t_array):
            # Find segment containing ti
            if ti <= self.t[0]:
                segment = 0
                tau = 0
            elif ti >= self.t[-1]:
                segment = self.n - 2
                tau = self.T[segment]
            else:
                segment = np.searchsorted(self.t, ti, side="right") - 1
                tau = ti - self.t[segment]

            # Evaluate second derivative: 2aₖ₂ + 6aₖ₃(t-tₖ)
            c = self.coeffs[segment]
            result[i] = 2 * c[2] + 6 * c[3] * tau

        return result[0] if len(result) == 1 else result

    def plot(self, num_points: int = 1000) -> None:
        """
        Plot the spline trajectory with velocity and acceleration profiles.

        Parameters
        ----------
        num_points : int, optional
            Number of points for smooth plotting. Default is 1000

        Returns
        -------
        None
            Displays the plot using matplotlib

        Notes
        -----
        This method creates a figure with three subplots showing:
        1. Position trajectory with original and extra waypoints
        2. Velocity profile with initial and final velocities marked
        3. Acceleration profile with initial and final accelerations marked

        Original waypoints are shown as red circles, while extra points
        are shown as green x-marks.
        """
        # Generate evaluation points
        t_eval = np.linspace(self.t[0], self.t[-1], num_points)

        # Evaluate the spline and its derivatives
        q = self.evaluate(t_eval)
        v = self.evaluate_velocity(t_eval)
        a = self.evaluate_acceleration(t_eval)

        # Get original points for plotting
        original_t = [self.t[i] for i in self.original_indices]
        original_q = [self.q[i] for i in self.original_indices]

        # Get extra points
        extra_indices = [i for i in range(self.n) if i not in self.original_indices]
        extra_t = [self.t[i] for i in extra_indices]
        extra_q = [self.q[i] for i in extra_indices]

        # Create a figure with three subplots
        _fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # Position plot
        ax1.plot(t_eval, q, "b-", linewidth=2, label="Spline")
        ax1.plot(original_t, original_q, "ro", markersize=8, label="Original Points")
        ax1.plot(extra_t, extra_q, "gx", markersize=8, label="Extra Points")
        ax1.set_ylabel("Position")
        ax1.grid(True)
        ax1.legend()
        ax1.set_title("Cubic Spline with Velocity and Acceleration Constraints")

        # Velocity plot
        ax2.plot(t_eval, v, "g-", linewidth=2)
        ax2.plot(self.t[0], self.v0, "bo", markersize=6, label=f"Initial v: {self.v0}")
        ax2.plot(self.t[-1], self.vn, "bo", markersize=6, label=f"Final v: {self.vn}")
        ax2.set_ylabel("Velocity")
        ax2.grid(True)
        ax2.legend()

        # Acceleration plot
        ax3.plot(t_eval, a, "r-", linewidth=2)
        ax3.plot(self.t[0], self.a0, "bo", markersize=6, label=f"Initial a: {self.a0}")
        ax3.plot(self.t[-1], self.an, "bo", markersize=6, label=f"Final a: {self.an}")
        ax3.set_ylabel("Acceleration")
        ax3.set_xlabel("Time")
        ax3.grid(True)
        ax3.legend()

        plt.tight_layout()
        plt.show()
