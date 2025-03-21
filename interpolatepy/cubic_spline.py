import matplotlib.pyplot as plt
import numpy as np

from interpolatepy.tridiagonal_inv import solve_tridiagonal


class CubicSpline:
    """
    Cubic spline trajectory planning implementation.

    This class implements the cubic spline algorithm described in the document.
    It generates a smooth trajectory passing through specified waypoints with
    continuous velocity and acceleration profiles.
    """

    def __init__(
        self,
        t_points: list[float] | np.ndarray,
        q_points: list[float] | np.ndarray,
        v0: float = 0.0,
        vn: float = 0.0,
        debug: bool = False,
    ) -> None:
        """
        Initialize a cubic spline trajectory.

        Args:
            t_points: List or array of time points (t0, t1, ..., tn)
            q_points: List or array of position points (q0, q1, ..., qn)
            v0: Initial velocity at t0 (default: 0.0)
            vn: Final velocity at tn (default: 0.0)
            debug: Whether to print debug information (default: False)
        """
        # Ensure inputs are numpy arrays
        self.t_points = np.array(t_points, dtype=float)
        self.q_points = np.array(q_points, dtype=float)
        self.v0 = float(v0)
        self.vn = float(vn)
        self.debug = debug

        # Check input validity
        if len(self.t_points) != len(self.q_points):
            raise ValueError("Time points and position points must have the same length")

        if not np.all(np.diff(self.t_points) > 0):
            raise ValueError("Time points must be strictly increasing")

        # Compute time intervals
        self.n = len(self.t_points) - 1
        self.t_intervals = np.diff(self.t_points)

        # Compute intermediate velocities and coefficients
        self.velocities = self._compute_velocities()
        self.coefficients = self._compute_coefficients()

    def _compute_velocities(self) -> np.ndarray:
        """
        Compute the velocities at intermediate points by solving
        the tridiagonal system described in the document.

        This method implements the mathematical formulation from pages 4-5 of the document,
        corresponding to equation (17). It sets up and solves the tridiagonal system A*v = c
        to find the intermediate velocities, which ensures C2 continuity of the spline.

        Returns:
            Array of velocities [v0, v1, ..., vn]
        """
        n = self.n
        t_intervals = self.t_intervals
        q = self.q_points

        # Create the tridiagonal matrix A as shown in the document:
        # Matrix A has the following structure:
        # [2(T0+T1)    T0         0         ...        0       ]
        # [T2        2(T1+T2)     T1        0         ...      ]
        # [0          T3        2(T2+T3)    T2        ...      ]
        # [...        ...        ...        ...       ...      ]
        # [0          ...        0         Tn-2    2(Tn-3+Tn-2) Tn-3]
        # [0          ...        0          0       Tn-1      2(Tn-2+Tn-1)]

        if n == 1:
            # Special case: only one segment
            # We know v0 and vn, so no need to solve system
            return np.array([self.v0, self.vn])

        # Create the right-hand side vector c from equation (17) in the document
        # c_i = 3/(T_i*T_{i+1}) * [T_i^2*(q_{i+2}-q_{i+1}) + T_{i+1}^2*(q_{i+1}-q_i)]
        rhs = np.zeros(n - 1)

        for i in range(n - 1):
            rhs[i] = (
                3
                / (t_intervals[i] * t_intervals[i + 1])
                * (
                    t_intervals[i] ** 2 * (q[i + 2] - q[i + 1])
                    + t_intervals[i + 1] ** 2 * (q[i + 1] - q[i])
                )
            )

        # Adjust the right-hand side to account for known boundary velocities v0 and vn
        # These adjustments come from moving the terms with known velocities to the RHS
        if n > 1:
            # First equation: subtract T_1*v_0 from RHS
            rhs[0] -= t_intervals[1] * self.v0

            # Last equation: subtract T_{n-2}*v_n from RHS
            rhs[-1] -= t_intervals[-2] * self.vn

        # Solve the system for the intermediate velocities v1, ..., v(n-1)
        # Case n=2 means we have exactly one intermediate velocity to solve (1x1 system)
        if n == 2:  # noqa: PLR2004
            # Special case: only one intermediate velocity to solve for
            # Simple division is sufficient (1x1 system)
            main_diag_value = 2 * (t_intervals[0] + t_intervals[1])
            v_intermediate = rhs / main_diag_value
        else:
            # Instead of building the full matrix, extract the diagonal elements for the
            # tridiagonal solver

            # Main diagonal: 2(T_i + T_{i+1})
            main_diag = np.zeros(n - 1)
            for i in range(n - 1):
                main_diag[i] = 2 * (t_intervals[i] + t_intervals[i + 1])

            # Lower diagonal: T_{i+1} (first element not used in solve_tridiagonal)
            lower_diag = np.zeros(n - 1)
            for i in range(1, n - 1):
                lower_diag[i] = t_intervals[i + 1]

            # Upper diagonal: T_i (last element not used in solve_tridiagonal)
            upper_diag = np.zeros(n - 1)
            for i in range(n - 2):
                upper_diag[i] = t_intervals[i]

            # Print debug information only if debug is enabled
            if self.debug:
                print("\nTridiagonal Matrix components:")
                print("Main diagonal:", main_diag)
                print("Lower diagonal:", lower_diag)
                print("Upper diagonal:", upper_diag)
                print("Right-hand side vector:", rhs)

            # Solve the tridiagonal system using the Thomas algorithm
            v_intermediate = solve_tridiagonal(lower_diag, main_diag, upper_diag, rhs)

            # Print the intermediate velocities if debug is enabled
            if self.debug:
                print("\nIntermediate velocities v_1 to v_{n-1}:")
                print(v_intermediate)

        # Construct the full velocity array by including the known boundary velocities
        velocities = np.zeros(n + 1)
        velocities[0] = self.v0  # Initial velocity (given)
        velocities[1:-1] = v_intermediate  # Intermediate velocities (computed)
        velocities[-1] = self.vn  # Final velocity (given)

        # Print the complete velocity vector if debug is enabled
        if self.debug:
            print("\nComplete velocity vector v:")
            print(velocities)

        return velocities

    def _compute_coefficients(self) -> np.ndarray:
        """
        Compute the coefficients for each cubic polynomial segment.
        For each segment k, we compute:
        - ak0: constant term
        - ak1: coefficient of (t-tk)
        - ak2: coefficient of (t-tk)^2
        - ak3: coefficient of (t-tk)^3

        Returns:
            Array of shape (n, 4) containing the coefficients
        """
        n = self.n
        t_intervals = self.t_intervals
        q = self.q_points
        v = self.velocities

        coeffs = np.zeros((n, 4))

        for k in range(n):
            coeffs[k, 0] = q[k]  # ak0 = qk
            coeffs[k, 1] = v[k]  # ak1 = vk

            # Compute ak2 and ak3
            coeffs[k, 2] = (1 / t_intervals[k]) * (
                (3 * (q[k + 1] - q[k]) / t_intervals[k]) - 2 * v[k] - v[k + 1]
            )
            coeffs[k, 3] = (1 / t_intervals[k] ** 2) * (
                (2 * (q[k] - q[k + 1]) / t_intervals[k]) + v[k] + v[k + 1]
            )

            # Print detailed coefficient calculation if debug is enabled
            if self.debug:
                print(f"\nCoefficient calculation for segment {k}:")
                print(f"  ak0 = {coeffs[k, 0]}")
                print(f"  ak1 = {coeffs[k, 1]}")
                print(f"  ak2 = {coeffs[k, 2]}")
                print(f"  ak3 = {coeffs[k, 3]}")

        return coeffs

    def evaluate(self, t: float | np.ndarray) -> float | np.ndarray:
        """
        Evaluate the spline at time t.

        Args:
            t: Time point or array of time points

        Returns:
            Position(s) at the specified time(s)
        """
        t = np.atleast_1d(t)
        result = np.zeros_like(t)

        for i, ti in enumerate(t):
            # Find the segment that contains ti
            if ti <= self.t_points[0]:
                # Before the start of the trajectory
                k = 0
                tau = 0
            elif ti >= self.t_points[-1]:
                # After the end of the trajectory
                k = self.n - 1
                tau = self.t_intervals[k]
            else:
                # Within the trajectory
                # Find the largest k such that t_k <= ti
                k = np.searchsorted(self.t_points, ti, side="right") - 1
                tau = ti - self.t_points[k]

            # Evaluate the polynomial
            a = self.coefficients[k]
            result[i] = a[0] + a[1] * tau + a[2] * tau**2 + a[3] * tau**3

        return result[0] if len(result) == 1 else result

    def evaluate_velocity(self, t: float | np.ndarray) -> float | np.ndarray:
        """
        Evaluate the velocity at time t.

        Args:
            t: Time point or array of time points

        Returns:
            Velocity at the specified time(s)
        """
        t = np.atleast_1d(t)
        result = np.zeros_like(t)

        for i, ti in enumerate(t):
            # Find the segment that contains ti
            if ti <= self.t_points[0]:
                # Before the start of the trajectory
                k = 0
                tau = 0
            elif ti >= self.t_points[-1]:
                # After the end of the trajectory
                k = self.n - 1
                tau = self.t_intervals[k]
            else:
                # Within the trajectory
                k = np.searchsorted(self.t_points, ti, side="right") - 1
                tau = ti - self.t_points[k]

            # Evaluate the derivative of the polynomial
            a = self.coefficients[k]
            result[i] = a[1] + 2 * a[2] * tau + 3 * a[3] * tau**2

        return result[0] if len(result) == 1 else result

    def evaluate_acceleration(self, t: float | np.ndarray) -> float | np.ndarray:
        """
        Evaluate the acceleration at time t.

        Args:
            t: Time point or array of time points

        Returns:
            Acceleration at the specified time(s)
        """
        t = np.atleast_1d(t)
        result = np.zeros_like(t)

        for i, ti in enumerate(t):
            # Find the segment that contains ti
            if ti <= self.t_points[0]:
                # Before the start of the trajectory
                k = 0
                tau = 0
            elif ti >= self.t_points[-1]:
                # After the end of the trajectory
                k = self.n - 1
                tau = self.t_intervals[k]
            else:
                # Within the trajectory
                k = np.searchsorted(self.t_points, ti, side="right") - 1
                tau = ti - self.t_points[k]

            # Evaluate the second derivative of the polynomial
            a = self.coefficients[k]
            result[i] = 2 * a[2] + 6 * a[3] * tau

        return result[0] if len(result) == 1 else result

    def plot(self, num_points: int = 1000) -> None:
        """
        Plot the spline trajectory along with its velocity and acceleration profiles.

        Args:
            num_points: Number of points to use for plotting
        """
        t_min, t_max = self.t_points[0], self.t_points[-1]
        t = np.linspace(t_min, t_max, num_points)

        q = self.evaluate(t)
        v = self.evaluate_velocity(t)
        a = self.evaluate_acceleration(t)

        _fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # Position plot
        ax1.plot(t, q, "b-", linewidth=2)
        ax1.plot(self.t_points, self.q_points, "ro", markersize=8)
        ax1.set_ylabel("Position")
        ax1.grid(True)
        ax1.set_title("Cubic Spline Trajectory")

        # Velocity plot
        ax2.plot(t, v, "g-", linewidth=2)
        ax2.plot(self.t_points, self.velocities, "ro", markersize=6)
        ax2.set_ylabel("Velocity")
        ax2.grid(True)

        # Acceleration plot
        ax3.plot(t, a, "r-", linewidth=2)
        ax3.set_ylabel("Acceleration")
        ax3.set_xlabel("Time")
        ax3.grid(True)

        plt.tight_layout()
        plt.show()
