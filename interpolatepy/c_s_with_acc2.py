from dataclasses import dataclass

import numpy as np

from interpolatepy.cubic_spline import CubicSpline


@dataclass
class SplineParameters:
    """
    Container for spline initialization parameters.

    Parameters
    ----------
    v0 : float, default=0.0
        Initial velocity constraint at the start of the spline.
    vn : float, default=0.0
        Final velocity constraint at the end of the spline.
    a0 : float, optional
        Initial acceleration constraint at the start of the spline.
        If provided, the first segment will use a quintic polynomial.
    an : float, optional
        Final acceleration constraint at the end of the spline.
        If provided, the last segment will use a quintic polynomial.
    debug : bool, default=False
        If True, prints detailed information about the spline calculation.
    """

    v0: float = 0.0
    vn: float = 0.0
    a0: float | None = None
    an: float | None = None
    debug: bool = False


class CubicSplineWithAcceleration2(CubicSpline):
    """
    Cubic spline trajectory planning with initial and final acceleration constraints.

    This class extends CubicSpline to handle initial and final acceleration constraints
    by using 5th degree polynomials for the first and last segments, as mentioned in
    section 4.4.4 of the paper.

    The spline consists of cubic polynomial segments for interior segments and
    optional quintic polynomial segments for the first and/or last segment when
    acceleration constraints are specified.

    Parameters
    ----------
    t_points : array_like
        Array of time points (t0, t1, ..., tn).
    q_points : array_like
        Array of position points (q0, q1, ..., qn).
    params : SplineParameters, optional
        Spline parameters including initial/final velocities and accelerations.
        If None, default parameters will be used.

    Attributes
    ----------
    t_points : ndarray
        Array of time points.
    q_points : ndarray
        Array of position points.
    velocities : ndarray
        Array of velocities at each point.
    t_intervals : ndarray
        Array of time intervals between consecutive points.
    n : int
        Number of segments.
    coefficients : ndarray
        Array of coefficients for each cubic segment.
    a0 : float or None
        Initial acceleration constraint.
    an : float or None
        Final acceleration constraint.
    quintic_coeffs : dict
        Dictionary containing quintic coefficients for first and/or last segments.
    """

    def __init__(
        self,
        t_points: list[float] | np.ndarray,
        q_points: list[float] | np.ndarray,
        params: SplineParameters | None = None,
    ) -> None:
        """
        Initialize a cubic spline trajectory with optional initial and final accelerations.

        Parameters
        ----------
        t_points : array_like
            Array of time points (t0, t1, ..., tn).
        q_points : array_like
            Array of position points (q0, q1, ..., qn).
        params : SplineParameters, optional
            Spline parameters including initial/final velocities and accelerations.
            If None, default parameters will be used.
        """
        # Set default parameters if not provided
        if params is None:
            params = SplineParameters()

        # Initialize the parent class to compute the basic cubic spline
        super().__init__(t_points, q_points, params.v0, params.vn, params.debug)

        # Store the acceleration constraints
        self.a0 = params.a0
        self.an = params.an

        # Replace the first and/or last segment with a 5th degree polynomial if needed
        if params.a0 is not None:
            self._replace_first_segment_with_quintic()

        if params.an is not None:
            self._replace_last_segment_with_quintic()

    def _replace_first_segment_with_quintic(self) -> None:
        """
        Replace the first segment with a 5th degree polynomial to satisfy
        initial acceleration constraint.

        This method computes the coefficients of a quintic polynomial for the first
        segment to satisfy the position, velocity, and acceleration constraints at
        both endpoints of the segment.

        The quintic polynomial has the form:
        p(tau) = b0 + b1*tau + b2*tau^2 + b3*tau^3 + b4*tau^4 + b5*tau^5

        With constraints:
        p(0) = q0, p'(0) = v0, p''(0) = a0
        p(T) = q1, p'(T) = v1, p''(T) = a1

        Where T is the duration of the first segment.

        The coefficients are stored in self.quintic_coeffs["first"].
        """
        # Get the time points and positions for the first segment
        t0, t1 = self.t_points[0], self.t_points[1]
        q0, q1 = self.q_points[0], self.q_points[1]
        v0, v1 = self.velocities[0], self.velocities[1]
        a0 = self.a0

        # Calculate the acceleration at the end of the first segment
        # For a cubic polynomial a(t) = 2*a2 + 6*a3*t
        a1 = 2 * self.coefficients[0, 2] + 6 * self.coefficients[0, 3] * self.t_intervals[0]

        # Compute the coefficients of the quintic polynomial
        # p(tau) = b0 + b1*tau + b2*tau^2 + b3*tau^3 + b4*tau^4 + b5*tau^5
        # with constraints:
        # p(0) = q0, p'(0) = v0, p''(0) = a0
        # p(T) = q1, p'(T) = v1, p''(T) = a1
        # where T = t1 - t0

        t_interval = t1 - t0

        # Set up the system of equations
        a_matrix = np.array(
            [
                [1, 0, 0, 0, 0, 0],  # p(0) = q0
                [0, 1, 0, 0, 0, 0],  # p'(0) = v0
                [0, 0, 2, 0, 0, 0],  # p''(0) = a0
                [
                    1,
                    t_interval,
                    t_interval**2,
                    t_interval**3,
                    t_interval**4,
                    t_interval**5,
                ],  # p(T) = q1
                [
                    0,
                    1,
                    2 * t_interval,
                    3 * t_interval**2,
                    4 * t_interval**3,
                    5 * t_interval**4,
                ],  # p'(T) = v1
                [0, 0, 2, 6 * t_interval, 12 * t_interval**2, 20 * t_interval**3],  # p''(T) = a1
            ]
        )

        b = np.array([q0, v0, a0, q1, v1, a1])

        # Solve for the coefficients
        quintic_coeffs = np.linalg.solve(a_matrix, b)

        # Store the quintic coefficients for later use
        if not hasattr(self, "quintic_coeffs"):
            self.quintic_coeffs = {}

        self.quintic_coeffs["first"] = quintic_coeffs

        if self.debug:
            print("\nReplaced first segment with quintic polynomial:")
            print(f"  b0 = {quintic_coeffs[0]}")
            print(f"  b1 = {quintic_coeffs[1]}")
            print(f"  b2 = {quintic_coeffs[2]}")
            print(f"  b3 = {quintic_coeffs[3]}")
            print(f"  b4 = {quintic_coeffs[4]}")
            print(f"  b5 = {quintic_coeffs[5]}")

    def _replace_last_segment_with_quintic(self) -> None:
        """
        Replace the last segment with a 5th degree polynomial to satisfy
        final acceleration constraint.

        This method computes the coefficients of a quintic polynomial for the last
        segment to satisfy the position, velocity, and acceleration constraints at
        both endpoints of the segment.

        The quintic polynomial has the form:
        p(tau) = b0 + b1*tau + b2*tau^2 + b3*tau^3 + b4*tau^4 + b5*tau^5

        With constraints:
        p(0) = qn_1, p'(0) = vn_1, p''(0) = an_1
        p(T) = qn, p'(T) = vn, p''(T) = an

        Where T is the duration of the last segment.

        The coefficients are stored in self.quintic_coeffs["last"].
        """
        # Get the time points and positions for the last segment
        tn_1, tn = self.t_points[-2], self.t_points[-1]
        qn_1, qn = self.q_points[-2], self.q_points[-1]
        vn_1, vn = self.velocities[-2], self.velocities[-1]
        an = self.an

        # Calculate the acceleration at the start of the last segment
        # For a cubic polynomial a(t) = 2*a2
        an_1 = 2 * self.coefficients[-1, 2]

        # Compute the coefficients of the quintic polynomial
        # p(tau) = b0 + b1*tau + b2*tau^2 + b3*tau^3 + b4*tau^4 + b5*tau^5
        # with constraints:
        # p(0) = qn_1, p'(0) = vn_1, p''(0) = an_1
        # p(T) = qn, p'(T) = vn, p''(T) = an
        # where T = tn - tn_1

        t_interval = tn - tn_1

        # Set up the system of equations
        a_matrix = np.array(
            [
                [1, 0, 0, 0, 0, 0],  # p(0) = qn_1
                [0, 1, 0, 0, 0, 0],  # p'(0) = vn_1
                [0, 0, 2, 0, 0, 0],  # p''(0) = an_1
                [
                    1,
                    t_interval,
                    t_interval**2,
                    t_interval**3,
                    t_interval**4,
                    t_interval**5,
                ],  # p(T) = qn
                [
                    0,
                    1,
                    2 * t_interval,
                    3 * t_interval**2,
                    4 * t_interval**3,
                    5 * t_interval**4,
                ],  # p'(T) = vn
                [0, 0, 2, 6 * t_interval, 12 * t_interval**2, 20 * t_interval**3],  # p''(T) = an
            ]
        )

        b = np.array([qn_1, vn_1, an_1, qn, vn, an])

        # Solve for the coefficients
        quintic_coeffs = np.linalg.solve(a_matrix, b)

        # Store the quintic coefficients for later use
        if not hasattr(self, "quintic_coeffs"):
            self.quintic_coeffs = {}

        self.quintic_coeffs["last"] = quintic_coeffs

        if self.debug:
            print("\nReplaced last segment with quintic polynomial:")
            print(f"  b0 = {quintic_coeffs[0]}")
            print(f"  b1 = {quintic_coeffs[1]}")
            print(f"  b2 = {quintic_coeffs[2]}")
            print(f"  b3 = {quintic_coeffs[3]}")
            print(f"  b4 = {quintic_coeffs[4]}")
            print(f"  b5 = {quintic_coeffs[5]}")

    def evaluate(self, t: float | np.ndarray) -> float | np.ndarray:
        """
        Evaluate the spline at time t.

        Parameters
        ----------
        t : float or ndarray
            Time point or array of time points at which to evaluate the spline.

        Returns
        -------
        float or ndarray
            Position(s) at the specified time(s). Returns a float if a single time
            point is provided, or an ndarray if an array of time points is provided.

        Notes
        -----
        For time values outside the spline range:
        - If t < t_points[0], returns the position at t_points[0]
        - If t > t_points[-1], returns the position at t_points[-1]

        For the first and last segments, quintic polynomials are used if acceleration
        constraints were specified. For all other segments, cubic polynomials are used.
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

            # Check if this segment uses a quintic polynomial
            if k == 0 and hasattr(self, "quintic_coeffs") and "first" in self.quintic_coeffs:
                # Use the quintic polynomial for the first segment
                b = self.quintic_coeffs["first"]
                result[i] = (
                    b[0]
                    + b[1] * tau
                    + b[2] * tau**2
                    + b[3] * tau**3
                    + b[4] * tau**4
                    + b[5] * tau**5
                )
            elif (
                k == self.n - 1
                and hasattr(self, "quintic_coeffs")
                and "last" in self.quintic_coeffs
            ):
                # Use the quintic polynomial for the last segment
                b = self.quintic_coeffs["last"]
                result[i] = (
                    b[0]
                    + b[1] * tau
                    + b[2] * tau**2
                    + b[3] * tau**3
                    + b[4] * tau**4
                    + b[5] * tau**5
                )
            else:
                # Use the cubic polynomial for other segments
                a = self.coefficients[k]
                result[i] = a[0] + a[1] * tau + a[2] * tau**2 + a[3] * tau**3

        return result[0] if len(result) == 1 else result

    def evaluate_velocity(self, t: float | np.ndarray) -> float | np.ndarray:
        """
        Evaluate the velocity at time t.

        Parameters
        ----------
        t : float or ndarray
            Time point or array of time points at which to evaluate the velocity.

        Returns
        -------
        float or ndarray
            Velocity at the specified time(s). Returns a float if a single time
            point is provided, or an ndarray if an array of time points is provided.

        Notes
        -----
        For time values outside the spline range:
        - If t < t_points[0], returns the velocity at t_points[0]
        - If t > t_points[-1], returns the velocity at t_points[-1]

        For the first and last segments, derivatives of quintic polynomials are used
        if acceleration constraints were specified. For all other segments, derivatives
        of cubic polynomials are used.
        """
        t = np.atleast_1d(t)
        result = np.zeros_like(t)

        for i, ti in enumerate(t):
            # Find the segment that contains ti
            if ti <= self.t_points[0]:
                k = 0
                tau = 0
            elif ti >= self.t_points[-1]:
                k = self.n - 1
                tau = self.t_intervals[k]
            else:
                k = np.searchsorted(self.t_points, ti, side="right") - 1
                tau = ti - self.t_points[k]

            # Check if this segment uses a quintic polynomial
            if k == 0 and hasattr(self, "quintic_coeffs") and "first" in self.quintic_coeffs:
                # Use the derivative of the quintic polynomial for the first segment
                b = self.quintic_coeffs["first"]
                result[i] = (
                    b[1]
                    + 2 * b[2] * tau
                    + 3 * b[3] * tau**2
                    + 4 * b[4] * tau**3
                    + 5 * b[5] * tau**4
                )
            elif (
                k == self.n - 1
                and hasattr(self, "quintic_coeffs")
                and "last" in self.quintic_coeffs
            ):
                # Use the derivative of the quintic polynomial for the last segment
                b = self.quintic_coeffs["last"]
                result[i] = (
                    b[1]
                    + 2 * b[2] * tau
                    + 3 * b[3] * tau**2
                    + 4 * b[4] * tau**3
                    + 5 * b[5] * tau**4
                )
            else:
                # Use the derivative of the cubic polynomial for other segments
                a = self.coefficients[k]
                result[i] = a[1] + 2 * a[2] * tau + 3 * a[3] * tau**2

        return result[0] if len(result) == 1 else result

    def evaluate_acceleration(self, t: float | np.ndarray) -> float | np.ndarray:
        """
        Evaluate the acceleration at time t.

        Parameters
        ----------
        t : float or ndarray
            Time point or array of time points at which to evaluate the acceleration.

        Returns
        -------
        float or ndarray
            Acceleration at the specified time(s). Returns a float if a single time
            point is provided, or an ndarray if an array of time points is provided.

        Notes
        -----
        For time values outside the spline range:
        - If t < t_points[0], returns the acceleration at t_points[0]
        - If t > t_points[-1], returns the acceleration at t_points[-1]

        For the first and last segments, second derivatives of quintic polynomials are
        used if acceleration constraints were specified. For all other segments, second
        derivatives of cubic polynomials are used.
        """
        t = np.atleast_1d(t)
        result = np.zeros_like(t)

        for i, ti in enumerate(t):
            # Find the segment that contains ti
            if ti <= self.t_points[0]:
                k = 0
                tau = 0
            elif ti >= self.t_points[-1]:
                k = self.n - 1
                tau = self.t_intervals[k]
            else:
                k = np.searchsorted(self.t_points, ti, side="right") - 1
                tau = ti - self.t_points[k]

            # Check if this segment uses a quintic polynomial
            if k == 0 and hasattr(self, "quintic_coeffs") and "first" in self.quintic_coeffs:
                # Use the second derivative of the quintic polynomial for the first segment
                b = self.quintic_coeffs["first"]
                result[i] = 2 * b[2] + 6 * b[3] * tau + 12 * b[4] * tau**2 + 20 * b[5] * tau**3
            elif (
                k == self.n - 1
                and hasattr(self, "quintic_coeffs")
                and "last" in self.quintic_coeffs
            ):
                # Use the second derivative of the quintic polynomial for the last segment
                b = self.quintic_coeffs["last"]
                result[i] = 2 * b[2] + 6 * b[3] * tau + 12 * b[4] * tau**2 + 20 * b[5] * tau**3
            else:
                # Use the second derivative of the cubic polynomial for other segments
                a = self.coefficients[k]
                result[i] = 2 * a[2] + 6 * a[3] * tau

        return result[0] if len(result) == 1 else result
