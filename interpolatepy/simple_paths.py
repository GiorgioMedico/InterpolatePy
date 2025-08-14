"""Module for simple geometric path primitives.

Provides basic geometric path classes for linear and circular trajectories
used in 3D path planning applications. These primitives support evaluation
of position, velocity, and acceleration profiles along parametric paths.
"""

import numpy as np


class LinearPath:
    """
    A linear path between two points in 3D space.

    This class represents a straight-line trajectory from an initial point to a final point,
    providing methods to evaluate position, velocity, and acceleration along the path.

    Parameters
    ----------
    pi : array_like
        Initial point coordinates [x, y, z].
    pf : array_like
        Final point coordinates [x, y, z].

    Attributes
    ----------
    pi : np.ndarray
        Initial point coordinates.
    pf : np.ndarray
        Final point coordinates.
    length : float
        Total length of the linear path.
    tangent : np.ndarray
        Unit tangent vector (constant for linear path).

    Examples
    --------
    >>> import numpy as np
    >>> # Create a linear path from origin to point (1, 1, 1)
    >>> pi = np.array([0, 0, 0])
    >>> pf = np.array([1, 1, 1])
    >>> path = LinearPath(pi, pf)
    >>>
    >>> # Evaluate position at half the path length
    >>> midpoint = path.position(path.length / 2)
    >>> print(midpoint)  # Should be [0.5, 0.5, 0.5]
    >>>
    >>> # Generate complete trajectory
    >>> trajectory = path.all_traj(num_points=10)
    >>> positions = trajectory['position']
    >>> velocities = trajectory['velocity']
    """

    def __init__(self, pi: np.ndarray, pf: np.ndarray) -> None:
        """
        Initialize a linear path from point pi to point pf.

        Parameters
        ----------
        pi : array_like
            Initial point coordinates [x, y, z].
        pf : array_like
            Final point coordinates [x, y, z].
        """
        self.pi: np.ndarray = np.array(pi)
        self.pf: np.ndarray = np.array(pf)
        self.length: float = float(np.linalg.norm(self.pf - self.pi))

        # Unit tangent vector (constant for linear path)
        if self.length > 0:
            self.tangent: np.ndarray = (self.pf - self.pi) / self.length
        else:
            self.tangent = np.zeros(3)

    def position(self, s: float) -> np.ndarray:
        """
        Calculate position at arc length s.

        Parameters
        ----------
        s : float or array_like
            Arc length parameter(s).

        Returns
        -------
        np.ndarray
            Position vector(s) at the specified arc length(s).
        """
        # Ensure s is within valid range
        s = np.clip(s, 0, self.length)

        # Equation 4.34: p(s) = pi + (s/||pf-pi||)(pf-pi)
        return self.pi + (s / self.length) * (self.pf - self.pi) if self.length > 0 else self.pi

    def velocity(self, _s: float | None = None) -> np.ndarray:
        """
        Calculate first derivative with respect to arc length.
        For linear path, this is constant and doesn't depend on s.

        Parameters
        ----------
        _s : float, optional
            Arc length parameter (not used for linear path).

        Returns
        -------
        np.ndarray
            Velocity (tangent) vector.
        """
        # Equation 4.35: dp/ds = (pf-pi)/||pf-pi||
        return self.tangent

    @staticmethod
    def acceleration(_s: float | None = None) -> np.ndarray:
        """
        Calculate second derivative with respect to arc length.
        For linear path, this is always zero.

        Parameters
        ----------
        _s : float, optional
            Arc length parameter (not used for linear path).

        Returns
        -------
        np.ndarray
            Acceleration vector (always zero for linear path).
        """
        # Equation 4.36: d²p/ds² = 0
        return np.zeros(3)

    def evaluate_at(self, s_values: float | list[float] | np.ndarray) -> dict[str, np.ndarray]:
        """
        Evaluate position, velocity, and acceleration at specific arc length values.

        Parameters
        ----------
        s_values : float or array_like
            Arc length parameter(s).

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing arrays for position, velocity, and acceleration.
            Each array has shape (n, 3) where n is the number of s values.
        """
        # Convert scalar to array if needed
        s_values_arr: np.ndarray = np.array([s_values]) if np.isscalar(s_values) else np.array(s_values)

        # Clip values to valid range
        s_clipped = np.clip(s_values_arr, 0, self.length)

        # Initialize result arrays
        n = len(s_clipped)
        positions = np.zeros((n, 3))
        velocities = np.zeros((n, 3))
        accelerations = np.zeros((n, 3))

        # Calculate positions for each s
        for i, s in enumerate(s_clipped):
            positions[i] = self.position(s)

        # For linear path, velocity is constant and acceleration is zero
        velocities[:] = self.velocity()
        # accelerations already initialized to zeros

        return {
            "position": positions,
            "velocity": velocities,
            "acceleration": accelerations,
            "s": s_clipped,
        }

    def all_traj(self, num_points: int = 100) -> dict[str, np.ndarray]:
        """
        Generate a complete trajectory along the entire linear path.

        Parameters
        ----------
        num_points : int, optional
            Number of points to generate along the path. Default is 100.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing arrays for position, velocity, and acceleration.
            Each array has shape (num_points, 3).
        """
        # Generate evenly spaced points along the entire path
        s_values = np.linspace(0, self.length, num_points)

        # Use evaluate_at to get the trajectory data
        return self.evaluate_at(s_values)


class CircularPath:
    """
    A circular path in 3D space defined by an axis and a point on the circle.

    This class represents a circular trajectory defined by an axis vector,
    a point on the axis, and a point on the circle. It provides methods to
    evaluate position, velocity, and acceleration along the circular arc.

    Parameters
    ----------
    r : array_like
        Unit vector of circle axis.
    d : array_like
        Position vector of a point on the circle axis.
    pi : array_like
        Position vector of a point on the circle.

    Attributes
    ----------
    r : np.ndarray
        Normalized axis vector of the circle.
    d : np.ndarray
        Position vector of a point on the circle axis.
    pi : np.ndarray
        Position vector of a point on the circle.
    center : np.ndarray
        Center point of the circle.
    radius : float
        Radius of the circle.
    R : np.ndarray
        Rotation matrix from local to global coordinates.

    Raises
    ------
    ValueError
        If the point pi lies on the circle axis.

    Examples
    --------
    >>> import numpy as np
    >>> # Create a circular path in the XY plane centered at origin
    >>> r = np.array([0, 0, 1])     # Z-axis
    >>> d = np.array([0, 0, 0])     # Origin on axis
    >>> pi = np.array([1, 0, 0])    # Point on circle
    >>> circle = CircularPath(r, d, pi)
    >>>
    >>> # Evaluate position at quarter circle
    >>> quarter_arc = np.pi * circle.radius / 2
    >>> pos = circle.position(quarter_arc)
    >>>
    >>> # Generate complete trajectory around the circle
    >>> trajectory = circle.all_traj(num_points=100)
    >>> positions = trajectory['position']
    """

    def __init__(self, r: np.ndarray, d: np.ndarray, pi: np.ndarray) -> None:
        """
        Initialize a circular path.

        Parameters
        ----------
        r : array_like
            Unit vector of circle axis.
        d : array_like
            Position vector of a point on the circle axis.
        pi : array_like
            Position vector of a point on the circle.
        """
        self.r: np.ndarray = np.array(r)
        self.d: np.ndarray = np.array(d)
        self.pi: np.ndarray = np.array(pi)

        # Normalize axis vector
        self.r /= np.linalg.norm(self.r)

        # Compute delta vector
        delta = self.pi - self.d

        # Check if pi is not on the axis
        if np.abs(np.dot(delta, self.r)) >= np.linalg.norm(delta):
            raise ValueError("The point pi must not be on the circle axis")

        # Compute center (equation 4.37)
        self.center = self.d + np.dot(delta, self.r) * self.r

        # Compute radius
        self.radius = np.linalg.norm(self.pi - self.center)

        # Compute rotation matrix
        x_prime = (self.pi - self.center) / self.radius  # Unit vector in x' direction
        z_prime = self.r  # Unit vector in z' direction
        y_prime = np.cross(z_prime, x_prime)  # Unit vector in y' direction

        # Rotation matrix R = [x' y' z']
        self.R = np.column_stack((x_prime, y_prime, z_prime))

    def position(self, s: float | np.ndarray) -> np.ndarray:
        """
        Calculate position at arc length s.

        Parameters
        ----------
        s : float or array_like
            Arc length parameter(s).

        Returns
        -------
        np.ndarray
            Position vector(s) at the specified arc length(s).
        """
        if np.isscalar(s):
            # Position in local coordinate system (equation 4.38)
            angle = float(np.asarray(s)) / self.radius
            p_prime = np.array(
                [
                    self.radius * np.cos(angle),
                    self.radius * np.sin(angle),
                    0.0,
                ]
            )

            # Position in global coordinate system (equation 4.39)
            return self.center + self.R @ p_prime

        # Ensure s is treated as an array/iterable
        s_array = np.asarray(s)
        positions = []
        for s_val in s_array:
            p_prime = np.array(
                [
                    self.radius * np.cos(s_val / self.radius),
                    self.radius * np.sin(s_val / self.radius),
                    0,
                ]
            )
            positions.append(self.center + self.R @ p_prime)
        return np.array(positions)

    def velocity(self, s: float) -> np.ndarray:
        """
        Calculate first derivative with respect to arc length.

        Parameters
        ----------
        s : float
            Arc length parameter.

        Returns
        -------
        np.ndarray
            Velocity (tangent) vector.
        """
        # Velocity in local coordinate system (equation 4.40)
        dp_prime_ds = np.array([-np.sin(s / self.radius), np.cos(s / self.radius), 0])

        # Velocity in global coordinate system
        return self.R @ dp_prime_ds

    def acceleration(self, s: float) -> np.ndarray:
        """
        Calculate second derivative with respect to arc length.

        Parameters
        ----------
        s : float
            Arc length parameter.

        Returns
        -------
        np.ndarray
            Acceleration vector.
        """
        # Acceleration in local coordinate system (equation 4.41)
        d2p_prime_ds2 = np.array(
            [
                -np.cos(s / self.radius) / self.radius,
                -np.sin(s / self.radius) / self.radius,
                0,
            ]
        )

        # Acceleration in global coordinate system
        return self.R @ d2p_prime_ds2

    def evaluate_at(self, s_values: float | list[float] | np.ndarray) -> dict[str, np.ndarray]:
        """
        Evaluate position, velocity, and acceleration at specific arc length values.

        Parameters
        ----------
        s_values : float or array_like
            Arc length parameter(s).

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing arrays for position, velocity, and acceleration.
            Each array has shape (n, 3) where n is the number of s values.
        """
        # Convert scalar to array if needed
        s_values_arr: np.ndarray = np.array([s_values]) if np.isscalar(s_values) else np.array(s_values)

        # Initialize result arrays
        n = len(s_values_arr)
        positions = np.zeros((n, 3))
        velocities = np.zeros((n, 3))
        accelerations = np.zeros((n, 3))

        # Calculate values for each s
        for i, s in enumerate(s_values_arr):
            positions[i] = self.position(s)
            velocities[i] = self.velocity(s)
            accelerations[i] = self.acceleration(s)

        return {
            "position": positions,
            "velocity": velocities,
            "acceleration": accelerations,
            "s": s_values_arr,
        }

    def all_traj(self, num_points: int = 100) -> dict[str, np.ndarray]:
        """
        Generate a complete trajectory around the entire circular path.

        Parameters
        ----------
        num_points : int, optional
            Number of points to generate around the circle. Default is 100.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing arrays for position, velocity, and acceleration.
            Each array has shape (num_points, 3).
        """
        # Generate evenly spaced points for a complete circle
        s_values = np.linspace(0, 2 * np.pi * self.radius, num_points)

        # Use evaluate_at to get the trajectory data
        return self.evaluate_at(s_values)
