import numpy as np


class LinearPath:
    def __init__(self, pi: np.ndarray, pf: np.ndarray) -> None:
        """
        Initialize a linear path from point pi to point pf.

        Parameters:
            pi (array-like): Initial point coordinates [x, y, z]
            pf (array-like): Final point coordinates [x, y, z]
        """
        self.pi: np.ndarray = np.array(pi)
        self.pf: np.ndarray = np.array(pf)
        self.length: float = np.linalg.norm(self.pf - self.pi)

        # Unit tangent vector (constant for linear path)
        if self.length > 0:
            self.tangent: np.ndarray = (self.pf - self.pi) / self.length
        else:
            self.tangent = np.zeros(3)

    def position(self, s: float) -> np.ndarray:
        """
        Calculate position at arc length s.

        Parameters:
            s (float or array): Arc length parameter(s)

        Returns:
            numpy.ndarray: Position vector(s)
        """
        # Ensure s is within valid range
        s = np.clip(s, 0, self.length)

        # Equation 4.34: p(s) = pi + (s/||pf-pi||)(pf-pi)
        return self.pi + (s / self.length) * (self.pf - self.pi) if self.length > 0 else self.pi

    def velocity(self, _s: float | None = None) -> np.ndarray:
        """
        Calculate first derivative with respect to arc length.
        For linear path, this is constant and doesn't depend on s.

        Parameters:
            _s (float, optional): Arc length parameter (not used for linear path)

        Returns:
            numpy.ndarray: Velocity (tangent) vector
        """
        # Equation 4.35: dp/ds = (pf-pi)/||pf-pi||
        return self.tangent

    @staticmethod
    def acceleration(_s: float | None = None) -> np.ndarray:
        """
        Calculate second derivative with respect to arc length.
        For linear path, this is always zero.

        Parameters:
            _s (float, optional): Arc length parameter (not used for linear path)

        Returns:
            numpy.ndarray: Acceleration vector (always zero for linear path)
        """
        # Equation 4.36: d²p/ds² = 0
        return np.zeros(3)

    def evaluate_at(self, s_values: float | list[float] | np.ndarray) -> dict[str, np.ndarray]:
        """
        Evaluate position, velocity, and acceleration at specific arc length values.

        Parameters:
            s_values (float or array-like): Arc length parameter(s)

        Returns:
            dict: Dictionary containing arrays for position, velocity, and acceleration
                 Each array has shape (n, 3) where n is the number of s values
        """
        # Convert scalar to array if needed
        s_values_arr: np.ndarray = (
            np.array([s_values]) if np.isscalar(s_values) else np.array(s_values)
        )

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

        Parameters:
            num_points (int): Number of points to generate along the path

        Returns:
            dict: Dictionary containing arrays for position, velocity, and acceleration
                 Each array has shape (num_points, 3)
        """
        # Generate evenly spaced points along the entire path
        s_values = np.linspace(0, self.length, num_points)

        # Use evaluate_at to get the trajectory data
        return self.evaluate_at(s_values)


class CircularPath:
    def __init__(self, r: np.ndarray, d: np.ndarray, pi: np.ndarray) -> None:
        """
        Initialize a circular path.

        Parameters:
            r (array-like): Unit vector of circle axis
            d (array-like): Position vector of a point on the circle axis
            pi (array-like): Position vector of a point on the circle
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

        Parameters:
            s (float or array): Arc length parameter(s)

        Returns:
            numpy.ndarray: Position vector(s)
        """
        if np.isscalar(s):
            # Position in local coordinate system (equation 4.38)
            p_prime = np.array(
                [self.radius * np.cos(s / self.radius), self.radius * np.sin(s / self.radius), 0]
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

        Parameters:
            s (float): Arc length parameter

        Returns:
            numpy.ndarray: Velocity (tangent) vector
        """
        # Velocity in local coordinate system (equation 4.40)
        dp_prime_ds = np.array([-np.sin(s / self.radius), np.cos(s / self.radius), 0])

        # Velocity in global coordinate system
        return self.R @ dp_prime_ds

    def acceleration(self, s: float) -> np.ndarray:
        """
        Calculate second derivative with respect to arc length.

        Parameters:
            s (float): Arc length parameter

        Returns:
            numpy.ndarray: Acceleration vector
        """
        # Acceleration in local coordinate system (equation 4.41)
        d2p_prime_ds2 = np.array(
            [-np.cos(s / self.radius) / self.radius, -np.sin(s / self.radius) / self.radius, 0]
        )

        # Acceleration in global coordinate system
        return self.R @ d2p_prime_ds2

    def evaluate_at(self, s_values: float | list[float] | np.ndarray) -> dict[str, np.ndarray]:
        """
        Evaluate position, velocity, and acceleration at specific arc length values.

        Parameters:
            s_values (float or array-like): Arc length parameter(s)

        Returns:
            dict: Dictionary containing arrays for position, velocity, and acceleration
                 Each array has shape (n, 3) where n is the number of s values
        """
        # Convert scalar to array if needed
        s_values_arr: np.ndarray = (
            np.array([s_values]) if np.isscalar(s_values) else np.array(s_values)
        )

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

        Parameters:
            num_points (int): Number of points to generate around the circle

        Returns:
            dict: Dictionary containing arrays for position, velocity, and acceleration
                 Each array has shape (num_points, 3)
        """
        # Generate evenly spaced points for a complete circle
        s_values = np.linspace(0, 2 * np.pi * self.radius, num_points)

        # Use evaluate_at to get the trajectory data
        return self.evaluate_at(s_values)
