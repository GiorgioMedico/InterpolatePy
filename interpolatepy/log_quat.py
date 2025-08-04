"""
Logarithmic Quaternion B-spline Interpolation

This module provides smooth quaternion trajectory generation using logarithmic
quaternion representation with cubic B-spline interpolation.

The algorithm:
1. Transform unit quaternions to logarithmic space using q.Log()
2. Interpolate the 3D vector parts using cubic B-splines
3. Transform back to unit quaternions using exp() mapping

This approach provides smooth, continuously differentiable quaternion trajectories
with precise control over rotational motion profiles.
"""

from __future__ import annotations

import warnings
import numpy as np

from .quat_core import Quaternion
from .b_spline_interpolate import BSplineInterpolator

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    plt = None  # type: ignore[assignment]
    Axes3D = None


class LogQuaternionBSpline:
    """
    Logarithmic Quaternion B-spline Interpolation.

    .. deprecated::
        LogQuaternionBSpline is deprecated and will be removed in a future version.
        Use LogQuaternionInterpolation or ModifiedLogQuaternionInterpolation instead,
        which provide improved algorithms based on Parker et al. (2023) with better
        handling of quaternion double-cover and axis-angle discontinuities.

    This class provides smooth quaternion interpolation by working in logarithmic
    quaternion space and using cubic B-splines for interpolation.

    Parameters
    ----------
    time_points : array_like
        Time values corresponding to each quaternion (must be strictly increasing).
    quaternions : array_like
        List of unit quaternions to interpolate between.
    degree : int, optional
        Degree of the B-spline (3, 4, or 5). Default is 3 (cubic).
    initial_velocity : array_like, optional
        Initial angular velocity constraint (3D vector). Default is None.
    final_velocity : array_like, optional
        Final angular velocity constraint (3D vector). Default is None.
    initial_acceleration : array_like, optional
        Initial angular acceleration constraint (3D vector). Default is None.
    final_acceleration : array_like, optional
        Final angular acceleration constraint (3D vector). Default is None.

    Attributes
    ----------
    time_points : ndarray
        Time values for the quaternion waypoints.
    quaternions : list[Quaternion]
        Original quaternion waypoints.
    degree : int
        Degree of the B-spline (3, 4, or 5).
    t_min : float
        Minimum valid time value.
    t_max : float
        Maximum valid time value.
    """

    # Constants
    EPSILON = 1e-10
    DEFAULT_DEGREE = 3

    def __init__(  # noqa: PLR0913
        self,
        time_points: list | np.ndarray,
        quaternions: list[Quaternion],
        degree: int = DEFAULT_DEGREE,
        initial_velocity: list | np.ndarray | None = None,
        final_velocity: list | np.ndarray | None = None,
        initial_acceleration: list | np.ndarray | None = None,
        final_acceleration: list | np.ndarray | None = None,
    ) -> None:
        """
        Initialize the logarithmic quaternion B-spline interpolator.

        Parameters
        ----------
        time_points : array_like
            Time values corresponding to each quaternion.
        quaternions : list[Quaternion]
            Unit quaternions to interpolate between.
        degree : int, optional
            Degree of the B-spline (3, 4, or 5). Default is 3 (cubic).
        initial_velocity : array_like, optional
            Initial angular velocity constraint (3D vector). Default is None.
        final_velocity : array_like, optional
            Final angular velocity constraint (3D vector). Default is None.
        initial_acceleration : array_like, optional
            Initial angular acceleration constraint (3D vector). Default is None.
        final_acceleration : array_like, optional
            Final angular acceleration constraint (3D vector). Default is None.

        Raises
        ------
        ValueError
            If inputs are invalid or quaternions are not unit quaternions.
        """
        warnings.warn(
            "LogQuaternionBSpline is deprecated and will be removed in a future version. "
            "Use LogQuaternionInterpolation or ModifiedLogQuaternionInterpolation instead, "
            "which provide improved algorithms with better handling of quaternion "
            "double-cover and axis-angle discontinuities.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Convert time points to numpy array
        self.time_points = np.array(time_points, dtype=np.float64)
        self.quaternions = list(quaternions)  # Keep original quaternions
        self.degree = degree

        # Validate inputs
        self._validate_inputs()

        # Ensure quaternions have consistent orientation (handle double-cover)
        self._ensure_quaternion_continuity()

        # Transform to logarithmic space
        log_quaternions = self._transform_to_log_space()

        # Create B-spline interpolator with direct time-based interpolation
        self.bspline_interpolator = BSplineInterpolator(
            degree=degree,
            points=log_quaternions,
            times=self.time_points,
            initial_velocity=initial_velocity,
            final_velocity=final_velocity,
            initial_acceleration=initial_acceleration,
            final_acceleration=final_acceleration,
        )

        # Store time range
        self.t_min = self.time_points[0]
        self.t_max = self.time_points[-1]

    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        if len(self.time_points) != len(self.quaternions):
            raise ValueError("Number of time points must match number of quaternions")

        min_quaternions = 2
        if len(self.quaternions) < min_quaternions:
            raise ValueError("At least 2 quaternions are required for interpolation")

        # Validate degree
        if self.degree not in {3, 4, 5}:
            raise ValueError(f"Degree must be 3, 4, or 5, got {self.degree}")

        # Check minimum points for the degree
        min_points = self.degree + 1
        if len(self.quaternions) < min_points:
            raise ValueError(
                f"Not enough quaternions for degree {self.degree} B-spline interpolation. "
                f"Need at least {min_points} quaternions, got {len(self.quaternions)}"
            )

        # Check time points are strictly increasing
        if not np.all(np.diff(self.time_points) > 0):
            raise ValueError("Time points must be strictly increasing")

        # Validate quaternions are unit quaternions
        for i, q in enumerate(self.quaternions):
            if not isinstance(q, Quaternion):
                raise TypeError(f"Element {i} is not a Quaternion instance")

            norm = q.norm()
            if abs(norm - 1.0) > self.EPSILON:
                print(f"Warning: Quaternion {i} is not unit (norm={norm:.6f}), normalizing...")
                self.quaternions[i] = q.unit()

    def _ensure_quaternion_continuity(self) -> None:
        """
        Ensure quaternion continuity by handling the double-cover property.
        Choose the sign of each quaternion to minimize the distance to the previous one.
        """
        for i in range(1, len(self.quaternions)):
            # Check both q and -q to see which is closer to the previous quaternion
            q_pos = self.quaternions[i]
            q_neg = -self.quaternions[i]

            # Use dot product to measure similarity (closer to 1 means more similar)
            dot_pos = self.quaternions[i - 1].dot_product(q_pos)
            dot_neg = self.quaternions[i - 1].dot_product(q_neg)

            # Choose the quaternion with higher dot product (smaller angle)
            if dot_neg > dot_pos:
                self.quaternions[i] = q_neg

    def _transform_to_log_space(self) -> np.ndarray:
        """
        Transform quaternions to logarithmic space.

        Returns
        -------
        ndarray
            3D control points (vector parts of log quaternions).
        """
        log_vectors = []

        for q in self.quaternions:
            # Get logarithm of unit quaternion
            log_q = q.Log()
            # Extract vector part (scalar part is always 0 for unit quaternions)
            log_vectors.append(log_q.v())

        return np.array(log_vectors)

    def evaluate(self, t: float) -> Quaternion:
        """
        Evaluate the quaternion trajectory at time t.

        Parameters
        ----------
        t : float
            Time value to evaluate at.

        Returns
        -------
        Quaternion
            Interpolated unit quaternion at time t.

        Raises
        ------
        ValueError
            If t is outside the valid time range.
        """
        # Validate time range
        if t < self.t_min - self.EPSILON or t > self.t_max + self.EPSILON:
            raise ValueError(f"Time {t} outside valid range [{self.t_min}, {self.t_max}]")

        # Clamp to valid range
        t = np.clip(t, self.t_min, self.t_max)

        # Handle boundary cases exactly
        if abs(t - self.t_min) <= self.EPSILON:
            return self.quaternions[0].copy()
        if abs(t - self.t_max) <= self.EPSILON:
            return self.quaternions[-1].copy()

        # Evaluate B-spline interpolator directly to get vector part in log space
        log_vector = self.bspline_interpolator.evaluate(t)

        # Create log quaternion (scalar part is 0)
        log_quaternion = Quaternion(0.0, log_vector[0], log_vector[1], log_vector[2])

        # Transform back to unit quaternion using exponential map
        return log_quaternion.exp()

    def evaluate_velocity(self, t: float) -> np.ndarray:
        """
        Evaluate the angular velocity at time t.

        The angular velocity is computed as the derivative of the log quaternion
        in the tangent space.

        Parameters
        ----------
        t : float
            Time value to evaluate at.

        Returns
        -------
        ndarray
            3D angular velocity vector.
        """
        # Validate time range
        if t < self.t_min - self.EPSILON or t > self.t_max + self.EPSILON:
            raise ValueError(f"Time {t} outside valid range [{self.t_min}, {self.t_max}]")

        # Clamp to valid range
        t = np.clip(t, self.t_min, self.t_max)

        # Get derivative of B-spline interpolator (first derivative in log space)
        return self.bspline_interpolator.evaluate_derivative(t, order=1)

    def evaluate_acceleration(self, t: float) -> np.ndarray:
        """
        Evaluate the angular acceleration at time t.

        Parameters
        ----------
        t : float
            Time value to evaluate at.

        Returns
        -------
        ndarray
            3D angular acceleration vector.
        """
        # Validate time range
        if t < self.t_min - self.EPSILON or t > self.t_max + self.EPSILON:
            raise ValueError(f"Time {t} outside valid range [{self.t_min}, {self.t_max}]")

        # Clamp to valid range
        t = np.clip(t, self.t_min, self.t_max)

        # Get second derivative of B-spline interpolator (second derivative in log space)
        return self.bspline_interpolator.evaluate_derivative(t, order=2)

    def generate_trajectory(self, num_points: int = 100) -> tuple[np.ndarray, list[Quaternion]]:
        """
        Generate a trajectory with evenly spaced time points.

        Parameters
        ----------
        num_points : int, optional
            Number of points to generate (default is 100).

        Returns
        -------
        time_values : ndarray
            Evaluation time points.
        quaternion_trajectory : list[Quaternion]
            Corresponding quaternions.
        """
        time_values = np.linspace(self.t_min, self.t_max, num_points)
        quaternion_trajectory = [self.evaluate(t) for t in time_values]

        return time_values, quaternion_trajectory


class LogQuaternionInterpolation:
    """
    Logarithmic Quaternion Interpolation (LQI) using axis-angle representation.

    This class implements the LQI method from Parker et al. (2023) which transforms
    quaternions to axis-angle space r = θ*n̂ and interpolates using B-splines.

    Key features:
    - Handles quaternion double-cover and axis-angle discontinuities
    - Uses Algorithm 1 from the paper for continuous axis-angle recovery
    - Provides C² continuous quaternion interpolation

    Parameters
    ----------
    time_points : array_like
        Time values corresponding to each quaternion (must be strictly increasing).
    quaternions : array_like
        List of unit quaternions to interpolate between.
    degree : int, optional
        Degree of the B-spline (3, 4, or 5). Default is 3 (cubic).
    initial_velocity : array_like, optional
        Initial angular velocity constraint (3D vector). Default is None.
    final_velocity : array_like, optional
        Final angular velocity constraint (3D vector). Default is None.
    initial_acceleration : array_like, optional
        Initial angular acceleration constraint (3D vector). Default is None.
    final_acceleration : array_like, optional
        Final angular acceleration constraint (3D vector). Default is None.
    """

    # Constants
    EPSILON = 1e-10
    DEFAULT_DEGREE = 3

    def __init__(  # noqa: PLR0913
        self,
        time_points: list | np.ndarray,
        quaternions: list[Quaternion],
        degree: int = DEFAULT_DEGREE,
        initial_velocity: list | np.ndarray | None = None,
        final_velocity: list | np.ndarray | None = None,
        initial_acceleration: list | np.ndarray | None = None,
        final_acceleration: list | np.ndarray | None = None,
    ) -> None:
        """Initialize the LQI interpolator."""
        # Convert time points to numpy array
        self.time_points = np.array(time_points, dtype=np.float64)
        self.quaternions = list(quaternions)  # Keep original quaternions
        self.degree = degree

        # Validate inputs
        self._validate_inputs()

        # Recover continuous axis-angle representation (Algorithm 1 from paper)
        axis_angle_vectors = self._recover_continuous_axis_angle()

        # Create B-spline interpolator for axis-angle vectors
        self.bspline_interpolator = BSplineInterpolator(
            degree=degree,
            points=axis_angle_vectors,
            times=self.time_points,
            initial_velocity=initial_velocity,
            final_velocity=final_velocity,
            initial_acceleration=initial_acceleration,
            final_acceleration=final_acceleration,
        )

        # Store time range
        self.t_min = self.time_points[0]
        self.t_max = self.time_points[-1]

    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        if len(self.time_points) != len(self.quaternions):
            raise ValueError("Number of time points must match number of quaternions")

        min_quaternions = 2
        if len(self.quaternions) < min_quaternions:
            raise ValueError("At least 2 quaternions are required for interpolation")

        # Validate degree
        if self.degree not in {3, 4, 5}:
            raise ValueError(f"Degree must be 3, 4, or 5, got {self.degree}")

        # Check minimum points for the degree
        min_points = self.degree + 1
        if len(self.quaternions) < min_points:
            raise ValueError(
                f"Not enough quaternions for degree {self.degree} B-spline interpolation. "
                f"Need at least {min_points} quaternions, got {len(self.quaternions)}"
            )

        # Check time points are strictly increasing
        if not np.all(np.diff(self.time_points) > 0):
            raise ValueError("Time points must be strictly increasing")

        # Validate quaternions are unit quaternions
        for i, q in enumerate(self.quaternions):
            if not isinstance(q, Quaternion):
                raise TypeError(f"Element {i} is not a Quaternion instance")

            norm = q.norm()
            if abs(norm - 1.0) > self.EPSILON:
                print(f"Warning: Quaternion {i} is not unit (norm={norm:.6f}), normalizing...")
                self.quaternions[i] = q.unit()

    def _recover_continuous_axis_angle(self) -> np.ndarray:
        """
        Implement Algorithm 1 from Parker et al. (2023) for recovering continuous axis-angle series.

        This method handles:
        1. Quaternion double-cover (q and -q represent same rotation)
        2. Axis-angle discontinuities
        3. Phase unwrapping around ±2π
        4. Special cases where θ ≈ 0 (indeterminate axis)

        Returns
        -------
        ndarray
            Array of continuous axis-angle vectors r = θ*n̂ for B-spline interpolation.
        """
        n = len(self.quaternions)
        axis_angle_vectors = []

        # Step 1: Extract initial (θ, n̂) from quaternions
        axes = []
        angles = []

        for q in self.quaternions:
            axis, angle = q.to_axis_angle()
            axes.append(axis)
            angles.append(angle)

        # Step 2: Handle quaternion double-cover and ensure continuity
        for i in range(1, n):
            # Check both q and -q to see which provides better continuity
            q_pos = self.quaternions[i]
            q_neg = -self.quaternions[i]

            # Use dot product to measure similarity (closer to 1 means more similar)
            dot_pos = self.quaternions[i - 1].dot_product(q_pos)
            dot_neg = self.quaternions[i - 1].dot_product(q_neg)

            # Choose the quaternion with higher dot product (smaller angle)
            if dot_neg > dot_pos:
                self.quaternions[i] = q_neg
                # Recalculate axis-angle for the flipped quaternion
                axis, angle = q_neg.to_axis_angle()
                axes[i] = axis
                angles[i] = angle

            # Now check if we need to flip the axis to maintain continuity
            if np.linalg.norm(axes[i - 1] - axes[i]) > np.linalg.norm(axes[i - 1] + axes[i]):
                angles[i] = -angles[i]
                axes[i] = -axes[i]

        # Step 3: Unwrap phase angles around ±2π
        angles = np.unwrap(angles).tolist()

        # Step 4: Convert to axis-angle vectors r = θ*n̂
        for i in range(n):
            if abs(angles[i]) < self.EPSILON:
                # For small angles, set axis-angle vector to zero
                axis_angle_vectors.append(np.array([0.0, 0.0, 0.0]))
            else:
                # r = θ * n̂
                axis_angle_vectors.append(angles[i] * axes[i])

        return np.array(axis_angle_vectors)

    def evaluate(self, t: float) -> Quaternion:
        """
        Evaluate the quaternion trajectory at time t.

        Parameters
        ----------
        t : float
            Time value to evaluate at.

        Returns
        -------
        Quaternion
            Interpolated unit quaternion at time t.
        """
        # Validate time range
        if t < self.t_min - self.EPSILON or t > self.t_max + self.EPSILON:
            raise ValueError(f"Time {t} outside valid range [{self.t_min}, {self.t_max}]")

        # Clamp to valid range
        t = np.clip(t, self.t_min, self.t_max)

        # Handle boundary cases exactly
        if abs(t - self.t_min) <= self.EPSILON:
            return self.quaternions[0].copy()
        if abs(t - self.t_max) <= self.EPSILON:
            return self.quaternions[-1].copy()

        # Evaluate B-spline to get axis-angle vector r = θ*n̂
        axis_angle_vector = self.bspline_interpolator.evaluate(t)

        # Convert back to quaternion
        theta = np.linalg.norm(axis_angle_vector)

        if theta < self.EPSILON:
            # For small angles, return identity quaternion
            return Quaternion.identity()

        # Extract axis and create quaternion
        axis = axis_angle_vector / theta
        return Quaternion.from_angle_axis(float(theta), axis)

    def evaluate_velocity(self, t: float) -> np.ndarray:
        """
        Evaluate the angular velocity at time t.

        Parameters
        ----------
        t : float
            Time value to evaluate at.

        Returns
        -------
        ndarray
            3D angular velocity vector.
        """
        # Validate time range
        if t < self.t_min - self.EPSILON or t > self.t_max + self.EPSILON:
            raise ValueError(f"Time {t} outside valid range [{self.t_min}, {self.t_max}]")

        # Clamp to valid range
        t = np.clip(t, self.t_min, self.t_max)

        # Get derivative of B-spline interpolator (first derivative of axis-angle vector)
        return self.bspline_interpolator.evaluate_derivative(t, order=1)

    def evaluate_acceleration(self, t: float) -> np.ndarray:
        """
        Evaluate the angular acceleration at time t.

        Parameters
        ----------
        t : float
            Time value to evaluate at.

        Returns
        -------
        ndarray
            3D angular acceleration vector.
        """
        # Validate time range
        if t < self.t_min - self.EPSILON or t > self.t_max + self.EPSILON:
            raise ValueError(f"Time {t} outside valid range [{self.t_min}, {self.t_max}]")

        # Clamp to valid range
        t = np.clip(t, self.t_min, self.t_max)

        # Get second derivative of B-spline interpolator (second derivative of axis-angle vector)
        return self.bspline_interpolator.evaluate_derivative(t, order=2)

    def generate_trajectory(self, num_points: int = 100) -> tuple[np.ndarray, list[Quaternion]]:
        """
        Generate a trajectory with evenly spaced time points.

        Parameters
        ----------
        num_points : int, optional
            Number of points to generate (default is 100).

        Returns
        -------
        time_values : ndarray
            Evaluation time points.
        quaternion_trajectory : list[Quaternion]
            Corresponding quaternions.
        """
        time_values = np.linspace(self.t_min, self.t_max, num_points)
        quaternion_trajectory = [self.evaluate(t) for t in time_values]

        return time_values, quaternion_trajectory


class ModifiedLogQuaternionInterpolation:
    """
    Modified Logarithmic Quaternion Interpolation (mLQI).

    This class implements the modified LQI method from Parker et al. (2023) which
    interpolates quaternions as (θ, X, Y, Z) where X²+Y²+Z²=1.

    Key features:
    - Decouples angle θ from unit vector components (X,Y,Z)
    - Uses separate B-spline interpolators for better numerical stability
    - Supports both normalized and unnormalized unit vector interpolation
    - Provides C² continuous quaternion interpolation

    Parameters
    ----------
    time_points : array_like
        Time values corresponding to each quaternion (must be strictly increasing).
    quaternions : array_like
        List of unit quaternions to interpolate between.
    degree : int, optional
        Degree of the B-spline (3, 4, or 5). Default is 3 (cubic).
    normalize_axis : bool, optional
        Whether to normalize (X,Y,Z) components after interpolation. Default is True.
    initial_velocity : array_like, optional
        Initial angular velocity constraint (4D vector: [θ̇, Ẋ, Ẏ, Ż]). Default is None.
    final_velocity : array_like, optional
        Final angular velocity constraint (4D vector: [θ̇, Ẋ, Ẏ, Ż]). Default is None.
    initial_acceleration : array_like, optional
        Initial angular acceleration constraint (4D vector). Default is None.
    final_acceleration : array_like, optional
        Final angular acceleration constraint (4D vector). Default is None.
    """

    # Constants
    EPSILON = 1e-10
    DEFAULT_DEGREE = 3

    def __init__(  # noqa: PLR0913
        self,
        time_points: list | np.ndarray,
        quaternions: list[Quaternion],
        degree: int = DEFAULT_DEGREE,
        normalize_axis: bool = True,
        initial_velocity: list | np.ndarray | None = None,
        final_velocity: list | np.ndarray | None = None,
        initial_acceleration: list | np.ndarray | None = None,
        final_acceleration: list | np.ndarray | None = None,
    ) -> None:
        """Initialize the mLQI interpolator."""
        # Convert time points to numpy array
        self.time_points = np.array(time_points, dtype=np.float64)
        self.quaternions = list(quaternions)  # Keep original quaternions
        self.degree = degree
        self.normalize_axis = normalize_axis

        # Validate inputs
        self._validate_inputs()

        # Ensure quaternion continuity by handling the double-cover
        self._ensure_quaternion_continuity()

        # Transform to (θ, X, Y, Z) representation
        theta_values, xyz_values = self._transform_to_theta_xyz_space()

        # Split velocity/acceleration constraints if provided
        theta_initial_vel = None
        xyz_initial_vel = None
        theta_final_vel = None
        xyz_final_vel = None

        if initial_velocity is not None:
            initial_velocity = np.array(initial_velocity)
            theta_initial_vel = np.array([initial_velocity[0]])  # θ̇
            xyz_initial_vel = initial_velocity[1:4]  # [Ẋ, Ẏ, Ż]

        if final_velocity is not None:
            final_velocity = np.array(final_velocity)
            theta_final_vel = np.array([final_velocity[0]])  # θ̇
            xyz_final_vel = final_velocity[1:4]  # [Ẋ, Ẏ, Ż]

        # Similar for acceleration
        theta_initial_acc = None
        xyz_initial_acc = None
        theta_final_acc = None
        xyz_final_acc = None

        if initial_acceleration is not None:
            initial_acceleration = np.array(initial_acceleration)
            theta_initial_acc = np.array([initial_acceleration[0]])
            xyz_initial_acc = initial_acceleration[1:4]

        if final_acceleration is not None:
            final_acceleration = np.array(final_acceleration)
            theta_final_acc = np.array([final_acceleration[0]])
            xyz_final_acc = final_acceleration[1:4]

        # Create separate B-spline interpolators
        # For θ (1D)
        self.theta_interpolator = BSplineInterpolator(
            degree=degree,
            points=theta_values.reshape(-1, 1),  # Make it 2D for BSplineInterpolator
            times=self.time_points,
            initial_velocity=theta_initial_vel,
            final_velocity=theta_final_vel,
            initial_acceleration=theta_initial_acc,
            final_acceleration=theta_final_acc,
        )

        # For (X, Y, Z) (3D)
        self.xyz_interpolator = BSplineInterpolator(
            degree=degree,
            points=xyz_values,
            times=self.time_points,
            initial_velocity=xyz_initial_vel,
            final_velocity=xyz_final_vel,
            initial_acceleration=xyz_initial_acc,
            final_acceleration=xyz_final_acc,
        )

        # Store time range
        self.t_min = self.time_points[0]
        self.t_max = self.time_points[-1]

    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        if len(self.time_points) != len(self.quaternions):
            raise ValueError("Number of time points must match number of quaternions")

        min_quaternions = 2
        if len(self.quaternions) < min_quaternions:
            raise ValueError("At least 2 quaternions are required for interpolation")

        # Validate degree
        if self.degree not in {3, 4, 5}:
            raise ValueError(f"Degree must be 3, 4, or 5, got {self.degree}")

        # Check minimum points for the degree
        min_points = self.degree + 1
        if len(self.quaternions) < min_points:
            raise ValueError(
                f"Not enough quaternions for degree {self.degree} B-spline interpolation. "
                f"Need at least {min_points} quaternions, got {len(self.quaternions)}"
            )

        # Check time points are strictly increasing
        if not np.all(np.diff(self.time_points) > 0):
            raise ValueError("Time points must be strictly increasing")

        # Validate quaternions are unit quaternions
        for i, q in enumerate(self.quaternions):
            if not isinstance(q, Quaternion):
                raise TypeError(f"Element {i} is not a Quaternion instance")

            norm = q.norm()
            if abs(norm - 1.0) > self.EPSILON:
                print(f"Warning: Quaternion {i} is not unit (norm={norm:.6f}), normalizing...")
                self.quaternions[i] = q.unit()

    def _ensure_quaternion_continuity(self) -> None:
        """
        Ensure quaternion continuity by handling the double-cover property.
        Choose the sign of each quaternion to minimize the distance to the previous one.
        """
        for i in range(1, len(self.quaternions)):
            # Check both q and -q to see which is closer to the previous quaternion
            q_pos = self.quaternions[i]
            q_neg = -self.quaternions[i]

            # Use dot product to measure similarity (closer to 1 means more similar)
            dot_pos = self.quaternions[i - 1].dot_product(q_pos)
            dot_neg = self.quaternions[i - 1].dot_product(q_neg)

            # Choose the quaternion with higher dot product (smaller angle)
            if dot_neg > dot_pos:
                self.quaternions[i] = q_neg

    def _transform_to_theta_xyz_space(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform quaternions to (θ, X, Y, Z) representation.

        Returns
        -------
        theta_values : ndarray
            Array of angles θ.
        xyz_values : ndarray
            Array of unit vector components (X, Y, Z).
        """
        theta_values = []
        xyz_values = []

        for q in self.quaternions:
            # Extract angle and axis from quaternion
            axis, angle = q.to_axis_angle()

            theta_values.append(angle)
            xyz_values.append(axis)

        return np.array(theta_values), np.array(xyz_values)

    def evaluate(self, t: float) -> Quaternion:
        """
        Evaluate the quaternion trajectory at time t.

        Parameters
        ----------
        t : float
            Time value to evaluate at.

        Returns
        -------
        Quaternion
            Interpolated unit quaternion at time t.
        """
        # Validate time range
        if t < self.t_min - self.EPSILON or t > self.t_max + self.EPSILON:
            raise ValueError(f"Time {t} outside valid range [{self.t_min}, {self.t_max}]")

        # Clamp to valid range
        t = np.clip(t, self.t_min, self.t_max)

        # Handle boundary cases exactly
        if abs(t - self.t_min) <= self.EPSILON:
            return self.quaternions[0].copy()
        if abs(t - self.t_max) <= self.EPSILON:
            return self.quaternions[-1].copy()

        # Evaluate B-spline interpolators
        theta = self.theta_interpolator.evaluate(t)[0]  # Extract scalar from 1D array
        xyz = self.xyz_interpolator.evaluate(t)  # 3D vector

        # Optionally normalize the axis components
        if self.normalize_axis:
            norm_xyz = np.linalg.norm(xyz)
            # If axis is zero, use default axis
            xyz = xyz / norm_xyz if norm_xyz > self.EPSILON else np.array([1.0, 0.0, 0.0])

        # Create quaternion: q = [cos(θ/2), sin(θ/2)*X, sin(θ/2)*Y, sin(θ/2)*Z]
        if abs(theta) < self.EPSILON:
            # For small angles, return identity quaternion
            return Quaternion.identity()

        cos_half_theta = np.cos(theta / 2.0)
        sin_half_theta = np.sin(theta / 2.0)

        return Quaternion(
            cos_half_theta,
            sin_half_theta * xyz[0],
            sin_half_theta * xyz[1],
            sin_half_theta * xyz[2],
        )

    def evaluate_velocity(self, t: float) -> np.ndarray:
        """
        Evaluate the angular velocity at time t.

        This returns the derivative of (θ, X, Y, Z) as a 4D vector.

        Parameters
        ----------
        t : float
            Time value to evaluate at.

        Returns
        -------
        ndarray
            4D vector [θ̇, Ẋ, Ẏ, Ż].
        """
        # Validate time range
        if t < self.t_min - self.EPSILON or t > self.t_max + self.EPSILON:
            raise ValueError(f"Time {t} outside valid range [{self.t_min}, {self.t_max}]")

        # Clamp to valid range
        t = np.clip(t, self.t_min, self.t_max)

        # Get derivatives from both interpolators
        theta_dot = self.theta_interpolator.evaluate_derivative(t, order=1)[0]  # Scalar
        xyz_dot = self.xyz_interpolator.evaluate_derivative(t, order=1)  # 3D vector

        # Combine into 4D vector
        return np.array([theta_dot, xyz_dot[0], xyz_dot[1], xyz_dot[2]])

    def evaluate_acceleration(self, t: float) -> np.ndarray:
        """
        Evaluate the angular acceleration at time t.

        This returns the second derivative of (θ, X, Y, Z) as a 4D vector.

        Parameters
        ----------
        t : float
            Time value to evaluate at.

        Returns
        -------
        ndarray
            4D vector [θ̈, Ẍ, Ÿ, Z̈].
        """
        # Validate time range
        if t < self.t_min - self.EPSILON or t > self.t_max + self.EPSILON:
            raise ValueError(f"Time {t} outside valid range [{self.t_min}, {self.t_max}]")

        # Clamp to valid range
        t = np.clip(t, self.t_min, self.t_max)

        # Get second derivatives from both interpolators
        theta_ddot = self.theta_interpolator.evaluate_derivative(t, order=2)[0]  # Scalar
        xyz_ddot = self.xyz_interpolator.evaluate_derivative(t, order=2)  # 3D vector

        # Combine into 4D vector
        return np.array([theta_ddot, xyz_ddot[0], xyz_ddot[1], xyz_ddot[2]])

    def generate_trajectory(self, num_points: int = 100) -> tuple[np.ndarray, list[Quaternion]]:
        """
        Generate a trajectory with evenly spaced time points.

        Parameters
        ----------
        num_points : int, optional
            Number of points to generate (default is 100).

        Returns
        -------
        time_values : ndarray
            Evaluation time points.
        quaternion_trajectory : list[Quaternion]
            Corresponding quaternions.
        """
        time_values = np.linspace(self.t_min, self.t_max, num_points)
        quaternion_trajectory = [self.evaluate(t) for t in time_values]

        return time_values, quaternion_trajectory
