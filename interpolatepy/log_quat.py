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

import numpy as np

from .quat_core import Quaternion
from .b_spline_interpolate import BSplineInterpolator

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    plt = None
    Axes3D = None


class LogQuaternionBSpline:
    """
    Logarithmic Quaternion B-spline Interpolation.

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
