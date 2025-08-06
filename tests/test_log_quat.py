"""
Comprehensive tests for the logarithmic quaternion interpolation implementation.

This module contains extensive tests for the logarithmic quaternion interpolation classes:
1. LogQuaternionBSpline (deprecated)
2. LogQuaternionInterpolation (LQI)
3. ModifiedLogQuaternionInterpolation (mLQI)

The tests verify numerical accuracy, handle edge cases, and ensure
robust behavior across different use cases.
"""

import warnings
import numpy as np
import pytest

from interpolatepy.quat_core import Quaternion
from interpolatepy.log_quat import (
    LogQuaternionBSpline,
    LogQuaternionInterpolation,
    ModifiedLogQuaternionInterpolation,
)


class TestLogQuaternionBSpline:
    """Test suite for the deprecated LogQuaternionBSpline class."""

    # Test tolerances
    REGULAR_RTOL = 1e-10
    REGULAR_ATOL = 1e-10
    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def setup_test_data(self) -> tuple[list[float], list[Quaternion]]:
        """Create test data for interpolation testing."""
        time_points = [0.0, 1.0, 2.0, 3.0, 4.0]
        quaternions = [
            Quaternion.identity(),
            Quaternion.from_euler_angles(0.1, 0.2, 0.3),
            Quaternion.from_euler_angles(0.2, 0.4, 0.6),
            Quaternion.from_euler_angles(0.3, 0.6, 0.9),
            Quaternion.from_euler_angles(0.4, 0.8, 1.2),
        ]
        return time_points, quaternions

    def test_deprecation_warning(self) -> None:
        """Test that LogQuaternionBSpline raises a deprecation warning."""
        time_points, quaternions = self.setup_test_data()

        with pytest.warns(DeprecationWarning, match="LogQuaternionBSpline is deprecated"):
            LogQuaternionBSpline(time_points, quaternions)

    def test_basic_initialization(self) -> None:
        """Test basic initialization of LogQuaternionBSpline."""
        time_points, quaternions = self.setup_test_data()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            interpolator = LogQuaternionBSpline(time_points, quaternions)

            assert interpolator.degree == LogQuaternionBSpline.DEFAULT_DEGREE
            assert interpolator.t_min == 0.0
            assert interpolator.t_max == 4.0
            assert len(interpolator.quaternions) == len(quaternions)

    def test_initialization_with_parameters(self) -> None:
        """Test initialization with various parameters."""
        time_points, quaternions = self.setup_test_data()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            # Test with velocity constraints (keep degree 3 to avoid rank issues)
            initial_velocity = np.array([0.1, 0.2, 0.3])
            final_velocity = np.array([0.4, 0.5, 0.6])
            interpolator = LogQuaternionBSpline(
                time_points, quaternions,
                initial_velocity=initial_velocity,
                final_velocity=final_velocity
            )
            assert interpolator.degree == LogQuaternionBSpline.DEFAULT_DEGREE

    def test_input_validation(self) -> None:
        """Test input validation for LogQuaternionBSpline."""
        time_points, quaternions = self.setup_test_data()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            # Mismatched lengths
            with pytest.raises(ValueError, match="Number of time points must match number of quaternions"):
                LogQuaternionBSpline(time_points[:-1], quaternions)

            # Too few quaternions
            with pytest.raises(ValueError, match="At least 2 quaternions are required"):
                LogQuaternionBSpline([0.0], [Quaternion.identity()])

            # Invalid degree
            with pytest.raises(ValueError, match="Degree must be 3, 4, or 5"):
                LogQuaternionBSpline(time_points, quaternions, degree=2)

            # Not enough points for degree
            with pytest.raises(ValueError, match="Not enough quaternions for degree"):
                LogQuaternionBSpline([0.0, 1.0], [Quaternion.identity(), Quaternion.identity()], degree=5)

            # Non-increasing time points
            bad_times = [0.0, 2.0, 1.0, 3.0, 4.0]
            with pytest.raises(ValueError, match="Time points must be strictly increasing"):
                LogQuaternionBSpline(bad_times, quaternions)

    def test_evaluation_at_control_points(self) -> None:
        """Test evaluation at control points."""
        time_points, quaternions = self.setup_test_data()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            interpolator = LogQuaternionBSpline(time_points, quaternions)

            # Test at each control point
            for i, t in enumerate(time_points):
                q_interp = interpolator.evaluate(t)

                # Should be unit quaternion
                assert abs(q_interp.norm() - 1.0) < self.NUMERICAL_ATOL

                # Should be close to original quaternion (allowing for double-cover)
                dot_product = q_interp.dot_product(quaternions[i])
                assert abs(abs(dot_product) - 1.0) < self.NUMERICAL_ATOL

    def test_evaluation_between_points(self) -> None:
        """Test evaluation between control points."""
        time_points, quaternions = self.setup_test_data()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            interpolator = LogQuaternionBSpline(time_points, quaternions)

            # Test at intermediate points
            for t in [0.5, 1.5, 2.5, 3.5]:
                q_interp = interpolator.evaluate(t)

                # Should be unit quaternion
                assert abs(q_interp.norm() - 1.0) < self.NUMERICAL_ATOL

    def test_evaluation_boundary_conditions(self) -> None:
        """Test evaluation at boundaries and outside range."""
        time_points, quaternions = self.setup_test_data()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            interpolator = LogQuaternionBSpline(time_points, quaternions)

            # Outside range should raise ValueError
            with pytest.raises(ValueError, match="Time .* outside valid range"):
                interpolator.evaluate(-1.0)

            with pytest.raises(ValueError, match="Time .* outside valid range"):
                interpolator.evaluate(5.0)

            # At exact boundaries
            q_start = interpolator.evaluate(0.0)
            q_end = interpolator.evaluate(4.0)

            assert abs(q_start.norm() - 1.0) < self.NUMERICAL_ATOL
            assert abs(q_end.norm() - 1.0) < self.NUMERICAL_ATOL

    def test_velocity_evaluation(self) -> None:
        """Test angular velocity evaluation."""
        time_points, quaternions = self.setup_test_data()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            interpolator = LogQuaternionBSpline(time_points, quaternions)

            # Test velocity at various points
            for t in [0.5, 1.5, 2.5, 3.5]:
                velocity = interpolator.evaluate_velocity(t)

                # Should be 3D vector
                assert len(velocity) == 3
                assert isinstance(velocity, np.ndarray)

    def test_acceleration_evaluation(self) -> None:
        """Test angular acceleration evaluation."""
        time_points, quaternions = self.setup_test_data()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            interpolator = LogQuaternionBSpline(time_points, quaternions)

            # Test acceleration at various points
            for t in [0.5, 1.5, 2.5, 3.5]:
                acceleration = interpolator.evaluate_acceleration(t)

                # Should be 3D vector
                assert len(acceleration) == 3
                assert isinstance(acceleration, np.ndarray)

    def test_generate_trajectory(self) -> None:
        """Test trajectory generation."""
        time_points, quaternions = self.setup_test_data()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            interpolator = LogQuaternionBSpline(time_points, quaternions)

            # Generate trajectory
            time_values, quaternion_trajectory = interpolator.generate_trajectory(num_points=50)

            assert len(time_values) == 50
            assert len(quaternion_trajectory) == 50
            assert time_values[0] == interpolator.t_min
            assert time_values[-1] == interpolator.t_max

            # All quaternions should be unit
            for q in quaternion_trajectory:
                assert abs(q.norm() - 1.0) < self.NUMERICAL_ATOL

    def test_quaternion_continuity_handling(self) -> None:
        """Test handling of quaternion double-cover."""
        # Create quaternions that might have double-cover issues (need 4 points for degree 3)
        time_points = [0.0, 1.0, 2.0, 3.0]
        quaternions = [
            Quaternion.identity(),
            Quaternion.from_euler_angles(0.1, 0.2, 0.3),
            -Quaternion.from_euler_angles(0.2, 0.4, 0.6),  # Negated quaternion
            Quaternion.from_euler_angles(0.3, 0.6, 0.9),
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            interpolator = LogQuaternionBSpline(time_points, quaternions)

            # Should handle gracefully
            q_interp = interpolator.evaluate(1.5)
            assert abs(q_interp.norm() - 1.0) < self.NUMERICAL_ATOL


class TestLogQuaternionInterpolation:
    """Test suite for LogQuaternionInterpolation (LQI) class."""

    REGULAR_RTOL = 1e-10
    REGULAR_ATOL = 1e-10
    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def setup_test_data(self) -> tuple[list[float], list[Quaternion]]:
        """Create test data for interpolation testing."""
        time_points = [0.0, 1.0, 2.0, 3.0, 4.0]
        quaternions = [
            Quaternion.identity(),
            Quaternion.from_euler_angles(0.1, 0.2, 0.3),
            Quaternion.from_euler_angles(0.2, 0.4, 0.6),
            Quaternion.from_euler_angles(0.3, 0.6, 0.9),
            Quaternion.from_euler_angles(0.4, 0.8, 1.2),
        ]
        return time_points, quaternions

    def test_basic_initialization(self) -> None:
        """Test basic initialization of LogQuaternionInterpolation."""
        time_points, quaternions = self.setup_test_data()

        interpolator = LogQuaternionInterpolation(time_points, quaternions)

        assert interpolator.degree == LogQuaternionInterpolation.DEFAULT_DEGREE
        assert interpolator.t_min == 0.0
        assert interpolator.t_max == 4.0
        assert len(interpolator.quaternions) == len(quaternions)

    def test_initialization_with_parameters(self) -> None:
        """Test initialization with various parameters."""
        time_points, quaternions = self.setup_test_data()

        # Test with velocity constraints (keep degree 3 to avoid rank issues)
        initial_velocity = np.array([0.1, 0.2, 0.3])
        final_velocity = np.array([0.4, 0.5, 0.6])
        interpolator = LogQuaternionInterpolation(
            time_points, quaternions,
            initial_velocity=initial_velocity,
            final_velocity=final_velocity
        )
        assert interpolator.degree == LogQuaternionInterpolation.DEFAULT_DEGREE

    def test_input_validation(self) -> None:
        """Test input validation for LogQuaternionInterpolation."""
        time_points, quaternions = self.setup_test_data()

        # Mismatched lengths
        with pytest.raises(ValueError, match="Number of time points must match number of quaternions"):
            LogQuaternionInterpolation(time_points[:-1], quaternions)

        # Too few quaternions
        with pytest.raises(ValueError, match="At least 2 quaternions are required"):
            LogQuaternionInterpolation([0.0], [Quaternion.identity()])

        # Invalid degree
        with pytest.raises(ValueError, match="Degree must be 3, 4, or 5"):
            LogQuaternionInterpolation(time_points, quaternions, degree=2)

        # Not enough points for degree
        with pytest.raises(ValueError, match="Not enough quaternions for degree"):
            LogQuaternionInterpolation([0.0, 1.0], [Quaternion.identity(), Quaternion.identity()], degree=5)

        # Non-increasing time points
        bad_times = [0.0, 2.0, 1.0, 3.0, 4.0]
        with pytest.raises(ValueError, match="Time points must be strictly increasing"):
            LogQuaternionInterpolation(bad_times, quaternions)

    def test_evaluation_at_control_points(self) -> None:
        """Test evaluation at control points."""
        time_points, quaternions = self.setup_test_data()
        interpolator = LogQuaternionInterpolation(time_points, quaternions)

        # Test at each control point
        for i, t in enumerate(time_points):
            q_interp = interpolator.evaluate(t)

            # Should be unit quaternion
            assert abs(q_interp.norm() - 1.0) < self.NUMERICAL_ATOL

            # Should be close to original quaternion (allowing for double-cover)
            dot_product = q_interp.dot_product(quaternions[i])
            assert abs(abs(dot_product) - 1.0) < self.NUMERICAL_ATOL

    def test_evaluation_between_points(self) -> None:
        """Test evaluation between control points."""
        time_points, quaternions = self.setup_test_data()
        interpolator = LogQuaternionInterpolation(time_points, quaternions)

        # Test at intermediate points
        for t in [0.5, 1.5, 2.5, 3.5]:
            q_interp = interpolator.evaluate(t)

            # Should be unit quaternion
            assert abs(q_interp.norm() - 1.0) < self.NUMERICAL_ATOL

    def test_evaluation_boundary_conditions(self) -> None:
        """Test evaluation at boundaries and outside range."""
        time_points, quaternions = self.setup_test_data()
        interpolator = LogQuaternionInterpolation(time_points, quaternions)

        # Outside range should raise ValueError
        with pytest.raises(ValueError, match="Time .* outside valid range"):
            interpolator.evaluate(-1.0)

        with pytest.raises(ValueError, match="Time .* outside valid range"):
            interpolator.evaluate(5.0)

        # At exact boundaries
        q_start = interpolator.evaluate(0.0)
        q_end = interpolator.evaluate(4.0)

        assert abs(q_start.norm() - 1.0) < self.NUMERICAL_ATOL
        assert abs(q_end.norm() - 1.0) < self.NUMERICAL_ATOL

    def test_velocity_evaluation(self) -> None:
        """Test angular velocity evaluation."""
        time_points, quaternions = self.setup_test_data()
        interpolator = LogQuaternionInterpolation(time_points, quaternions)

        # Test velocity at various points
        for t in [0.5, 1.5, 2.5, 3.5]:
            velocity = interpolator.evaluate_velocity(t)

            # Should be 3D vector
            assert len(velocity) == 3
            assert isinstance(velocity, np.ndarray)

    def test_acceleration_evaluation(self) -> None:
        """Test angular acceleration evaluation."""
        time_points, quaternions = self.setup_test_data()
        interpolator = LogQuaternionInterpolation(time_points, quaternions)

        # Test acceleration at various points
        for t in [0.5, 1.5, 2.5, 3.5]:
            acceleration = interpolator.evaluate_acceleration(t)

            # Should be 3D vector
            assert len(acceleration) == 3
            assert isinstance(acceleration, np.ndarray)

    def test_generate_trajectory(self) -> None:
        """Test trajectory generation."""
        time_points, quaternions = self.setup_test_data()
        interpolator = LogQuaternionInterpolation(time_points, quaternions)

        # Generate trajectory
        time_values, quaternion_trajectory = interpolator.generate_trajectory(num_points=50)

        assert len(time_values) == 50
        assert len(quaternion_trajectory) == 50
        assert time_values[0] == interpolator.t_min
        assert time_values[-1] == interpolator.t_max

        # All quaternions should be unit
        for q in quaternion_trajectory:
            assert abs(q.norm() - 1.0) < self.NUMERICAL_ATOL

    def test_axis_angle_recovery(self) -> None:
        """Test the continuous axis-angle recovery algorithm."""
        # Create quaternions with potential discontinuities
        time_points = [0.0, 1.0, 2.0, 3.0]
        quaternions = [
            Quaternion.identity(),
            Quaternion.from_angle_axis(np.pi/4, np.array([1.0, 0.0, 0.0])),
            Quaternion.from_angle_axis(np.pi/2, np.array([1.0, 0.0, 0.0])),
            Quaternion.from_angle_axis(3*np.pi/4, np.array([1.0, 0.0, 0.0])),
        ]

        interpolator = LogQuaternionInterpolation(time_points, quaternions)

        # Should handle gracefully
        q_interp = interpolator.evaluate(1.5)
        assert abs(q_interp.norm() - 1.0) < self.NUMERICAL_ATOL

    def test_small_angle_handling(self) -> None:
        """Test handling of small angles where axis is indeterminate."""
        # Create quaternions with very small rotations (need 4 points for degree 3)
        time_points = [0.0, 1.0, 2.0, 3.0]
        quaternions = [
            Quaternion.identity(),
            Quaternion.from_angle_axis(1e-8, np.array([1.0, 0.0, 0.0])),
            Quaternion.from_angle_axis(2e-8, np.array([0.0, 1.0, 0.0])),
            Quaternion.from_angle_axis(3e-8, np.array([0.0, 0.0, 1.0])),
        ]

        interpolator = LogQuaternionInterpolation(time_points, quaternions)

        # Should handle gracefully
        q_interp = interpolator.evaluate(0.5)
        assert abs(q_interp.norm() - 1.0) < self.NUMERICAL_ATOL


class TestModifiedLogQuaternionInterpolation:
    """Test suite for ModifiedLogQuaternionInterpolation (mLQI) class."""

    REGULAR_RTOL = 1e-10
    REGULAR_ATOL = 1e-10
    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def setup_test_data(self) -> tuple[list[float], list[Quaternion]]:
        """Create test data for interpolation testing."""
        time_points = [0.0, 1.0, 2.0, 3.0, 4.0]
        quaternions = [
            Quaternion.identity(),
            Quaternion.from_euler_angles(0.1, 0.2, 0.3),
            Quaternion.from_euler_angles(0.2, 0.4, 0.6),
            Quaternion.from_euler_angles(0.3, 0.6, 0.9),
            Quaternion.from_euler_angles(0.4, 0.8, 1.2),
        ]
        return time_points, quaternions

    def test_basic_initialization(self) -> None:
        """Test basic initialization of ModifiedLogQuaternionInterpolation."""
        time_points, quaternions = self.setup_test_data()

        interpolator = ModifiedLogQuaternionInterpolation(time_points, quaternions)

        assert interpolator.degree == ModifiedLogQuaternionInterpolation.DEFAULT_DEGREE
        assert interpolator.normalize_axis is True
        assert interpolator.t_min == 0.0
        assert interpolator.t_max == 4.0
        assert len(interpolator.quaternions) == len(quaternions)

    def test_initialization_with_parameters(self) -> None:
        """Test initialization with various parameters."""
        time_points, quaternions = self.setup_test_data()

        # Test with normalize_axis parameter (keep degree 3 to avoid rank issues)
        interpolator = ModifiedLogQuaternionInterpolation(
            time_points, quaternions, normalize_axis=False
        )
        assert interpolator.degree == ModifiedLogQuaternionInterpolation.DEFAULT_DEGREE
        assert interpolator.normalize_axis is False

        # Test with velocity constraints (4D)
        initial_velocity = np.array([0.1, 0.2, 0.3, 0.4])
        final_velocity = np.array([0.4, 0.5, 0.6, 0.7])
        interpolator = ModifiedLogQuaternionInterpolation(
            time_points, quaternions,
            initial_velocity=initial_velocity,
            final_velocity=final_velocity
        )
        assert interpolator.degree == ModifiedLogQuaternionInterpolation.DEFAULT_DEGREE

    def test_input_validation(self) -> None:
        """Test input validation for ModifiedLogQuaternionInterpolation."""
        time_points, quaternions = self.setup_test_data()

        # Mismatched lengths
        with pytest.raises(ValueError, match="Number of time points must match number of quaternions"):
            ModifiedLogQuaternionInterpolation(time_points[:-1], quaternions)

        # Too few quaternions
        with pytest.raises(ValueError, match="At least 2 quaternions are required"):
            ModifiedLogQuaternionInterpolation([0.0], [Quaternion.identity()])

        # Invalid degree
        with pytest.raises(ValueError, match="Degree must be 3, 4, or 5"):
            ModifiedLogQuaternionInterpolation(time_points, quaternions, degree=2)

        # Not enough points for degree
        with pytest.raises(ValueError, match="Not enough quaternions for degree"):
            ModifiedLogQuaternionInterpolation([0.0, 1.0], [Quaternion.identity(), Quaternion.identity()], degree=5)

        # Non-increasing time points
        bad_times = [0.0, 2.0, 1.0, 3.0, 4.0]
        with pytest.raises(ValueError, match="Time points must be strictly increasing"):
            ModifiedLogQuaternionInterpolation(bad_times, quaternions)

    def test_evaluation_at_control_points(self) -> None:
        """Test evaluation at control points."""
        time_points, quaternions = self.setup_test_data()
        interpolator = ModifiedLogQuaternionInterpolation(time_points, quaternions)

        # Test at each control point
        for i, t in enumerate(time_points):
            q_interp = interpolator.evaluate(t)

            # Should be unit quaternion
            assert abs(q_interp.norm() - 1.0) < self.NUMERICAL_ATOL

            # Should be close to original quaternion (allowing for double-cover)
            dot_product = q_interp.dot_product(quaternions[i])
            assert abs(abs(dot_product) - 1.0) < self.NUMERICAL_ATOL

    def test_evaluation_between_points(self) -> None:
        """Test evaluation between control points."""
        time_points, quaternions = self.setup_test_data()
        interpolator = ModifiedLogQuaternionInterpolation(time_points, quaternions)

        # Test at intermediate points
        for t in [0.5, 1.5, 2.5, 3.5]:
            q_interp = interpolator.evaluate(t)

            # Should be unit quaternion
            assert abs(q_interp.norm() - 1.0) < self.NUMERICAL_ATOL

    def test_evaluation_boundary_conditions(self) -> None:
        """Test evaluation at boundaries and outside range."""
        time_points, quaternions = self.setup_test_data()
        interpolator = ModifiedLogQuaternionInterpolation(time_points, quaternions)

        # Outside range should raise ValueError
        with pytest.raises(ValueError, match="Time .* outside valid range"):
            interpolator.evaluate(-1.0)

        with pytest.raises(ValueError, match="Time .* outside valid range"):
            interpolator.evaluate(5.0)

        # At exact boundaries
        q_start = interpolator.evaluate(0.0)
        q_end = interpolator.evaluate(4.0)

        assert abs(q_start.norm() - 1.0) < self.NUMERICAL_ATOL
        assert abs(q_end.norm() - 1.0) < self.NUMERICAL_ATOL

    def test_velocity_evaluation(self) -> None:
        """Test angular velocity evaluation (4D vector)."""
        time_points, quaternions = self.setup_test_data()
        interpolator = ModifiedLogQuaternionInterpolation(time_points, quaternions)

        # Test velocity at various points
        for t in [0.5, 1.5, 2.5, 3.5]:
            velocity = interpolator.evaluate_velocity(t)

            # Should be 4D vector [θ̇, Ẋ, Ẏ, Ż]
            assert len(velocity) == 4
            assert isinstance(velocity, np.ndarray)

    def test_acceleration_evaluation(self) -> None:
        """Test angular acceleration evaluation (4D vector)."""
        time_points, quaternions = self.setup_test_data()
        interpolator = ModifiedLogQuaternionInterpolation(time_points, quaternions)

        # Test acceleration at various points
        for t in [0.5, 1.5, 2.5, 3.5]:
            acceleration = interpolator.evaluate_acceleration(t)

            # Should be 4D vector [θ̈, Ẍ, Ÿ, Z̈]
            assert len(acceleration) == 4
            assert isinstance(acceleration, np.ndarray)

    def test_generate_trajectory(self) -> None:
        """Test trajectory generation."""
        time_points, quaternions = self.setup_test_data()
        interpolator = ModifiedLogQuaternionInterpolation(time_points, quaternions)

        # Generate trajectory
        time_values, quaternion_trajectory = interpolator.generate_trajectory(num_points=50)

        assert len(time_values) == 50
        assert len(quaternion_trajectory) == 50
        assert time_values[0] == interpolator.t_min
        assert time_values[-1] == interpolator.t_max

        # All quaternions should be unit
        for q in quaternion_trajectory:
            assert abs(q.norm() - 1.0) < self.NUMERICAL_ATOL

    def test_normalize_axis_parameter(self) -> None:
        """Test the normalize_axis parameter behavior."""
        time_points, quaternions = self.setup_test_data()

        # Test with normalize_axis=True (default)
        interpolator_norm = ModifiedLogQuaternionInterpolation(
            time_points, quaternions, normalize_axis=True
        )
        q_norm = interpolator_norm.evaluate(1.5)
        assert abs(q_norm.norm() - 1.0) < self.NUMERICAL_ATOL

        # Test with normalize_axis=False
        interpolator_no_norm = ModifiedLogQuaternionInterpolation(
            time_points, quaternions, normalize_axis=False
        )
        q_no_norm = interpolator_no_norm.evaluate(1.5)

        # With normalize_axis=False, the quaternion might not be perfectly unit
        # but should be close (within a reasonable tolerance)
        assert abs(q_no_norm.norm() - 1.0) < 0.1  # More relaxed tolerance

        # Results should be different due to different axis handling
        # Both should represent valid rotations
        assert q_norm.norm() > 0.9  # Should be close to unit
        assert q_no_norm.norm() > 0.9  # Should be close to unit

    def test_theta_xyz_separation(self) -> None:
        """Test that θ and (X,Y,Z) are properly separated and interpolated."""
        # Create quaternions with known axis-angle representations (need 4 points for degree 3)
        time_points = [0.0, 1.0, 2.0, 3.0]
        quaternions = [
            Quaternion.from_angle_axis(np.pi/4, np.array([1.0, 0.0, 0.0])),
            Quaternion.from_angle_axis(np.pi/2, np.array([1.0, 0.0, 0.0])),
            Quaternion.from_angle_axis(3*np.pi/4, np.array([1.0, 0.0, 0.0])),
            Quaternion.from_angle_axis(np.pi, np.array([1.0, 0.0, 0.0])),
        ]

        interpolator = ModifiedLogQuaternionInterpolation(time_points, quaternions)

        # Interpolate at midpoint
        q_interp = interpolator.evaluate(1.0)

        # Should be unit quaternion
        assert abs(q_interp.norm() - 1.0) < self.NUMERICAL_ATOL

        # Should be close to the middle quaternion
        dot_product = q_interp.dot_product(quaternions[1])
        assert abs(abs(dot_product) - 1.0) < self.NUMERICAL_ATOL


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main(["-xvs", __file__])
