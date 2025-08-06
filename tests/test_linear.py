"""
Comprehensive tests for linear trajectory implementation.

This module contains extensive tests for the linear trajectory function covering:
1. Basic linear interpolation functionality
2. Scalar and vector position handling
3. Mathematical accuracy verification
4. Edge cases and boundary conditions
5. Performance benchmarks

The tests verify that linear trajectories correctly generate positions,
velocities, and accelerations with proper vectorization and broadcasting.
"""

from typing import Any

import numpy as np
import pytest

from interpolatepy.linear import linear_traj


# Type alias for pytest benchmark fixture
if not hasattr(pytest, "FixtureFunction"):
    pytest.FixtureFunction = Any


class TestLinearTrajectoryScalar:
    """Test suite for scalar linear trajectories."""

    NUMERICAL_RTOL = 1e-10
    NUMERICAL_ATOL = 1e-10

    def test_basic_scalar_trajectory(self) -> None:
        """Test basic scalar linear trajectory."""
        p0, p1 = 0.0, 10.0
        t0, t1 = 0.0, 2.0
        time_array = np.array([0.0, 1.0, 2.0])

        positions, velocities, accelerations = linear_traj(p0, p1, t0, t1, time_array)

        # Check positions
        expected_positions = np.array([0.0, 5.0, 10.0])
        assert np.allclose(positions, expected_positions, atol=self.NUMERICAL_ATOL)

        # Check velocities (should be constant)
        expected_velocity = 5.0  # (10-0)/(2-0)
        assert np.allclose(velocities, expected_velocity, atol=self.NUMERICAL_ATOL)

        # Check accelerations (should be zero)
        assert np.allclose(accelerations, 0.0, atol=self.NUMERICAL_ATOL)

    def test_scalar_trajectory_negative_displacement(self) -> None:
        """Test scalar trajectory with negative displacement."""
        p0, p1 = 5.0, 2.0
        t0, t1 = 0.0, 3.0
        time_array = np.array([0.0, 1.5, 3.0])

        positions, velocities, accelerations = linear_traj(p0, p1, t0, t1, time_array)

        # Check positions
        expected_positions = np.array([5.0, 3.5, 2.0])
        assert np.allclose(positions, expected_positions, atol=self.NUMERICAL_ATOL)

        # Check velocities (should be negative)
        expected_velocity = -1.0  # (2-5)/(3-0)
        assert np.allclose(velocities, expected_velocity, atol=self.NUMERICAL_ATOL)

        # Check accelerations
        assert np.allclose(accelerations, 0.0, atol=self.NUMERICAL_ATOL)

    def test_scalar_trajectory_zero_displacement(self) -> None:
        """Test scalar trajectory with zero displacement."""
        p0, p1 = 3.0, 3.0
        t0, t1 = 0.0, 2.0
        time_array = np.array([0.0, 1.0, 2.0])

        positions, velocities, accelerations = linear_traj(p0, p1, t0, t1, time_array)

        # Check positions (should remain constant)
        expected_positions = np.array([3.0, 3.0, 3.0])
        assert np.allclose(positions, expected_positions, atol=self.NUMERICAL_ATOL)

        # Check velocities (should be zero)
        assert np.allclose(velocities, 0.0, atol=self.NUMERICAL_ATOL)

        # Check accelerations
        assert np.allclose(accelerations, 0.0, atol=self.NUMERICAL_ATOL)

    def test_scalar_trajectory_non_zero_start_time(self) -> None:
        """Test scalar trajectory with non-zero start time."""
        p0, p1 = 1.0, 7.0
        t0, t1 = 2.0, 5.0
        time_array = np.array([2.0, 3.5, 5.0])

        positions, velocities, accelerations = linear_traj(p0, p1, t0, t1, time_array)

        # Check positions
        expected_positions = np.array([1.0, 4.0, 7.0])
        assert np.allclose(positions, expected_positions, atol=self.NUMERICAL_ATOL)

        # Check velocities
        expected_velocity = 2.0  # (7-1)/(5-2)
        assert np.allclose(velocities, expected_velocity, atol=self.NUMERICAL_ATOL)

    def test_scalar_trajectory_extrapolation(self) -> None:
        """Test scalar trajectory extrapolation outside time range."""
        p0, p1 = 0.0, 4.0
        t0, t1 = 1.0, 3.0
        time_array = np.array([0.0, 1.0, 2.0, 3.0, 4.0])  # Includes extrapolation

        positions, velocities, accelerations = linear_traj(p0, p1, t0, t1, time_array)

        # Check positions (including extrapolated values)
        expected_positions = np.array([-2.0, 0.0, 2.0, 4.0, 6.0])
        assert np.allclose(positions, expected_positions, atol=self.NUMERICAL_ATOL)

        # Velocity should remain constant even during extrapolation
        expected_velocity = 2.0  # (4-0)/(3-1)
        assert np.allclose(velocities, expected_velocity, atol=self.NUMERICAL_ATOL)

    def test_scalar_trajectory_single_time_point(self) -> None:
        """Test scalar trajectory with single time point."""
        p0, p1 = 2.0, 8.0
        t0, t1 = 0.0, 3.0
        time_array = np.array([1.5])

        positions, velocities, accelerations = linear_traj(p0, p1, t0, t1, time_array)

        # Check dimensions
        assert positions.shape == (1,)
        assert velocities.shape == (1,)
        assert accelerations.shape == (1,)

        # Check values
        expected_position = 5.0  # 2 + (8-2)*(1.5-0)/(3-0)
        assert abs(positions[0] - expected_position) < self.NUMERICAL_ATOL

        expected_velocity = 2.0  # (8-2)/(3-0)
        assert abs(velocities[0] - expected_velocity) < self.NUMERICAL_ATOL

        assert abs(accelerations[0]) < self.NUMERICAL_ATOL


class TestLinearTrajectoryVector:
    """Test suite for vector linear trajectories."""

    NUMERICAL_RTOL = 1e-10
    NUMERICAL_ATOL = 1e-10

    def test_basic_2d_trajectory(self) -> None:
        """Test basic 2D vector trajectory."""
        p0 = [0.0, 0.0]
        p1 = [4.0, 6.0]
        t0, t1 = 0.0, 2.0
        time_array = np.array([0.0, 1.0, 2.0])

        positions, velocities, accelerations = linear_traj(p0, p1, t0, t1, time_array)

        # Check positions
        expected_positions = np.array([[0.0, 0.0], [2.0, 3.0], [4.0, 6.0]])
        assert np.allclose(positions, expected_positions, atol=self.NUMERICAL_ATOL)

        # Check velocities (should be constant)
        expected_velocity = [2.0, 3.0]  # (p1-p0)/(t1-t0)
        expected_velocities = np.tile(expected_velocity, (3, 1))
        assert np.allclose(velocities, expected_velocities, atol=self.NUMERICAL_ATOL)

        # Check accelerations (should be zero)
        assert np.allclose(accelerations, 0.0, atol=self.NUMERICAL_ATOL)

    def test_3d_trajectory(self) -> None:
        """Test 3D vector trajectory."""
        p0 = [1.0, 2.0, 3.0]
        p1 = [4.0, 8.0, 0.0]
        t0, t1 = 0.0, 3.0
        time_array = np.array([0.0, 1.5, 3.0])

        positions, velocities, accelerations = linear_traj(p0, p1, t0, t1, time_array)

        # Check positions
        expected_positions = np.array([[1.0, 2.0, 3.0], [2.5, 5.0, 1.5], [4.0, 8.0, 0.0]])
        assert np.allclose(positions, expected_positions, atol=self.NUMERICAL_ATOL)

        # Check velocities
        expected_velocity = [1.0, 2.0, -1.0]  # (p1-p0)/(t1-t0)
        expected_velocities = np.tile(expected_velocity, (3, 1))
        assert np.allclose(velocities, expected_velocities, atol=self.NUMERICAL_ATOL)

    def test_vector_trajectory_numpy_arrays(self) -> None:
        """Test vector trajectory with numpy array inputs."""
        p0 = np.array([0.0, 5.0])
        p1 = np.array([8.0, 1.0])
        t0, t1 = 1.0, 3.0
        time_array = np.array([1.0, 2.0, 3.0])

        positions, velocities, accelerations = linear_traj(p0, p1, t0, t1, time_array)

        # Check positions
        expected_positions = np.array([[0.0, 5.0], [4.0, 3.0], [8.0, 1.0]])
        assert np.allclose(positions, expected_positions, atol=self.NUMERICAL_ATOL)

        # Check shapes
        assert positions.shape == (3, 2)
        assert velocities.shape == (3, 2)
        assert accelerations.shape == (3, 2)

    def test_vector_trajectory_different_dimensions(self) -> None:
        """Test vector trajectories with different dimensions."""
        for dim in [1, 2, 3, 5, 10]:
            p0 = np.zeros(dim)
            p1 = np.ones(dim) * 2.0
            t0, t1 = 0.0, 2.0
            time_array = np.array([0.0, 1.0, 2.0])

            positions, velocities, accelerations = linear_traj(p0, p1, t0, t1, time_array)

            # Check shapes
            assert positions.shape == (3, dim)
            assert velocities.shape == (3, dim)
            assert accelerations.shape == (3, dim)

            # Check values
            expected_velocity = np.ones(dim)  # (2-0)/(2-0) = 1 for each dimension
            assert np.allclose(velocities[0], expected_velocity, atol=self.NUMERICAL_ATOL)
            assert np.allclose(accelerations, 0.0, atol=self.NUMERICAL_ATOL)

    def test_vector_trajectory_mixed_directions(self) -> None:
        """Test vector trajectory with mixed positive/negative directions."""
        p0 = [1.0, 5.0, -2.0]
        p1 = [3.0, 2.0, 4.0]
        t0, t1 = 0.0, 2.0
        time_array = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

        positions, velocities, accelerations = linear_traj(p0, p1, t0, t1, time_array)

        # Check that trajectory moves correctly in each dimension
        expected_velocity = [1.0, -1.5, 3.0]  # (p1-p0)/(t1-t0)

        for i in range(len(time_array)):
            assert np.allclose(velocities[i], expected_velocity, atol=self.NUMERICAL_ATOL)


class TestLinearTrajectoryEdgeCases:
    """Test suite for edge cases."""

    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def test_zero_time_duration(self) -> None:
        """Test trajectory with zero time duration."""
        p0, p1 = 0.0, 5.0
        t0 = t1 = 2.0  # Zero duration
        time_array = np.array([2.0])

        # This should produce warnings but not crash
        with pytest.warns(RuntimeWarning):
            positions, velocities, accelerations = linear_traj(p0, p1, t0, t1, time_array)

            # Should still return arrays of correct shape
            assert positions.shape == (1,)
            assert velocities.shape == (1,)
            assert accelerations.shape == (1,)

    def test_very_small_time_duration(self) -> None:
        """Test trajectory with very small time duration."""
        p0, p1 = 0.0, 1.0
        t0, t1 = 0.0, 1e-10
        time_array = np.array([0.0, 5e-11, 1e-10])

        positions, velocities, accelerations = linear_traj(p0, p1, t0, t1, time_array)

        # Should handle small durations
        assert np.all(np.isfinite(positions))
        assert np.all(np.isfinite(velocities))
        assert np.allclose(accelerations, 0.0, atol=self.NUMERICAL_ATOL)

    def test_large_time_duration(self) -> None:
        """Test trajectory with very large time duration."""
        p0, p1 = 0.0, 1.0
        t0, t1 = 0.0, 1e6
        time_array = np.array([0.0, 5e5, 1e6])

        positions, velocities, accelerations = linear_traj(p0, p1, t0, t1, time_array)

        # Should handle large durations
        expected_positions = np.array([0.0, 0.5, 1.0])
        assert np.allclose(positions, expected_positions, atol=self.NUMERICAL_ATOL)

        expected_velocity = 1e-6  # 1/(1e6)
        assert np.allclose(velocities, expected_velocity, atol=self.NUMERICAL_ATOL)

    def test_negative_time_values(self) -> None:
        """Test trajectory with negative time values."""
        p0, p1 = 2.0, 6.0
        t0, t1 = -3.0, -1.0
        time_array = np.array([-3.0, -2.0, -1.0])

        positions, velocities, accelerations = linear_traj(p0, p1, t0, t1, time_array)

        # Should handle negative times correctly
        expected_positions = np.array([2.0, 4.0, 6.0])
        assert np.allclose(positions, expected_positions, atol=self.NUMERICAL_ATOL)

        expected_velocity = 2.0  # (6-2)/(-1-(-3))
        assert np.allclose(velocities, expected_velocity, atol=self.NUMERICAL_ATOL)

    def test_empty_time_array(self) -> None:
        """Test trajectory with empty time array."""
        p0, p1 = 0.0, 5.0
        t0, t1 = 0.0, 2.0
        time_array = np.array([])

        positions, velocities, accelerations = linear_traj(p0, p1, t0, t1, time_array)

        # Should return empty arrays
        assert positions.shape == (0,)
        assert velocities.shape == (0,)
        assert accelerations.shape == (0,)

    def test_large_position_values(self) -> None:
        """Test trajectory with very large position values."""
        p0, p1 = 1e6, 2e6
        t0, t1 = 0.0, 10.0
        time_array = np.array([0.0, 5.0, 10.0])

        positions, velocities, accelerations = linear_traj(p0, p1, t0, t1, time_array)

        # Should handle large values correctly
        expected_positions = np.array([1e6, 1.5e6, 2e6])
        assert np.allclose(positions, expected_positions, rtol=self.NUMERICAL_RTOL)

        expected_velocity = 1e5  # (2e6-1e6)/10
        assert np.allclose(velocities, expected_velocity, rtol=self.NUMERICAL_RTOL)


class TestLinearTrajectoryInputValidation:
    """Test suite for input validation and type handling."""

    def test_list_inputs(self) -> None:
        """Test trajectory with list inputs."""
        p0 = [1.0, 2.0]
        p1 = [3.0, 4.0]
        t0, t1 = 0.0, 1.0
        time_array = [0.0, 0.5, 1.0]  # List instead of array

        positions, velocities, accelerations = linear_traj(p0, p1, t0, t1, time_array)

        # Should convert to numpy arrays internally
        assert isinstance(positions, np.ndarray)
        assert isinstance(velocities, np.ndarray)
        assert isinstance(accelerations, np.ndarray)

    def test_mixed_input_types(self) -> None:
        """Test trajectory with mixed input types."""
        p0 = 0  # Integer
        p1 = 5.0  # Float
        t0, t1 = 0.0, 2.0
        time_array = np.array([0.0, 1.0, 2.0])

        positions, velocities, accelerations = linear_traj(p0, p1, t0, t1, time_array)

        # Should handle mixed types correctly
        expected_positions = np.array([0.0, 2.5, 5.0])
        assert np.allclose(positions, expected_positions)

    def test_single_element_arrays(self) -> None:
        """Test trajectory with single-element position arrays."""
        p0 = np.array([3.0])
        p1 = np.array([7.0])
        t0, t1 = 0.0, 2.0
        time_array = np.array([0.0, 1.0, 2.0])

        positions, velocities, accelerations = linear_traj(p0, p1, t0, t1, time_array)

        # Should treat as vector trajectory
        assert positions.shape == (3, 1)
        assert velocities.shape == (3, 1)
        assert accelerations.shape == (3, 1)


class TestLinearTrajectoryMathematicalProperties:
    """Test suite for mathematical properties verification."""

    NUMERICAL_RTOL = 1e-10
    NUMERICAL_ATOL = 1e-10

    def test_linearity_property(self) -> None:
        """Test that trajectory is truly linear."""
        p0, p1 = 1.0, 9.0
        t0, t1 = 0.0, 4.0
        time_array = np.linspace(0.0, 4.0, 20)

        positions, velocities, accelerations = linear_traj(p0, p1, t0, t1, time_array)

        # Verify linearity: second differences should be zero
        second_diffs = np.diff(positions, n=2)
        assert np.allclose(second_diffs, 0.0, atol=self.NUMERICAL_ATOL)

        # Verify constant velocity
        first_diffs = np.diff(positions)
        dt = time_array[1] - time_array[0]
        numerical_velocities = first_diffs / dt
        expected_velocity = (p1 - p0) / (t1 - t0)
        assert np.allclose(numerical_velocities, expected_velocity, atol=1e-6)

    def test_interpolation_property(self) -> None:
        """Test that trajectory interpolates correctly at boundary points."""
        p0, p1 = 3.0, 11.0
        t0, t1 = 1.0, 5.0

        # Test exactly at boundary points
        time_array = np.array([t0, t1])
        positions, _, _ = linear_traj(p0, p1, t0, t1, time_array)

        assert abs(positions[0] - p0) < self.NUMERICAL_ATOL
        assert abs(positions[1] - p1) < self.NUMERICAL_ATOL

    def test_midpoint_property(self) -> None:
        """Test that midpoint has correct value."""
        p0, p1 = 2.0, 8.0
        t0, t1 = 0.0, 4.0
        t_mid = (t0 + t1) / 2

        time_array = np.array([t_mid])
        positions, _, _ = linear_traj(p0, p1, t0, t1, time_array)

        expected_mid_position = (p0 + p1) / 2
        assert abs(positions[0] - expected_mid_position) < self.NUMERICAL_ATOL

    def test_superposition_property(self) -> None:
        """Test linear superposition for vector trajectories."""
        # Create two separate 1D trajectories
        p0_x, p1_x = 0.0, 4.0
        p0_y, p1_y = 1.0, 5.0
        t0, t1 = 0.0, 2.0
        time_array = np.array([0.0, 1.0, 2.0])

        # Separate trajectories
        pos_x, vel_x, acc_x = linear_traj(p0_x, p1_x, t0, t1, time_array)
        pos_y, vel_y, acc_y = linear_traj(p0_y, p1_y, t0, t1, time_array)

        # Combined trajectory
        p0_combined = [p0_x, p0_y]
        p1_combined = [p1_x, p1_y]
        pos_combined, vel_combined, acc_combined = linear_traj(
            p0_combined, p1_combined, t0, t1, time_array
        )

        # Should satisfy superposition
        assert np.allclose(pos_combined[:, 0], pos_x, atol=self.NUMERICAL_ATOL)
        assert np.allclose(pos_combined[:, 1], pos_y, atol=self.NUMERICAL_ATOL)
        assert np.allclose(vel_combined[:, 0], vel_x, atol=self.NUMERICAL_ATOL)
        assert np.allclose(vel_combined[:, 1], vel_y, atol=self.NUMERICAL_ATOL)


class TestLinearTrajectoryPerformance:
    """Test suite for performance benchmarks."""

    @pytest.mark.parametrize("n_points", [100, 1000, 10000])
    def test_scalar_trajectory_performance(
        self, n_points: int, benchmark: pytest.FixtureFunction
    ) -> None:
        """Benchmark scalar trajectory performance."""
        p0, p1 = 0.0, 100.0
        t0, t1 = 0.0, 10.0
        time_array = np.linspace(t0, t1, n_points)

        def compute_trajectory():
            return linear_traj(p0, p1, t0, t1, time_array)

        positions, velocities, accelerations = benchmark(compute_trajectory)
        assert len(positions) == n_points

    @pytest.mark.parametrize("dimension", [2, 3, 10])
    def test_vector_trajectory_performance(
        self, dimension: int, benchmark: pytest.FixtureFunction
    ) -> None:
        """Benchmark vector trajectory performance."""
        p0 = np.zeros(dimension)
        p1 = np.ones(dimension) * 10.0
        t0, t1 = 0.0, 5.0
        time_array = np.linspace(t0, t1, 1000)

        def compute_trajectory():
            return linear_traj(p0, p1, t0, t1, time_array)

        positions, velocities, accelerations = benchmark(compute_trajectory)
        assert positions.shape == (1000, dimension)

    def test_large_dataset_performance(self, benchmark: pytest.FixtureFunction) -> None:
        """Benchmark performance with large datasets."""
        p0 = np.random.rand(100)  # 100D trajectory
        p1 = np.random.rand(100) * 10
        t0, t1 = 0.0, 10.0
        time_array = np.linspace(t0, t1, 5000)  # 5000 time points

        def compute_large_trajectory():
            return linear_traj(p0, p1, t0, t1, time_array)

        positions, velocities, accelerations = benchmark(compute_large_trajectory)
        assert positions.shape == (5000, 100)


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main(["-xvs", __file__])
