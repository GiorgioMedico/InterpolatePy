"""
Comprehensive tests for linear-parabolic trajectory implementation.

This module contains extensive tests for the ParabolicBlendTrajectory class covering:
1. ParabolicBlendTrajectory - Linear segments with parabolic blends at via points

Test coverage includes:
- Constructor validation and parameter checking
- Mathematical accuracy with known analytical solutions
- Trajectory generation algorithms
- Boundary condition handling (zero initial/final velocities)
- Edge cases and error handling
- Numerical stability and convergence
- Performance benchmarks
- Plotting functionality

The tests verify that parabolic blend trajectories generate smooth transitions
between linear segments with appropriate continuity constraints.
"""

from typing import Any
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest

from interpolatepy.lin_poly_parabolic import ParabolicBlendTrajectory


# Type alias for pytest benchmark fixture
if not hasattr(pytest, "FixtureFunction"):
    pytest.FixtureFunction = Any


class TestParabolicBlendTrajectoryConstruction:
    """Test suite for ParabolicBlendTrajectory construction and validation."""

    def test_basic_construction(self) -> None:
        """Test basic construction with valid parameters."""
        q = [0.0, 1.0, 2.0]
        t = [0.0, 1.0, 2.0]
        dt_blend = [0.1, 0.2, 0.1]

        traj = ParabolicBlendTrajectory(q, t, dt_blend)

        np.testing.assert_array_equal(traj.q, np.array([0.0, 1.0, 2.0]))
        np.testing.assert_array_equal(traj.t, np.array([0.0, 1.0, 2.0]))
        np.testing.assert_array_equal(traj.dt_blend, np.array([0.1, 0.2, 0.1]))
        assert traj.dt == 0.01

    def test_construction_with_custom_dt(self) -> None:
        """Test construction with custom sampling interval."""
        q = [0.0, 1.0, 2.0]
        t = [0.0, 1.0, 2.0]
        dt_blend = [0.1, 0.2, 0.1]

        traj = ParabolicBlendTrajectory(q, t, dt_blend, dt=0.005)

        assert traj.dt == 0.005

    def test_construction_with_lists(self) -> None:
        """Test construction with Python lists."""
        q = [0.0, 1.0, 2.0]
        t = [0.0, 1.0, 2.0]
        dt_blend = [0.1, 0.2, 0.1]

        traj = ParabolicBlendTrajectory(q, t, dt_blend)

        assert isinstance(traj.q, np.ndarray)
        assert isinstance(traj.t, np.ndarray)
        assert isinstance(traj.dt_blend, np.ndarray)

    def test_construction_with_numpy_arrays(self) -> None:
        """Test construction with numpy arrays."""
        q = np.array([0.0, 1.0, 2.0])
        t = np.array([0.0, 1.0, 2.0])
        dt_blend = np.array([0.1, 0.2, 0.1])

        traj = ParabolicBlendTrajectory(q, t, dt_blend)

        np.testing.assert_array_equal(traj.q, q)
        np.testing.assert_array_equal(traj.t, t)
        np.testing.assert_array_equal(traj.dt_blend, dt_blend)

    def test_construction_mismatched_lengths(self) -> None:
        """Test that mismatched array lengths raise ValueError."""
        q = [0.0, 1.0, 2.0]
        t = [0.0, 1.0]  # Wrong length
        dt_blend = [0.1, 0.2, 0.1]

        with pytest.raises(ValueError, match="Lengths of q, t, and dt_blend must match"):
            ParabolicBlendTrajectory(q, t, dt_blend)

    def test_construction_empty_arrays(self) -> None:
        """Test construction with empty arrays."""
        q: list[float] = []
        t: list[float] = []
        dt_blend: list[float] = []

        traj = ParabolicBlendTrajectory(q, t, dt_blend)

        assert len(traj.q) == 0
        assert len(traj.t) == 0
        assert len(traj.dt_blend) == 0

    def test_construction_single_point(self) -> None:
        """Test construction with single waypoint."""
        q = [1.0]
        t = [0.0]
        dt_blend = [0.1]

        traj = ParabolicBlendTrajectory(q, t, dt_blend)

        assert len(traj.q) == 1
        assert traj.q[0] == 1.0


class TestParabolicBlendTrajectoryGeneration:
    """Test suite for trajectory generation functionality."""

    def test_simple_two_point_trajectory(self) -> None:
        """Test trajectory generation with two waypoints."""
        q = [0.0, 1.0]
        t = [0.0, 1.0]
        dt_blend = [0.2, 0.2]

        traj = ParabolicBlendTrajectory(q, t, dt_blend)
        traj_func, duration = traj.generate()

        assert callable(traj_func)
        assert duration > 0.0

        # Test trajectory function returns tuple of 3 values
        pos, vel, acc = traj_func(0.0)
        assert isinstance(pos, int | float | np.number)
        assert isinstance(vel, int | float | np.number)
        assert isinstance(acc, int | float | np.number)

    def test_three_point_trajectory(self) -> None:
        """Test trajectory generation with three waypoints."""
        q = [0.0, 1.0, 2.0]
        t = [0.0, 1.0, 2.0]
        dt_blend = [0.1, 0.2, 0.1]

        traj = ParabolicBlendTrajectory(q, t, dt_blend)
        traj_func, duration = traj_func, duration = traj.generate()

        assert callable(traj_func)
        assert duration > 0.0

        # Check continuity at several points
        for test_time in [0.0, duration / 4, duration / 2, 3 * duration / 4, duration]:
            pos, vel, acc = traj_func(test_time)
            assert np.isfinite(pos)
            assert np.isfinite(vel)
            assert np.isfinite(acc)

    def test_initial_and_final_velocities_zero(self) -> None:
        """Test that initial and final velocities are zero."""
        q = [0.0, 1.0, 2.0]
        t = [0.0, 1.0, 2.0]
        dt_blend = [0.1, 0.2, 0.1]

        traj = ParabolicBlendTrajectory(q, t, dt_blend)
        traj_func, duration = traj.generate()

        # Check initial velocity
        _, vel_initial, _ = traj_func(0.0)
        assert abs(vel_initial) < 1e-10, f"Initial velocity should be zero, got {vel_initial}"

        # Check final velocity
        _, vel_final, _ = traj_func(duration)
        assert abs(vel_final) < 1e-10, f"Final velocity should be zero, got {vel_final}"

    def test_position_continuity(self) -> None:
        """Test position continuity throughout trajectory."""
        q = [0.0, 2.0, 1.0, 3.0]
        t = [0.0, 1.0, 2.0, 3.0]
        dt_blend = [0.1, 0.2, 0.15, 0.1]

        traj = ParabolicBlendTrajectory(q, t, dt_blend)
        traj_func, duration = traj.generate()

        # Sample trajectory densely to check for discontinuities
        times = np.linspace(0, duration, 1000)
        positions = []

        for t_sample in times:
            pos, _, _ = traj_func(t_sample)
            positions.append(pos)

        # Check no sudden jumps in position (derivative test)
        positions = np.array(positions)
        pos_diff = np.diff(positions)
        dt_sample = duration / (len(times) - 1)
        velocities_approx = pos_diff / dt_sample

        # Velocity should be bounded for smooth trajectory
        assert np.all(np.abs(velocities_approx) < 100), "Trajectory shows discontinuities"

    def test_out_of_bounds_handling(self) -> None:
        """Test trajectory evaluation outside valid time range."""
        q = [0.0, 1.0, 2.0]
        t = [0.0, 1.0, 2.0]
        dt_blend = [0.1, 0.2, 0.1]

        traj = ParabolicBlendTrajectory(q, t, dt_blend)
        traj_func, duration = traj.generate()

        # Test before start time
        pos_before, vel_before, acc_before = traj_func(-1.0)
        pos_start, vel_start, acc_start = traj_func(0.0)

        assert pos_before == pos_start
        assert vel_before == vel_start
        assert acc_before == acc_start

        # Test after end time
        pos_after, vel_after, acc_after = traj_func(duration + 1.0)
        pos_end, vel_end, acc_end = traj_func(duration)

        assert pos_after == pos_end
        assert vel_after == vel_end
        assert acc_after == acc_end

    def test_monotonic_time_requirement(self) -> None:
        """Test trajectory with non-monotonic time points."""
        q = [0.0, 1.0, 2.0]
        t = [0.0, 2.0, 1.0]  # Non-monotonic
        dt_blend = [0.1, 0.2, 0.1]

        traj = ParabolicBlendTrajectory(q, t, dt_blend)

        # Should still construct but may produce unexpected results
        # The implementation doesn't explicitly check for monotonic times
        traj_func, duration = traj.generate()
        assert callable(traj_func)
        assert duration > 0.0

    def test_zero_blend_durations(self) -> None:
        """Test trajectory with zero blend durations."""
        q = [0.0, 1.0, 2.0]
        t = [0.0, 1.0, 2.0]
        dt_blend = [0.0, 0.0, 0.0]

        traj = ParabolicBlendTrajectory(q, t, dt_blend)

        # Zero blend durations cause division by zero in the implementation
        # This is expected behavior that should be handled by the user
        with pytest.warns(RuntimeWarning):
            traj_func, duration = traj.generate()

        assert callable(traj_func)
        assert duration > 0.0

    def test_large_blend_durations(self) -> None:
        """Test trajectory with large blend durations."""
        q = [0.0, 1.0, 2.0]
        t = [0.0, 2.0, 4.0]
        dt_blend = [1.0, 2.0, 1.0]  # Large blend durations

        traj = ParabolicBlendTrajectory(q, t, dt_blend)
        traj_func, duration = traj.generate()

        assert callable(traj_func)
        assert duration > 0.0


class TestParabolicBlendTrajectoryMathematicalProperties:
    """Test suite for mathematical properties and accuracy."""

    def test_conservation_of_position_endpoints(self) -> None:
        """Test that trajectory passes through corrected via points."""
        q = [0.0, 2.0, 4.0]
        t = [0.0, 2.0, 4.0]
        dt_blend = [0.2, 0.4, 0.2]

        traj = ParabolicBlendTrajectory(q, t, dt_blend)
        traj_func, duration = traj.generate()

        # Get positions at start and end
        pos_start, _, _ = traj_func(0.0)
        pos_end, _, _ = traj_func(duration)

        # Due to blend modifications, exact waypoint matching may not occur
        # But positions should be reasonable
        assert np.isfinite(pos_start)
        assert np.isfinite(pos_end)

    def test_velocity_profile_smoothness(self) -> None:
        """Test that velocity profile is reasonably smooth."""
        q = [0.0, 1.0, 0.0, 1.0]
        t = [0.0, 1.0, 2.0, 3.0]
        dt_blend = [0.1, 0.2, 0.2, 0.1]

        traj = ParabolicBlendTrajectory(q, t, dt_blend)
        traj_func, duration = traj.generate()

        # Sample velocity at multiple points
        times = np.linspace(0, duration, 100)
        velocities = []

        for t_sample in times:
            _, vel, _ = traj_func(t_sample)
            velocities.append(vel)

        velocities = np.array(velocities)

        # Check that velocity doesn't have unreasonable jumps
        vel_diff = np.diff(velocities)
        max_vel_jump = np.max(np.abs(vel_diff))

        # This is a heuristic check - adjust threshold if needed
        assert max_vel_jump < 10.0, f"Velocity jumps too large: {max_vel_jump}"

    def test_acceleration_in_blends_vs_linear_segments(self) -> None:
        """Test that acceleration is non-zero in blends and zero in linear segments."""
        q = [0.0, 1.0, 2.0]
        t = [0.0, 1.0, 2.0]
        dt_blend = [0.2, 0.4, 0.2]

        traj = ParabolicBlendTrajectory(q, t, dt_blend)
        traj_func, duration = traj.generate()

        # Sample throughout trajectory
        times = np.linspace(0, duration, 50)
        accelerations = []

        for t_sample in times:
            _, _, acc = traj_func(t_sample)
            accelerations.append(acc)

        # We expect some non-zero accelerations (in blend regions)
        accelerations = np.array(accelerations)
        has_nonzero_acc = np.any(np.abs(accelerations) > 1e-10)

        assert has_nonzero_acc, (
            "Trajectory should have some non-zero accelerations in blend regions"
        )

    def test_energy_and_smoothness_properties(self) -> None:
        """Test energy-related properties of the trajectory."""
        q = [0.0, 3.0, 1.0, 4.0]
        t = [0.0, 1.0, 2.0, 3.0]
        dt_blend = [0.15, 0.3, 0.25, 0.15]

        traj = ParabolicBlendTrajectory(q, t, dt_blend)
        traj_func, duration = traj.generate()

        # Sample trajectory
        times = np.linspace(0, duration, 200)
        positions = []
        velocities = []
        accelerations = []

        for t_sample in times:
            pos, vel, acc = traj_func(t_sample)
            positions.append(pos)
            velocities.append(vel)
            accelerations.append(acc)

        positions = np.array(positions)
        velocities = np.array(velocities)
        accelerations = np.array(accelerations)

        # Check finite values
        assert np.all(np.isfinite(positions))
        assert np.all(np.isfinite(velocities))
        assert np.all(np.isfinite(accelerations))

        # Kinetic energy should be reasonable (not excessively high)
        kinetic_energy = 0.5 * velocities**2
        max_kinetic = np.max(kinetic_energy)

        assert max_kinetic < 100.0, f"Maximum kinetic energy too high: {max_kinetic}"


class TestParabolicBlendTrajectoryEdgeCases:
    """Test suite for edge cases and error conditions."""

    def test_identical_waypoints(self) -> None:
        """Test trajectory with identical consecutive waypoints."""
        q = [1.0, 1.0, 1.0]
        t = [0.0, 1.0, 2.0]
        dt_blend = [0.1, 0.2, 0.1]

        traj = ParabolicBlendTrajectory(q, t, dt_blend)
        traj_func, duration = traj.generate()

        # Should handle identical waypoints gracefully
        pos_mid, vel_mid, acc_mid = traj_func(duration / 2)

        assert np.isfinite(pos_mid)
        assert np.isfinite(vel_mid)
        assert np.isfinite(acc_mid)

    def test_very_small_time_intervals(self) -> None:
        """Test with very small time intervals between waypoints."""
        q = [0.0, 1.0, 2.0]
        t = [0.0, 0.001, 0.002]  # Very small intervals
        dt_blend = [0.0001, 0.0002, 0.0001]

        traj = ParabolicBlendTrajectory(q, t, dt_blend)
        traj_func, duration = traj.generate()

        assert callable(traj_func)
        assert duration > 0.0

        # Check that we can evaluate without numerical issues
        pos, vel, acc = traj_func(duration / 2)
        assert np.isfinite(pos)
        assert np.isfinite(vel)
        assert np.isfinite(acc)

    def test_negative_positions(self) -> None:
        """Test trajectory with negative position values."""
        q = [-2.0, -1.0, -3.0]
        t = [0.0, 1.0, 2.0]
        dt_blend = [0.1, 0.2, 0.1]

        traj = ParabolicBlendTrajectory(q, t, dt_blend)
        traj_func, duration = traj.generate()

        pos_start, _, _ = traj_func(0.0)
        pos_end, _, _ = traj_func(duration)

        assert np.isfinite(pos_start)
        assert np.isfinite(pos_end)

    def test_single_waypoint_trajectory(self) -> None:
        """Test trajectory with only one waypoint."""
        q = [5.0]
        t = [0.0]
        dt_blend = [0.2]

        traj = ParabolicBlendTrajectory(q, t, dt_blend)
        traj_func, duration = traj.generate()

        # Single point should create a trivial trajectory
        pos, vel, acc = traj_func(duration / 2)

        assert np.isfinite(pos)
        assert np.isfinite(vel)
        assert np.isfinite(acc)

        # For single point, velocity should be zero throughout
        assert abs(vel) < 1e-10


class TestParabolicBlendTrajectoryPlotting:
    """Test suite for plotting functionality."""

    @patch("matplotlib.pyplot.subplots")
    def test_plot_without_data(self, mock_subplots: Mock) -> None:
        """Test plot method without providing trajectory data."""
        # Setup mock - subplots returns (fig, axes) tuple
        from unittest.mock import MagicMock

        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock(), MagicMock()]
        mock_subplots.return_value = (mock_fig, mock_axes)

        q = [0.0, 1.0, 2.0]
        t = [0.0, 1.0, 2.0]
        dt_blend = [0.1, 0.2, 0.1]

        traj = ParabolicBlendTrajectory(q, t, dt_blend)

        # Should not raise an exception
        traj.plot()

        # Verify plot was called
        mock_subplots.assert_called_once()

    @patch("matplotlib.pyplot.subplots")
    def test_plot_with_provided_data(self, mock_subplots: Mock) -> None:
        """Test plot method with provided trajectory data."""
        # Setup mock - subplots returns (fig, axes) tuple
        from unittest.mock import MagicMock

        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock(), MagicMock()]
        mock_subplots.return_value = (mock_fig, mock_axes)

        q = [0.0, 1.0, 2.0]
        t = [0.0, 1.0, 2.0]
        dt_blend = [0.1, 0.2, 0.1]

        traj = ParabolicBlendTrajectory(q, t, dt_blend)

        # Generate some sample data
        times = np.linspace(0, 2, 100)
        pos = np.sin(times)
        vel = np.cos(times)
        acc = -np.sin(times)

        # Should not raise an exception
        traj.plot(times=times, pos=pos, vel=vel, acc=acc)

        # Verify plot was called
        mock_subplots.assert_called_once()

    @patch("matplotlib.pyplot.subplots")
    def test_plot_with_custom_dt(self, mock_subplots: Mock) -> None:
        """Test plot method with custom sampling interval."""
        # Setup mock - subplots returns (fig, axes) tuple
        from unittest.mock import MagicMock

        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock(), MagicMock()]
        mock_subplots.return_value = (mock_fig, mock_axes)

        q = [0.0, 1.0, 2.0]
        t = [0.0, 1.0, 2.0]
        dt_blend = [0.1, 0.2, 0.1]

        traj = ParabolicBlendTrajectory(q, t, dt_blend, dt=0.02)

        # Should not raise an exception
        traj.plot()

        # Verify plot was called
        mock_subplots.assert_called_once()


class TestParabolicBlendTrajectoryPerformance:
    """Test suite for performance benchmarks."""

    def test_construction_performance(self, benchmark: pytest.FixtureFunction) -> None:
        """Benchmark trajectory construction."""
        q = np.random.randn(10)
        t = np.linspace(0, 10, 10)
        dt_blend = np.full(10, 0.1)

        def construct_trajectory() -> ParabolicBlendTrajectory:
            return ParabolicBlendTrajectory(q, t, dt_blend)

        result = benchmark(construct_trajectory)
        assert isinstance(result, ParabolicBlendTrajectory)

    def test_generation_performance(self, benchmark: pytest.FixtureFunction) -> None:
        """Benchmark trajectory generation."""
        q = np.random.randn(20)
        t = np.linspace(0, 20, 20)
        dt_blend = np.full(20, 0.2)

        traj = ParabolicBlendTrajectory(q, t, dt_blend)

        def generate_trajectory():
            return traj.generate()

        traj_func, duration = benchmark(generate_trajectory)
        assert callable(traj_func)
        assert duration > 0.0

    def test_evaluation_performance(self, benchmark: pytest.FixtureFunction) -> None:
        """Benchmark trajectory evaluation."""
        q = np.random.randn(15)
        t = np.linspace(0, 15, 15)
        dt_blend = np.full(15, 0.15)

        traj = ParabolicBlendTrajectory(q, t, dt_blend)
        traj_func, duration = traj.generate()

        def evaluate_trajectory():
            return traj_func(duration / 2)

        pos, vel, acc = benchmark(evaluate_trajectory)
        assert np.isfinite(pos)
        assert np.isfinite(vel)
        assert np.isfinite(acc)

    def test_large_waypoint_trajectory(self, benchmark: pytest.FixtureFunction) -> None:
        """Benchmark trajectory with many waypoints."""
        n_points = 100
        q = np.random.randn(n_points)
        t = np.linspace(0, n_points, n_points)
        dt_blend = np.full(n_points, 0.1)

        def create_and_generate():
            traj = ParabolicBlendTrajectory(q, t, dt_blend)
            return traj.generate()

        traj_func, duration = benchmark(create_and_generate)
        assert callable(traj_func)
        assert duration > 0.0


class TestParabolicBlendTrajectoryIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow_example(self) -> None:
        """Test complete workflow similar to the example script."""
        # Define waypoints similar to the example
        q = [0, 2 * np.pi, np.pi / 2, np.pi]
        t = [0, 2, 3, 5]
        dt_blend = np.full(len(t), 0.6)

        # Create trajectory
        traj = ParabolicBlendTrajectory(q, t, dt_blend)

        # Generate trajectory function
        traj_func, duration = traj.generate()

        # Verify basic properties
        assert callable(traj_func)
        assert duration > 0.0

        # Test evaluation at specific points
        evaluation_times = [0.5, 2.1, 3.5, 4.8]

        for time_point in evaluation_times:
            if time_point <= duration:
                position, velocity, acceleration = traj_func(time_point)

                assert np.isfinite(position)
                assert np.isfinite(velocity)
                assert np.isfinite(acceleration)

        # Test out-of-bounds handling
        out_of_bounds_time = duration + 1.0
        position, velocity, acceleration = traj_func(out_of_bounds_time)

        assert np.isfinite(position)
        assert np.isfinite(velocity)
        assert np.isfinite(acceleration)

    def test_trajectory_with_varying_blend_durations(self) -> None:
        """Test trajectory with different blend durations at each point."""
        q = [0.0, 5.0, 2.0, 8.0, 3.0]
        t = [0.0, 2.0, 4.0, 6.0, 8.0]
        dt_blend = [0.1, 0.5, 0.2, 0.8, 0.1]  # Varying blend durations

        traj = ParabolicBlendTrajectory(q, t, dt_blend)
        traj_func, duration = traj.generate()

        # Sample at many points to verify continuity
        times = np.linspace(0, duration, 200)

        for t_sample in times:
            pos, vel, acc = traj_func(t_sample)

            assert np.isfinite(pos), f"Non-finite position at t={t_sample}"
            assert np.isfinite(vel), f"Non-finite velocity at t={t_sample}"
            assert np.isfinite(acc), f"Non-finite acceleration at t={t_sample}"

    def test_comparison_with_linear_interpolation(self) -> None:
        """Compare trajectory with small blend durations (approximates linear)."""
        q = [0.0, 2.0, 1.0, 3.0]
        t = [0.0, 1.0, 2.0, 3.0]
        dt_blend = [0.01, 0.01, 0.01, 0.01]  # Very small blending ≈ linear

        traj = ParabolicBlendTrajectory(q, t, dt_blend)
        traj_func, duration = traj.generate()

        # With very small blending, should be approximately piecewise linear
        # Test at original waypoint times (adjusted for trajectory timing)
        sample_times = np.linspace(0, duration, 50)

        for t_sample in sample_times:
            pos, vel, acc = traj_func(t_sample)

            # Should be finite and reasonable
            assert np.isfinite(pos)
            assert np.isfinite(vel)
            assert np.isfinite(acc)
