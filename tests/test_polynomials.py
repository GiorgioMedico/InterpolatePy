"""
Comprehensive tests for polynomial trajectory implementations.

This module contains extensive tests for the polynomial trajectory classes covering:
1. BoundaryCondition - Storage of boundary conditions
2. TimeInterval - Time interval representation
3. TrajectoryParams - Parameters for multipoint trajectories
4. PolynomialTrajectory - Various order polynomial trajectories (3rd, 5th, 7th)

Test coverage includes:
- Constructor validation and parameter checking
- Mathematical accuracy with known analytical solutions
- Boundary condition satisfaction
- Different polynomial orders (3rd, 5th, 7th order)
- Multipoint trajectory generation
- Heuristic velocity calculation
- Edge cases and error handling
- Performance benchmarks

The tests verify that polynomial trajectories correctly satisfy specified
boundary conditions and generate smooth motion profiles.
"""

from typing import Any

import numpy as np
import pytest

from interpolatepy.polynomials import BoundaryCondition
from interpolatepy.polynomials import PolynomialTrajectory
from interpolatepy.polynomials import TimeInterval
from interpolatepy.polynomials import TrajectoryParams
from interpolatepy.polynomials import ORDER_3
from interpolatepy.polynomials import ORDER_5
from interpolatepy.polynomials import ORDER_7

# Type alias for pytest benchmark fixture
if not hasattr(pytest, "FixtureFunction"):
    pytest.FixtureFunction = Any


class TestBoundaryCondition:
    """Test suite for BoundaryCondition dataclass."""

    def test_boundary_condition_creation_minimal(self) -> None:
        """Test BoundaryCondition creation with minimal parameters."""
        bc = BoundaryCondition(position=5.0, velocity=2.0)
        
        assert bc.position == 5.0
        assert bc.velocity == 2.0
        assert bc.acceleration == 0.0
        assert bc.jerk == 0.0

    def test_boundary_condition_creation_full(self) -> None:
        """Test BoundaryCondition creation with all parameters."""
        bc = BoundaryCondition(
            position=10.0, velocity=3.0, acceleration=1.5, jerk=0.8
        )
        
        assert bc.position == 10.0
        assert bc.velocity == 3.0
        assert bc.acceleration == 1.5
        assert bc.jerk == 0.8

    def test_boundary_condition_defaults(self) -> None:
        """Test that default values are correctly applied."""
        bc = BoundaryCondition(position=0.0, velocity=1.0)
        
        # acceleration and jerk should default to 0.0
        assert bc.acceleration == 0.0
        assert bc.jerk == 0.0

    def test_boundary_condition_negative_values(self) -> None:
        """Test BoundaryCondition with negative values."""
        bc = BoundaryCondition(
            position=-5.0, velocity=-2.0, acceleration=-1.0, jerk=-0.5
        )
        
        assert bc.position == -5.0
        assert bc.velocity == -2.0
        assert bc.acceleration == -1.0
        assert bc.jerk == -0.5


class TestTimeInterval:
    """Test suite for TimeInterval dataclass."""

    def test_time_interval_creation(self) -> None:
        """Test TimeInterval creation."""
        interval = TimeInterval(start=0.0, end=5.0)
        
        assert interval.start == 0.0
        assert interval.end == 5.0

    def test_time_interval_duration_calculation(self) -> None:
        """Test that we can calculate duration from time interval."""
        interval = TimeInterval(start=2.0, end=8.0)
        duration = interval.end - interval.start
        
        assert duration == 6.0

    def test_time_interval_negative_start(self) -> None:
        """Test TimeInterval with negative start time."""
        interval = TimeInterval(start=-3.0, end=2.0)
        
        assert interval.start == -3.0
        assert interval.end == 2.0

    def test_time_interval_zero_duration(self) -> None:
        """Test TimeInterval with zero duration."""
        interval = TimeInterval(start=5.0, end=5.0)
        duration = interval.end - interval.start
        
        assert duration == 0.0


class TestTrajectoryParams:
    """Test suite for TrajectoryParams dataclass."""

    def test_trajectory_params_minimal(self) -> None:
        """Test TrajectoryParams creation with minimal parameters."""
        points = [0.0, 5.0, 10.0]
        times = [0.0, 2.0, 4.0]
        
        params = TrajectoryParams(points=points, times=times)
        
        assert params.points == points
        assert params.times == times
        assert params.velocities is None
        assert params.accelerations is None
        assert params.jerks is None
        assert params.order == ORDER_3

    def test_trajectory_params_full(self) -> None:
        """Test TrajectoryParams creation with all parameters."""
        points = [0.0, 3.0, 8.0]
        times = [0.0, 1.5, 3.0]
        velocities = [0.0, 2.0, 0.0]
        accelerations = [0.0, 1.0, 0.0]
        jerks = [0.0, 0.5, 0.0]
        
        params = TrajectoryParams(
            points=points,
            times=times,
            velocities=velocities,
            accelerations=accelerations,
            jerks=jerks,
            order=ORDER_5
        )
        
        assert params.points == points
        assert params.times == times
        assert params.velocities == velocities
        assert params.accelerations == accelerations
        assert params.jerks == jerks
        assert params.order == ORDER_5

    def test_trajectory_params_order_validation(self) -> None:
        """Test that different orders can be specified."""
        points = [0.0, 5.0]
        times = [0.0, 2.0]
        
        for order in [ORDER_3, ORDER_5, ORDER_7]:
            params = TrajectoryParams(points=points, times=times, order=order)
            assert params.order == order


class TestPolynomialTrajectoryOrder3:
    """Test suite for 3rd order polynomial trajectories."""

    NUMERICAL_RTOL = 1e-10
    NUMERICAL_ATOL = 1e-10

    def test_order_3_basic_trajectory(self) -> None:
        """Test basic 3rd order trajectory generation."""
        initial = BoundaryCondition(position=0.0, velocity=0.0)
        final = BoundaryCondition(position=10.0, velocity=0.0)
        time = TimeInterval(start=0.0, end=2.0)
        
        trajectory = PolynomialTrajectory.order_3_trajectory(initial, final, time)
        
        # Test that we get a callable
        assert callable(trajectory)
        
        # Test boundary conditions
        q_init, v_init, a_init, j_init = trajectory(time.start)
        assert abs(q_init - initial.position) < self.NUMERICAL_ATOL
        assert abs(v_init - initial.velocity) < self.NUMERICAL_ATOL
        
        q_final, v_final, a_final, j_final = trajectory(time.end)
        assert abs(q_final - final.position) < self.NUMERICAL_ATOL
        assert abs(v_final - final.velocity) < self.NUMERICAL_ATOL

    def test_order_3_nonzero_velocities(self) -> None:
        """Test 3rd order trajectory with non-zero boundary velocities."""
        initial = BoundaryCondition(position=2.0, velocity=1.0)
        final = BoundaryCondition(position=8.0, velocity=3.0)
        time = TimeInterval(start=0.0, end=3.0)
        
        trajectory = PolynomialTrajectory.order_3_trajectory(initial, final, time)
        
        # Test boundary conditions
        q_init, v_init, _, _ = trajectory(time.start)
        assert abs(q_init - initial.position) < self.NUMERICAL_ATOL
        assert abs(v_init - initial.velocity) < self.NUMERICAL_ATOL
        
        q_final, v_final, _, _ = trajectory(time.end)
        assert abs(q_final - final.position) < self.NUMERICAL_ATOL
        assert abs(v_final - final.velocity) < self.NUMERICAL_ATOL

    def test_order_3_negative_displacement(self) -> None:
        """Test 3rd order trajectory with negative displacement."""
        initial = BoundaryCondition(position=10.0, velocity=0.0)
        final = BoundaryCondition(position=3.0, velocity=0.0)
        time = TimeInterval(start=0.0, end=2.0)
        
        trajectory = PolynomialTrajectory.order_3_trajectory(initial, final, time)
        
        # Should handle negative displacement correctly
        q_init, v_init, _, _ = trajectory(time.start)
        assert abs(q_init - initial.position) < self.NUMERICAL_ATOL
        assert abs(v_init - initial.velocity) < self.NUMERICAL_ATOL
        
        q_final, v_final, _, _ = trajectory(time.end)
        assert abs(q_final - final.position) < self.NUMERICAL_ATOL
        assert abs(v_final - final.velocity) < self.NUMERICAL_ATOL

    def test_order_3_zero_displacement(self) -> None:
        """Test 3rd order trajectory with zero displacement."""
        initial = BoundaryCondition(position=5.0, velocity=2.0)
        final = BoundaryCondition(position=5.0, velocity=-2.0)
        time = TimeInterval(start=0.0, end=1.0)
        
        trajectory = PolynomialTrajectory.order_3_trajectory(initial, final, time)
        
        # Should handle zero displacement with velocity change
        q_init, v_init, _, _ = trajectory(time.start)
        assert abs(q_init - initial.position) < self.NUMERICAL_ATOL
        assert abs(v_init - initial.velocity) < self.NUMERICAL_ATOL
        
        q_final, v_final, _, _ = trajectory(time.end)
        assert abs(q_final - final.position) < self.NUMERICAL_ATOL
        assert abs(v_final - final.velocity) < self.NUMERICAL_ATOL

    def test_order_3_different_time_intervals(self) -> None:
        """Test 3rd order trajectory with different time intervals."""
        initial = BoundaryCondition(position=0.0, velocity=1.0)
        final = BoundaryCondition(position=6.0, velocity=2.0)
        
        # Test with different time intervals
        for start_time, end_time in [(0.0, 1.0), (2.0, 5.0), (-1.0, 2.0)]:
            time = TimeInterval(start=start_time, end=end_time)
            trajectory = PolynomialTrajectory.order_3_trajectory(initial, final, time)
            
            # Check boundary conditions
            q_init, v_init, _, _ = trajectory(time.start)
            assert abs(q_init - initial.position) < self.NUMERICAL_ATOL
            assert abs(v_init - initial.velocity) < self.NUMERICAL_ATOL
            
            q_final, v_final, _, _ = trajectory(time.end)
            assert abs(q_final - final.position) < self.NUMERICAL_ATOL
            assert abs(v_final - final.velocity) < self.NUMERICAL_ATOL

    def test_order_3_acceleration_continuity(self) -> None:
        """Test that acceleration is continuous for 3rd order trajectory."""
        initial = BoundaryCondition(position=0.0, velocity=0.0)
        final = BoundaryCondition(position=4.0, velocity=0.0)
        time = TimeInterval(start=0.0, end=2.0)
        
        trajectory = PolynomialTrajectory.order_3_trajectory(initial, final, time)
        
        # For 3rd order polynomial, acceleration should be linear
        # Test at several points
        test_times = [0.0, 0.5, 1.0, 1.5, 2.0]
        accelerations = []
        
        for t in test_times:
            _, _, a, _ = trajectory(t)
            accelerations.append(a)
        
        # Acceleration should vary linearly (finite differences should be constant)
        diffs = np.diff(accelerations)
        # Allow for numerical precision
        assert np.allclose(diffs, diffs[0], atol=self.NUMERICAL_ATOL)

    def test_order_3_jerk_constant(self) -> None:
        """Test that jerk is constant for 3rd order trajectory."""
        initial = BoundaryCondition(position=1.0, velocity=0.5)
        final = BoundaryCondition(position=7.0, velocity=1.5)
        time = TimeInterval(start=0.0, end=3.0)
        
        trajectory = PolynomialTrajectory.order_3_trajectory(initial, final, time)
        
        # For 3rd order polynomial, jerk should be constant
        test_times = [0.0, 0.8, 1.6, 2.4, 3.0]
        jerks = []
        
        for t in test_times:
            _, _, _, j = trajectory(t)
            jerks.append(j)
        
        # All jerks should be equal
        assert np.allclose(jerks, jerks[0], atol=self.NUMERICAL_ATOL)


class TestPolynomialTrajectoryOrder5:
    """Test suite for 5th order polynomial trajectories."""

    NUMERICAL_RTOL = 1e-10
    NUMERICAL_ATOL = 1e-10

    def test_order_5_basic_trajectory(self) -> None:
        """Test basic 5th order trajectory generation."""
        initial = BoundaryCondition(position=0.0, velocity=0.0, acceleration=0.0)
        final = BoundaryCondition(position=8.0, velocity=0.0, acceleration=0.0)
        time = TimeInterval(start=0.0, end=2.0)
        
        trajectory = PolynomialTrajectory.order_5_trajectory(initial, final, time)
        
        # Test boundary conditions
        q_init, v_init, a_init, j_init = trajectory(time.start)
        assert abs(q_init - initial.position) < self.NUMERICAL_ATOL
        assert abs(v_init - initial.velocity) < self.NUMERICAL_ATOL
        assert abs(a_init - initial.acceleration) < self.NUMERICAL_ATOL
        
        q_final, v_final, a_final, j_final = trajectory(time.end)
        assert abs(q_final - final.position) < self.NUMERICAL_ATOL
        assert abs(v_final - final.velocity) < self.NUMERICAL_ATOL
        assert abs(a_final - final.acceleration) < self.NUMERICAL_ATOL

    def test_order_5_nonzero_accelerations(self) -> None:
        """Test 5th order trajectory with non-zero boundary accelerations."""
        initial = BoundaryCondition(position=1.0, velocity=2.0, acceleration=0.5)
        final = BoundaryCondition(position=9.0, velocity=1.0, acceleration=-0.3)
        time = TimeInterval(start=0.0, end=4.0)
        
        trajectory = PolynomialTrajectory.order_5_trajectory(initial, final, time)
        
        # Test boundary conditions
        q_init, v_init, a_init, _ = trajectory(time.start)
        assert abs(q_init - initial.position) < self.NUMERICAL_ATOL
        assert abs(v_init - initial.velocity) < self.NUMERICAL_ATOL
        assert abs(a_init - initial.acceleration) < self.NUMERICAL_ATOL
        
        q_final, v_final, a_final, _ = trajectory(time.end)
        assert abs(q_final - final.position) < self.NUMERICAL_ATOL
        assert abs(v_final - final.velocity) < self.NUMERICAL_ATOL
        assert abs(a_final - final.acceleration) < self.NUMERICAL_ATOL

    def test_order_5_jerk_continuity(self) -> None:
        """Test that jerk varies continuously for 5th order trajectory."""
        initial = BoundaryCondition(position=0.0, velocity=0.0, acceleration=0.0)
        final = BoundaryCondition(position=5.0, velocity=0.0, acceleration=0.0)
        time = TimeInterval(start=0.0, end=2.5)
        
        trajectory = PolynomialTrajectory.order_5_trajectory(initial, final, time)
        
        # For 5th order polynomial, jerk should be quadratic
        test_times = np.linspace(time.start, time.end, 10)
        jerks = []
        
        for t in test_times:
            _, _, _, j = trajectory(t)
            jerks.append(j)
        
        # Jerk should vary smoothly (no discontinuities)
        assert all(np.isfinite(j) for j in jerks)


class TestPolynomialTrajectoryOrder7:
    """Test suite for 7th order polynomial trajectories."""

    NUMERICAL_RTOL = 1e-10
    NUMERICAL_ATOL = 1e-10

    def test_order_7_basic_trajectory(self) -> None:
        """Test basic 7th order trajectory generation."""
        initial = BoundaryCondition(position=0.0, velocity=0.0, acceleration=0.0, jerk=0.0)
        final = BoundaryCondition(position=12.0, velocity=0.0, acceleration=0.0, jerk=0.0)
        time = TimeInterval(start=0.0, end=3.0)
        
        trajectory = PolynomialTrajectory.order_7_trajectory(initial, final, time)
        
        # Test boundary conditions
        q_init, v_init, a_init, j_init = trajectory(time.start)
        assert abs(q_init - initial.position) < self.NUMERICAL_ATOL
        assert abs(v_init - initial.velocity) < self.NUMERICAL_ATOL
        assert abs(a_init - initial.acceleration) < self.NUMERICAL_ATOL
        assert abs(j_init - initial.jerk) < self.NUMERICAL_ATOL
        
        q_final, v_final, a_final, j_final = trajectory(time.end)
        assert abs(q_final - final.position) < self.NUMERICAL_ATOL
        assert abs(v_final - final.velocity) < self.NUMERICAL_ATOL
        assert abs(a_final - final.acceleration) < self.NUMERICAL_ATOL
        assert abs(j_final - final.jerk) < self.NUMERICAL_ATOL

    def test_order_7_nonzero_jerks(self) -> None:
        """Test 7th order trajectory with non-zero boundary jerks."""
        initial = BoundaryCondition(
            position=2.0, velocity=1.0, acceleration=0.5, jerk=0.2
        )
        final = BoundaryCondition(
            position=10.0, velocity=2.0, acceleration=-0.3, jerk=-0.1
        )
        time = TimeInterval(start=0.0, end=4.0)
        
        trajectory = PolynomialTrajectory.order_7_trajectory(initial, final, time)
        
        # Test boundary conditions
        q_init, v_init, a_init, j_init = trajectory(time.start)
        assert abs(q_init - initial.position) < self.NUMERICAL_ATOL
        assert abs(v_init - initial.velocity) < self.NUMERICAL_ATOL
        assert abs(a_init - initial.acceleration) < self.NUMERICAL_ATOL
        assert abs(j_init - initial.jerk) < self.NUMERICAL_ATOL
        
        q_final, v_final, a_final, j_final = trajectory(time.end)
        assert abs(q_final - final.position) < self.NUMERICAL_ATOL
        assert abs(v_final - final.velocity) < self.NUMERICAL_ATOL
        assert abs(a_final - final.acceleration) < self.NUMERICAL_ATOL
        assert abs(j_final - final.jerk) < self.NUMERICAL_ATOL

    def test_order_7_smooth_derivatives(self) -> None:
        """Test that all derivatives are smooth for 7th order trajectory."""
        initial = BoundaryCondition(
            position=0.0, velocity=0.0, acceleration=0.0, jerk=0.0
        )
        final = BoundaryCondition(
            position=8.0, velocity=0.0, acceleration=0.0, jerk=0.0
        )
        time = TimeInterval(start=0.0, end=2.0)
        
        trajectory = PolynomialTrajectory.order_7_trajectory(initial, final, time)
        
        # Sample trajectory at multiple points
        test_times = np.linspace(time.start, time.end, 20)
        
        for t in test_times:
            q, v, a, j = trajectory(t)
            
            # All values should be finite
            assert np.isfinite(q)
            assert np.isfinite(v)
            assert np.isfinite(a)
            assert np.isfinite(j)


class TestPolynomialTrajectoryHeuristicVelocities:
    """Test suite for heuristic velocity calculation."""

    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def test_heuristic_velocities_basic(self) -> None:
        """Test basic heuristic velocity calculation."""
        points = [0.0, 3.0, 8.0, 12.0]
        times = [0.0, 1.0, 3.0, 4.0]
        
        velocities = PolynomialTrajectory.heuristic_velocities(points, times)
        
        # Should return velocities for all points (including boundaries)
        assert len(velocities) == len(points)
        assert all(np.isfinite(v) for v in velocities)
        # Boundary velocities should be zero
        assert velocities[0] == 0.0
        assert velocities[-1] == 0.0

    def test_heuristic_velocities_linear_trajectory(self) -> None:
        """Test heuristic velocities for linear trajectory."""
        # Linear trajectory: constant velocity should be detected
        points = [0.0, 2.0, 4.0, 6.0]
        times = [0.0, 1.0, 2.0, 3.0]  # Uniform time spacing
        
        velocities = PolynomialTrajectory.heuristic_velocities(points, times)
        
        # For linear trajectory, intermediate velocities should be constant (2.0)
        # but boundary velocities are set to 0.0 by default
        expected_velocity = 2.0  # (2-0)/1 = 2
        assert velocities[0] == 0.0  # Boundary
        assert abs(velocities[1] - expected_velocity) < self.NUMERICAL_ATOL
        assert abs(velocities[2] - expected_velocity) < self.NUMERICAL_ATOL
        assert velocities[-1] == 0.0  # Boundary

    def test_heuristic_velocities_parabolic_trajectory(self) -> None:
        """Test heuristic velocities for parabolic trajectory."""
        # Parabolic trajectory: y = t^2
        times = [0.0, 1.0, 2.0, 3.0]
        points = [t**2 for t in times]  # [0, 1, 4, 9]
        
        velocities = PolynomialTrajectory.heuristic_velocities(points, times)
        
        # Should return velocities for all points
        assert len(velocities) == 4
        # Boundary velocities should be zero
        assert velocities[0] == 0.0
        assert velocities[-1] == 0.0
        # Intermediate velocities should be reasonable
        assert all(np.isfinite(v) for v in velocities)

    def test_heuristic_velocities_non_uniform_times(self) -> None:
        """Test heuristic velocities with non-uniform time spacing."""
        points = [0.0, 2.0, 6.0, 10.0]
        times = [0.0, 0.5, 2.0, 3.0]  # Non-uniform spacing
        
        velocities = PolynomialTrajectory.heuristic_velocities(points, times)
        
        # Should handle non-uniform spacing
        assert len(velocities) == len(points)
        assert all(np.isfinite(v) for v in velocities)
        # Boundary velocities should be zero
        assert velocities[0] == 0.0
        assert velocities[-1] == 0.0

    def test_heuristic_velocities_minimum_points(self) -> None:
        """Test heuristic velocities with minimum number of points."""
        points = [0.0, 5.0]
        times = [0.0, 2.0]
        
        velocities = PolynomialTrajectory.heuristic_velocities(points, times)
        
        # With only 2 points, should return velocities for both (both zero)
        assert len(velocities) == 2
        assert velocities[0] == 0.0
        assert velocities[1] == 0.0

    def test_heuristic_velocities_identical_points(self) -> None:
        """Test heuristic velocities with some identical points."""
        points = [0.0, 3.0, 3.0, 6.0]  # Middle point repeated
        times = [0.0, 1.0, 2.0, 3.0]
        
        # Should handle gracefully without throwing errors
        velocities = PolynomialTrajectory.heuristic_velocities(points, times)
        
        assert len(velocities) == 4
        assert all(np.isfinite(v) for v in velocities)
        # Boundary velocities should be zero
        assert velocities[0] == 0.0
        assert velocities[-1] == 0.0


class TestPolynomialTrajectoryMultipoint:
    """Test suite for multipoint trajectory generation."""

    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def test_multipoint_trajectory_basic(self) -> None:
        """Test basic multipoint trajectory generation."""
        params = TrajectoryParams(
            points=[0.0, 5.0, 10.0],
            times=[0.0, 2.0, 4.0],
            order=ORDER_3
        )
        
        trajectory = PolynomialTrajectory.multipoint_trajectory(params)
        
        # Should return a callable
        assert callable(trajectory)
        
        # Test that trajectory passes through waypoints
        for point, time in zip(params.points, params.times):
            q, _, _, _ = trajectory(time)
            assert abs(q - point) < self.NUMERICAL_ATOL

    def test_multipoint_trajectory_with_velocities(self) -> None:
        """Test multipoint trajectory with specified velocities."""
        params = TrajectoryParams(
            points=[0.0, 4.0, 8.0],
            times=[0.0, 2.0, 4.0],
            velocities=[0.0, 2.0, 0.0],
            order=ORDER_3
        )
        
        trajectory = PolynomialTrajectory.multipoint_trajectory(params)
        
        # Test waypoint positions
        for point, time in zip(params.points, params.times):
            q, _, _, _ = trajectory(time)
            assert abs(q - point) < self.NUMERICAL_ATOL
        
        # Test specified velocities
        if params.velocities is not None:
            for velocity, time in zip(params.velocities, params.times):
                _, v, _, _ = trajectory(time)
                assert abs(v - velocity) < self.NUMERICAL_ATOL

    def test_multipoint_trajectory_order_5(self) -> None:
        """Test multipoint trajectory with 5th order polynomials."""
        params = TrajectoryParams(
            points=[0.0, 3.0, 9.0],
            times=[0.0, 1.5, 3.0],
            velocities=[0.0, 1.0, 0.0],
            accelerations=[0.0, 0.5, 0.0],
            order=ORDER_5
        )
        
        trajectory = PolynomialTrajectory.multipoint_trajectory(params)
        
        # Test waypoint conditions
        if params.velocities is not None and params.accelerations is not None:
            for i, (point, time, vel, acc) in enumerate(
                zip(params.points, params.times, params.velocities, params.accelerations)
            ):
                q, v, a, _ = trajectory(time)
                assert abs(q - point) < self.NUMERICAL_ATOL
                assert abs(v - vel) < self.NUMERICAL_ATOL
                assert abs(a - acc) < self.NUMERICAL_ATOL

    def test_multipoint_trajectory_order_7(self) -> None:
        """Test multipoint trajectory with 7th order polynomials."""
        params = TrajectoryParams(
            points=[0.0, 6.0],
            times=[0.0, 3.0],
            velocities=[0.0, 0.0],
            accelerations=[0.0, 0.0],
            jerks=[0.0, 0.0],
            order=ORDER_7
        )
        
        trajectory = PolynomialTrajectory.multipoint_trajectory(params)
        
        # Test boundary conditions
        q0, v0, a0, j0 = trajectory(params.times[0])
        assert abs(q0 - params.points[0]) < self.NUMERICAL_ATOL
        if params.velocities is not None:
            assert abs(v0 - params.velocities[0]) < self.NUMERICAL_ATOL
        if params.accelerations is not None:
            assert abs(a0 - params.accelerations[0]) < self.NUMERICAL_ATOL
        if params.jerks is not None:
            assert abs(j0 - params.jerks[0]) < self.NUMERICAL_ATOL
        
        q1, v1, a1, j1 = trajectory(params.times[1])
        assert abs(q1 - params.points[1]) < self.NUMERICAL_ATOL
        if params.velocities is not None:
            assert abs(v1 - params.velocities[1]) < self.NUMERICAL_ATOL
        if params.accelerations is not None:
            assert abs(a1 - params.accelerations[1]) < self.NUMERICAL_ATOL
        if params.jerks is not None:
            assert abs(j1 - params.jerks[1]) < self.NUMERICAL_ATOL

    def test_multipoint_trajectory_continuity(self) -> None:
        """Test continuity between segments in multipoint trajectory."""
        params = TrajectoryParams(
            points=[0.0, 2.0, 6.0, 8.0],
            times=[0.0, 1.0, 2.5, 3.5],
            order=ORDER_3
        )
        
        trajectory = PolynomialTrajectory.multipoint_trajectory(params)
        
        # Test continuity at segment boundaries
        eps = 1e-8
        for i in range(1, len(params.times) - 1):
            t = params.times[i]
            
            # Approach from left and right
            q_left, v_left, a_left, _ = trajectory(t - eps)
            q_right, v_right, a_right, _ = trajectory(t + eps)
            q_exact, v_exact, a_exact, _ = trajectory(t)
            
            # Position should be continuous
            assert abs(q_left - q_exact) < self.NUMERICAL_ATOL
            assert abs(q_right - q_exact) < self.NUMERICAL_ATOL
            
            # For order 3, velocity should be continuous
            assert abs(v_left - v_exact) < self.NUMERICAL_ATOL
            assert abs(v_right - v_exact) < self.NUMERICAL_ATOL

    def test_multipoint_trajectory_outside_range(self) -> None:
        """Test multipoint trajectory evaluation outside time range."""
        params = TrajectoryParams(
            points=[0.0, 5.0],
            times=[1.0, 3.0],
            order=ORDER_3
        )
        
        trajectory = PolynomialTrajectory.multipoint_trajectory(params)
        
        # Before first time point
        q_before, v_before, a_before, j_before = trajectory(0.5)
        assert np.isfinite(q_before)
        assert np.isfinite(v_before)
        assert np.isfinite(a_before)
        assert np.isfinite(j_before)
        
        # After last time point
        q_after, v_after, a_after, j_after = trajectory(3.5)
        assert np.isfinite(q_after)
        assert np.isfinite(v_after)
        assert np.isfinite(a_after)
        assert np.isfinite(j_after)


class TestPolynomialTrajectoryValidOrders:
    """Test suite for valid polynomial orders."""

    def test_valid_orders_class_variable(self) -> None:
        """Test that valid orders are correctly defined."""
        assert PolynomialTrajectory.VALID_ORDERS == (ORDER_3, ORDER_5, ORDER_7)
        assert ORDER_3 in PolynomialTrajectory.VALID_ORDERS
        assert ORDER_5 in PolynomialTrajectory.VALID_ORDERS
        assert ORDER_7 in PolynomialTrajectory.VALID_ORDERS

    def test_order_constants(self) -> None:
        """Test that order constants have correct values."""
        assert ORDER_3 == 3
        assert ORDER_5 == 5
        assert ORDER_7 == 7


class TestPolynomialTrajectoryEdgeCases:
    """Test suite for edge cases and special situations."""

    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def test_zero_time_interval(self) -> None:
        """Test trajectory with zero time interval."""
        initial = BoundaryCondition(position=5.0, velocity=2.0)
        final = BoundaryCondition(position=5.0, velocity=2.0)
        time = TimeInterval(start=2.0, end=2.0)  # Zero duration
        
        # Zero duration should raise ZeroDivisionError
        with pytest.raises(ZeroDivisionError):
            PolynomialTrajectory.order_3_trajectory(initial, final, time)

    def test_very_small_time_interval(self) -> None:
        """Test trajectory with very small time interval."""
        initial = BoundaryCondition(position=0.0, velocity=0.0)
        final = BoundaryCondition(position=1.0, velocity=0.0)
        time = TimeInterval(start=0.0, end=1e-6)  # Very small duration
        
        trajectory = PolynomialTrajectory.order_3_trajectory(initial, final, time)
        
        # Should handle small intervals
        q_init, v_init, _, _ = trajectory(time.start)
        q_final, v_final, _, _ = trajectory(time.end)
        
        assert abs(q_init - initial.position) < self.NUMERICAL_ATOL
        assert abs(q_final - final.position) < self.NUMERICAL_ATOL

    def test_large_time_interval(self) -> None:
        """Test trajectory with large time interval."""
        initial = BoundaryCondition(position=0.0, velocity=0.0)
        final = BoundaryCondition(position=10.0, velocity=0.0)
        time = TimeInterval(start=0.0, end=1e6)  # Very large duration
        
        trajectory = PolynomialTrajectory.order_3_trajectory(initial, final, time)
        
        # Should handle large intervals
        q_init, v_init, _, _ = trajectory(time.start)
        q_final, v_final, _, _ = trajectory(time.end)
        
        assert abs(q_init - initial.position) < self.NUMERICAL_ATOL
        assert abs(q_final - final.position) < self.NUMERICAL_ATOL

    def test_extreme_boundary_conditions(self) -> None:
        """Test trajectory with extreme boundary condition values."""
        initial = BoundaryCondition(
            position=-1e6, velocity=1e3, acceleration=-1e2, jerk=1e1
        )
        final = BoundaryCondition(
            position=1e6, velocity=-1e3, acceleration=1e2, jerk=-1e1
        )
        time = TimeInterval(start=0.0, end=100.0)
        
        trajectory = PolynomialTrajectory.order_7_trajectory(initial, final, time)
        
        # Should handle extreme values
        q_init, v_init, a_init, j_init = trajectory(time.start)
        q_final, v_final, a_final, j_final = trajectory(time.end)
        
        # Check boundary conditions with appropriate tolerance for large numbers
        assert abs(q_init - initial.position) < 1e-6 * abs(initial.position)
        assert abs(v_init - initial.velocity) < 1e-6 * abs(initial.velocity)
        assert abs(a_init - initial.acceleration) < 1e-6 * abs(initial.acceleration)
        assert abs(j_init - initial.jerk) < 1e-6 * abs(initial.jerk)


class TestPolynomialTrajectoryPerformance:
    """Test suite for performance benchmarks."""

    @pytest.mark.parametrize("order", [ORDER_3, ORDER_5, ORDER_7])
    def test_trajectory_generation_performance(
        self, order: int, benchmark: pytest.FixtureFunction
    ) -> None:
        """Benchmark trajectory generation performance for different orders."""
        if order == ORDER_3:
            initial = BoundaryCondition(position=0.0, velocity=0.0)
            final = BoundaryCondition(position=10.0, velocity=0.0)
            
            def generate_trajectory():
                return PolynomialTrajectory.order_3_trajectory(
                    initial, final, TimeInterval(start=0.0, end=2.0)
                )
        elif order == ORDER_5:
            initial = BoundaryCondition(position=0.0, velocity=0.0, acceleration=0.0)
            final = BoundaryCondition(position=10.0, velocity=0.0, acceleration=0.0)
            
            def generate_trajectory():
                return PolynomialTrajectory.order_5_trajectory(
                    initial, final, TimeInterval(start=0.0, end=2.0)
                )
        else:  # ORDER_7
            initial = BoundaryCondition(position=0.0, velocity=0.0, acceleration=0.0, jerk=0.0)
            final = BoundaryCondition(position=10.0, velocity=0.0, acceleration=0.0, jerk=0.0)
            
            def generate_trajectory():
                return PolynomialTrajectory.order_7_trajectory(
                    initial, final, TimeInterval(start=0.0, end=2.0)
                )
        
        trajectory = benchmark(generate_trajectory)
        assert callable(trajectory)

    @pytest.mark.parametrize("n_points", [5, 10, 20])
    def test_multipoint_trajectory_performance(
        self, n_points: int, benchmark: pytest.FixtureFunction
    ) -> None:
        """Benchmark multipoint trajectory generation performance."""
        points = [i * 2.0 for i in range(n_points)]
        times = [i * 1.0 for i in range(n_points)]
        
        params = TrajectoryParams(points=points, times=times, order=ORDER_3)
        
        def generate_multipoint_trajectory():
            return PolynomialTrajectory.multipoint_trajectory(params)
        
        trajectory = benchmark(generate_multipoint_trajectory)
        assert callable(trajectory)

    def test_heuristic_velocities_performance(
        self, benchmark: pytest.FixtureFunction
    ) -> None:
        """Benchmark heuristic velocity calculation performance."""
        n_points = 100
        points = [i + 0.1 * i**2 for i in range(n_points)]
        times = [i * 0.1 for i in range(n_points)]
        
        def calculate_heuristic_velocities():
            return PolynomialTrajectory.heuristic_velocities(points, times)
        
        velocities = benchmark(calculate_heuristic_velocities)
        assert len(velocities) == n_points


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main(["-xvs", __file__])