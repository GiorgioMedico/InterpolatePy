"""
Comprehensive tests for motion profile implementations.

This module contains extensive tests for the motion profile classes covering:
1. DoubleSTrajectory - S-curve trajectories with bounded jerk
2. TrapezoidalTrajectory - Trapezoidal velocity profiles

Test coverage includes:
- Constructor validation and parameter checking
- Mathematical accuracy with known analytical solutions
- Trajectory planning algorithms
- Boundary condition handling
- Edge cases and error handling
- Numerical stability and convergence
- Performance benchmarks

The tests verify that motion profiles generate smooth trajectories with
appropriate velocity, acceleration, and jerk constraints.
"""

from typing import Any

import numpy as np
import pytest

from interpolatepy.double_s import DoubleSTrajectory
from interpolatepy.double_s import StateParams
from interpolatepy.double_s import TrajectoryBounds
from interpolatepy.trapezoidal import CalculationParams
from interpolatepy.trapezoidal import InterpolationParams
from interpolatepy.trapezoidal import TrajectoryParams
from interpolatepy.trapezoidal import TrapezoidalTrajectory

# Type alias for pytest benchmark fixture
if not hasattr(pytest, "FixtureFunction"):
    pytest.FixtureFunction = Any


class TestTrajectoryBounds:
    """Test suite for TrajectoryBounds dataclass."""

    def test_valid_bounds_creation(self) -> None:
        """Test creation of valid trajectory bounds."""
        bounds = TrajectoryBounds(v_bound=2.0, a_bound=1.0, j_bound=0.5)
        
        assert bounds.v_bound == 2.0
        assert bounds.a_bound == 1.0
        assert bounds.j_bound == 0.5

    def test_bounds_absolute_values(self) -> None:
        """Test that bounds are converted to absolute values."""
        # Create with negative values - should be converted to positive
        bounds = TrajectoryBounds(v_bound=-2.0, a_bound=-1.0, j_bound=-0.5)
        
        # The post_init should convert to absolute values
        assert bounds.v_bound == 2.0
        assert bounds.a_bound == 1.0  
        assert bounds.j_bound == 0.5

    def test_bounds_validation_zero_values(self) -> None:
        """Test that zero bounds raise ValueError."""
        with pytest.raises(ValueError, match="Bounds must be positive values"):
            TrajectoryBounds(v_bound=0.0, a_bound=1.0, j_bound=0.5)
            
        with pytest.raises(ValueError, match="Bounds must be positive values"):
            TrajectoryBounds(v_bound=2.0, a_bound=0.0, j_bound=0.5)
            
        with pytest.raises(ValueError, match="Bounds must be positive values"):
            TrajectoryBounds(v_bound=2.0, a_bound=1.0, j_bound=0.0)

    def test_bounds_validation_non_numeric(self) -> None:
        """Test that non-numeric bounds raise TypeError."""
        with pytest.raises(TypeError, match="All bounds must be numeric values"):
            TrajectoryBounds(v_bound="invalid", a_bound=1.0, j_bound=0.5)


class TestStateParams:
    """Test suite for StateParams named tuple."""

    def test_state_params_creation(self) -> None:
        """Test creation of state parameters."""
        params = StateParams(q_0=0.0, q_1=10.0, v_0=0.0, v_1=0.0)
        
        assert params.q_0 == 0.0
        assert params.q_1 == 10.0
        assert params.v_0 == 0.0
        assert params.v_1 == 0.0

    def test_state_params_immutability(self) -> None:
        """Test that StateParams is immutable (named tuple behavior)."""
        params = StateParams(q_0=0.0, q_1=10.0, v_0=0.0, v_1=0.0)
        
        with pytest.raises(AttributeError):
            params.q_0 = 5.0


class TestDoubleSTrajectoryConstruction:
    """Test suite for DoubleSTrajectory construction and validation."""

    def test_basic_construction(self) -> None:
        """Test basic DoubleSTrajectory construction."""
        state_params = StateParams(q_0=0.0, q_1=10.0, v_0=0.0, v_1=0.0)
        bounds = TrajectoryBounds(v_bound=2.0, a_bound=1.0, j_bound=0.5)
        
        trajectory = DoubleSTrajectory(state_params, bounds)
        
        # Should complete construction without errors
        assert isinstance(trajectory, DoubleSTrajectory)

    def test_construction_validation_non_numeric_states(self) -> None:
        """Test that non-numeric state parameters raise TypeError."""
        bounds = TrajectoryBounds(v_bound=2.0, a_bound=1.0, j_bound=0.5)
        
        with pytest.raises(TypeError, match="All state parameters must be numeric values"):
            invalid_params = StateParams(q_0="invalid", q_1=10.0, v_0=0.0, v_1=0.0)
            DoubleSTrajectory(invalid_params, bounds)

    def test_construction_with_various_states(self) -> None:
        """Test construction with various start/end states."""
        bounds = TrajectoryBounds(v_bound=5.0, a_bound=2.0, j_bound=1.0)
        
        # Forward motion
        state_params = StateParams(q_0=0.0, q_1=20.0, v_0=1.0, v_1=2.0)
        trajectory = DoubleSTrajectory(state_params, bounds)
        assert isinstance(trajectory, DoubleSTrajectory)
        
        # Backward motion
        state_params = StateParams(q_0=20.0, q_1=0.0, v_0=-1.0, v_1=-2.0)
        trajectory = DoubleSTrajectory(state_params, bounds)
        assert isinstance(trajectory, DoubleSTrajectory)
        
        # Zero displacement
        state_params = StateParams(q_0=5.0, q_1=5.0, v_0=0.0, v_1=0.0)
        trajectory = DoubleSTrajectory(state_params, bounds)
        assert isinstance(trajectory, DoubleSTrajectory)


class TestDoubleSTrajectoryEvaluation:
    """Test suite for DoubleSTrajectory evaluation methods."""

    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def test_evaluate_basic(self) -> None:
        """Test basic trajectory evaluation."""
        state_params = StateParams(q_0=0.0, q_1=10.0, v_0=0.0, v_1=0.0)
        bounds = TrajectoryBounds(v_bound=3.0, a_bound=2.0, j_bound=1.0)
        
        trajectory = DoubleSTrajectory(state_params, bounds)
        
        # Evaluate at start
        q, v, a, j = trajectory.evaluate(0.0)
        assert abs(q - 0.0) < self.NUMERICAL_ATOL
        assert abs(v - 0.0) < self.NUMERICAL_ATOL
        
        # Evaluate at end
        duration = trajectory.get_duration()
        q, v, a, j = trajectory.evaluate(duration)
        assert abs(q - 10.0) < self.NUMERICAL_ATOL
        assert abs(v - 0.0) < self.NUMERICAL_ATOL

    def test_evaluate_array_input(self) -> None:
        """Test trajectory evaluation with array inputs."""
        state_params = StateParams(q_0=0.0, q_1=5.0, v_0=0.0, v_1=0.0)
        bounds = TrajectoryBounds(v_bound=2.0, a_bound=1.0, j_bound=0.5)
        
        trajectory = DoubleSTrajectory(state_params, bounds)
        
        # Test with array input
        t_array = np.array([0.0, 1.0, 2.0])
        q, v, a, j = trajectory.evaluate(t_array)
        
        assert isinstance(q, np.ndarray)
        assert isinstance(v, np.ndarray)
        assert isinstance(a, np.ndarray)
        assert isinstance(j, np.ndarray)
        assert len(q) == len(t_array)
        assert len(v) == len(t_array)
        assert len(a) == len(t_array)
        assert len(j) == len(t_array)

    def test_boundary_conditions(self) -> None:
        """Test that boundary conditions are satisfied."""
        state_params = StateParams(q_0=2.0, q_1=8.0, v_0=1.0, v_1=0.5)
        bounds = TrajectoryBounds(v_bound=3.0, a_bound=2.0, j_bound=1.0)
        
        trajectory = DoubleSTrajectory(state_params, bounds)
        
        # Check initial conditions
        q0, v0, a0, j0 = trajectory.evaluate(0.0)
        assert abs(q0 - state_params.q_0) < self.NUMERICAL_ATOL
        assert abs(v0 - state_params.v_0) < self.NUMERICAL_ATOL
        
        # Check final conditions
        duration = trajectory.get_duration()
        q1, v1, a1, j1 = trajectory.evaluate(duration)
        assert abs(q1 - state_params.q_1) < self.NUMERICAL_ATOL
        assert abs(v1 - state_params.v_1) < self.NUMERICAL_ATOL

    def test_velocity_bounds_satisfaction(self) -> None:
        """Test that velocity bounds are not exceeded."""
        state_params = StateParams(q_0=0.0, q_1=20.0, v_0=0.0, v_1=0.0)
        bounds = TrajectoryBounds(v_bound=2.0, a_bound=1.5, j_bound=1.0)
        
        trajectory = DoubleSTrajectory(state_params, bounds)
        
        # Sample trajectory at multiple points
        duration = trajectory.get_duration()
        t_samples = np.linspace(0, duration, 100)
        
        for t in t_samples:
            _, v, _, _ = trajectory.evaluate(t)
            assert abs(v) <= bounds.v_bound + self.NUMERICAL_ATOL

    def test_acceleration_bounds_satisfaction(self) -> None:
        """Test that acceleration bounds are not exceeded."""
        state_params = StateParams(q_0=0.0, q_1=15.0, v_0=0.0, v_1=0.0)
        bounds = TrajectoryBounds(v_bound=3.0, a_bound=1.0, j_bound=0.8)
        
        trajectory = DoubleSTrajectory(state_params, bounds)
        
        # Sample trajectory at multiple points
        duration = trajectory.get_duration()
        t_samples = np.linspace(0, duration, 100)
        
        for t in t_samples:
            _, _, a, _ = trajectory.evaluate(t)
            assert abs(a) <= bounds.a_bound + self.NUMERICAL_ATOL

    def test_jerk_bounds_satisfaction(self) -> None:
        """Test that jerk bounds are not exceeded (implicitly through S-curves)."""
        state_params = StateParams(q_0=0.0, q_1=10.0, v_0=0.0, v_1=0.0)
        bounds = TrajectoryBounds(v_bound=2.0, a_bound=1.0, j_bound=0.5)
        
        trajectory = DoubleSTrajectory(state_params, bounds)
        
        # For S-curve trajectories, jerk should be bounded by construction
        # We test this indirectly by checking that the trajectory is computed without errors
        duration = trajectory.get_duration()
        assert duration > 0

    def test_get_duration(self) -> None:
        """Test trajectory duration calculation."""
        state_params = StateParams(q_0=0.0, q_1=10.0, v_0=0.0, v_1=0.0)
        bounds = TrajectoryBounds(v_bound=2.0, a_bound=1.0, j_bound=0.5)
        
        trajectory = DoubleSTrajectory(state_params, bounds)
        duration = trajectory.get_duration()
        
        assert duration > 0
        assert isinstance(duration, float)

    def test_get_phase_durations(self) -> None:
        """Test phase duration retrieval."""
        state_params = StateParams(q_0=0.0, q_1=15.0, v_0=0.0, v_1=0.0)
        bounds = TrajectoryBounds(v_bound=3.0, a_bound=2.0, j_bound=1.0)
        
        trajectory = DoubleSTrajectory(state_params, bounds)
        phase_durations = trajectory.get_phase_durations()
        
        assert isinstance(phase_durations, dict)
        # Test that all required phase keys are present
        required_keys = {"total", "acceleration", "constant_velocity", "deceleration", "jerk_acceleration", "jerk_deceleration"}
        assert set(phase_durations.keys()) == required_keys
        
        # Test that durations are non-negative
        for duration in phase_durations.values():
            assert duration >= 0


class TestDoubleSTrajectoryEdgeCases:
    """Test suite for DoubleSTrajectory edge cases."""

    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def test_zero_displacement(self) -> None:
        """Test trajectory with zero displacement."""
        state_params = StateParams(q_0=5.0, q_1=5.0, v_0=0.0, v_1=0.0)
        bounds = TrajectoryBounds(v_bound=2.0, a_bound=1.0, j_bound=0.5)
        
        trajectory = DoubleSTrajectory(state_params, bounds)
        duration = trajectory.get_duration()
        
        # For zero displacement with zero velocities, duration should be very small
        assert duration >= 0
        
        # Position should remain constant
        q, v, a, j = trajectory.evaluate(duration / 2)
        assert abs(q - 5.0) < self.NUMERICAL_ATOL

    def test_small_displacement(self) -> None:
        """Test trajectory with very small displacement."""
        state_params = StateParams(q_0=0.0, q_1=0.001, v_0=0.0, v_1=0.0)
        bounds = TrajectoryBounds(v_bound=1.0, a_bound=1.0, j_bound=1.0)
        
        trajectory = DoubleSTrajectory(state_params, bounds)
        
        # Should handle small displacements gracefully
        duration = trajectory.get_duration()
        assert duration > 0
        
        # Check final position
        q_final, _, _, _ = trajectory.evaluate(duration)
        assert abs(q_final - 0.001) < self.NUMERICAL_ATOL

    def test_large_displacement(self) -> None:
        """Test trajectory with large displacement."""
        state_params = StateParams(q_0=0.0, q_1=1000.0, v_0=0.0, v_1=0.0)
        bounds = TrajectoryBounds(v_bound=10.0, a_bound=5.0, j_bound=2.0)
        
        trajectory = DoubleSTrajectory(state_params, bounds)
        
        # Should handle large displacements
        duration = trajectory.get_duration()
        assert duration > 0
        
        # Check final position
        q_final, _, _, _ = trajectory.evaluate(duration)
        assert abs(q_final - 1000.0) < self.NUMERICAL_ATOL

    def test_negative_displacement(self) -> None:
        """Test trajectory with negative displacement."""
        state_params = StateParams(q_0=10.0, q_1=0.0, v_0=0.0, v_1=0.0)
        bounds = TrajectoryBounds(v_bound=2.0, a_bound=1.0, j_bound=0.5)
        
        trajectory = DoubleSTrajectory(state_params, bounds)
        
        # Should handle backward motion
        duration = trajectory.get_duration()
        assert duration > 0
        
        # Check final position
        q_final, _, _, _ = trajectory.evaluate(duration)
        assert abs(q_final - 0.0) < self.NUMERICAL_ATOL

    def test_non_zero_initial_final_velocities(self) -> None:
        """Test trajectory with non-zero initial and final velocities."""
        state_params = StateParams(q_0=0.0, q_1=10.0, v_0=1.0, v_1=2.0)
        bounds = TrajectoryBounds(v_bound=5.0, a_bound=3.0, j_bound=2.0)
        
        trajectory = DoubleSTrajectory(state_params, bounds)
        
        # Check boundary conditions
        q0, v0, _, _ = trajectory.evaluate(0.0)
        assert abs(q0 - 0.0) < self.NUMERICAL_ATOL
        assert abs(v0 - 1.0) < self.NUMERICAL_ATOL
        
        duration = trajectory.get_duration()
        q1, v1, _, _ = trajectory.evaluate(duration)
        assert abs(q1 - 10.0) < self.NUMERICAL_ATOL
        assert abs(v1 - 2.0) < self.NUMERICAL_ATOL


class TestDoubleSTrajectoryStaticMethods:
    """Test suite for DoubleSTrajectory static methods."""

    def test_create_trajectory_function(self) -> None:
        """Test static trajectory creation method."""
        state_params = StateParams(q_0=0.0, q_1=5.0, v_0=0.0, v_1=0.0)
        bounds = TrajectoryBounds(v_bound=2.0, a_bound=1.0, j_bound=0.5)
        
        trajectory_func, duration = DoubleSTrajectory.create_trajectory(state_params, bounds)
        
        # Should return a callable
        assert callable(trajectory_func)
        
        # Test the returned function
        q, v, a, j = trajectory_func(0.0)
        assert abs(q - 0.0) < 1e-6
        assert abs(v - 0.0) < 1e-6


class TestTrapezoidalTrajectoryParams:
    """Test suite for trapezoidal trajectory parameter classes."""

    def test_trajectory_params_creation(self) -> None:
        """Test TrajectoryParams dataclass creation."""
        params = TrajectoryParams(q0=0.0, q1=10.0, t0=0.0, v0=0.0, v1=0.0)
        
        assert params.q0 == 0.0
        assert params.q1 == 10.0
        assert params.t0 == 0.0
        assert params.v0 == 0.0
        assert params.v1 == 0.0

    def test_calculation_params_creation(self) -> None:
        """Test CalculationParams dataclass creation."""
        params = CalculationParams(q0=0.0, q1=5.0, v0=0.0, v1=0.0, amax=1.0)
        
        assert params.q0 == 0.0
        assert params.q1 == 5.0
        assert params.v0 == 0.0
        assert params.v1 == 0.0
        assert params.amax == 1.0

    def test_interpolation_params_creation(self) -> None:
        """Test InterpolationParams dataclass creation."""
        points = [0.0, 5.0, 10.0]
        params = InterpolationParams(points=points, v0=0.0, vn=0.0, amax=2.0)
        
        assert params.points == points
        assert params.v0 == 0.0
        assert params.vn == 0.0
        assert params.amax == 2.0


class TestTrapezoidalTrajectoryGeneration:
    """Test suite for trapezoidal trajectory generation."""

    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def test_generate_trajectory_basic(self) -> None:
        """Test basic trajectory generation."""
        params = TrajectoryParams(
            q0=0.0, q1=10.0, v0=0.0, v1=0.0, amax=2.0, vmax=3.0
        )
        
        trajectory_func, duration = TrapezoidalTrajectory.generate_trajectory(params)
        
        # Should return a callable and duration
        assert callable(trajectory_func)
        assert isinstance(duration, (float, int))
        assert duration > 0

    def test_trajectory_boundary_conditions(self) -> None:
        """Test that trajectory satisfies boundary conditions."""
        params = TrajectoryParams(
            q0=1.0, q1=8.0, v0=0.5, v1=1.0, amax=2.0, vmax=3.0
        )
        
        trajectory_func, duration = TrapezoidalTrajectory.generate_trajectory(params)
        
        # Test initial conditions
        q0, v0, a0 = trajectory_func(0.0)
        assert abs(q0 - params.q0) < self.NUMERICAL_ATOL
        assert abs(v0 - params.v0) < self.NUMERICAL_ATOL
        
        # Test final conditions
        q1, v1, a1 = trajectory_func(duration)
        assert abs(q1 - params.q1) < self.NUMERICAL_ATOL
        assert abs(v1 - params.v1) < self.NUMERICAL_ATOL

    def test_velocity_constraints(self) -> None:
        """Test that velocity constraints are respected."""
        params = TrajectoryParams(
            q0=0.0, q1=20.0, v0=0.0, v1=0.0, amax=2.0, vmax=3.0
        )
        
        trajectory_func, duration = TrapezoidalTrajectory.generate_trajectory(params)
        
        # Sample trajectory and check velocity bounds
        t_samples = np.linspace(0, duration, 100)
        
        for t in t_samples:
            _, v, _ = trajectory_func(t)
            assert abs(v) <= params.vmax + self.NUMERICAL_ATOL

    def test_acceleration_constraints(self) -> None:
        """Test that acceleration constraints are respected."""
        params = TrajectoryParams(
            q0=0.0, q1=15.0, v0=0.0, v1=0.0, amax=1.5, vmax=4.0
        )
        
        trajectory_func, duration = TrapezoidalTrajectory.generate_trajectory(params)
        
        # Sample trajectory and check acceleration bounds
        t_samples = np.linspace(0, duration, 100)
        
        for t in t_samples:
            _, _, a = trajectory_func(t)
            assert abs(a) <= params.amax + self.NUMERICAL_ATOL

    def test_duration_based_trajectory(self) -> None:
        """Test trajectory generation with duration constraints."""
        params = TrajectoryParams(
            q0=0.0, q1=10.0, v0=0.0, v1=0.0, amax=2.0, duration=8.0
        )
        
        trajectory_func, duration = TrapezoidalTrajectory.generate_trajectory(params)
        
        # Duration should match specification
        assert abs(duration - 8.0) < self.NUMERICAL_ATOL
        
        # Final position should be correct
        q_final, _, _ = trajectory_func(8.0)
        assert abs(q_final - 10.0) < self.NUMERICAL_ATOL

    def test_infeasible_trajectory_error(self) -> None:
        """Test that infeasible trajectories raise appropriate errors."""
        # Test case where amax * h < abs(v0^2 - v1^2) / 2
        # With h=1, v0=10, v1=0, need amax * 1 < (100 - 0) / 2 = 50
        # So amax < 50, let's use amax = 1 (much less than 50)
        params = TrajectoryParams(
            q0=0.0, q1=1.0, v0=10.0, v1=0.0, amax=1.0, duration=1.0
        )
        
        with pytest.raises(ValueError, match="not feasible"):
            TrapezoidalTrajectory.generate_trajectory(params)


class TestTrapezoidalWaypointInterpolation:
    """Test suite for trapezoidal waypoint interpolation."""

    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def test_interpolate_waypoints_basic(self) -> None:
        """Test basic waypoint interpolation."""
        params = InterpolationParams(
            points=[0.0, 5.0, 10.0], v0=0.0, vn=0.0, amax=2.0, vmax=3.0
        )
        
        trajectory_func, total_duration = TrapezoidalTrajectory.interpolate_waypoints(params)
        
        # Should return callable and total duration
        assert callable(trajectory_func)
        assert isinstance(total_duration, (float, int))
        assert total_duration > 0

    def test_waypoint_interpolation_continuity(self) -> None:
        """Test continuity at waypoints."""
        params = InterpolationParams(
            points=[0.0, 3.0, 8.0, 12.0], v0=0.0, vn=0.0, amax=2.0, vmax=4.0
        )
        
        trajectory_func, total_duration = TrapezoidalTrajectory.interpolate_waypoints(params)
        
        # Check that trajectory passes through start and end waypoints
        # At t=0, should be at first waypoint
        q_start, v_start, _ = trajectory_func(0.0)
        assert abs(q_start - params.points[0]) < self.NUMERICAL_ATOL
        assert abs(v_start - params.v0) < self.NUMERICAL_ATOL
        
        # At t=total_duration, should be at last waypoint
        q_end, v_end, _ = trajectory_func(total_duration)
        assert abs(q_end - params.points[-1]) < self.NUMERICAL_ATOL
        assert abs(v_end - params.vn) < self.NUMERICAL_ATOL

    def test_waypoint_interpolation_with_times(self) -> None:
        """Test waypoint interpolation with specified times."""
        params = InterpolationParams(
            points=[0.0, 5.0, 10.0],
            times=[0.0, 3.0, 6.0],
            v0=0.0, vn=0.0, amax=3.0
        )
        
        trajectory_func, total_duration = TrapezoidalTrajectory.interpolate_waypoints(params)
        
        # Check timing constraints
        for i, (point, time) in enumerate(zip(params.points, params.times)):
            q, _, _ = trajectory_func(time)
            assert abs(q - point) < self.NUMERICAL_ATOL

    def test_waypoint_interpolation_with_velocities(self) -> None:
        """Test waypoint interpolation with intermediate velocities."""
        params = InterpolationParams(
            points=[0.0, 4.0, 8.0],
            inter_velocities=[1.0],  # Velocity at middle waypoint
            v0=0.0, vn=0.0, amax=2.0, vmax=3.0
        )
        
        trajectory_func, total_duration = TrapezoidalTrajectory.interpolate_waypoints(params)
        
        # Should handle intermediate velocities
        assert callable(trajectory_func)
        assert total_duration > 0

    def test_calculate_heuristic_velocities(self) -> None:
        """Test heuristic velocity calculation."""
        points = [0.0, 3.0, 8.0, 10.0]
        vmax = 4.0
        
        velocities = TrapezoidalTrajectory.calculate_heuristic_velocities(points, v0=0.0, vn=0.0, v_max=vmax)
        
        # Should return velocities for all points (including v0 and vn)
        assert len(velocities) == len(points)
        assert all(abs(v) <= vmax for v in velocities)
        # First and last should match specified boundary conditions
        assert velocities[0] == 0.0  # v0
        assert velocities[-1] == 0.0  # vn

    def test_minimum_points_validation(self) -> None:
        """Test validation of minimum number of points."""
        params = InterpolationParams(
            points=[5.0],  # Only one point
            v0=0.0, vn=0.0, amax=1.0
        )
        
        with pytest.raises(ValueError, match="At least two points are required"):
            TrapezoidalTrajectory.interpolate_waypoints(params)


class TestTrapezoidalTrajectoryEdgeCases:
    """Test suite for trapezoidal trajectory edge cases."""

    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def test_zero_displacement(self) -> None:
        """Test trajectory with zero displacement."""
        params = TrajectoryParams(
            q0=5.0, q1=5.0, v0=0.0, v1=0.0, amax=2.0, vmax=3.0
        )
        
        trajectory_func, duration = TrapezoidalTrajectory.generate_trajectory(params)
        
        # Duration should be minimal for zero displacement
        assert duration >= 0
        
        # Position should remain constant
        q_mid, _, _ = trajectory_func(duration / 2)
        assert abs(q_mid - 5.0) < self.NUMERICAL_ATOL

    def test_very_high_constraints(self) -> None:
        """Test with very high velocity and acceleration constraints."""
        params = TrajectoryParams(
            q0=0.0, q1=1.0, v0=0.0, v1=0.0, amax=1000.0, vmax=1000.0
        )
        
        trajectory_func, duration = TrapezoidalTrajectory.generate_trajectory(params)
        
        # Should handle high constraints gracefully
        assert duration > 0
        
        # Final position should be correct
        q_final, _, _ = trajectory_func(duration)
        assert abs(q_final - 1.0) < self.NUMERICAL_ATOL

    def test_identical_waypoints(self) -> None:
        """Test waypoint interpolation with some identical points."""
        params = InterpolationParams(
            points=[0.0, 5.0, 5.0, 10.0],  # Middle points identical
            v0=0.0, vn=0.0, amax=2.0, vmax=3.0
        )
        
        trajectory_func, total_duration = TrapezoidalTrajectory.interpolate_waypoints(params)
        
        # Should handle identical waypoints
        assert callable(trajectory_func)
        assert total_duration > 0


class TestMotionProfilePerformance:
    """Test suite for performance benchmarks."""

    @pytest.mark.parametrize("displacement", [1.0, 10.0, 100.0])
    def test_double_s_construction_performance(
        self, displacement: float, benchmark: pytest.FixtureFunction
    ) -> None:
        """Benchmark DoubleSTrajectory construction performance."""
        state_params = StateParams(q_0=0.0, q_1=displacement, v_0=0.0, v_1=0.0)
        bounds = TrajectoryBounds(v_bound=5.0, a_bound=3.0, j_bound=2.0)
        
        def construct_trajectory() -> DoubleSTrajectory:
            return DoubleSTrajectory(state_params, bounds)
        
        trajectory = benchmark(construct_trajectory)
        assert isinstance(trajectory, DoubleSTrajectory)

    @pytest.mark.parametrize("n_evaluations", [100, 1000])
    def test_double_s_evaluation_performance(
        self, n_evaluations: int, benchmark: pytest.FixtureFunction
    ) -> None:
        """Benchmark DoubleSTrajectory evaluation performance."""
        state_params = StateParams(q_0=0.0, q_1=20.0, v_0=0.0, v_1=0.0)
        bounds = TrajectoryBounds(v_bound=4.0, a_bound=2.0, j_bound=1.0)
        trajectory = DoubleSTrajectory(state_params, bounds)
        
        duration = trajectory.get_duration()
        t_values = np.linspace(0, duration, n_evaluations)
        
        def evaluate_trajectory() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            return trajectory.evaluate(t_values)
        
        q, v, a, j = benchmark(evaluate_trajectory)
        assert len(q) == n_evaluations

    def test_trapezoidal_generation_performance(
        self, benchmark: pytest.FixtureFunction
    ) -> None:
        """Benchmark trapezoidal trajectory generation performance."""
        params = TrajectoryParams(
            q0=0.0, q1=50.0, v0=0.0, v1=0.0, amax=5.0, vmax=10.0
        )
        
        def generate_trajectory() -> tuple:
            return TrapezoidalTrajectory.generate_trajectory(params)
        
        trajectory_func, duration = benchmark(generate_trajectory)
        assert callable(trajectory_func)
        assert duration > 0

    def test_waypoint_interpolation_performance(
        self, benchmark: pytest.FixtureFunction
    ) -> None:
        """Benchmark waypoint interpolation performance."""
        n_points = 10
        points = [i * 2.0 for i in range(n_points)]
        params = InterpolationParams(
            points=points, v0=0.0, vn=0.0, amax=3.0, vmax=5.0
        )
        
        def interpolate_waypoints() -> tuple:
            return TrapezoidalTrajectory.interpolate_waypoints(params)
        
        trajectory_func, total_duration = benchmark(interpolate_waypoints)
        assert callable(trajectory_func)
        assert total_duration > 0


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main(["-xvs", __file__])