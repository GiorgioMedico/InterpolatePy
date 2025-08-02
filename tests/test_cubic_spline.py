"""
Comprehensive tests for the CubicSpline trajectory planning implementation.

This module contains extensive tests for the CubicSpline class covering:
1. Constructor validation and parameter checking
2. Mathematical accuracy with known analytical solutions
3. Continuity properties (C0, C1, C2 continuity)
4. Boundary condition handling
5. Edge cases and error handling
6. Numerical stability and convergence
7. Performance benchmarks

The tests verify that the cubic spline implementation provides smooth
trajectories with continuous position, velocity, and acceleration profiles.
"""

from typing import Any

import numpy as np
import pytest

from interpolatepy.cubic_spline import CubicSpline


# Type alias for pytest benchmark fixture
if not hasattr(pytest, "FixtureFunction"):
    pytest.FixtureFunction = Any


class TestCubicSplineConstruction:
    """Test suite for cubic spline construction and validation."""

    REGULAR_RTOL = 1e-10
    REGULAR_ATOL = 1e-10
    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def test_basic_construction(self) -> None:
        """Test basic cubic spline construction with valid inputs."""
        t_points = [0.0, 1.0, 2.0, 3.0]
        q_points = [0.0, 1.0, 0.0, 1.0]

        spline = CubicSpline(t_points, q_points)

        # Check basic attributes
        assert len(spline.t_points) == 4
        assert len(spline.q_points) == 4
        assert spline.v0 == 0.0
        assert spline.vn == 0.0
        assert spline.n == 3
        assert len(spline.t_intervals) == 3
        assert len(spline.velocities) == 4
        assert spline.coefficients.shape == (3, 4)

    def test_construction_with_boundary_velocities(self) -> None:
        """Test construction with specified boundary velocities."""
        t_points = [0.0, 1.0, 2.0]
        q_points = [0.0, 1.0, 2.0]
        v0, vn = 0.5, -0.5

        spline = CubicSpline(t_points, q_points, v0=v0, vn=vn)

        assert spline.v0 == v0
        assert spline.vn == vn
        assert abs(spline.velocities[0] - v0) < self.REGULAR_ATOL
        assert abs(spline.velocities[-1] - vn) < self.REGULAR_ATOL

    def test_construction_with_numpy_arrays(self) -> None:
        """Test construction with numpy arrays instead of lists."""
        t_points = np.array([0.0, 1.0, 2.0, 3.0])
        q_points = np.array([1.0, 2.0, 3.0, 4.0])

        spline = CubicSpline(t_points, q_points)

        assert isinstance(spline.t_points, np.ndarray)
        assert isinstance(spline.q_points, np.ndarray)
        assert spline.n == 3

    def test_construction_validation_mismatched_lengths(self) -> None:
        """Test that mismatched lengths raise ValueError."""
        t_points = [0.0, 1.0, 2.0]
        q_points = [0.0, 1.0]  # One less point

        with pytest.raises(ValueError, match="must have the same length"):
            CubicSpline(t_points, q_points)

    def test_construction_validation_non_increasing_time(self) -> None:
        """Test that non-increasing time points raise ValueError."""
        # Decreasing time points
        t_points = [3.0, 2.0, 1.0, 0.0]
        q_points = [0.0, 1.0, 2.0, 3.0]

        with pytest.raises(ValueError, match="must be strictly increasing"):
            CubicSpline(t_points, q_points)

        # Repeated time points
        t_points_repeated = [0.0, 1.0, 1.0, 2.0]

        with pytest.raises(ValueError, match="must be strictly increasing"):
            CubicSpline(t_points_repeated, q_points)

    def test_single_segment_construction(self) -> None:
        """Test construction with only two points (single segment)."""
        t_points = [0.0, 2.0]
        q_points = [1.0, 3.0]
        v0, vn = 0.5, 1.5

        spline = CubicSpline(t_points, q_points, v0=v0, vn=vn)

        assert spline.n == 1
        assert len(spline.velocities) == 2
        assert spline.coefficients.shape == (1, 4)
        assert abs(spline.velocities[0] - v0) < self.REGULAR_ATOL
        assert abs(spline.velocities[1] - vn) < self.REGULAR_ATOL

    def test_debug_mode(self) -> None:
        """Test that debug mode can be enabled without errors."""
        t_points = [0.0, 1.0, 2.0, 3.0]
        q_points = [0.0, 1.0, 0.0, 1.0]

        # Should not raise any exceptions
        spline = CubicSpline(t_points, q_points, debug=True)
        assert spline.debug is True


class TestCubicSplineMathematicalAccuracy:
    """Test suite for mathematical accuracy and analytical solutions."""

    REGULAR_RTOL = 1e-10
    REGULAR_ATOL = 1e-10
    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def test_linear_function_exact(self) -> None:
        """Test that spline exactly represents a linear function."""
        # y = 2x + 1, so dy/dt = 2
        t_points = [0.0, 1.0, 2.0, 3.0]
        q_points = [1.0, 3.0, 5.0, 7.0]  # Linear with slope 2

        spline = CubicSpline(t_points, q_points, v0=2.0, vn=2.0)

        # Check at waypoints
        for _i, (t, q) in enumerate(zip(t_points, q_points)):
            assert abs(spline.evaluate(t) - q) < self.REGULAR_ATOL
            assert abs(spline.evaluate_velocity(t) - 2.0) < self.NUMERICAL_ATOL
            assert abs(spline.evaluate_acceleration(t) - 0.0) < self.NUMERICAL_ATOL

        # Check at intermediate points
        test_times = [0.5, 1.5, 2.5]
        expected_positions = [2.0, 4.0, 6.0]

        for t, expected_q in zip(test_times, expected_positions):
            assert abs(spline.evaluate(t) - expected_q) < self.NUMERICAL_ATOL
            assert abs(spline.evaluate_velocity(t) - 2.0) < self.NUMERICAL_ATOL
            assert abs(spline.evaluate_acceleration(t) - 0.0) < self.NUMERICAL_ATOL

    def test_quadratic_function_representation(self) -> None:
        """Test spline representation of a quadratic function."""
        # y = x^2, so dy/dt = 2x, d²y/dt² = 2
        t_points = [0.0, 1.0, 2.0, 3.0]
        q_points = [0.0, 1.0, 4.0, 9.0]  # x^2
        v0, vn = 0.0, 6.0  # Velocities at boundaries for x^2

        spline = CubicSpline(t_points, q_points, v0=v0, vn=vn)

        # Check at intermediate points
        test_times = [0.5, 1.5, 2.5]
        expected_positions = [0.25, 2.25, 6.25]
        expected_velocities = [1.0, 3.0, 5.0]
        expected_accelerations = [2.0, 2.0, 2.0]

        for t, exp_q, exp_v, exp_a in zip(
            test_times, expected_positions, expected_velocities, expected_accelerations
        ):
            assert abs(spline.evaluate(t) - exp_q) < self.NUMERICAL_ATOL
            assert abs(spline.evaluate_velocity(t) - exp_v) < self.NUMERICAL_ATOL
            assert abs(spline.evaluate_acceleration(t) - exp_a) < self.NUMERICAL_ATOL

    def test_waypoint_interpolation_exact(self) -> None:
        """Test that spline passes exactly through all waypoints."""
        t_points = [0.0, 0.5, 1.2, 2.1, 3.5]
        q_points = [1.5, -0.8, 2.3, -1.1, 0.7]

        spline = CubicSpline(t_points, q_points)

        for t, q in zip(t_points, q_points):
            assert abs(spline.evaluate(t) - q) < self.REGULAR_ATOL

    def test_boundary_conditions_exact(self) -> None:
        """Test that boundary velocities are exactly satisfied."""
        t_points = [0.0, 1.0, 2.0, 3.0]
        q_points = [0.0, 1.0, 0.0, 1.0]
        v0, vn = 1.5, -2.3

        spline = CubicSpline(t_points, q_points, v0=v0, vn=vn)

        assert abs(spline.evaluate_velocity(t_points[0]) - v0) < self.REGULAR_ATOL
        assert abs(spline.evaluate_velocity(t_points[-1]) - vn) < self.REGULAR_ATOL

    def test_symmetric_trajectory(self) -> None:
        """Test spline with symmetric trajectory."""
        # Symmetric waypoints should produce symmetric velocity profile
        t_points = [0.0, 1.0, 2.0, 3.0, 4.0]
        q_points = [0.0, 1.0, 2.0, 1.0, 0.0]  # Symmetric

        spline = CubicSpline(t_points, q_points, v0=0.0, vn=0.0)

        # Check symmetry around t=2.0
        test_times = [0.5, 1.5]
        for dt in test_times:
            t1, t2 = 2.0 - dt, 2.0 + dt
            q1, q2 = spline.evaluate(t1), spline.evaluate(t2)
            v1, v2 = spline.evaluate_velocity(t1), spline.evaluate_velocity(t2)

            # Positions should be equal (symmetric)
            assert abs(q1 - q2) < self.NUMERICAL_ATOL
            # Velocities should be opposite (antisymmetric)
            assert abs(v1 + v2) < self.NUMERICAL_ATOL


class TestCubicSplineContinuity:
    """Test suite for continuity properties of cubic splines."""

    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def test_position_continuity_c0(self) -> None:
        """Test C0 continuity (continuous position) at waypoints."""
        t_points = [0.0, 1.0, 2.0, 3.0, 4.0]
        q_points = [1.0, 3.0, 0.5, 2.8, 1.2]

        spline = CubicSpline(t_points, q_points)

        # Test continuity at internal waypoints
        for i in range(1, len(t_points) - 1):
            t = t_points[i]
            eps = 1e-8

            # Approach from left and right
            q_left = spline.evaluate(t - eps)
            q_right = spline.evaluate(t + eps)
            q_exact = spline.evaluate(t)

            assert abs(q_left - q_exact) < self.NUMERICAL_ATOL
            assert abs(q_right - q_exact) < self.NUMERICAL_ATOL
            assert abs(q_left - q_right) < self.NUMERICAL_ATOL

    def test_velocity_continuity_c1(self) -> None:
        """Test C1 continuity (continuous velocity) at waypoints."""
        t_points = [0.0, 1.0, 2.0, 3.0]
        q_points = [0.0, 1.0, 0.0, 1.0]

        spline = CubicSpline(t_points, q_points)

        # Test continuity at internal waypoints
        for i in range(1, len(t_points) - 1):
            t = t_points[i]
            eps = 1e-8

            # Approach from left and right
            v_left = spline.evaluate_velocity(t - eps)
            v_right = spline.evaluate_velocity(t + eps)
            v_exact = spline.evaluate_velocity(t)

            assert abs(v_left - v_exact) < self.NUMERICAL_ATOL
            assert abs(v_right - v_exact) < self.NUMERICAL_ATOL
            assert abs(v_left - v_right) < self.NUMERICAL_ATOL

    def test_acceleration_continuity_c2(self) -> None:
        """Test C2 continuity (continuous acceleration) at waypoints."""
        t_points = [0.0, 1.0, 2.0, 3.0, 4.0]
        q_points = [1.0, 2.0, 1.5, 3.0, 2.0]

        spline = CubicSpline(t_points, q_points)

        # Test continuity at internal waypoints
        for i in range(1, len(t_points) - 1):
            t = t_points[i]
            eps = 1e-8

            # Approach from left and right
            a_left = spline.evaluate_acceleration(t - eps)
            a_right = spline.evaluate_acceleration(t + eps)
            a_exact = spline.evaluate_acceleration(t)

            assert abs(a_left - a_exact) < self.NUMERICAL_ATOL
            assert abs(a_right - a_exact) < self.NUMERICAL_ATOL
            assert abs(a_left - a_right) < self.NUMERICAL_ATOL


class TestCubicSplineEvaluationMethods:
    """Test suite for spline evaluation methods."""

    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def test_scalar_evaluation(self) -> None:
        """Test evaluation with scalar time inputs."""
        t_points = [0.0, 1.0, 2.0]
        q_points = [0.0, 1.0, 4.0]

        spline = CubicSpline(t_points, q_points)

        # Test scalar inputs return scalars
        t_test = 1.5
        q = spline.evaluate(t_test)
        v = spline.evaluate_velocity(t_test)
        a = spline.evaluate_acceleration(t_test)

        assert isinstance(q, int | float | np.number)
        assert isinstance(v, int | float | np.number)
        assert isinstance(a, int | float | np.number)

    def test_array_evaluation(self) -> None:
        """Test evaluation with array time inputs."""
        t_points = [0.0, 1.0, 2.0, 3.0]
        q_points = [0.0, 1.0, 0.0, 1.0]

        spline = CubicSpline(t_points, q_points)

        # Test array inputs return arrays
        t_test = np.array([0.5, 1.0, 1.5, 2.5])
        q = spline.evaluate(t_test)
        v = spline.evaluate_velocity(t_test)
        a = spline.evaluate_acceleration(t_test)

        assert isinstance(q, np.ndarray)
        assert isinstance(v, np.ndarray)
        assert isinstance(a, np.ndarray)
        assert len(q) == len(t_test)
        assert len(v) == len(t_test)
        assert len(a) == len(t_test)

    def test_boundary_extrapolation(self) -> None:
        """Test behavior outside the time range."""
        t_points = [1.0, 2.0, 3.0]
        q_points = [1.0, 2.0, 3.0]

        spline = CubicSpline(t_points, q_points, v0=0.5, vn=0.5)

        # Before first point - should extrapolate using first segment
        t_before = 0.5
        q_before = spline.evaluate(t_before)
        v_before = spline.evaluate_velocity(t_before)
        a_before = spline.evaluate_acceleration(t_before)

        # Should evaluate without errors and return finite values
        assert np.isfinite(q_before)
        assert np.isfinite(v_before)
        assert np.isfinite(a_before)

        # After last point - should extrapolate using last segment
        t_after = 3.5
        q_after = spline.evaluate(t_after)
        v_after = spline.evaluate_velocity(t_after)
        a_after = spline.evaluate_acceleration(t_after)

        # Should evaluate without errors and return finite values
        assert np.isfinite(q_after)
        assert np.isfinite(v_after)
        assert np.isfinite(a_after)

    def test_evaluation_consistency(self) -> None:
        """Test that evaluation methods are mutually consistent."""
        t_points = [0.0, 1.0, 2.0, 3.0]
        q_points = [0.0, 1.0, 0.0, 1.0]

        spline = CubicSpline(t_points, q_points)

        # Test finite difference approximation of derivatives
        t_test = 1.5
        dt = 1e-6

        # Approximate velocity using finite differences
        q_plus = spline.evaluate(t_test + dt)
        q_minus = spline.evaluate(t_test - dt)
        v_numeric = (q_plus - q_minus) / (2 * dt)
        v_analytic = spline.evaluate_velocity(t_test)

        assert abs(v_numeric - v_analytic) < 1e-4

        # Approximate acceleration using finite differences
        v_plus = spline.evaluate_velocity(t_test + dt)
        v_minus = spline.evaluate_velocity(t_test - dt)
        a_numeric = (v_plus - v_minus) / (2 * dt)
        a_analytic = spline.evaluate_acceleration(t_test)

        assert abs(a_numeric - a_analytic) < 1e-4


class TestCubicSplineEdgeCases:
    """Test suite for edge cases and special situations."""

    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def test_minimum_points(self) -> None:
        """Test spline with minimum number of points (2)."""
        t_points = [0.0, 1.0]
        q_points = [1.0, 2.0]

        spline = CubicSpline(t_points, q_points, v0=0.5, vn=1.5)

        assert spline.n == 1
        assert spline.coefficients.shape == (1, 4)

        # Should exactly match endpoints
        assert abs(spline.evaluate(0.0) - 1.0) < self.NUMERICAL_ATOL
        assert abs(spline.evaluate(1.0) - 2.0) < self.NUMERICAL_ATOL
        assert abs(spline.evaluate_velocity(0.0) - 0.5) < self.NUMERICAL_ATOL
        assert abs(spline.evaluate_velocity(1.0) - 1.5) < self.NUMERICAL_ATOL

    def test_identical_positions(self) -> None:
        """Test spline with some identical position values."""
        t_points = [0.0, 1.0, 2.0, 3.0]
        q_points = [1.0, 1.0, 1.0, 2.0]  # Constant then jump

        # Should not raise exceptions
        spline = CubicSpline(t_points, q_points)

        # Should still pass through waypoints
        for t, q in zip(t_points, q_points):
            assert abs(spline.evaluate(t) - q) < self.NUMERICAL_ATOL

    def test_large_time_intervals(self) -> None:
        """Test spline with very different time interval sizes."""
        t_points = [0.0, 0.1, 10.0, 10.1]
        q_points = [0.0, 1.0, 2.0, 3.0]

        spline = CubicSpline(t_points, q_points)

        # Should handle large differences gracefully
        assert spline.n == 3
        for t, q in zip(t_points, q_points):
            assert abs(spline.evaluate(t) - q) < self.NUMERICAL_ATOL

    def test_negative_time_values(self) -> None:
        """Test spline with negative time values."""
        t_points = [-2.0, -1.0, 0.0, 1.0]
        q_points = [1.0, 2.0, 3.0, 4.0]

        spline = CubicSpline(t_points, q_points)

        # Should work normally with negative times
        for t, q in zip(t_points, q_points):
            assert abs(spline.evaluate(t) - q) < self.NUMERICAL_ATOL

    def test_zero_boundary_velocities(self) -> None:
        """Test natural spline (zero boundary velocities)."""
        t_points = [0.0, 1.0, 2.0, 3.0]
        q_points = [0.0, 1.0, 0.0, 1.0]

        spline = CubicSpline(t_points, q_points, v0=0.0, vn=0.0)

        assert abs(spline.evaluate_velocity(t_points[0])) < self.NUMERICAL_ATOL
        assert abs(spline.evaluate_velocity(t_points[-1])) < self.NUMERICAL_ATOL


class TestCubicSplineNumericalStability:
    """Test suite for numerical stability and convergence."""

    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def test_high_precision_waypoints(self) -> None:
        """Test spline with high precision waypoint data."""
        # Use high precision values that could cause numerical issues
        t_points = [0.0, 1e-3, 2e-3, 3e-3]
        q_points = [1e-12, 2e-12, 3e-12, 4e-12]

        spline = CubicSpline(t_points, q_points)

        # Should handle small values correctly
        for t, q in zip(t_points, q_points):
            relative_error = abs(spline.evaluate(t) - q) / (abs(q) + 1e-15)
            assert relative_error < 1e-10

    def test_large_scale_values(self) -> None:
        """Test spline with large scale values."""
        t_points = [0.0, 1e6, 2e6, 3e6]
        q_points = [1e9, 2e9, 3e9, 4e9]

        spline = CubicSpline(t_points, q_points)

        # Should handle large values correctly
        for t, q in zip(t_points, q_points):
            relative_error = abs(spline.evaluate(t) - q) / abs(q)
            assert relative_error < 1e-10

    def test_convergence_with_increasing_points(self) -> None:
        """Test convergence behavior as number of waypoints increases."""

        # Test function: sin(x) on [0, π]
        def test_function(t: float) -> float:
            return np.sin(t)

        t_min, t_max = 0.0, np.pi

        # Test with increasing number of points
        errors = []
        point_counts = [5, 10, 20, 40]

        for n_points in point_counts:
            t_points = np.linspace(t_min, t_max, n_points)
            q_points = [test_function(t) for t in t_points]

            # Use natural boundary conditions (zero second derivatives)
            v0 = np.cos(t_points[0])  # True derivative at start
            vn = np.cos(t_points[-1])  # True derivative at end

            spline = CubicSpline(t_points, q_points, v0=v0, vn=vn)

            # Compute error at intermediate points
            t_test = np.linspace(t_min, t_max, 100)
            q_true = [test_function(t) for t in t_test]
            q_spline = spline.evaluate(t_test)

            error = np.max(np.abs(np.array(q_spline) - np.array(q_true)))
            errors.append(error)

        # Error should generally decrease as we add more points
        # (though cubic splines may not always converge monotonically)
        assert errors[-1] < errors[0], "Error should decrease with more points"

    def test_ill_conditioned_system_handling(self) -> None:
        """Test handling of potentially ill-conditioned systems."""
        # Create a case that might be challenging numerically
        t_points = [0.0, 1e-10, 1.0, 1.0 + 1e-10]
        q_points = [0.0, 1.0, 1.0, 0.0]

        # Should not raise exceptions or produce NaN values
        spline = CubicSpline(t_points, q_points)

        # All computed values should be finite
        assert np.all(np.isfinite(spline.velocities))
        assert np.all(np.isfinite(spline.coefficients))

        # Evaluation should produce finite results
        test_times = [0.5e-10, 0.5, 1.0 + 0.5e-10]
        for t in test_times:
            q = spline.evaluate(t)
            v = spline.evaluate_velocity(t)
            a = spline.evaluate_acceleration(t)

            assert np.isfinite(q)
            assert np.isfinite(v)
            assert np.isfinite(a)


class TestCubicSplinePerformance:
    """Test suite for performance benchmarks."""

    @pytest.mark.parametrize("n_points", [10, 50, 100, 500])
    def test_construction_performance(
        self, n_points: int, benchmark: pytest.FixtureFunction
    ) -> None:
        """Benchmark spline construction performance."""
        t_points = np.linspace(0, 10, n_points)
        q_points = np.sin(t_points)

        def construct_spline() -> CubicSpline:
            return CubicSpline(t_points, q_points)

        spline = benchmark(construct_spline)
        assert len(spline.t_points) == n_points

    @pytest.mark.parametrize("n_evaluations", [100, 1000, 10000])
    def test_evaluation_performance(
        self, n_evaluations: int, benchmark: pytest.FixtureFunction
    ) -> None:
        """Benchmark spline evaluation performance."""
        t_points = np.linspace(0, 10, 20)
        q_points = np.sin(t_points)
        spline = CubicSpline(t_points, q_points)

        t_eval = np.linspace(0, 10, n_evaluations)

        def evaluate_spline() -> np.ndarray:
            return spline.evaluate(t_eval)

        result = benchmark(evaluate_spline)
        assert len(result) == n_evaluations

    def test_derivative_evaluation_performance(self, benchmark: pytest.FixtureFunction) -> None:
        """Benchmark derivative evaluation performance."""
        t_points = np.linspace(0, 10, 50)
        q_points = np.sin(t_points)
        spline = CubicSpline(t_points, q_points)

        t_eval = np.linspace(0, 10, 1000)

        def evaluate_derivatives() -> tuple[np.ndarray, np.ndarray]:
            v = spline.evaluate_velocity(t_eval)
            a = spline.evaluate_acceleration(t_eval)
            return v, a

        v, a = benchmark(evaluate_derivatives)
        assert len(v) == 1000
        assert len(a) == 1000


# class TestCubicSplinePlotting:
#     """Test suite for plotting functionality."""

#     def test_plot_method_exists(self) -> None:
#         """Test that plot method exists and can be called."""
#         t_points = [0.0, 1.0, 2.0, 3.0]
#         q_points = [0.0, 1.0, 0.0, 1.0]

#         spline = CubicSpline(t_points, q_points)

#         # Should not raise exceptions
#         # Note: We don't actually display the plot in tests
#         try:
#             # Temporarily redirect to non-interactive backend
#             plt.ioff()
#             spline.plot()
#             plt.close('all')
#         except Exception as e:
#             pytest.fail(f"Plot method raised exception: {e}")

#     def test_plot_with_custom_points(self) -> None:
#         """Test plot method with custom number of points."""
#         t_points = [0.0, 1.0, 2.0]
#         q_points = [1.0, 2.0, 3.0]

#         spline = CubicSpline(t_points, q_points)

#         try:
#             plt.ioff()
#             spline.plot(num_points=100)
#             plt.close('all')
#         except Exception as e:
#             pytest.fail(f"Plot method with custom points raised exception: {e}")


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main(["-xvs", __file__])
