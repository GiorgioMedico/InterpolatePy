"""
Comprehensive tests for the B-spline curve implementation.

This module contains extensive tests for the BSpline class covering:
1. Constructor validation and parameter checking
2. Mathematical accuracy with known analytical solutions
3. Basis function calculations and properties
4. Curve evaluation and derivative computation
5. Knot vector handling and span finding
6. Edge cases and error handling
7. Plotting functionality
8. Performance benchmarks

The tests verify that the B-spline implementation correctly handles various
degrees, knot vectors, and control point configurations.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest

from interpolatepy.b_spline import BSpline


# Type alias for pytest benchmark fixture
if not hasattr(pytest, "FixtureFunction"):
    pytest.FixtureFunction = Any


class TestBSplineConstruction:
    """Test suite for B-spline construction and validation."""

    REGULAR_RTOL = 1e-10
    REGULAR_ATOL = 1e-10
    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def test_basic_construction_1d(self) -> None:
        """Test basic B-spline construction with 1D control points."""
        degree = 2
        knots = [0, 0, 0, 1, 2, 3, 3, 3]
        control_points = [1, 2, 3, 4, 5]

        spline = BSpline(degree, knots, control_points)

        assert spline.degree == degree
        assert len(spline.knots) == len(knots)
        assert len(spline.control_points) == len(control_points)
        assert spline.dimension == 1
        assert spline.u_min == knots[degree]
        assert spline.u_max == knots[-(degree + 1)]

    def test_basic_construction_2d(self) -> None:
        """Test basic B-spline construction with 2D control points."""
        degree = 2
        knots = [0, 0, 0, 1, 2, 3, 3, 3]
        control_points = [[0, 0], [1, 1], [2, 0], [3, 1], [4, 0]]

        spline = BSpline(degree, knots, control_points)

        assert spline.degree == degree
        assert spline.dimension == 2
        assert spline.control_points.shape == (5, 2)

    def test_basic_construction_3d(self) -> None:
        """Test basic B-spline construction with 3D control points."""
        degree = 1
        knots = [0, 0, 1, 2, 2]
        control_points = [[0, 0, 0], [1, 1, 1], [2, 0, 2]]

        spline = BSpline(degree, knots, control_points)

        assert spline.degree == degree
        assert spline.dimension == 3
        assert spline.control_points.shape == (3, 3)

    def test_construction_with_numpy_arrays(self) -> None:
        """Test construction with numpy arrays."""
        degree = 1
        knots = np.array([0, 0, 1, 1])
        control_points = np.array([[0, 0], [1, 1]])

        spline = BSpline(degree, knots, control_points)

        assert isinstance(spline.knots, np.ndarray)
        assert isinstance(spline.control_points, np.ndarray)
        assert spline.knots.dtype == np.float64
        assert spline.control_points.dtype == np.float64

    def test_construction_validation_negative_degree(self) -> None:
        """Test that negative degree raises ValueError."""
        degree = -1
        knots = [0, 0, 1, 1]
        control_points = [1, 2]

        with pytest.raises(ValueError, match="Degree must be non-negative"):
            BSpline(degree, knots, control_points)

    def test_construction_validation_non_decreasing_knots(self) -> None:
        """Test that non-decreasing knots raise ValueError."""
        degree = 1
        knots = [0, 1, 0.5, 1]  # Not non-decreasing
        control_points = [1, 2]

        with pytest.raises(ValueError, match="Knot vector must be non-decreasing"):
            BSpline(degree, knots, control_points)

    def test_construction_validation_knot_relationship(self) -> None:
        """Test that invalid knot-control point relationship raises ValueError."""
        degree = 2
        knots = [0, 0, 0, 1, 1, 1]  # 6 knots
        control_points = [1, 2, 3, 4]  # 4 control points
        # For degree=2, should need 4+2+1=7 knots, but have 6

        with pytest.raises(ValueError, match="Invalid knot vector length"):
            BSpline(degree, knots, control_points)

    def test_degree_zero_construction(self) -> None:
        """Test construction with degree 0 (constant basis functions)."""
        degree = 0
        knots = [0, 1, 2, 3]
        control_points = [1, 2, 3]

        spline = BSpline(degree, knots, control_points)

        assert spline.degree == 0
        assert spline.u_min == knots[0]
        assert spline.u_max == knots[-1]

    def test_high_degree_construction(self) -> None:
        """Test construction with higher degree B-spline."""
        degree = 5
        knots = [0] * 6 + [1] * 6  # Multiplicity at endpoints
        control_points = [[i, i**2] for i in range(6)]

        spline = BSpline(degree, knots, control_points)

        assert spline.degree == 5
        assert spline.dimension == 2


class TestBSplineKnotHandling:
    """Test suite for knot vector handling and span finding."""

    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def test_find_knot_span_basic(self) -> None:
        """Test basic knot span finding."""
        degree = 2
        knots = [0, 0, 0, 1, 2, 3, 3, 3]
        control_points = [1, 2, 3, 4, 5]

        spline = BSpline(degree, knots, control_points)

        # Test various parameter values
        assert spline.find_knot_span(0.0) == 2
        assert spline.find_knot_span(0.5) == 2
        assert spline.find_knot_span(1.0) == 3
        assert spline.find_knot_span(1.5) == 3
        assert spline.find_knot_span(2.0) == 4
        assert spline.find_knot_span(3.0) == 4

    def test_find_knot_span_caching(self) -> None:
        """Test that knot span finding uses caching correctly."""
        degree = 1
        knots = [0, 0, 1, 2, 2]
        control_points = [1, 2, 3]

        spline = BSpline(degree, knots, control_points)

        # First call should compute and cache
        span1 = spline.find_knot_span(0.5)

        # Second call should use cache
        span2 = spline.find_knot_span(0.5)

        assert span1 == span2
        assert 0.5 in spline._cached_spans

    def test_find_knot_span_boundary_conditions(self) -> None:
        """Test knot span finding at boundary conditions."""
        degree = 2
        knots = [0, 0, 0, 1, 2, 2, 2]
        control_points = [1, 2, 3, 4]

        spline = BSpline(degree, knots, control_points)

        # Test at exact boundaries
        assert spline.find_knot_span(spline.u_min) == degree
        assert spline.find_knot_span(spline.u_max) == len(spline.knots) - degree - 2

    def test_find_knot_span_out_of_range(self) -> None:
        """Test that out-of-range parameters raise ValueError."""
        degree = 1
        knots = [0, 0, 1, 1]
        control_points = [1, 2]

        spline = BSpline(degree, knots, control_points)

        with pytest.raises(ValueError, match="Parameter u=.* outside valid range"):
            spline.find_knot_span(-0.1)

        with pytest.raises(ValueError, match="Parameter u=.* outside valid range"):
            spline.find_knot_span(1.1)

    def test_uniform_knot_creation(self) -> None:
        """Test uniform knot vector creation."""
        degree = 2
        num_control_points = 5

        knots = BSpline.create_uniform_knots(degree, num_control_points)

        # Check knot vector properties
        assert len(knots) == num_control_points + degree + 1
        assert np.all(np.diff(knots) >= 0)  # Non-decreasing
        assert knots[0] == 0.0  # Default domain start
        assert knots[-1] == 1.0  # Default domain end
        assert np.sum(knots == 0.0) == degree + 1  # Multiplicity at start
        assert np.sum(knots == 1.0) == degree + 1  # Multiplicity at end

    def test_uniform_knot_creation_custom_domain(self) -> None:
        """Test uniform knot vector creation with custom domain."""
        degree = 1
        num_control_points = 3
        domain_min = -2.0
        domain_max = 5.0

        knots = BSpline.create_uniform_knots(degree, num_control_points, domain_min, domain_max)

        assert knots[0] == domain_min
        assert knots[-1] == domain_max

    def test_uniform_knot_creation_validation(self) -> None:
        """Test uniform knot creation input validation."""
        with pytest.raises(ValueError, match="Degree must be non-negative"):
            BSpline.create_uniform_knots(-1, 3)

        with pytest.raises(ValueError, match="must be greater than the degree"):
            BSpline.create_uniform_knots(2, 2)

    def test_periodic_knot_creation(self) -> None:
        """Test periodic knot vector creation."""
        degree = 2
        num_control_points = 4

        knots = BSpline.create_periodic_knots(degree, num_control_points)

        assert len(knots) == num_control_points + degree + 1
        assert np.all(np.diff(knots) >= 0)  # Non-decreasing

    def test_periodic_knot_creation_validation(self) -> None:
        """Test periodic knot creation input validation."""
        with pytest.raises(ValueError, match="Degree must be non-negative"):
            BSpline.create_periodic_knots(-1, 3)

        with pytest.raises(ValueError, match="must be at least degree\\+1"):
            BSpline.create_periodic_knots(3, 2)


class TestBSplineBasisFunctions:
    """Test suite for basis function calculations."""

    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def test_basis_functions_degree_zero(self) -> None:
        """Test basis functions for degree 0 (constant functions)."""
        degree = 0
        knots = [0, 1, 2, 3]
        control_points = [1, 2, 3]

        spline = BSpline(degree, knots, control_points)

        # For degree 0, only one basis function should be 1, others 0
        span = spline.find_knot_span(0.5)
        basis = spline.basis_functions(0.5, span)

        assert len(basis) == degree + 1
        assert np.sum(basis) == pytest.approx(1.0, abs=self.NUMERICAL_ATOL)

    def test_basis_functions_degree_one(self) -> None:
        """Test basis functions for degree 1 (linear functions)."""
        degree = 1
        knots = [0, 0, 1, 2, 2]
        control_points = [1, 2, 3]

        spline = BSpline(degree, knots, control_points)

        # Test at parameter value 0.5
        span = spline.find_knot_span(0.5)
        basis = spline.basis_functions(0.5, span)

        assert len(basis) == degree + 1
        assert np.sum(basis) == pytest.approx(1.0, abs=self.NUMERICAL_ATOL)

    def test_basis_functions_degree_two(self) -> None:
        """Test basis functions for degree 2 (quadratic functions)."""
        degree = 2
        knots = [0, 0, 0, 1, 2, 2, 2]
        control_points = [1, 2, 3, 4]

        spline = BSpline(degree, knots, control_points)

        # Test at parameter value 0.5
        span = spline.find_knot_span(0.5)
        basis = spline.basis_functions(0.5, span)

        assert len(basis) == degree + 1
        assert np.sum(basis) == pytest.approx(1.0, abs=self.NUMERICAL_ATOL)
        assert np.all(basis >= 0)  # Basis functions should be non-negative

    def test_basis_functions_partition_of_unity(self) -> None:
        """Test that basis functions form a partition of unity."""
        degree = 3
        knots = [0, 0, 0, 0, 1, 2, 3, 3, 3, 3]
        control_points = [1, 2, 3, 4, 5, 6]

        spline = BSpline(degree, knots, control_points)

        # Test at multiple parameter values
        test_params = [0.0, 0.3, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]

        for u in test_params:
            span = spline.find_knot_span(u)
            basis = spline.basis_functions(u, span)

            # Sum should be 1 (partition of unity)
            assert np.sum(basis) == pytest.approx(1.0, abs=self.NUMERICAL_ATOL)
            # All basis functions should be non-negative
            assert np.all(basis >= -self.NUMERICAL_ATOL)

    def test_basis_function_derivatives(self) -> None:
        """Test basis function derivative calculations."""
        degree = 2
        knots = [0, 0, 0, 1, 2, 2, 2]
        control_points = [1, 2, 3, 4]

        spline = BSpline(degree, knots, control_points)

        u = 0.5
        span = spline.find_knot_span(u)

        # Test derivatives up to degree
        for order in range(degree + 1):
            ders = spline.basis_function_derivatives(u, span, order)

            assert ders.shape == (order + 1, degree + 1)

            # Zero-th derivative should match basis functions
            if order >= 0:
                basis = spline.basis_functions(u, span)
                assert np.allclose(ders[0], basis, atol=self.NUMERICAL_ATOL)

    def test_basis_function_derivatives_partition_property(self) -> None:
        """Test that derivatives of basis functions sum correctly."""
        degree = 2
        knots = [0, 0, 0, 1, 2, 3, 3, 3]
        control_points = [1, 2, 3, 4, 5]

        spline = BSpline(degree, knots, control_points)

        u = 1.5
        span = spline.find_knot_span(u)

        # First derivatives should sum to zero
        ders = spline.basis_function_derivatives(u, span, 1)
        first_derivatives_sum = np.sum(ders[1])

        assert abs(first_derivatives_sum) < self.NUMERICAL_ATOL


class TestBSplineEvaluation:
    """Test suite for B-spline curve evaluation."""

    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def test_evaluate_1d_linear(self) -> None:
        """Test evaluation of 1D linear B-spline."""
        degree = 1
        knots = [0, 0, 1, 1]
        control_points = [0, 2]  # Linear from 0 to 2

        spline = BSpline(degree, knots, control_points)

        # Test at known points
        assert spline.evaluate(0.0) == pytest.approx(0.0, abs=self.NUMERICAL_ATOL)
        assert spline.evaluate(0.5) == pytest.approx(1.0, abs=self.NUMERICAL_ATOL)
        assert spline.evaluate(1.0) == pytest.approx(2.0, abs=self.NUMERICAL_ATOL)

    def test_evaluate_2d_curve(self) -> None:
        """Test evaluation of 2D B-spline curve."""
        degree = 2
        knots = [0, 0, 0, 1, 1, 1]
        control_points = [[0, 0], [1, 1], [2, 0]]

        spline = BSpline(degree, knots, control_points)

        # Test at endpoints
        start_point = spline.evaluate(0.0)
        end_point = spline.evaluate(1.0)

        assert np.allclose(start_point, [0, 0], atol=self.NUMERICAL_ATOL)
        assert np.allclose(end_point, [2, 0], atol=self.NUMERICAL_ATOL)

        # Test at midpoint
        mid_point = spline.evaluate(0.5)
        assert len(mid_point) == 2

    def test_evaluate_3d_curve(self) -> None:
        """Test evaluation of 3D B-spline curve."""
        degree = 1
        knots = [0, 0, 1, 2, 2]
        control_points = [[0, 0, 0], [1, 1, 1], [2, 0, 2]]

        spline = BSpline(degree, knots, control_points)

        point = spline.evaluate(0.5)
        assert len(point) == 3
        assert np.all(np.isfinite(point))

    def test_evaluate_endpoint_handling(self) -> None:
        """Test that endpoints are handled correctly."""
        degree = 2
        knots = [0, 0, 0, 1, 1, 1]
        control_points = [[0, 1], [1, 0], [2, 1]]

        spline = BSpline(degree, knots, control_points)

        # Exactly at endpoints
        start = spline.evaluate(spline.u_min)
        end = spline.evaluate(spline.u_max)

        assert np.allclose(start, control_points[0], atol=self.NUMERICAL_ATOL)
        assert np.allclose(end, control_points[-1], atol=self.NUMERICAL_ATOL)

    def test_evaluate_derivative_order_zero(self) -> None:
        """Test that zero-order derivative equals evaluation."""
        degree = 2
        knots = [0, 0, 0, 1, 2, 2, 2]
        control_points = [[0, 0], [1, 1], [2, 0], [3, 1]]

        spline = BSpline(degree, knots, control_points)

        u = 0.7
        point = spline.evaluate(u)
        derivative_0 = spline.evaluate_derivative(u, order=0)

        assert np.allclose(point, derivative_0, atol=self.NUMERICAL_ATOL)

    def test_evaluate_derivative_basic(self) -> None:
        """Test basic derivative evaluation."""
        degree = 2
        knots = [0, 0, 0, 1, 1, 1]
        control_points = [[0, 0], [1, 1], [2, 0]]

        spline = BSpline(degree, knots, control_points)

        # First derivative
        derivative = spline.evaluate_derivative(0.5, order=1)

        assert len(derivative) == 2
        assert np.all(np.isfinite(derivative))

    def test_evaluate_derivative_order_validation(self) -> None:
        """Test that derivative order validation works."""
        degree = 2
        knots = [0, 0, 0, 1, 1, 1]
        control_points = [1, 2, 3]

        spline = BSpline(degree, knots, control_points)

        # Order greater than degree should raise error
        with pytest.raises(ValueError, match="Derivative order .* exceeds B-spline degree"):
            spline.evaluate_derivative(0.5, order=3)

    def test_evaluate_curve_consistency(self) -> None:
        """Test consistency between different evaluation methods."""
        degree = 2
        knots = [0, 0, 0, 1, 2, 2, 2]
        control_points = [1, 2, 3, 4]

        spline = BSpline(degree, knots, control_points)

        # Test finite difference approximation of derivative
        u = 1.0
        h = 1e-6

        # Approximate first derivative
        f_plus = spline.evaluate(u + h)
        f_minus = spline.evaluate(u - h)
        derivative_approx = (f_plus - f_minus) / (2 * h)

        derivative_exact = spline.evaluate_derivative(u, order=1)

        # Should be close for smooth curves
        assert abs(derivative_approx - derivative_exact) < 1e-4


class TestBSplineEdgeCases:
    """Test suite for edge cases and special situations."""

    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def test_single_control_point(self) -> None:
        """Test B-spline with single control point (degree 0)."""
        degree = 0
        knots = [0, 1]
        control_points = [5.0]

        spline = BSpline(degree, knots, control_points)

        # Should be constant
        assert spline.evaluate(0.0) == pytest.approx(5.0, abs=self.NUMERICAL_ATOL)
        assert spline.evaluate(0.5) == pytest.approx(5.0, abs=self.NUMERICAL_ATOL)
        assert spline.evaluate(1.0) == pytest.approx(5.0, abs=self.NUMERICAL_ATOL)

    def test_identical_control_points(self) -> None:
        """Test B-spline with identical control points."""
        degree = 2
        knots = [0, 0, 0, 1, 1, 1]
        control_points = [[1, 2], [1, 2], [1, 2]]  # All identical

        spline = BSpline(degree, knots, control_points)

        # Should be constant curve
        for u in [0.0, 0.3, 0.7, 1.0]:
            point = spline.evaluate(u)
            assert np.allclose(point, [1, 2], atol=self.NUMERICAL_ATOL)

    def test_repeated_knots(self) -> None:
        """Test B-spline with repeated internal knots."""
        degree = 2
        knots = [0, 0, 0, 0.5, 0.5, 1, 1, 1]
        control_points = [1, 2, 3, 4, 5]

        spline = BSpline(degree, knots, control_points)

        # Should handle repeated knots gracefully
        point = spline.evaluate(0.5)
        assert np.isfinite(point)

    def test_high_precision_evaluation(self) -> None:
        """Test evaluation with high precision requirements."""
        degree = 3
        knots = [0, 0, 0, 0, 1, 1, 1, 1]
        control_points = [[0, 0], [1 / 3, 1], [2 / 3, 1], [1, 0]]

        spline = BSpline(degree, knots, control_points)

        # Test evaluation at rational points
        u = 1 / 3
        point = spline.evaluate(u)

        assert np.all(np.isfinite(point))
        assert not np.any(np.isnan(point))

    def test_parameter_clamping(self) -> None:
        """Test that parameters are correctly clamped to valid range."""
        degree = 1
        knots = [0, 0, 1, 1]
        control_points = [[0, 0], [1, 1]]

        spline = BSpline(degree, knots, control_points)

        # These should work due to clamping (within epsilon)
        start_point = spline.evaluate(spline.u_min - spline.eps / 2)
        end_point = spline.evaluate(spline.u_max + spline.eps / 2)

        assert np.allclose(start_point, [0, 0], atol=self.NUMERICAL_ATOL)
        assert np.allclose(end_point, [1, 1], atol=self.NUMERICAL_ATOL)


class TestBSplineCurveGeneration:
    """Test suite for curve point generation and sampling."""

    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def test_generate_curve_points_basic(self) -> None:
        """Test basic curve point generation."""
        degree = 2
        knots = [0, 0, 0, 1, 1, 1]
        control_points = [[0, 0], [1, 1], [2, 0]]

        spline = BSpline(degree, knots, control_points)

        u_values, curve_points = spline.generate_curve_points(num_points=10)

        assert len(u_values) == 10
        assert curve_points.shape == (10, 2)
        assert u_values[0] == spline.u_min
        assert u_values[-1] == spline.u_max

    def test_generate_curve_points_1d(self) -> None:
        """Test curve point generation for 1D case."""
        degree = 1
        knots = [0, 0, 1, 1]
        control_points = [0, 5]

        spline = BSpline(degree, knots, control_points)

        u_values, curve_points = spline.generate_curve_points(num_points=5)

        assert len(u_values) == 5
        assert curve_points.shape == (5, 1)

    def test_generate_curve_points_3d(self) -> None:
        """Test curve point generation for 3D case."""
        degree = 1
        knots = [0, 0, 1, 2, 2]
        control_points = [[0, 0, 0], [1, 1, 1], [2, 0, 2]]

        spline = BSpline(degree, knots, control_points)

        u_values, curve_points = spline.generate_curve_points(num_points=6)

        assert len(u_values) == 6
        assert curve_points.shape == (6, 3)

    def test_generate_curve_points_different_counts(self) -> None:
        """Test curve point generation with different point counts."""
        degree = 2
        knots = [0, 0, 0, 1, 1, 1]
        control_points = [[0, 0], [1, 1], [2, 0]]

        spline = BSpline(degree, knots, control_points)

        for num_points in [5, 50, 100]:
            u_values, curve_points = spline.generate_curve_points(num_points=num_points)

            assert len(u_values) == num_points
            assert curve_points.shape[0] == num_points
            assert curve_points.shape[1] == 2


class TestBSplinePlotting:
    """Test suite for plotting functionality."""

    def test_plot_2d_basic(self) -> None:
        """Test basic 2D plotting functionality."""
        degree = 2
        knots = [0, 0, 0, 1, 1, 1]
        control_points = [[0, 0], [1, 2], [2, 0]]

        spline = BSpline(degree, knots, control_points)

        try:
            plt.ioff()
            ax = spline.plot_2d()
            assert ax is not None
            plt.close("all")
        except Exception as e:
            pytest.fail(f"2D plot method raised exception: {e}")

    def test_plot_2d_dimension_validation(self) -> None:
        """Test that 2D plotting validates dimension."""
        degree = 1
        knots = [0, 0, 1, 1]
        control_points = [1, 2]  # 1D control points

        spline = BSpline(degree, knots, control_points)

        with pytest.raises(ValueError, match="Control points must be 2D"):
            spline.plot_2d()

    def test_plot_2d_custom_options(self) -> None:
        """Test 2D plotting with custom options."""
        degree = 1
        knots = [0, 0, 1, 1]
        control_points = [[0, 0], [1, 1]]

        spline = BSpline(degree, knots, control_points)

        try:
            plt.ioff()
            ax = spline.plot_2d(num_points=50, show_control_polygon=True, show_knots=True)
            assert ax is not None
            plt.close("all")
        except Exception as e:
            pytest.fail(f"2D plot with options raised exception: {e}")

    def test_plot_3d_basic(self) -> None:
        """Test basic 3D plotting functionality."""
        degree = 1
        knots = [0, 0, 1, 1]
        control_points = [[0, 0, 0], [1, 1, 1]]

        spline = BSpline(degree, knots, control_points)

        try:
            plt.ioff()
            ax = spline.plot_3d()
            assert ax is not None
            plt.close("all")
        except Exception as e:
            pytest.fail(f"3D plot method raised exception: {e}")

    def test_plot_3d_dimension_validation(self) -> None:
        """Test that 3D plotting validates dimension."""
        degree = 1
        knots = [0, 0, 1, 1]
        control_points = [[0, 0], [1, 1]]  # 2D control points

        spline = BSpline(degree, knots, control_points)

        with pytest.raises(ValueError, match="Control points must be 3D"):
            spline.plot_3d()


class TestBSplineStringRepresentation:
    """Test suite for string representation."""

    def test_repr_basic(self) -> None:
        """Test basic string representation."""
        degree = 2
        knots = [0, 0, 0, 1, 1, 1]
        control_points = [[0, 0], [1, 1], [2, 0]]

        spline = BSpline(degree, knots, control_points)

        repr_str = repr(spline)

        assert "BSpline" in repr_str
        assert "degree=2" in repr_str
        assert "control_points=3" in repr_str
        assert "dimension=2" in repr_str


class TestBSplineNumericalStability:
    """Test suite for numerical stability."""

    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def test_nearly_coincident_knots(self) -> None:
        """Test stability with nearly coincident knots."""
        degree = 2
        eps = 1e-12
        knots = [0, 0, 0, 0.5, 0.5 + eps, 1, 1, 1]
        control_points = [1, 2, 3, 4, 5]

        spline = BSpline(degree, knots, control_points)

        # Should handle without throwing exceptions
        point = spline.evaluate(0.5)
        assert np.isfinite(point)

    def test_extreme_parameter_values(self) -> None:
        """Test with extreme parameter values."""
        degree = 1
        knots = [0, 0, 1e6, 1e6]
        control_points = [[0, 0], [1e6, 1e6]]

        spline = BSpline(degree, knots, control_points)

        # Should handle large parameter values
        point = spline.evaluate(5e5)
        assert np.all(np.isfinite(point))

    def test_high_degree_stability(self) -> None:
        """Test numerical stability with high degree."""
        degree = 7
        n_control = 10
        knots = BSpline.create_uniform_knots(degree, n_control)
        control_points = [[i, np.sin(i)] for i in range(n_control)]

        spline = BSpline(degree, knots, control_points)

        # Should evaluate without numerical issues
        test_params = np.linspace(spline.u_min, spline.u_max, 20)
        for u in test_params:
            point = spline.evaluate(u)
            assert np.all(np.isfinite(point))


class TestBSplinePerformance:
    """Test suite for performance benchmarks."""

    @pytest.mark.parametrize("degree", [1, 2, 3, 5])
    def test_construction_performance(self, degree: int, benchmark: pytest.FixtureFunction) -> None:
        """Benchmark B-spline construction performance."""
        n_control = 20
        knots = BSpline.create_uniform_knots(degree, n_control)
        control_points = [[i, np.sin(i)] for i in range(n_control)]

        def construct_spline() -> BSpline:
            return BSpline(degree, knots, control_points)

        spline = benchmark(construct_spline)
        assert spline.degree == degree

    @pytest.mark.parametrize("n_evaluations", [100, 1000])
    def test_evaluation_performance(
        self, n_evaluations: int, benchmark: pytest.FixtureFunction
    ) -> None:
        """Benchmark B-spline evaluation performance."""
        degree = 3
        n_control = 10
        knots = BSpline.create_uniform_knots(degree, n_control)
        control_points = [[i, np.sin(i)] for i in range(n_control)]

        spline = BSpline(degree, knots, control_points)
        u_values = np.linspace(spline.u_min, spline.u_max, n_evaluations)

        def evaluate_spline() -> list[np.ndarray]:
            return [spline.evaluate(u) for u in u_values]

        results = benchmark(evaluate_spline)
        assert len(results) == n_evaluations

    def test_basis_function_performance(self, benchmark: pytest.FixtureFunction) -> None:
        """Benchmark basis function calculation performance."""
        degree = 4
        n_control = 15
        knots = BSpline.create_uniform_knots(degree, n_control)
        control_points = [[i, i**2] for i in range(n_control)]

        spline = BSpline(degree, knots, control_points)
        u_values = np.linspace(spline.u_min, spline.u_max, 100)

        def compute_basis_functions() -> list[np.ndarray]:
            results = []
            for u in u_values:
                span = spline.find_knot_span(u)
                basis = spline.basis_functions(u, span)
                results.append(basis)
            return results

        results = benchmark(compute_basis_functions)
        assert len(results) == 100


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main(["-xvs", __file__])
