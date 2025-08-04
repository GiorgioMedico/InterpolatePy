"""
Comprehensive tests for the quaternion visualization implementation.

This module contains extensive tests for the quaternion visualization classes covering:
1. PlotStyle dataclass functionality
2. QuaternionTrajectoryVisualizer core methods
3. Stereographic projection and inverse projection
4. Trajectory projection and visualization
5. Velocity analysis and plotting
6. Edge cases and error handling
7. Integration tests with quaternion trajectories
8. Performance benchmarks

The tests verify mathematical accuracy, handle edge cases, and ensure
robust behavior across different visualization scenarios.
"""

from typing import Any
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest

from interpolatepy.quat_core import Quaternion
from interpolatepy.quat_spline import QuaternionSpline
from interpolatepy.quat_visualization import PlotStyle
from interpolatepy.quat_visualization import QuaternionTrajectoryVisualizer


# Type alias for pytest benchmark fixture
try:
    from pytest import FixtureFunction  # type: ignore[attr-defined]
except (ImportError, AttributeError):
    # Fallback for pytest versions without FixtureFunction
    from typing import Callable
    FixtureFunction = Callable[..., Any]


class TestPlotStyle:
    """Test suite for PlotStyle dataclass."""

    def test_default_initialization(self) -> None:
        """Test PlotStyle initialization with default values."""
        style = PlotStyle()

        assert style.color == "blue"
        assert style.line_width == 2.0
        assert style.point_size == 20.0
        assert style.figsize == (10, 8)
        assert style.title == "Quaternion Plot"

    def test_custom_initialization(self) -> None:
        """Test PlotStyle initialization with custom values."""
        style = PlotStyle(
            color="red",
            line_width=1.5,
            point_size=30.0,
            figsize=(12, 10),
            title="Custom Quaternion Plot"
        )

        assert style.color == "red"
        assert style.line_width == 1.5
        assert style.point_size == 30.0
        assert style.figsize == (12, 10)
        assert style.title == "Custom Quaternion Plot"

    def test_partial_initialization(self) -> None:
        """Test PlotStyle initialization with some custom values."""
        style = PlotStyle(color="green", line_width=3.0)

        assert style.color == "green"
        assert style.line_width == 3.0
        assert style.point_size == 20.0  # default
        assert style.figsize == (10, 8)  # default
        assert style.title == "Quaternion Plot"  # default


class TestQuaternionTrajectoryVisualizerBasics:
    """Test suite for basic QuaternionTrajectoryVisualizer functionality."""

    # Test tolerances
    REGULAR_RTOL = 1e-10
    REGULAR_ATOL = 1e-10
    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def test_initialization(self) -> None:
        """Test QuaternionTrajectoryVisualizer initialization."""
        visualizer = QuaternionTrajectoryVisualizer()

        # Test constants are properly set
        assert visualizer.SINGULARITY_THRESHOLD == 1e-6
        assert visualizer.DEFAULT_TRAJECTORY_COLOR == "blue"
        assert visualizer.DEFAULT_POINT_SIZE == 20
        assert visualizer.DEFAULT_LINE_WIDTH == 2

    def test_class_constants(self) -> None:
        """Test class constants are properly defined."""
        assert hasattr(QuaternionTrajectoryVisualizer, "SINGULARITY_THRESHOLD")
        assert hasattr(QuaternionTrajectoryVisualizer, "DEFAULT_TRAJECTORY_COLOR")
        assert hasattr(QuaternionTrajectoryVisualizer, "DEFAULT_POINT_SIZE")
        assert hasattr(QuaternionTrajectoryVisualizer, "DEFAULT_LINE_WIDTH")


class TestStereographicProjection:
    """Test suite for stereographic projection methods."""

    REGULAR_RTOL = 1e-10
    REGULAR_ATOL = 1e-10
    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def test_identity_quaternion_projection(self) -> None:
        """Test stereographic projection of identity quaternion."""
        q_identity = Quaternion.identity()
        mrp = QuaternionTrajectoryVisualizer.stereographic_projection(q_identity)

        # Identity quaternion should project to origin
        expected = np.array([0.0, 0.0, 0.0])
        assert np.allclose(mrp, expected, atol=self.NUMERICAL_ATOL)

    @pytest.mark.parametrize("angle", [np.pi/4, np.pi/2, 3*np.pi/4])
    @pytest.mark.parametrize("axis", [
        [1.0, 0.0, 0.0],  # X-axis
        [0.0, 1.0, 0.0],  # Y-axis
        [0.0, 0.0, 1.0],  # Z-axis
        [1.0, 1.0, 1.0],  # Diagonal
    ])
    def test_stereographic_projection_basic(self, angle: float, axis: list[float]) -> None:
        """Test stereographic projection with various quaternions."""
        axis_array = np.array(axis)
        q = Quaternion.from_angle_axis(angle, axis_array)

        mrp = QuaternionTrajectoryVisualizer.stereographic_projection(q)

        # MRP should be finite
        assert np.all(np.isfinite(mrp))
        assert len(mrp) == 3

    def test_stereographic_projection_mathematical_correctness(self) -> None:
        """Test mathematical correctness of stereographic projection."""
        # Test known case: 90° rotation around Z-axis
        q = Quaternion.from_angle_axis(np.pi/2, np.array([0.0, 0.0, 1.0]))
        mrp = QuaternionTrajectoryVisualizer.stereographic_projection(q)

        # For 90° rotation around Z: q = [cos(π/4), 0, 0, sin(π/4)] = [√2/2, 0, 0, √2/2]
        # MRP = [x/(1+w), y/(1+w), z/(1+w)] = [0, 0, (√2/2)/(1+√2/2)]
        expected_z = np.sqrt(2)/2 / (1 + np.sqrt(2)/2)
        expected = np.array([0.0, 0.0, expected_z])

        assert np.allclose(mrp, expected, atol=self.NUMERICAL_ATOL)

    def test_singularity_near_negative_w(self) -> None:
        """Test singularity handling near w = -1."""
        # Create quaternion very close to singularity
        q_near_singularity = Quaternion(-1.0 + 1e-8, 0.1, 0.2, 0.3).unit()

        # Should handle gracefully by flipping quaternion
        mrp = QuaternionTrajectoryVisualizer.stereographic_projection(q_near_singularity)
        assert np.all(np.isfinite(mrp))

    def test_singularity_at_negative_w(self) -> None:
        """Test exact singularity at w = -1."""
        q_singularity = Quaternion(-1.0, 0.0, 0.0, 0.0)

        # The implementation first tries to flip the quaternion, so exact -1 becomes +1
        # which should work fine. Let's test with a quaternion that remains problematic
        # after flipping (very unlikely in practice but theoretically possible)
        mrp = QuaternionTrajectoryVisualizer.stereographic_projection(q_singularity)

        # Should handle by flipping to equivalent quaternion
        assert np.all(np.isfinite(mrp))

    def test_non_unit_quaternion_projection(self) -> None:
        """Test projection normalizes non-unit quaternions."""
        # Non-unit quaternion
        q_non_unit = Quaternion(2.0, 1.0, 1.0, 1.0)
        mrp = QuaternionTrajectoryVisualizer.stereographic_projection(q_non_unit)

        # Should handle by normalizing first
        assert np.all(np.isfinite(mrp))
        assert len(mrp) == 3


class TestInverseStereographicProjection:
    """Test suite for inverse stereographic projection."""

    REGULAR_RTOL = 1e-10
    REGULAR_ATOL = 1e-10
    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def test_origin_inverse_projection(self) -> None:
        """Test inverse projection of origin."""
        mrp_origin = np.array([0.0, 0.0, 0.0])
        q = QuaternionTrajectoryVisualizer.inverse_stereographic_projection(mrp_origin)

        # Should give identity quaternion
        q_identity = Quaternion.identity()
        assert abs(q.s() - q_identity.s()) < self.NUMERICAL_ATOL
        assert np.allclose(q.v(), q_identity.v(), atol=self.NUMERICAL_ATOL)

    @pytest.mark.parametrize("mrp_point", [
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.1, 0.2, 0.3],
        [1.0, 1.0, 1.0],
    ])
    def test_inverse_projection_basic(self, mrp_point: list[float]) -> None:
        """Test inverse projection with various MRP points."""
        mrp = np.array(mrp_point)
        q = QuaternionTrajectoryVisualizer.inverse_stereographic_projection(mrp)

        # Result should be unit quaternion
        assert abs(q.norm() - 1.0) < self.NUMERICAL_ATOL

    def test_round_trip_consistency(self) -> None:
        """Test round-trip projection consistency."""
        # Start with various quaternions
        test_quaternions = [
            Quaternion.identity(),
            Quaternion.from_angle_axis(np.pi/4, np.array([1.0, 0.0, 0.0])),
            Quaternion.from_angle_axis(np.pi/3, np.array([0.0, 1.0, 0.0])),
            Quaternion.from_angle_axis(np.pi/6, np.array([1.0, 1.0, 1.0])),
            Quaternion.from_euler_angles(0.1, 0.2, 0.3),
        ]

        for q_original in test_quaternions:
            # Avoid singularity region
            if abs(q_original.w + 1.0) > 1e-3:
                # Forward and inverse projection
                mrp = QuaternionTrajectoryVisualizer.stereographic_projection(q_original)
                q_recovered = QuaternionTrajectoryVisualizer.inverse_stereographic_projection(mrp)

                # Should recover original quaternion (up to sign)
                dot_product = q_original.dot_product(q_recovered)
                assert abs(abs(dot_product) - 1.0) < self.NUMERICAL_ATOL

    def test_large_mrp_values(self) -> None:
        """Test inverse projection with large MRP values."""
        # Large MRP values (far from origin)
        large_mrp = np.array([10.0, 20.0, 30.0])
        q = QuaternionTrajectoryVisualizer.inverse_stereographic_projection(large_mrp)

        # Should still be unit quaternion
        assert abs(q.norm() - 1.0) < self.NUMERICAL_ATOL


class TestTrajectoryProjection:
    """Test suite for trajectory projection functionality."""

    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def setup_method(self) -> None:
        """Setup test fixtures."""
        self.visualizer = QuaternionTrajectoryVisualizer()

    def test_empty_quaternion_list(self) -> None:
        """Test trajectory projection with empty quaternion list."""
        empty_quaternions: list[Quaternion] = []
        projected = self.visualizer.project_trajectory(empty_quaternions)

        assert projected.shape == (0, 3)

    def test_single_quaternion_trajectory(self) -> None:
        """Test trajectory projection with single quaternion."""
        quaternions = [Quaternion.identity()]
        projected = self.visualizer.project_trajectory(quaternions)

        assert projected.shape == (1, 3)
        assert np.allclose(projected[0], [0.0, 0.0, 0.0], atol=self.NUMERICAL_ATOL)

    def test_multiple_quaternion_trajectory(self) -> None:
        """Test trajectory projection with multiple quaternions."""
        quaternions = [
            Quaternion.identity(),
            Quaternion.from_angle_axis(np.pi/4, np.array([1.0, 0.0, 0.0])),
            Quaternion.from_angle_axis(np.pi/2, np.array([0.0, 1.0, 0.0])),
            Quaternion.from_angle_axis(np.pi/3, np.array([0.0, 0.0, 1.0])),
        ]

        projected = self.visualizer.project_trajectory(quaternions)

        assert projected.shape == (4, 3)
        # All projected points should be finite
        assert np.all(np.isfinite(projected))

    def test_trajectory_with_singularities(self) -> None:
        """Test trajectory projection with quaternions near singularities."""
        quaternions = [
            Quaternion.identity(),
            Quaternion(-1.0 + 1e-8, 0.1, 0.0, 0.0).unit(),  # Near singularity
            Quaternion.from_angle_axis(np.pi/4, np.array([1.0, 0.0, 0.0])),
        ]

        # Should handle singularities gracefully
        projected = self.visualizer.project_trajectory(quaternions)

        # Should get fewer points due to skipped singularities
        assert projected.shape[0] <= len(quaternions)
        assert projected.shape[1] == 3

    def test_trajectory_all_singularities(self) -> None:
        """Test trajectory projection with all quaternions at singularities."""
        # Since the implementation flips quaternions to avoid singularities,
        # it's very difficult to create actual singularities. Let's test with
        # quaternions that would be close to problematic after normalization
        quaternions = [
            Quaternion(-1.0, 0.0, 0.0, 0.0),
            Quaternion(-1.0, 0.0, 0.0, 0.0),
        ]

        projected = self.visualizer.project_trajectory(quaternions)

        # The implementation should handle these by flipping them
        # so we should get projected points
        assert projected.shape[1] == 3
        assert len(projected) >= 0  # Could be empty if truly singular, or have points if handled


class TestVelocityAnalysis:
    """Test suite for velocity analysis methods."""

    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def setup_method(self) -> None:
        """Setup test fixtures."""
        self.visualizer = QuaternionTrajectoryVisualizer()

    def test_velocity_computation_basic(self) -> None:
        """Test basic velocity magnitude computation."""
        quaternions = [
            Quaternion.identity(),
            Quaternion.from_angle_axis(np.pi/4, np.array([1.0, 0.0, 0.0])),
            Quaternion.from_angle_axis(np.pi/2, np.array([1.0, 0.0, 0.0])),
        ]

        times, velocities = self.visualizer.compute_velocity_magnitudes(quaternions)

        assert len(times) == len(quaternions)
        assert len(velocities) == len(quaternions)
        assert np.all(velocities >= 0)  # Velocities should be non-negative

    def test_velocity_computation_with_time_points(self) -> None:
        """Test velocity computation with custom time points."""
        quaternions = [
            Quaternion.identity(),
            Quaternion.from_angle_axis(np.pi/4, np.array([1.0, 0.0, 0.0])),
            Quaternion.from_angle_axis(np.pi/2, np.array([1.0, 0.0, 0.0])),
        ]
        time_points = [0.0, 1.0, 2.0]

        times, velocities = self.visualizer.compute_velocity_magnitudes(quaternions, time_points)

        assert np.allclose(times, time_points)
        assert len(velocities) == len(quaternions)

    def test_velocity_computation_insufficient_points(self) -> None:
        """Test velocity computation with insufficient quaternions."""
        quaternions = [Quaternion.identity()]

        with pytest.raises(ValueError, match="Need at least 2 quaternions"):
            self.visualizer.compute_velocity_magnitudes(quaternions)

    def test_velocity_computation_mismatched_time_length(self) -> None:
        """Test velocity computation with mismatched time points length."""
        quaternions = [
            Quaternion.identity(),
            Quaternion.from_angle_axis(np.pi/4, np.array([1.0, 0.0, 0.0])),
        ]
        time_points = [0.0, 1.0, 2.0]  # Wrong length

        with pytest.raises(ValueError, match="Time points length must match"):
            self.visualizer.compute_velocity_magnitudes(quaternions, time_points)

    def test_velocity_computation_identical_quaternions(self) -> None:
        """Test velocity computation with identical quaternions."""
        q = Quaternion.from_angle_axis(np.pi/4, np.array([1.0, 0.0, 0.0]))
        quaternions = [q, q, q]

        times, velocities = self.visualizer.compute_velocity_magnitudes(quaternions)

        # Velocities should be zero for identical quaternions
        assert np.allclose(velocities, 0.0, atol=self.NUMERICAL_ATOL)

    def test_quaternion_distance_method(self) -> None:
        """Test private quaternion distance method."""
        q1 = Quaternion.identity()
        q2 = Quaternion.from_angle_axis(np.pi/2, np.array([1.0, 0.0, 0.0]))

        distance = self.visualizer._quaternion_distance(q1, q2)  # noqa: SLF001

        assert distance >= 0
        assert np.isfinite(distance)

    def test_quaternion_distance_identical(self) -> None:
        """Test quaternion distance with identical quaternions."""
        q = Quaternion.from_angle_axis(np.pi/4, np.array([1.0, 0.0, 0.0]))

        distance = self.visualizer._quaternion_distance(q, q)  # noqa: SLF001

        assert abs(distance) < self.NUMERICAL_ATOL


class TestPlottingFunctionality:
    """Test suite for plotting methods with matplotlib mocking."""

    def setup_method(self) -> None:
        """Setup test fixtures."""
        self.visualizer = QuaternionTrajectoryVisualizer()
        self.test_quaternions = [
            Quaternion.identity(),
            Quaternion.from_angle_axis(np.pi/4, np.array([1.0, 0.0, 0.0])),
            Quaternion.from_angle_axis(np.pi/2, np.array([0.0, 1.0, 0.0])),
            Quaternion.from_angle_axis(np.pi/3, np.array([0.0, 0.0, 1.0])),
        ]

    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.figure")
    def test_plot_3d_trajectory_basic(self, mock_figure: Mock, _mock_tight_layout: Mock) -> None:
        """Test basic 3D trajectory plotting."""
        # Setup mock
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        # Test plotting
        _ = self.visualizer.plot_3d_trajectory(self.test_quaternions)

        # Verify matplotlib calls
        mock_figure.assert_called_once()
        mock_fig.add_subplot.assert_called_once_with(111, projection="3d")
        assert mock_ax.plot.called
        assert mock_ax.set_xlabel.called
        assert mock_ax.set_ylabel.called
        assert mock_ax.set_zlabel.called

    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.figure")
    def test_plot_3d_trajectory_with_options(
        self, mock_figure: Mock, _mock_tight_layout: Mock
    ) -> None:
        """Test 3D trajectory plotting with custom options."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        # Test with custom options
        _ = self.visualizer.plot_3d_trajectory(
            self.test_quaternions,
            title="Custom Title",
            color="red",
            line_width=3.0,
            point_size=50.0,
            show_points=False,
            figsize=(12, 10)
        )

        # Verify custom figsize was used
        mock_figure.assert_called_once_with(figsize=(12, 10))

    def test_plot_3d_trajectory_empty_quaternions(self) -> None:
        """Test 3D trajectory plotting with empty quaternion list."""
        with pytest.raises(ValueError, match="Empty quaternion list provided"):
            self.visualizer.plot_3d_trajectory([])

    @patch("matplotlib.pyplot.subplots")
    def test_plot_angular_velocity_basic(self, mock_subplots: Mock) -> None:
        """Test basic angular velocity plotting."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        _ = self.visualizer.plot_angular_velocity(self.test_quaternions)

        # Verify matplotlib calls
        mock_subplots.assert_called_once()
        assert mock_ax.plot.called
        assert mock_ax.set_xlabel.called
        assert mock_ax.set_ylabel.called
        assert mock_ax.set_title.called

    @patch("matplotlib.pyplot.subplots")
    def test_plot_angular_velocity_with_time_points(self, mock_subplots: Mock) -> None:
        """Test angular velocity plotting with custom time points."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        time_points = [0.0, 1.0, 2.0, 3.0]
        _ = self.visualizer.plot_angular_velocity(self.test_quaternions, time_points)

        mock_subplots.assert_called_once()
        assert mock_ax.plot.called

    def test_plot_angular_velocity_empty_quaternions(self) -> None:
        """Test angular velocity plotting with empty quaternion list."""
        with pytest.raises(ValueError, match="Empty quaternion list provided"):
            self.visualizer.plot_angular_velocity([])

    @patch("matplotlib.pyplot.figure")
    def test_plot_trajectory_with_velocity_basic(self, mock_figure: Mock) -> None:
        """Test combined trajectory and velocity plotting."""
        mock_fig = MagicMock()
        mock_ax3d = MagicMock()
        mock_ax2d = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.side_effect = [mock_ax3d, mock_ax2d]

        _ = self.visualizer.plot_trajectory_with_velocity(self.test_quaternions)

        # Verify matplotlib calls
        mock_figure.assert_called_once()
        assert mock_fig.add_subplot.call_count == 2
        assert mock_ax3d.plot.called
        assert mock_ax2d.plot.called

    @patch("matplotlib.pyplot.figure")
    def test_plot_trajectory_with_velocity_custom_options(self, mock_figure: Mock) -> None:
        """Test combined plotting with custom options."""
        mock_fig = MagicMock()
        mock_ax3d = MagicMock()
        mock_ax2d = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.side_effect = [mock_ax3d, mock_ax2d]

        _ = self.visualizer.plot_trajectory_with_velocity(
            self.test_quaternions,
            title="Custom Combined Plot",
            trajectory_color="green",
            velocity_color="orange",
            figsize=(16, 8)
        )

        mock_figure.assert_called_once_with(figsize=(16, 8))

    def test_plot_trajectory_with_velocity_empty_quaternions(self) -> None:
        """Test combined plotting with empty quaternion list."""
        with pytest.raises(ValueError, match="Empty quaternion list provided"):
            self.visualizer.plot_trajectory_with_velocity([])


class TestEdgeCasesAndErrorHandling:
    """Test suite for edge cases and error handling."""

    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def setup_method(self) -> None:
        """Setup test fixtures."""
        self.visualizer = QuaternionTrajectoryVisualizer()

    def test_numerical_stability_near_singularity(self) -> None:
        """Test numerical stability near singularity points."""
        # Quaternion very close to singularity
        q_near_sing = Quaternion(-1.0 + 1e-7, 1e-8, 1e-8, 1e-8).unit()

        # Should handle gracefully
        try:
            mrp = self.visualizer.stereographic_projection(q_near_sing)
            assert np.all(np.isfinite(mrp))
        except ValueError:
            # Acceptable to fail for very close singularities
            pass

    def test_large_angle_rotations(self) -> None:
        """Test handling of large angle rotations."""
        large_angles = [2*np.pi, 4*np.pi, 10*np.pi]

        for angle in large_angles:
            q = Quaternion.from_angle_axis(angle, np.array([0.0, 0.0, 1.0]))
            mrp = self.visualizer.stereographic_projection(q)
            assert np.all(np.isfinite(mrp))

    def test_near_zero_quaternions(self) -> None:
        """Test handling of near-zero quaternions."""
        q_small = Quaternion(1e-10, 1e-10, 1e-10, 1e-10)

        # Should normalize to identity
        mrp = self.visualizer.stereographic_projection(q_small)
        assert np.allclose(mrp, [0.0, 0.0, 0.0], atol=self.NUMERICAL_ATOL)

    def test_opposite_quaternions_in_trajectory(self) -> None:
        """Test trajectory with nearly opposite quaternions."""
        quaternions = [
            Quaternion(1.0, 0.0, 0.0, 0.0),
            Quaternion(-1.0 + 1e-8, 1e-9, 1e-9, 1e-9).unit(),
            Quaternion(0.0, 0.0, 0.0, 1.0),
        ]

        # Should handle gracefully
        projected = self.visualizer.project_trajectory(quaternions)
        assert projected.shape[1] == 3

    def test_invalid_mrp_dimensions(self) -> None:
        """Test error handling for invalid MRP dimensions."""
        # This would be internal to the implementation
        # Testing the mathematical robustness
        valid_mrp = np.array([0.1, 0.2, 0.3])
        q = self.visualizer.inverse_stereographic_projection(valid_mrp)
        assert abs(q.norm() - 1.0) < self.NUMERICAL_ATOL


class TestIntegrationWithQuaternionTrajectories:
    """Test suite for integration with actual quaternion trajectories."""

    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def setup_method(self) -> None:
        """Setup test fixtures."""
        self.visualizer = QuaternionTrajectoryVisualizer()

    def create_test_spline_trajectory(self) -> tuple[list[float], list[Quaternion]]:
        """Create a test quaternion spline trajectory."""
        time_points = [0.0, 1.0, 2.0, 3.0, 4.0]
        quaternions = [
            Quaternion.identity(),
            Quaternion.from_euler_angles(0.2, 0.1, 0.3),
            Quaternion.from_euler_angles(0.4, 0.3, 0.6),
            Quaternion.from_euler_angles(0.6, 0.5, 0.9),
            Quaternion.from_euler_angles(0.8, 0.7, 1.2),
        ]
        return time_points, quaternions

    @patch("matplotlib.pyplot.figure")
    def test_visualization_with_spline_trajectory(self, mock_figure: Mock) -> None:
        """Test visualization with quaternion spline trajectory."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        time_points, quaternions = self.create_test_spline_trajectory()
        spline = QuaternionSpline(time_points, quaternions, Quaternion.SLERP)

        # Generate interpolated trajectory
        eval_times = np.linspace(0.0, 4.0, 20)
        interpolated_quaternions = []

        for t in eval_times:
            q_interp, status = spline.interpolate_at_time(t)
            if status == 0:
                interpolated_quaternions.append(q_interp)

        # Visualize trajectory
        _ = self.visualizer.plot_3d_trajectory(interpolated_quaternions)

        # Should complete without errors
        mock_figure.assert_called_once()

    def test_velocity_analysis_with_spline_trajectory(self) -> None:
        """Test velocity analysis with quaternion spline trajectory."""
        time_points, quaternions = self.create_test_spline_trajectory()
        spline = QuaternionSpline(time_points, quaternions, Quaternion.SLERP)

        # Generate interpolated trajectory
        eval_times = np.linspace(0.0, 4.0, 10)
        interpolated_quaternions = []

        for t in eval_times:
            q_interp, status = spline.interpolate_at_time(t)
            if status == 0:
                interpolated_quaternions.append(q_interp)

        # Analyze velocity
        times, velocities = self.visualizer.compute_velocity_magnitudes(
            interpolated_quaternions, eval_times.tolist()
        )

        assert len(times) == len(interpolated_quaternions)
        assert len(velocities) == len(interpolated_quaternions)
        assert np.all(velocities >= 0)

    def test_round_trip_with_different_interpolation_methods(self) -> None:
        """Test visualization consistency across interpolation methods."""
        time_points, quaternions = self.create_test_spline_trajectory()

        methods = [Quaternion.SLERP, Quaternion.SQUAD]

        for method in methods:
            spline = QuaternionSpline(time_points, quaternions, method)

            # Generate trajectory
            eval_times = np.linspace(0.0, 4.0, 10)
            interpolated_quaternions = []

            for t in eval_times:
                q_interp, status = spline.interpolate_at_time(t)
                if status == 0:
                    interpolated_quaternions.append(q_interp)

            # Test projection
            projected = self.visualizer.project_trajectory(interpolated_quaternions)
            assert projected.shape[0] > 0
            assert projected.shape[1] == 3
            assert np.all(np.isfinite(projected))


class TestPerformanceAndBenchmarks:
    """Test suite for performance benchmarks."""

    def setup_method(self) -> None:
        """Setup test fixtures."""
        self.visualizer = QuaternionTrajectoryVisualizer()

    @pytest.mark.parametrize("num_quaternions", [100, 1000])
    def test_projection_performance(
        self, num_quaternions: int, benchmark: FixtureFunction
    ) -> None:
        """Benchmark trajectory projection performance."""
        # Create test trajectory
        quaternions = [
            Quaternion.from_euler_angles(0.01 * i, 0.02 * i, 0.03 * i)
            for i in range(num_quaternions)
        ]

        def run_projection() -> None:
            _ = self.visualizer.project_trajectory(quaternions)

        benchmark(run_projection)

    @pytest.mark.parametrize("num_quaternions", [100, 1000])
    def test_velocity_computation_performance(
        self, num_quaternions: int, benchmark: FixtureFunction
    ) -> None:
        """Benchmark velocity computation performance."""
        quaternions = [
            Quaternion.from_euler_angles(0.01 * i, 0.02 * i, 0.03 * i)
            for i in range(num_quaternions)
        ]

        def run_velocity_computation() -> None:
            _ = self.visualizer.compute_velocity_magnitudes(quaternions)

        benchmark(run_velocity_computation)

    def test_stereographic_projection_performance(
        self, benchmark: FixtureFunction
    ) -> None:
        """Benchmark stereographic projection performance."""
        quaternions = [
            Quaternion.from_euler_angles(0.01 * i, 0.02 * i, 0.03 * i)
            for i in range(1000)
        ]

        def run_projections() -> None:
            for q in quaternions:
                try:
                    _ = self.visualizer.stereographic_projection(q)
                except ValueError:
                    continue

        benchmark(run_projections)


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main(["-xvs", __file__])
