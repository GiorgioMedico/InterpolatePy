"""Tests for protocol conformance and functional behavior.

Verifies that all expected classes satisfy their respective protocol interfaces
using both ``isinstance`` checks (runtime_checkable) and functional invocation.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import numpy.testing as npt
import pytest

from interpolatepy import (
    ApproximationBSpline,
    BSpline,
    BSplineInterpolator,
    CircularPath,
    CubicBSplineInterpolation,
    CubicSmoothingSpline,
    CubicSpline,
    CubicSplineWithAcceleration1,
    CubicSplineWithAcceleration2,
    CurveEvaluator,
    DoubleSTrajectory,
    GeometricPath,
    LinearPath,
    LogQuaternionInterpolation,
    ModifiedLogQuaternionInterpolation,
    Quaternion,
    QuaternionSpline,
    QuaternionTrajectory,
    ScalarTrajectory,
    SmoothingCubicBSpline,
    SquadC2,
    StateParams,
    TrajectoryBounds,
    TrajectoryFunction,
)


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def cubic_spline() -> CubicSpline:
    """Simple cubic spline for testing."""
    return CubicSpline(
        t_points=[0.0, 1.0, 2.0, 3.0],
        q_points=[0.0, 1.0, 0.5, 2.0],
    )


@pytest.fixture()
def double_s_trajectory() -> DoubleSTrajectory:
    """Standard double-S trajectory."""
    state = StateParams(q_0=0.0, q_1=10.0, v_0=0.0, v_1=0.0)
    bounds = TrajectoryBounds(v_bound=3.0, a_bound=2.0, j_bound=1.0)
    return DoubleSTrajectory(state, bounds)


@pytest.fixture()
def linear_path() -> LinearPath:
    """Linear path from origin to (1, 1, 1)."""
    return LinearPath(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))


@pytest.fixture()
def circular_path() -> CircularPath:
    """Circular path in XY plane."""
    return CircularPath(
        r=np.array([0.0, 0.0, 1.0]),
        d=np.array([0.0, 0.0, 0.0]),
        pi=np.array([1.0, 0.0, 0.0]),
    )


@pytest.fixture()
def quaternion_spline() -> QuaternionSpline:
    """Quaternion spline with 5 waypoints for SQUAD support."""
    quats = [
        Quaternion.identity(),
        Quaternion.from_euler_angles(0.1, 0.0, 0.0),
        Quaternion.from_euler_angles(0.2, 0.1, 0.0),
        Quaternion.from_euler_angles(0.1, 0.2, 0.1),
        Quaternion.identity(),
    ]
    return QuaternionSpline([0.0, 1.0, 2.0, 3.0, 4.0], quats)


@pytest.fixture()
def squad_c2() -> SquadC2:
    """SquadC2 interpolator."""
    quats = [
        Quaternion.identity(),
        Quaternion.from_euler_angles(0.1, 0.0, 0.0),
        Quaternion.from_euler_angles(0.2, 0.1, 0.0),
        Quaternion.from_euler_angles(0.1, 0.2, 0.1),
        Quaternion.identity(),
    ]
    return SquadC2([0.0, 1.0, 2.0, 3.0, 4.0], quats)


@pytest.fixture()
def log_quat_interp() -> LogQuaternionInterpolation:
    """LogQuaternionInterpolation instance."""
    quats = [
        Quaternion.identity(),
        Quaternion.from_euler_angles(0.1, 0.0, 0.0),
        Quaternion.from_euler_angles(0.2, 0.1, 0.0),
        Quaternion.from_euler_angles(0.1, 0.2, 0.1),
        Quaternion.identity(),
    ]
    return LogQuaternionInterpolation([0.0, 1.0, 2.0, 3.0, 4.0], quats)


@pytest.fixture()
def modified_log_quat_interp() -> ModifiedLogQuaternionInterpolation:
    """ModifiedLogQuaternionInterpolation instance."""
    quats = [
        Quaternion.identity(),
        Quaternion.from_euler_angles(0.1, 0.0, 0.0),
        Quaternion.from_euler_angles(0.2, 0.1, 0.0),
        Quaternion.from_euler_angles(0.1, 0.2, 0.1),
        Quaternion.identity(),
    ]
    return ModifiedLogQuaternionInterpolation([0.0, 1.0, 2.0, 3.0, 4.0], quats)


# ---------------------------------------------------------------------------
#  ScalarTrajectory conformance
# ---------------------------------------------------------------------------


class TestScalarTrajectoryProtocol:
    """Tests for ScalarTrajectory protocol conformance."""

    def test_cubic_spline_isinstance(self, cubic_spline: CubicSpline) -> None:
        assert isinstance(cubic_spline, ScalarTrajectory)

    def test_double_s_isinstance(self, double_s_trajectory: DoubleSTrajectory) -> None:
        assert isinstance(double_s_trajectory, ScalarTrajectory)

    def test_double_s_evaluate_returns_scalar(self, double_s_trajectory: DoubleSTrajectory) -> None:
        """DoubleSTrajectory.evaluate should return position only (not a tuple)."""
        result = double_s_trajectory.evaluate(0.0)
        assert isinstance(result, float | np.floating)

    def test_double_s_evaluate_full_returns_tuple(self, double_s_trajectory: DoubleSTrajectory) -> None:
        """evaluate_full should still return the 4-tuple."""
        result = double_s_trajectory.evaluate_full(0.0)
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_double_s_individual_methods_match_full(self, double_s_trajectory: DoubleSTrajectory) -> None:
        """Individual methods should match evaluate_full output."""
        t = 1.0
        q_full, v_full, a_full, j_full = double_s_trajectory.evaluate_full(t)

        npt.assert_allclose(double_s_trajectory.evaluate(t), q_full, rtol=1e-12)
        npt.assert_allclose(double_s_trajectory.evaluate_velocity(t), v_full, rtol=1e-12)
        npt.assert_allclose(double_s_trajectory.evaluate_acceleration(t), a_full, rtol=1e-12)
        npt.assert_allclose(double_s_trajectory.evaluate_jerk(t), j_full, rtol=1e-12)

    def test_cubic_spline_functional(self, cubic_spline: CubicSpline) -> None:
        """CubicSpline methods should be callable through protocol."""

        def use_scalar_trajectory(traj: ScalarTrajectory, t: float) -> tuple[float, float, float]:
            pos = traj.evaluate(t)
            vel = traj.evaluate_velocity(t)
            acc = traj.evaluate_acceleration(t)
            return pos, vel, acc

        pos, vel, acc = use_scalar_trajectory(cubic_spline, 1.0)
        assert pos is not None
        assert vel is not None
        assert acc is not None

    def test_negative_isinstance(self, linear_path: LinearPath) -> None:
        """LinearPath should NOT satisfy ScalarTrajectory."""
        assert not isinstance(linear_path, ScalarTrajectory)


# ---------------------------------------------------------------------------
#  CurveEvaluator conformance
# ---------------------------------------------------------------------------


class TestCurveEvaluatorProtocol:
    """Tests for CurveEvaluator protocol conformance."""

    def test_bspline_interpolator_isinstance(self) -> None:
        """BSplineInterpolator should satisfy CurveEvaluator."""
        points = np.array([[0, 0, 0], [1, 2, 0], [3, 1, 0], [4, 3, 0], [5, 0, 0]], dtype=float)
        interp = BSplineInterpolator(degree=3, points=points)
        assert isinstance(interp, CurveEvaluator)

    def test_bspline_isinstance(self) -> None:
        """BSpline should satisfy CurveEvaluator."""
        control_points = np.array([[0, 0], [1, 2], [3, 1], [4, 0]], dtype=float)
        knots = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
        bspline = BSpline(degree=3, knots=knots, control_points=control_points)
        assert isinstance(bspline, CurveEvaluator)

    def test_negative_isinstance(self, cubic_spline: CubicSpline) -> None:
        """CubicSpline should NOT satisfy CurveEvaluator (no evaluate_derivative)."""
        assert not isinstance(cubic_spline, CurveEvaluator)


# ---------------------------------------------------------------------------
#  GeometricPath conformance
# ---------------------------------------------------------------------------


class TestGeometricPathProtocol:
    """Tests for GeometricPath protocol conformance."""

    def test_linear_path_isinstance(self, linear_path: LinearPath) -> None:
        assert isinstance(linear_path, GeometricPath)

    def test_circular_path_isinstance(self, circular_path: CircularPath) -> None:
        assert isinstance(circular_path, GeometricPath)

    def test_linear_path_functional(self, linear_path: LinearPath) -> None:
        """LinearPath should work through GeometricPath interface."""

        def evaluate_path(path: GeometricPath, s: float) -> dict[str, np.ndarray]:
            return {
                "position": path.position(s),
                "velocity": path.velocity(s),
                "acceleration": path.acceleration(s),
            }

        result = evaluate_path(linear_path, 0.5)
        assert result["position"].shape == (3,)
        assert result["velocity"].shape == (3,)
        assert result["acceleration"].shape == (3,)

    def test_circular_path_functional(self, circular_path: CircularPath) -> None:
        """CircularPath should work through GeometricPath interface."""

        def evaluate_path(path: GeometricPath, s: float) -> dict[str, np.ndarray]:
            return {
                "position": path.position(s),
                "velocity": path.velocity(s),
                "acceleration": path.acceleration(s),
            }

        result = evaluate_path(circular_path, 0.5)
        assert result["position"].shape == (3,)
        assert result["velocity"].shape == (3,)
        assert result["acceleration"].shape == (3,)

    def test_negative_isinstance(self, cubic_spline: CubicSpline) -> None:
        """CubicSpline should NOT satisfy GeometricPath."""
        assert not isinstance(cubic_spline, GeometricPath)


# ---------------------------------------------------------------------------
#  QuaternionTrajectory conformance
# ---------------------------------------------------------------------------


class TestQuaternionTrajectoryProtocol:
    """Tests for QuaternionTrajectory protocol conformance."""

    def test_squad_c2_isinstance(self, squad_c2: SquadC2) -> None:
        assert isinstance(squad_c2, QuaternionTrajectory)

    def test_log_quat_isinstance(self, log_quat_interp: LogQuaternionInterpolation) -> None:
        assert isinstance(log_quat_interp, QuaternionTrajectory)

    def test_modified_log_quat_isinstance(
        self, modified_log_quat_interp: ModifiedLogQuaternionInterpolation
    ) -> None:
        assert isinstance(modified_log_quat_interp, QuaternionTrajectory)

    def test_quaternion_spline_isinstance(self, quaternion_spline: QuaternionSpline) -> None:
        assert isinstance(quaternion_spline, QuaternionTrajectory)

    def test_squad_c2_functional(self, squad_c2: SquadC2) -> None:
        """SquadC2 should work through QuaternionTrajectory interface."""

        def use_quat_traj(traj: QuaternionTrajectory, t: float) -> Quaternion:
            q = traj.evaluate(t)
            vel = traj.evaluate_velocity(t)
            acc = traj.evaluate_acceleration(t)
            assert isinstance(q, Quaternion)
            assert isinstance(vel, np.ndarray)
            assert isinstance(acc, np.ndarray)
            return q

        q = use_quat_traj(squad_c2, 2.0)
        assert isinstance(q, Quaternion)

    def test_quaternion_spline_evaluate(self, quaternion_spline: QuaternionSpline) -> None:
        """QuaternionSpline.evaluate should return a Quaternion."""
        q = quaternion_spline.evaluate(2.0)
        assert isinstance(q, Quaternion)

    def test_quaternion_spline_evaluate_velocity(self, quaternion_spline: QuaternionSpline) -> None:
        """QuaternionSpline.evaluate_velocity should return an ndarray."""
        omega = quaternion_spline.evaluate_velocity(2.0)
        assert isinstance(omega, np.ndarray)
        assert omega.shape == (3,)

    def test_quaternion_spline_evaluate_acceleration(self, quaternion_spline: QuaternionSpline) -> None:
        """QuaternionSpline.evaluate_acceleration should return an ndarray."""
        alpha = quaternion_spline.evaluate_acceleration(2.0)
        assert isinstance(alpha, np.ndarray)
        assert alpha.shape == (3,)

    def test_quaternion_spline_evaluate_raises_on_empty(self) -> None:
        """evaluate should raise on empty spline."""
        spline = QuaternionSpline.__new__(QuaternionSpline)
        spline.quat_data = OrderedDict()
        spline.interpolation_method = "auto"
        spline.intermediate_quaternions = {}
        with pytest.raises(ValueError, match="empty"):
            spline.evaluate(0.0)

    def test_negative_isinstance(self, linear_path: LinearPath) -> None:
        """LinearPath should NOT satisfy QuaternionTrajectory (no evaluate method)."""
        assert not isinstance(linear_path, QuaternionTrajectory)


# ---------------------------------------------------------------------------
#  TrajectoryFunction conformance
# ---------------------------------------------------------------------------


class TestTrajectoryFunctionProtocol:
    """Tests for TrajectoryFunction protocol conformance."""

    def test_callable_isinstance(self) -> None:
        """A simple callable returning a 3-tuple should satisfy TrajectoryFunction."""

        def my_traj(t: float) -> tuple[float, float, float]:
            return t, 1.0, 0.0

        assert isinstance(my_traj, TrajectoryFunction)

    def test_lambda_isinstance(self) -> None:
        """Lambda should also satisfy TrajectoryFunction."""
        f = lambda t: (t, 0.0, 0.0)  # noqa: E731
        assert isinstance(f, TrajectoryFunction)
