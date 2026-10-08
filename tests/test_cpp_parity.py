"""Parity of the C++ backend against the pure-Python reference at small n."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

import interpolatepy as ip
from interpolatepy.bsplines.cubic import CubicBSplineInterpolation as PyCubicBSplineInterpolation
from interpolatepy.bsplines.smoothing import BSplineParams as PyBSplineParams
from interpolatepy.bsplines.smoothing import SmoothingCubicBSpline as PySmoothingCubicBSpline
from interpolatepy.quaternion import logarithmic as py_logarithmic
from interpolatepy.quaternion.core import Quaternion as PyQuaternion
from interpolatepy.splines.acceleration1 import CubicSplineWithAcceleration1 as PyCubicSplineWithAcceleration1
from interpolatepy.splines.acceleration2 import CubicSplineWithAcceleration2 as PyCubicSplineWithAcceleration2
from interpolatepy.splines.acceleration2 import SplineParameters as PySplineParameters
from interpolatepy.splines.cubic import CubicSpline as PyCubicSpline
from interpolatepy.splines.smoothing import CubicSmoothingSpline as PyCubicSmoothingSpline

pytestmark = pytest.mark.skipif(not ip.HAS_CPP, reason="C++ backend not available")

TOL = 1e-9
METHODS = ["equally_spaced", "chord_length", "centripetal"]
FIELD_POINTS = [[0.0, 0.0, 0.0], [0.5, 0.2, 0.4], [1.0, 0.0, 1.0]]


def _points(n: int) -> np.ndarray:
    extra = [[1.4, 0.6, 0.7], [2.0, 0.1, 0.3], [2.3, -0.5, 0.9]]
    return np.array(FIELD_POINTS + extra[: n - 3])


def _bspline_grid(curve: Any) -> np.ndarray:
    return np.array([curve.evaluate(u) for u in np.linspace(curve.u_min, curve.u_max, 201)])


class TestCubicBSplineInterpolationParity:
    @pytest.mark.parametrize("n", [3, 4, 5, 6])
    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("velocities", [False, True])
    def test_matches_python_and_hits_waypoints(self, n: int, method: str, velocities: bool) -> None:
        points = _points(n)
        kwargs: dict[str, Any] = {"v0": [0.3, -0.2, 0.5], "vn": [-0.4, 0.1, 0.2]} if velocities else {}
        cpp = ip.CubicBSplineInterpolation(points, method=method, **kwargs)
        py = PyCubicBSplineInterpolation(points, method=method, **kwargs)

        npt.assert_allclose(cpp.control_points, py.control_points, atol=TOL)
        npt.assert_allclose(_bspline_grid(cpp), _bspline_grid(py), atol=TOL)
        for u, point in zip(cpp.u_bars, points):
            npt.assert_allclose(cpp.evaluate(u), point, atol=TOL)

    @pytest.mark.parametrize("method", METHODS)
    def test_auto_derivatives_three_points(self, method: str) -> None:
        cpp = ip.CubicBSplineInterpolation(FIELD_POINTS, method=method, auto_derivatives=True)
        py = PyCubicBSplineInterpolation(FIELD_POINTS, method=method, auto_derivatives=True)
        npt.assert_allclose(_bspline_grid(cpp), _bspline_grid(py), atol=TOL)


class TestSmallestNSplineParity:
    """Tridiagonal/banded solvers with boundary-row corrections, at their smallest n."""

    t = np.linspace(0.0, 1.0, 301)

    @pytest.mark.parametrize(("v0", "vn"), [(0.0, 0.0), (0.7, -0.3)])
    def test_cubic_spline(self, v0: float, vn: float) -> None:
        args = ([0.0, 1.0, 2.5], [0.0, 1.0, 0.2])
        cpp, py = ip.CubicSpline(*args, v0=v0, vn=vn), PyCubicSpline(*args, v0=v0, vn=vn)
        t = self.t * 2.5
        npt.assert_allclose(cpp.evaluate(t), py.evaluate(t), atol=TOL)

    @pytest.mark.parametrize("n", [3, 4])
    def test_cubic_spline_with_acceleration1(self, n: int) -> None:
        args = ([0.0, 1.0, 2.5, 3.0][:n], [0.0, 1.0, 0.2, 0.8][:n])
        kw: dict[str, Any] = {"v0": 0.4, "vn": -0.2, "a0": 0.1, "an": 0.3}
        cpp, py = ip.CubicSplineWithAcceleration1(*args, **kw), PyCubicSplineWithAcceleration1(*args, **kw)
        t = self.t * args[0][-1]
        npt.assert_allclose(cpp.evaluate(t), py.evaluate(t), atol=TOL)

    @pytest.mark.parametrize("n", [2, 3])
    def test_cubic_spline_with_acceleration2(self, n: int) -> None:
        args = ([0.0, 1.0, 2.5][:n], [0.0, 1.0, 0.2][:n])
        kw: dict[str, Any] = {"v0": 0.4, "vn": -0.2, "a0": 0.1, "an": 0.3}
        cpp = ip.CubicSplineWithAcceleration2(*args, ip.SplineParameters(**kw))
        py = PyCubicSplineWithAcceleration2(*args, PySplineParameters(**kw))
        t = self.t * args[0][-1]
        npt.assert_allclose(cpp.evaluate(t), py.evaluate(t), atol=TOL)

    def test_cubic_smoothing_spline(self) -> None:
        args = ([0.0, 1.0, 2.5], [0.0, 1.0, 0.2])
        kw: dict[str, Any] = {"mu": 0.6, "v0": 0.4, "vn": -0.2}
        cpp, py = ip.CubicSmoothingSpline(*args, **kw), PyCubicSmoothingSpline(*args, **kw)
        t = self.t * 2.5
        npt.assert_allclose(cpp.evaluate(t), py.evaluate(t), atol=TOL)

    @pytest.mark.parametrize("enforce_endpoints", [False, True])
    @pytest.mark.parametrize("method", METHODS)
    def test_smoothing_cubic_bspline(self, enforce_endpoints: bool, method: str) -> None:
        kw: dict[str, Any] = {
            "mu": 0.7,
            "method": method,
            "enforce_endpoints": enforce_endpoints,
            "v0": [0.3, 0.0, 0.1],
        }
        cpp = ip.SmoothingCubicBSpline(FIELD_POINTS, ip.BSplineParams(**kw))
        py = PySmoothingCubicBSpline(FIELD_POINTS, PyBSplineParams(**kw))
        npt.assert_allclose(_bspline_grid(cpp), _bspline_grid(py), atol=TOL)


def _z_rotation(angle: float) -> PyQuaternion:
    return PyQuaternion(np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0))


def _as_array(q: PyQuaternion) -> np.ndarray:
    return np.array([q.w, q.x, q.y, q.z])


@pytest.mark.parametrize("name", ["ModifiedLogQuaternionInterpolation", "LogQuaternionInterpolation"])
class TestLogQuaternionParity:
    times = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

    def _pair(self, name: str, quats: list[PyQuaternion]) -> tuple[Any, Any]:
        return getattr(ip, name)(self.times, quats), getattr(py_logarithmic, name)(self.times, quats)

    def test_rotation_past_pi_is_monotone(self, name: str) -> None:
        quats = [_z_rotation(a) for a in np.linspace(0.0, 3.6, len(self.times))]
        cpp, py = self._pair(name, quats)
        t = np.linspace(0.0, 1.0, 241)
        cpp_q = np.array([_as_array(cpp.evaluate(ti)) for ti in t])
        npt.assert_allclose(cpp_q, [_as_array(py.evaluate(ti)) for ti in t], atol=TOL)

        angles = np.unwrap(2.0 * np.arctan2(cpp_q[:, 3], cpp_q[:, 0]))
        assert np.all(np.diff(angles) > 0.0)
        npt.assert_allclose(angles[-1], 3.6, atol=TOL)

    def test_varying_axis_matches_python(self, name: str) -> None:
        rng = np.random.default_rng(7)
        quats = [PyQuaternion(*(q / np.linalg.norm(q))) for q in rng.normal(size=(len(self.times), 4))]
        cpp, py = self._pair(name, quats)
        for ti in np.linspace(0.0, 1.0, 241):
            npt.assert_allclose(_as_array(cpp.evaluate(ti)), _as_array(py.evaluate(ti)), atol=TOL)
            npt.assert_allclose(cpp.evaluate_velocity(ti), py.evaluate_velocity(ti), atol=1e-8)


@pytest.mark.parametrize("normalize_axis", [True, False])
def test_modified_log_angular_velocity_matches_python(normalize_axis: bool) -> None:
    times = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    rng = np.random.default_rng(11)
    quats = [PyQuaternion(*(q / np.linalg.norm(q))) for q in rng.normal(size=(len(times), 4))]
    cpp = ip.ModifiedLogQuaternionInterpolation(times, quats, normalize_axis=normalize_axis)
    py = py_logarithmic.ModifiedLogQuaternionInterpolation(times, quats, normalize_axis=normalize_axis)
    for ti in np.linspace(0.0, 1.0, 50):
        npt.assert_allclose(cpp.angular_velocity(ti), py.get_physical_kinematics(ti)[0], atol=1e-9)
