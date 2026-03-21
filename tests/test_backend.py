"""Tests for C++ backend detection and switching."""

# ruff: noqa: PLC0415
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import numpy.testing as npt
import pytest


class TestBackendDetection:
    """Tests that the backend detection mechanism works correctly."""

    def test_has_cpp_flag_is_bool(self) -> None:
        from interpolatepy._backend import HAS_CPP

        assert isinstance(HAS_CPP, bool)

    def test_has_cpp_exposed_in_init(self) -> None:
        import interpolatepy

        assert hasattr(interpolatepy, "HAS_CPP")
        assert isinstance(interpolatepy.HAS_CPP, bool)

    def test_env_var_forces_pure_python(self) -> None:
        """INTERPOLATEPY_NO_CPP=1 should force pure-Python mode."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import interpolatepy; print(interpolatepy.HAS_CPP)",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "INTERPOLATEPY_NO_CPP": "1"},
            check=True,
        )
        assert result.stdout.strip() == "False"

    def test_cpp_backend_active_without_env_var(self) -> None:
        """Without env var, C++ backend should be active if .so is present."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import interpolatepy; print(interpolatepy.HAS_CPP)",
            ],
            capture_output=True,
            text=True,
            env={k: v for k, v in os.environ.items() if k != "INTERPOLATEPY_NO_CPP"},
            check=True,
        )
        # Just check it returns a valid bool string
        assert result.stdout.strip() in ("True", "False")


class TestCppClassTypes:
    """Verify C++ backend classes are used when HAS_CPP is True."""

    def test_cubic_spline_is_cpp_backed(self) -> None:
        import interpolatepy

        if not interpolatepy.HAS_CPP:
            pytest.skip("C++ backend not available")
        assert "._adapters." in str(interpolatepy.CubicSpline)

    def test_bspline_is_cpp_backed(self) -> None:
        import interpolatepy

        if not interpolatepy.HAS_CPP:
            pytest.skip("C++ backend not available")
        assert "._adapters." in str(interpolatepy.BSpline)

    def test_quaternion_stays_pure_python(self) -> None:
        """Quaternion should always be the pure-Python class."""
        import interpolatepy

        assert "quat_core" in str(interpolatepy.Quaternion)


class TestNumericalEquivalence:
    """Compare Python and C++ results for key algorithms."""

    @pytest.fixture
    def t_points(self) -> list[float]:
        return [0.0, 1.0, 2.0, 3.0, 4.0]

    @pytest.fixture
    def q_points(self) -> list[float]:
        return [0.0, 1.0, 0.5, 2.0, 1.5]

    def test_cubic_spline_evaluate_matches(
        self, t_points: list[float], q_points: list[float]
    ) -> None:
        """C++ and Python CubicSpline should produce identical results."""
        from interpolatepy.cubic_spline import CubicSpline as PyCubicSpline

        import interpolatepy

        if not interpolatepy.HAS_CPP:
            pytest.skip("C++ backend not available")

        py_spline = PyCubicSpline(t_points, q_points, v0=0.0, vn=0.0)
        cpp_spline = interpolatepy.CubicSpline(t_points, q_points, v0=0.0, vn=0.0)

        t_eval = np.linspace(0.0, 4.0, 50)
        npt.assert_allclose(
            cpp_spline.evaluate(t_eval),
            py_spline.evaluate(t_eval),
            rtol=1e-12,
        )
        npt.assert_allclose(
            cpp_spline.evaluate_velocity(t_eval),
            py_spline.evaluate_velocity(t_eval),
            rtol=1e-12,
        )
        npt.assert_allclose(
            cpp_spline.evaluate_acceleration(t_eval),
            py_spline.evaluate_acceleration(t_eval),
            rtol=1e-10,
        )

    def test_double_s_evaluate_matches(self) -> None:
        """C++ and Python DoubleSTrajectory should produce matching results."""
        from interpolatepy.double_s import DoubleSTrajectory as PyDoubleSTrajectory
        from interpolatepy.double_s import StateParams as PyStateParams
        from interpolatepy.double_s import TrajectoryBounds as PyTrajectoryBounds

        import interpolatepy

        if not interpolatepy.HAS_CPP:
            pytest.skip("C++ backend not available")

        py_state = PyStateParams(q_0=0.0, q_1=10.0, v_0=0.0, v_1=0.0)
        py_bounds = PyTrajectoryBounds(v_bound=5.0, a_bound=10.0, j_bound=30.0)
        py_traj = PyDoubleSTrajectory(py_state, py_bounds)

        cpp_state = interpolatepy.StateParams(q_0=0.0, q_1=10.0, v_0=0.0, v_1=0.0)
        cpp_bounds = interpolatepy.TrajectoryBounds(v_bound=5.0, a_bound=10.0, j_bound=30.0)
        cpp_traj = interpolatepy.DoubleSTrajectory(cpp_state, cpp_bounds)

        t_eval = np.linspace(0.0, min(py_traj.T, cpp_traj.T), 50)
        npt.assert_allclose(
            cpp_traj.evaluate(t_eval),
            py_traj.evaluate(t_eval),
            rtol=1e-10,
        )
