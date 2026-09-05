"""Tests for the installed package layout and compatibility surface."""

from __future__ import annotations

import os
import subprocess
import sys
from importlib.resources import files

import interpolatepy


def test_domain_namespaces_match_top_level_api() -> None:
    """New domain namespaces expose the backend-neutral public classes."""
    from interpolatepy.bsplines import BSpline
    from interpolatepy.motion import DoubleSTrajectory
    from interpolatepy.paths import LinearPath
    from interpolatepy.quaternion import Quaternion
    from interpolatepy.splines import CubicSpline

    assert BSpline is interpolatepy.BSpline
    assert CubicSpline is interpolatepy.CubicSpline
    assert DoubleSTrajectory is interpolatepy.DoubleSTrajectory
    assert LinearPath is interpolatepy.LinearPath
    assert Quaternion is interpolatepy.Quaternion


def test_flat_compatibility_modules_are_removed() -> None:
    """Only the modern domain modules are shipped at the package root."""
    package_root = files("interpolatepy")
    removed_modules = {
        "b_spline.py",
        "b_spline_approx.py",
        "b_spline_cubic.py",
        "b_spline_interpolate.py",
        "b_spline_smooth.py",
        "c_s_smoot_search.py",
        "c_s_smoothing.py",
        "c_s_with_acc1.py",
        "c_s_with_acc2.py",
        "cubic_spline.py",
        "double_s.py",
        "frenet_frame.py",
        "lin_poly_parabolic.py",
        "linear.py",
        "log_quat.py",
        "polynomials.py",
        "quat_core.py",
        "quat_spline.py",
        "quat_visualization.py",
        "simple_paths.py",
        "squad_c2.py",
        "trapezoidal.py",
        "tridiagonal_inv.py",
    }

    assert all(not package_root.joinpath(module).is_file() for module in removed_modules)


def test_package_declares_inline_typing() -> None:
    """PEP 561 marker is present in the installed package."""
    assert files("interpolatepy").joinpath("py.typed").is_file()


def test_core_import_does_not_eagerly_load_matplotlib() -> None:
    """Numerical use should not import the optional plotting stack."""
    environment = {**os.environ, "INTERPOLATEPY_NO_CPP": "1"}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import interpolatepy; print('matplotlib.pyplot' in sys.modules)",
        ],
        capture_output=True,
        check=True,
        env=environment,
        text=True,
    )
    assert result.stdout.strip() == "False"
