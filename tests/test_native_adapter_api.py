"""Regression tests for the package-level API with the C++ backend active."""

from functools import partial

from matplotlib import pyplot as plt
import numpy as np
import pytest

import interpolatepy as ip


pytestmark = pytest.mark.skipif(not ip.HAS_CPP, reason="C++ backend not available")


def test_double_s_convenience_api() -> None:
    state = ip.StateParams(q_0=0.0, q_1=5.0, v_0=0.0, v_1=0.0)
    bounds = ip.TrajectoryBounds(v_bound=2.0, a_bound=2.0, j_bound=4.0)
    trajectory = ip.DoubleSTrajectory(state, bounds)

    assert trajectory.get_duration() == trajectory.T
    assert trajectory.get_phase_durations()["total"] == trajectory.T

    evaluate, duration = ip.DoubleSTrajectory.create_trajectory(state, bounds)
    assert duration == trajectory.T
    assert np.allclose(evaluate(duration)[0], state.q_1)


def test_bspline_scalar_inputs_and_python_config() -> None:
    points = np.array([0.0, 1.0, -0.5, 2.0, 1.0])
    curve = ip.BSplineInterpolator(
        degree=5,
        points=points,
        initial_velocity=0.0,
        final_velocity=0.0,
    )
    assert np.allclose(curve.evaluate(curve.u_min), [points[0]])
    assert np.allclose(curve.evaluate(curve.u_max), [points[-1]])

    smooth = ip.SmoothingCubicBSpline(
        points,
        ip.BSplineParams(mu=0.8, enforce_endpoints=True, auto_derivatives=True),
    )
    assert smooth.calculate_approximation_error().shape == points.shape
    assert smooth.calculate_smoothness_measure() >= 0.0

    cubic = ip.CubicBSplineInterpolation(
        points,
        method="chord_length",
        auto_derivatives=True,
    )
    assert cubic.v0.shape == cubic.vn.shape == (1,)


def test_polynomial_parameter_objects_and_smoothing_result() -> None:
    initial = ip.BoundaryCondition(position=0.0, velocity=0.0, acceleration=0.0)
    final = ip.BoundaryCondition(position=2.0, velocity=0.0, acceleration=0.0)
    interval = ip.TimeInterval(start=0.0, end=3.0)
    evaluate = ip.PolynomialTrajectory.order_5_trajectory(initial, final, interval)
    assert np.allclose(evaluate(interval.end)[:3], [2.0, 0.0, 0.0])

    t = np.linspace(0.0, 4.0, 5)
    q = np.array([0.0, 1.1, 0.1, 1.9, 1.0])
    spline, mu, error, iterations = ip.smoothing_spline_with_tolerance(
        t,
        q,
        tolerance=0.25,
        config=ip.SplineConfig(max_iterations=40),
    )
    evaluated = spline.evaluate(t)
    assert isinstance(evaluated, np.ndarray)
    assert evaluated.shape == t.shape
    assert 0.0 < mu <= 1.0
    assert error <= 0.25 + 1e-6
    assert iterations <= 40


def test_motion_and_path_helper_return_types() -> None:
    planner = ip.ParabolicBlendTrajectory(
        q=[0.0, 2.0, 1.0, 4.0],
        t=[0.0, 1.0, 2.5, 4.0],
        dt_blend=[0.2, 0.3, 0.3, 0.2],
        dt=0.02,
    )
    assert np.array_equal(planner.q, [0.0, 2.0, 1.0, 4.0])
    assert np.array_equal(planner.t, [0.0, 1.0, 2.5, 4.0])
    assert np.array_equal(planner.dt_blend, [0.2, 0.3, 0.3, 0.2])
    assert planner.dt == 0.02
    evaluate, duration = planner.generate()
    assert len(evaluate(duration / 2.0)) == 3
    planner.plot()
    plt.close("all")

    times = np.linspace(0.0, 2.0, 5)
    positions, velocities, accelerations = ip.linear_traj(0.0, 4.0, 0.0, 2.0, times)
    assert positions.shape == velocities.shape == accelerations.shape == times.shape

    circle = partial(ip.circular_trajectory_with_derivatives, r=1.5)
    parameters = np.linspace(0.0, 2.0 * np.pi, 20)
    points, frames = ip.compute_trajectory_frames(
        circle,
        parameters,
        tool_orientation=(0.1, 0.2, 0.0),
    )
    assert points.shape == (20, 3)
    assert frames.shape == (20, 3, 3)


def test_quaternion_compatibility_helpers() -> None:
    times = [0.0, 1.0, 2.0, 3.0]
    quaternions = [
        ip.Quaternion.from_angle_axis(angle, np.array([0.0, 0.0, 1.0]))
        for angle in (0.0, 0.4, 0.9, 1.3)
    ]
    spline = ip.QuaternionSpline(times, quaternions, interpolation_method="squad")
    interpolated, status = spline.interpolate_at_time(1.5)
    assert isinstance(interpolated, ip.Quaternion)
    assert status == 0

    spline.set_interpolation_method("slerp")
    assert spline.get_interpolation_method() == "slerp"
    assert isinstance(spline.evaluate(1.5), ip.Quaternion)

    logarithmic = ip.LogQuaternionInterpolation(times, quaternions, degree=3)
    sample_times, samples = logarithmic.generate_trajectory(num_points=5)
    assert logarithmic.degree == 3
    assert sample_times.shape == (5,)
    assert len(samples) == 5
