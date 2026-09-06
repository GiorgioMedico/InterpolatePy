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
    quaternions = [ip.Quaternion.from_angle_axis(angle, np.array([0.0, 0.0, 1.0])) for angle in (0.0, 0.4, 0.9, 1.3)]
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


def test_spring_native_adapter_matches_python_reference() -> None:
    from interpolatepy.quaternion.spring import (
        SpringQuaternionInterpolation as PythonSpringQuaternionInterpolation,
    )

    times = [0.0, 1.0, 2.0, 3.0]
    quaternions = [
        ip.Quaternion.identity(),
        ip.Quaternion.from_euler_angles(0.9, 0.1, 0.2),
        ip.Quaternion.from_euler_angles(0.2, 1.1, 0.5),
        ip.Quaternion.from_euler_angles(-0.4, 0.3, 1.4),
    ]
    config = ip.SpringConfig(num_samples=41, iterations=150)
    native = ip.SpringQuaternionInterpolation(times, quaternions, config)
    reference = PythonSpringQuaternionInterpolation(times, quaternions, config)

    assert type(native) is not PythonSpringQuaternionInterpolation
    assert np.array_equal(native.sample_times, reference.sample_times)
    assert np.array_equal(native.keyframe_indices, reference.keyframe_indices)
    assert native.refinement_sample_counts == reference.refinement_sample_counts
    assert len(native.stage_energy_history) == len(reference.stage_energy_history)
    assert native.energy_history is native.stage_energy_history[-1]
    assert np.isclose(native.initial_energy, reference.initial_energy, rtol=1e-11)
    assert np.isclose(native.final_energy, reference.final_energy, rtol=1e-9)
    for native_history, reference_history in zip(
        native.stage_energy_history,
        reference.stage_energy_history,
    ):
        assert np.allclose(native_history, reference_history, rtol=1e-8, atol=1e-12)

    for time in np.linspace(times[0], times[-1], 21):
        native_value = native.evaluate(float(time))
        reference_value = reference.evaluate(float(time))
        assert abs(native_value.dot_product(reference_value)) > 1.0 - 1e-8
        assert np.allclose(
            native.evaluate_velocity(float(time)),
            reference.evaluate_velocity(float(time)),
            rtol=1e-7,
            atol=1e-8,
        )
        assert np.allclose(
            native.evaluate_acceleration(float(time)),
            reference.evaluate_acceleration(float(time)),
            rtol=1e-6,
            atol=1e-7,
        )

    sample_times, samples = native.generate_trajectory(17)
    assert sample_times.shape == (17,)
    assert len(samples) == 17
    assert all(isinstance(sample, ip.Quaternion) for sample in samples)


def test_shooting_native_adapter_matches_python_reference() -> None:
    from interpolatepy.quaternion.shooting import (
        ShootingQuaternionInterpolation as PythonShootingQuaternionInterpolation,
    )

    times = [0.0, 0.7, 2.1, 3.0]
    quaternions = [
        ip.Quaternion.identity(),
        ip.Quaternion.from_euler_angles(0.9, 0.1, 0.2),
        ip.Quaternion.from_euler_angles(0.2, 1.1, 0.5),
        ip.Quaternion.from_euler_angles(-0.4, 0.3, 1.4),
    ]
    config = ip.ShootingConfig()
    native = ip.ShootingQuaternionInterpolation(times, quaternions, config)
    reference = PythonShootingQuaternionInterpolation(times, quaternions, config)
    assert type(native) is not PythonShootingQuaternionInterpolation
    assert np.array_equal(native.time_points, reference.time_points)
    assert native.iterations_run == reference.iterations_run
    assert native.integration_steps == reference.integration_steps
    assert native.num_variables == reference.num_variables
    assert np.isclose(native.residual_norm, reference.residual_norm, atol=1e-12, rtol=0.0)
    assert np.isclose(native.acceleration_energy, reference.acceleration_energy, atol=1e-10, rtol=0.0)
    for time in np.linspace(times[0], times[-1], 21):
        assert abs(native.evaluate(float(time)).dot_product(reference.evaluate(float(time)))) > 1.0 - 1e-12
        assert np.allclose(native.evaluate_velocity(float(time)), reference.evaluate_velocity(float(time)), atol=1e-10)
        assert np.allclose(
            native.evaluate_acceleration(float(time)), reference.evaluate_acceleration(float(time)), atol=1e-10
        )
    sample_times, samples = native.generate_trajectory(17)
    assert sample_times.shape == (17,)
    assert all(isinstance(sample, ip.Quaternion) for sample in samples)


@pytest.mark.parametrize("samples", [31, 101, 1001])
def test_spring_gauss_newton_native_adapter_matches_python(samples: int) -> None:
    from interpolatepy.quaternion.spring import SpringQuaternionInterpolation as PythonSpring

    times = [0.0, 0.7, 2.1, 3.0]
    keyframes = [
        ip.Quaternion.identity(),
        ip.Quaternion.from_euler_angles(0.9, 0.1, 0.2),
        ip.Quaternion.from_euler_angles(0.2, 1.1, 0.5),
        ip.Quaternion.from_euler_angles(-0.4, 0.3, 1.4),
    ]
    config = ip.SpringConfig(num_samples=samples, solver="gauss_newton", final_iterations=50)
    native = ip.SpringQuaternionInterpolation(times, keyframes, config)
    reference = PythonSpring(times, keyframes, config)
    assert native.converged
    assert reference.converged
    assert np.array_equal(native.sample_times, reference.sample_times)
    assert np.allclose(native.stage_gradient_norms, reference.stage_gradient_norms, atol=1e-8)
    assert np.isclose(native.final_energy, reference.final_energy, rtol=1e-8, atol=1e-12)
    for time in np.linspace(times[0], times[-1], 31):
        difference = native.evaluate(float(time)).inverse() * reference.evaluate(float(time))
        assert 2.0 * np.arctan2(np.linalg.norm(difference.v_), abs(difference.w)) < 1e-7
