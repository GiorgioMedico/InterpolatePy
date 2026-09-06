"""Multiple-shooting regression tests against independent mathematical cases."""

from typing import Any

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

import interpolatepy as ip
from interpolatepy import Quaternion
from interpolatepy import ShootingConfig
from interpolatepy import ShootingQuaternionInterpolation
from interpolatepy.quaternion._shooting_solver import matching_system
from interpolatepy.quaternion.shooting import ShootingQuaternionInterpolation as PythonShooting


def curved_keyframes() -> tuple[list[float], list[Quaternion]]:
    return [0.0, 1.0, 2.0, 3.0], [
        Quaternion.identity(),
        Quaternion.from_euler_angles(0.9, 0.1, 0.2),
        Quaternion.from_euler_angles(0.2, 1.1, 0.5),
        Quaternion.from_euler_angles(-0.4, 0.3, 1.4),
    ]


def test_shooting_exports_and_protocol() -> None:
    from interpolatepy.quaternion import ShootingConfig as GroupedConfig
    from interpolatepy.quaternion import ShootingQuaternionInterpolation as GroupedShooting

    assert GroupedConfig is ShootingConfig
    assert GroupedShooting is ShootingQuaternionInterpolation
    assert "ShootingQuaternionInterpolation" in ip.__all__
    assert "ShootingConfig" in ip.__all__
    times, quaternions = curved_keyframes()
    curve = ShootingQuaternionInterpolation(times, quaternions)
    assert isinstance(curve, ip.QuaternionTrajectory)


def test_shooting_preserves_rotated_constant_speed_geodesic() -> None:
    start = Quaternion.from_euler_angles(0.8, -0.2, 0.4)
    axis = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
    end = start * Quaternion.from_angle_axis(1.6, axis)
    curve = ShootingQuaternionInterpolation([0.0, 2.0], [start, end])

    for time in np.linspace(0.0, 2.0, 17):
        expected = start.slerp(end, float(time / 2.0))
        assert abs(curve.evaluate(float(time)).dot_product(expected)) > 1.0 - 1e-12
        assert np.allclose(curve.evaluate_velocity(float(time)), 0.8 * axis, atol=1e-8, rtol=0.0)
        assert np.allclose(curve.evaluate_acceleration(float(time)), 0.0, atol=1e-8, rtol=0.0)
    assert curve.acceleration_energy < 1e-12
    assert curve.num_variables == 9


def test_shooting_matches_natural_scalar_cubic_for_commuting_rotations() -> None:
    times = [-0.3, 0.0, 0.8, 1.9, 3.0]
    angles = [0.0, 0.4, 0.1, 1.1, 0.9]
    curve = ShootingQuaternionInterpolation(times, [Quaternion.from_euler_angles(0.0, 0.0, a) for a in angles])
    reference = CubicSpline(times, angles, bc_type="natural")

    for time in np.linspace(times[0], times[-1], 41):
        value = curve.evaluate(float(time))
        assert np.isclose(2.0 * np.arctan2(value.z, value.w), reference(time), atol=1e-8, rtol=0.0)
        assert np.allclose(curve.evaluate_velocity(float(time)), [0.0, 0.0, reference(time, 1)], atol=1e-8, rtol=0.0)
        assert np.allclose(
            curve.evaluate_acceleration(float(time)), [0.0, 0.0, reference(time, 2)], atol=1e-8, rtol=0.0
        )


def test_shooting_keyframes_continuity_and_natural_boundaries() -> None:
    times, quaternions = curved_keyframes()
    curve = ShootingQuaternionInterpolation(times, quaternions)
    assert curve.residual_norm <= curve.config.tolerance
    assert curve.iterations_run <= curve.config.max_iterations
    assert curve.integration_steps > curve.config.integration_steps
    for time, quaternion in zip(times, quaternions):
        assert abs(curve.evaluate(time).dot_product(quaternion)) > 1.0 - 1e-12
    for time in times[1:-1]:
        assert np.allclose(curve.evaluate_velocity(time - 1e-8), curve.evaluate_velocity(time + 1e-8), atol=1e-7)
        assert np.allclose(
            curve.evaluate_acceleration(time - 1e-8), curve.evaluate_acceleration(time + 1e-8), atol=1e-7
        )
    assert np.linalg.norm(curve.evaluate_acceleration(times[0])) < 1e-7
    assert np.linalg.norm(curve.evaluate_acceleration(times[-1])) < 1e-7
    assert curve.acceleration_energy > 0.0


def test_shooting_derivatives_agree_with_orientation_curve() -> None:
    times, quaternions = curved_keyframes()
    curve = ShootingQuaternionInterpolation(times, quaternions)
    step = 1e-5
    for time in (0.37, 1.23, 2.71):
        relative = curve.evaluate(time - step).inverse() * curve.evaluate(time + step)
        velocity = relative.Log().v_ / step
        acceleration = (curve.evaluate_velocity(time + step) - curve.evaluate_velocity(time - step)) / (2.0 * step)
        assert np.allclose(curve.evaluate_velocity(time), velocity, atol=2e-7, rtol=0.0)
        assert np.allclose(curve.evaluate_acceleration(time), acceleration, atol=2e-7, rtol=0.0)


def test_shooting_matches_independent_high_accuracy_integration() -> None:
    _, quaternions = curved_keyframes()
    times = np.array([0.0, 0.7, 2.1, 3.0])
    curve = PythonShooting(times, quaternions)
    for index, parameters in enumerate(curve._parameters):

        def rhs(_time: float, state: np.ndarray, constant: np.ndarray = parameters[6:]) -> np.ndarray:
            quaternion, velocity, acceleration = state[:4], state[4:7], state[7:]
            return np.r_[
                -np.dot(quaternion[1:], velocity),
                quaternion[0] * velocity + np.cross(quaternion[1:], velocity),
                acceleration,
                constant - 2.0 * np.cross(velocity, acceleration),
            ]

        quaternion = curve.quaternions[index]
        initial = np.r_[quaternion.w, quaternion.x, quaternion.y, quaternion.z, parameters[:6]]
        reference = solve_ivp(rhs, (0.0, 1.0), initial, method="DOP853", rtol=1e-12, atol=1e-13, dense_output=True)
        assert reference.success
        for fraction in np.linspace(0.03, 0.97, 17):
            time = float(times[index] + fraction * (times[index + 1] - times[index]))
            state, _ = curve._evaluate_state(time)
            assert np.allclose(state, reference.sol(fraction), atol=1e-8, rtol=0.0)


def test_shooting_analytic_matching_jacobian() -> None:
    random = np.random.default_rng(3)
    quaternions = random.normal(size=(4, 4))
    quaternions /= np.linalg.norm(quaternions, axis=1)[:, None]
    parameters = random.normal(scale=0.1, size=(3, 9))
    durations = np.array([0.2, 0.7, 1.8])
    _, jacobian = matching_system(parameters, quaternions, durations, 8, jacobian=True)
    assert jacobian is not None
    matrix = jacobian.toarray()
    delta = 1e-6
    for column in range(parameters.size):
        above, below = parameters.copy(), parameters.copy()
        above.flat[column] += delta
        below.flat[column] -= delta
        numerical = (
            matching_system(above, quaternions, durations, 8)[0] - matching_system(below, quaternions, durations, 8)[0]
        ) / (2.0 * delta)
        assert np.allclose(numerical, matrix[:, column], atol=1e-8, rtol=1e-7)


def test_shooting_sign_scale_and_left_rotation_invariance() -> None:
    times, quaternions = curved_keyframes()
    base = ShootingQuaternionInterpolation(times, quaternions)
    rotation = Quaternion.from_euler_angles(-0.3, 0.7, 1.1)
    transformed = ShootingQuaternionInterpolation(
        times, [(rotation * quaternion) * ((-1.0) ** index * 1e-8) for index, quaternion in enumerate(quaternions)]
    )
    for time in np.linspace(times[0], times[-1], 13):
        assert abs(transformed.evaluate(float(time)).dot_product(rotation * base.evaluate(float(time)))) > 1.0 - 1e-12
        assert np.allclose(transformed.evaluate_velocity(float(time)), base.evaluate_velocity(float(time)), atol=1e-8)
        assert np.allclose(
            transformed.evaluate_acceleration(float(time)), base.evaluate_acceleration(float(time)), atol=1e-8
        )


def test_shooting_respects_physical_time_scaling() -> None:
    times, quaternions = curved_keyframes()
    base = ShootingQuaternionInterpolation(times, quaternions)
    scaled = ShootingQuaternionInterpolation(10.0 + 2.0 * np.array(times), quaternions)
    for time in (0.4, 1.1, 2.8):
        assert abs(base.evaluate(time).dot_product(scaled.evaluate(10.0 + 2.0 * time))) > 1.0 - 1e-12
        assert np.allclose(base.evaluate_velocity(time) / 2.0, scaled.evaluate_velocity(10.0 + 2.0 * time), atol=1e-8)
        assert np.allclose(
            base.evaluate_acceleration(time) / 4.0, scaled.evaluate_acceleration(10.0 + 2.0 * time), atol=1e-8
        )
    assert np.isclose(scaled.acceleration_energy, base.acceleration_energy / 8.0, atol=1e-8)


@pytest.mark.parametrize("scale", [1e-160, 1e-110, 1e110, 1e160])
def test_shooting_geodesic_has_finite_zero_energy_at_extreme_time_scales(scale: float) -> None:
    quaternions = [Quaternion.from_euler_angles(angle, 0.0, 0.0) for angle in (0.0, 0.4, 0.8)]
    curve = ShootingQuaternionInterpolation([0.0, scale, 2.0 * scale], quaternions)
    assert curve.acceleration_energy == 0.0
    assert np.isfinite(curve.residual_norm)
    for fraction in (0.5, 1.5):
        time = fraction * scale
        expected = Quaternion.from_euler_angles(0.4 * fraction, 0.0, 0.0)
        assert abs(curve.evaluate(time).dot_product(expected)) > 1.0 - 1e-12
        np.testing.assert_allclose(curve.evaluate_velocity(time) * scale, [0.4, 0.0, 0.0], atol=1e-8)
        np.testing.assert_array_equal(curve.evaluate_acceleration(time), np.zeros(3))


@pytest.mark.parametrize("scale", [1e-160, 1e160])
def test_shooting_matching_system_is_invariant_to_extreme_time_scaling(scale: float) -> None:
    random = np.random.default_rng(3)
    quaternions = random.normal(size=(4, 4))
    quaternions /= np.linalg.norm(quaternions, axis=1)[:, None]
    parameters = random.normal(scale=0.1, size=(3, 9))
    durations = np.array([0.2, 0.7, 1.8])
    residual, jacobian = matching_system(parameters, quaternions, durations, 8, jacobian=True)
    scaled_residual, scaled_jacobian = matching_system(parameters, quaternions, durations * scale, 8, jacobian=True)
    np.testing.assert_allclose(scaled_residual, residual, atol=1e-15, rtol=1e-14)
    assert jacobian is not None
    assert scaled_jacobian is not None
    np.testing.assert_allclose(scaled_jacobian.toarray(), jacobian.toarray(), atol=1e-15, rtol=1e-14)


def test_shooting_preserves_representable_acceleration_on_very_long_intervals() -> None:
    _, quaternions = curved_keyframes()
    times = np.array([0.0, 0.2, 0.9, 2.7])
    base = ShootingQuaternionInterpolation(times, quaternions)
    scaled = ShootingQuaternionInterpolation(times * 1e160, quaternions)
    assert scaled.residual_norm <= scaled.config.tolerance
    for time in (0.1, 0.5, 1.8):
        assert abs(base.evaluate(time).dot_product(scaled.evaluate(time * 1e160))) > 1.0 - 1e-12
        np.testing.assert_allclose(
            scaled.evaluate_acceleration(time * 1e160) * 1e160 * 1e160,
            base.evaluate_acceleration(time),
            atol=1e-3,
            rtol=0.0,
        )


def test_shooting_handles_many_distant_keyframes() -> None:
    random = np.random.default_rng(3)
    values = random.normal(size=(10, 4))
    values /= np.linalg.norm(values, axis=1)[:, None]
    keyframes = [Quaternion(*value) for value in values]
    curve = ShootingQuaternionInterpolation(np.arange(10, dtype=float), keyframes)
    assert curve.num_variables == 81
    assert curve.residual_norm <= curve.config.tolerance
    for index, quaternion in enumerate(keyframes):
        assert abs(curve.evaluate(float(index)).dot_product(quaternion)) > 1.0 - 1e-12


def test_shooting_output_sampling_preserves_solution_and_exact_endpoints() -> None:
    curve = ShootingQuaternionInterpolation(
        [-100.0, 0.1], [Quaternion.identity(), Quaternion.from_euler_angles(0.0, 0.0, 1.2)]
    )
    iterations = curve.iterations_run
    for count in (17, 1001):
        times, values = curve.generate_trajectory(count)
        assert len(times) == len(values) == count
        assert times[0] == -100.0
        assert times[-1] == 0.1
        assert all(np.isclose(value.norm(), 1.0, atol=1e-12) for value in values)
    assert curve.iterations_run == iterations
    assert curve.num_variables == 9
    assert np.isclose(curve.evaluate(float(np.nextafter(0.1, -100.0))).norm(), 1.0)


def test_shooting_static_curve() -> None:
    quaternion = Quaternion.from_euler_angles(0.1, 0.2, 0.3)
    curve = ShootingQuaternionInterpolation([0.0, 0.4, 2.0], [quaternion, -quaternion, quaternion])
    assert curve.iterations_run == 0
    assert curve.acceleration_energy < 1e-20
    assert np.allclose(curve.evaluate_velocity(1.0), 0.0, atol=1e-12)


@pytest.mark.parametrize(
    "config", [ShootingConfig(max_iterations=1), ShootingConfig(integration_steps=1, max_integration_steps=2)]
)
def test_shooting_reports_nonconvergence(config: ShootingConfig) -> None:
    times, quaternions = curved_keyframes()
    with pytest.raises(RuntimeError, match="Multiple shooting"):
        ShootingQuaternionInterpolation(times, quaternions, config)


@pytest.mark.parametrize("time", [-0.1, 3.1, np.nan, np.inf])
def test_shooting_rejects_invalid_evaluation_times(time: float) -> None:
    times, quaternions = curved_keyframes()
    curve = ShootingQuaternionInterpolation(times, quaternions)
    for evaluate in (curve.evaluate, curve.evaluate_velocity, curve.evaluate_acceleration):
        with pytest.raises(ValueError, match="Time"):
            evaluate(time)


@pytest.mark.parametrize(
    "arguments",
    [
        {"tolerance": 0.0},
        {"tolerance": np.nan},
        {"max_iterations": 0},
        {"integration_steps": 0},
        {"integration_steps": 2.5},
        {"max_integration_steps": 16},
    ],
)
def test_shooting_validates_configuration(arguments: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="must"):
        ShootingConfig(**arguments)


@pytest.mark.parametrize(
    ("times", "quaternions"),
    [
        ([0.0], [Quaternion.identity()]),
        ([0.0, 1.0], [Quaternion.identity()]),
        ([0.0, 0.0], [Quaternion.identity(), Quaternion.identity()]),
        ([0.0, np.nan], [Quaternion.identity(), Quaternion.identity()]),
        ([0.0, 1.0], [Quaternion.identity(), Quaternion(0.0, 0.0, 0.0, 0.0)]),
    ],
)
def test_shooting_validates_keyframes(times: list[float], quaternions: list[Quaternion]) -> None:
    with pytest.raises(ValueError, match="required|match|spacing|finite|non-zero"):
        ShootingQuaternionInterpolation(times, quaternions)


@pytest.mark.parametrize("count", [0, 1, True, 2.5])
def test_shooting_validates_output_count(count: Any) -> None:
    curve = ShootingQuaternionInterpolation([0.0, 1.0], [Quaternion.identity(), Quaternion.identity()])
    with pytest.raises(ValueError, match="num_points"):
        curve.generate_trajectory(count)
