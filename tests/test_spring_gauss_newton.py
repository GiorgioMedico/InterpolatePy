"""Same-objective and convergence checks for the two discrete SPRING solvers."""

from dataclasses import replace

import numpy as np
import pytest

from interpolatepy import Quaternion
from interpolatepy import SpringConfig
from interpolatepy import SpringQuaternionInterpolation
from interpolatepy.quaternion._spring_gauss_newton import normal_matrix
from interpolatepy.quaternion._spring_solver import curvature_energy_gradient


def keyframes() -> list[Quaternion]:
    return [
        Quaternion.identity(),
        Quaternion.from_euler_angles(0.9, 0.1, 0.2),
        Quaternion.from_euler_angles(0.2, 1.1, 0.5),
        Quaternion.from_euler_angles(-0.4, 0.3, 1.4),
    ]


def orientation_error(first: Quaternion, second: Quaternion) -> float:
    relative = first.inverse() * second
    return float(2.0 * np.arctan2(np.linalg.norm(relative.v_), abs(relative.w)))


@pytest.mark.parametrize("indices", [np.arange(7), np.array([0, 1, 3, 6, 9, 10, 15])])
def test_banded_model_matches_original_energy_and_numerical_jacobian(indices: np.ndarray) -> None:
    random = np.random.default_rng(4)
    frames = random.normal(size=(7, 4))
    frames /= np.linalg.norm(frames, axis=1)[:, None]
    frames *= random.uniform(0.9, 1.1, size=(7, 1))
    weights = np.array([1.0, 1.2, 1.0, 1.2, 1.0])
    penalty = 73.0
    fixed = np.array([True, False, False, True, False, False, True])

    def residual(values: np.ndarray) -> np.ndarray:
        # Independently evaluate the existing weighted divided-difference
        # energy, without using the new solver's analytic Jacobian.
        spacing = np.diff(indices) / np.mean(np.diff(indices))
        slopes = np.diff(values, axis=0) / spacing[:, None]
        difference = 2.0 * np.diff(slopes, axis=0) / (spacing[:-1] + spacing[1:])[:, None]
        center = values[1:-1]
        curvature = (
            difference - (np.sum(difference * center, axis=1) / np.sum(center * center, axis=1))[:, None] * center
        )
        return np.r_[
            (np.sqrt(weights)[:, None] * curvature).ravel(),
            np.sqrt(penalty) * (np.sum(values * values, axis=1) - 1.0),
        ]

    errors = residual(frames)
    jacobian = np.empty((len(errors), frames.size))
    delta = 1e-6
    for column in range(frames.size):
        above, below = frames.copy(), frames.copy()
        above.flat[column] += delta
        below.flat[column] -= delta
        jacobian[:, column] = (residual(above) - residual(below)) / (2.0 * delta)
    energy, gradient = curvature_energy_gradient(frames, weights, penalty, indices)
    assert np.isclose(errors @ errors, energy, rtol=1e-12)
    assert np.allclose(2.0 * jacobian.T @ errors, gradient.ravel(), rtol=1e-7, atol=1e-7)
    jacobian[:, np.repeat(fixed, 4)] = 0.0
    expected = 2.0 * jacobian.T @ jacobian
    expected[np.repeat(fixed, 4), np.repeat(fixed, 4)] = 1.0
    band = normal_matrix(frames, weights, penalty, fixed, indices)
    actual = np.zeros_like(expected)
    for diagonal, entries in enumerate(band):
        columns = np.arange(frames.size - diagonal)
        actual[columns + diagonal, columns] = entries[: len(columns)]
        actual[columns, columns + diagonal] = entries[: len(columns)]
    assert np.allclose(actual, expected, rtol=1e-7, atol=1e-6)


def test_solvers_have_identical_coarse_states_and_final_starting_energy() -> None:
    config = SpringConfig(num_samples=101, iterations=100)
    original = SpringQuaternionInterpolation([0.0, 0.4, 1.7, 3.0], keyframes(), config)
    accelerated = SpringQuaternionInterpolation(
        [0.0, 0.4, 1.7, 3.0], keyframes(), replace(config, solver="gauss_newton")
    )
    assert original.stage_energy_history[:-1] == accelerated.stage_energy_history[:-1]
    assert original.energy_history[0] == accelerated.energy_history[0]
    assert np.array_equal(original.sample_times, accelerated.sample_times)
    assert np.array_equal(original.keyframe_indices, accelerated.keyframe_indices)
    assert original.stage_gradient_norms[:-1] == accelerated.stage_gradient_norms[:-1]
    assert accelerated.converged
    assert accelerated.stage_gradient_norms[-1] <= config.tolerance


def test_solvers_agree_at_a_common_convergence_tolerance() -> None:
    config = SpringConfig(num_samples=11, iterations=100, final_iterations=10000, tolerance=1e-5)
    original = SpringQuaternionInterpolation([0.0, 1.0, 2.0, 3.0], keyframes(), config)
    accelerated = SpringQuaternionInterpolation(
        [0.0, 1.0, 2.0, 3.0], keyframes(), replace(config, solver="gauss_newton")
    )
    assert original.converged
    assert accelerated.converged
    assert original.energy_history[0] == accelerated.energy_history[0]
    assert np.isclose(original.energy_history[-1], accelerated.energy_history[-1], rtol=1e-7, atol=1e-10)
    for time in np.linspace(0.0, 3.0, 101):
        assert orientation_error(original.evaluate(float(time)), accelerated.evaluate(float(time))) < 1e-5
    assert len(accelerated.energy_history) < len(original.energy_history)


@pytest.mark.parametrize("samples", [31, 101, 1001])
def test_gauss_newton_solves_single_grid_without_changing_keyframes(samples: int) -> None:
    config = SpringConfig(num_samples=samples, refinement_levels=1, solver="gauss_newton")
    curve = SpringQuaternionInterpolation([0.0, 1.0, 2.0, 3.0], keyframes(), config)
    assert curve.converged
    assert curve.stage_gradient_norms[-1] <= config.tolerance
    assert curve.final_energy < curve.initial_energy
    assert all(
        orientation_error(curve.evaluate(float(i)), quaternion) < 1e-12 for i, quaternion in enumerate(keyframes())
    )
    history = np.asarray(curve.energy_history)
    assert np.all(np.diff(history) <= 1e-13 * np.max(history))


def test_zero_final_budget_leaves_the_common_initial_final_grid_unchanged() -> None:
    config = SpringConfig(num_samples=31, final_iterations=0)
    original = SpringQuaternionInterpolation([0.0, 1.0, 2.0, 3.0], keyframes(), config)
    accelerated = SpringQuaternionInterpolation(
        [0.0, 1.0, 2.0, 3.0], keyframes(), replace(config, solver="gauss_newton")
    )
    assert len(original.energy_history) == len(accelerated.energy_history) == 1
    assert original.stage_energy_history == accelerated.stage_energy_history
    assert not accelerated.converged
    assert all(orientation_error(a, b) < 1e-12 for a, b in zip(original.samples, accelerated.samples))


@pytest.mark.parametrize("solver", ["gradient_descent", "gauss_newton"])
def test_stationary_geodesic_reports_convergence(solver: str) -> None:
    curve = SpringQuaternionInterpolation(
        [0.0, 2.0], [Quaternion.identity(), Quaternion.from_euler_angles(0.0, 0.0, 1.6)], SpringConfig(solver=solver)
    )
    assert curve.converged
    assert curve.iterations_run == 0
    assert np.allclose(curve.evaluate_velocity(1.0), [0.0, 0.0, 0.8], atol=1e-9)


@pytest.mark.parametrize("invalid", [-1, 1.5, True])
def test_final_iteration_budget_validation(invalid: float) -> None:
    with pytest.raises(ValueError, match="final_iterations"):
        SpringConfig(final_iterations=invalid)  # type: ignore[arg-type]


def test_unknown_solver_is_rejected() -> None:
    with pytest.raises(ValueError, match="solver"):
        SpringConfig(solver="shooting")
