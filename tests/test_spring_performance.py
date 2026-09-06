"""Regression checks for result-preserving SPRING implementation optimizations."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from interpolatepy.quaternion import _spring_solver
from interpolatepy.quaternion._spring_energy import EnergyEvaluation
from interpolatepy.quaternion._spring_energy import SpringEnergy
from interpolatepy.quaternion._spring_gauss_newton import descent_direction
from interpolatepy.quaternion._spring_gauss_newton import normal_matrix
from interpolatepy.quaternion._spring_slerp import slerp_segments
from interpolatepy.quaternion.core import Quaternion
from interpolatepy.quaternion.spring import SpringConfig
from interpolatepy.quaternion.spring import SpringQuaternionInterpolation


def values(quaternion: Quaternion) -> np.ndarray:
    return np.array([quaternion.s_, *quaternion.v_])


@pytest.mark.parametrize("seed", range(5))
def test_batched_slerp_matches_scalar_operations(seed: int) -> None:
    random = np.random.default_rng(seed)
    frames = random.normal(size=(12, 4))
    frames /= np.linalg.norm(frames, axis=1)[:, None]
    quaternions = [Quaternion(*frame) for frame in frames]
    segments = np.repeat(np.arange(len(frames) - 1), 17)
    fractions = np.tile(np.linspace(0.0, 1.0, 17), len(frames) - 1)
    expected = np.array([values(quaternions[i].slerp(quaternions[i + 1], t)) for i, t in zip(segments, fractions)])
    actual = slerp_segments(quaternions, segments, fractions)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)
    # In particular, do not align or renormalize exact endpoints.
    np.testing.assert_array_equal(actual[::17], frames[:-1])
    np.testing.assert_array_equal(actual[16::17], frames[1:])


@pytest.mark.parametrize("half_angle", [0.0, 1e-12, 0.999e-7, 1.001e-7, 1e-5, np.pi / 2.0])
def test_batched_slerp_retains_small_angle_and_antipodal_rules(half_angle: float) -> None:
    start = Quaternion.identity()
    end = -Quaternion(float(np.cos(half_angle)), float(np.sin(half_angle)), 0.0, 0.0)
    fractions = np.array([0.0, 1e-10, 0.3, 0.999, 1.0])
    actual = slerp_segments([start, end], np.zeros(len(fractions), dtype=int), fractions)
    expected = np.array([values(start.slerp(end, t)) for t in fractions])
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)


def test_refinement_keeps_nonunit_coarse_anchors_and_scalar_slerp() -> None:
    random = np.random.default_rng(22)
    frames = random.normal(size=(5, 4))
    indices = np.array([0, 1, 7, 10, 19])
    targets = np.arange(20)
    refined = SpringQuaternionInterpolation._refine_initial_curve(indices, frames, targets)
    np.testing.assert_array_equal(refined[indices], frames)
    for index in targets:
        if index in indices:
            continue
        upper = np.searchsorted(indices, index)
        lower = upper - 1
        start, end = Quaternion(*frames[lower]).unit(), Quaternion(*frames[upper]).unit()
        expected = start.slerp(end, float(index - indices[lower]) / float(indices[upper] - indices[lower]))
        np.testing.assert_allclose(refined[index], values(expected), rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(SpringQuaternionInterpolation._refine_initial_curve(indices, frames, indices), frames)


@pytest.mark.parametrize("indices", [None, np.array([0, 1, 3, 4, 8, 12, 13])])
def test_cached_energy_and_gradients_are_independent(indices: np.ndarray | None) -> None:
    random = np.random.default_rng(11)
    frames = random.normal(size=(7, 4))
    weights = random.uniform(0.7, 1.5, size=5)
    model = SpringEnergy(7, weights, 81.0, indices)
    trial = model.evaluate(frames)
    gradient = trial.gradient()
    energy, expected = _spring_solver.curvature_energy_gradient(frames, weights, 81.0, indices)
    assert trial.energy == energy
    np.testing.assert_array_equal(gradient, expected)
    model.evaluate(frames * 0.8).gradient()
    np.testing.assert_array_equal(trial.gradient(), gradient)
    gradient[:] = 0.0
    np.testing.assert_array_equal(trial.gradient(), expected)


def eager_minimize(
    initial: np.ndarray, fixed: np.ndarray, weights: np.ndarray, settings: SpringConfig, indices: np.ndarray
) -> tuple[np.ndarray, tuple[float, ...]]:
    """Frozen pre-optimization line search, computing every trial gradient."""
    frames = initial.copy()
    energy, gradient = _spring_solver.curvature_energy_gradient(frames, weights, settings.norm_penalty, indices)
    history = [energy]
    for _ in range(settings.iterations):
        gradient[fixed] = 0.0
        gradient_norm = float(np.linalg.norm(gradient))
        if gradient_norm <= settings.tolerance:
            break
        if settings.solver == "gauss_newton":
            direction = -descent_direction(
                normal_matrix(frames, weights, settings.norm_penalty, fixed, indices), gradient
            )
            step = 1.0
        else:
            direction = gradient / gradient_norm
            step = settings.step_size
        accepted = False
        for _ in range(30):
            if settings.solver == "gauss_newton":
                lengths = np.linalg.norm(frames, axis=1)
                radial = np.sum(frames * direction, axis=1) / lengths
                target_lengths = lengths - step * radial
                if np.any(target_lengths <= 0.0):
                    step *= 0.5
                    continue
                tangent = direction - (radial / lengths)[:, None] * frames
                candidate = frames - step * tangent
                candidate *= (target_lengths / np.linalg.norm(candidate, axis=1))[:, None]
            else:
                candidate = frames - step * direction
            candidate[fixed] = initial[fixed]
            candidate_energy, candidate_gradient = _spring_solver.curvature_energy_gradient(
                candidate, weights, settings.norm_penalty, indices
            )
            roundoff_progress = (
                settings.solver == "gauss_newton"
                and abs(candidate_energy - energy) <= 64.0 * np.finfo(float).eps * max(abs(energy), 1e-30)
                and np.linalg.norm(candidate_gradient[~fixed]) < 0.5 * gradient_norm
            )
            if candidate_energy < energy or roundoff_progress:
                frames, energy, gradient = candidate, candidate_energy, candidate_gradient
                history.append(energy)
                accepted = True
                break
            step *= 0.5
            if step < 1e-12:
                break
        if not accepted:
            break
    return frames, tuple(history)


@pytest.mark.parametrize("solver", ["gradient_descent", "gauss_newton"])
@pytest.mark.parametrize("seed", range(3))
def test_lazy_backtracking_matches_eager_solver(solver: str, seed: int) -> None:
    random = np.random.default_rng(seed)
    frames = random.normal(size=(11, 4))
    frames /= np.linalg.norm(frames, axis=1)[:, None]
    frames *= random.uniform(0.9, 1.1, size=(11, 1))
    indices = np.cumsum(random.integers(1, 5, size=11))
    fixed = np.zeros(len(frames), dtype=bool)
    fixed[[0, 4, 10]] = True
    weights = np.ones(len(frames) - 2)
    settings = SpringConfig(iterations=30, solver=solver, step_size=1.0)
    expected, history = eager_minimize(frames, fixed, weights, settings, indices)
    actual, actual_history = _spring_solver.minimize(frames, fixed, weights, settings, 30, indices, solver)
    np.testing.assert_array_equal(actual, expected)
    assert actual_history == history


def test_rejected_descent_trials_skip_gradient_accumulation() -> None:
    frames = np.array([[1.0, 0.0, 0.0, 0.0], [0.8, 0.6, 0.0, 0.0], [0.7, 0.0, 0.7, 0.0]])
    with (
        patch.object(EnergyEvaluation, "gradient", autospec=True, side_effect=EnergyEvaluation.gradient) as gradient,
        patch.object(SpringEnergy, "evaluate", autospec=True, side_effect=SpringEnergy.evaluate) as evaluate,
    ):
        _, history = _spring_solver.minimize(frames, np.array([True, False, True]), np.ones(1), SpringConfig(), 10)
    assert gradient.call_count == len(history)
    assert evaluate.call_count > gradient.call_count


@pytest.mark.parametrize(("gradient_scale", "accepted"), [(0.25, True), (0.75, False)])
def test_roundoff_trials_still_check_free_gradient(gradient_scale: float, accepted: bool) -> None:
    frames = np.tile([1.0, 0.0, 0.0, 0.0], (3, 1))
    fixed = np.array([True, False, True])
    original_gradient = np.zeros_like(frames)
    original_gradient[1, 1] = 1.0
    candidate_gradient = original_gradient * gradient_scale
    candidate_gradient[fixed] = 1e6  # Fixed-node derivatives must not enter acceptance.
    first = SimpleNamespace(energy=1.0, gradient=original_gradient.copy)
    trial = SimpleNamespace(energy=1.0 + np.finfo(float).eps, gradient=candidate_gradient.copy)
    with patch.object(SpringEnergy, "evaluate", side_effect=[first, *([trial] * 30)]):
        _, history = _spring_solver.minimize(frames, fixed, np.ones(1), SpringConfig(), 1, solver="gauss_newton")
    assert len(history) == (2 if accepted else 1)


def test_banded_factorization_does_not_overwrite_input_model() -> None:
    band = np.zeros((12, 16))
    band[0] = 2.0
    gradient = np.arange(16.0).reshape(4, 4)
    saved_band, saved_gradient = band.copy(), gradient.copy()
    np.testing.assert_allclose(descent_direction(band, gradient), -gradient / 2.0, rtol=1e-15, atol=0.0)
    np.testing.assert_array_equal(band, saved_band)
    np.testing.assert_array_equal(gradient, saved_gradient)
