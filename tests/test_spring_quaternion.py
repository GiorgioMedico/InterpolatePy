"""Tests for the SPRING quaternion interpolation algorithm."""

from typing import Any

import numpy as np
import pytest

import interpolatepy
from interpolatepy import Quaternion
from interpolatepy import QuaternionTrajectory
from interpolatepy import SpringConfig
from interpolatepy import SpringQuaternionInterpolation
from interpolatepy.quaternion._spring_solver import curvature_energy_gradient
from interpolatepy.quaternion._spring_solver import iteration_budgets
from interpolatepy.quaternion._spring_solver import nested_level_indices
from interpolatepy.quaternion._spring_solver import refinement_sample_counts


def test_spring_exports_match_grouped_namespace() -> None:
    from interpolatepy.quaternion import SpringConfig as GroupedSpringConfig
    from interpolatepy.quaternion import (
        SpringQuaternionInterpolation as GroupedSpringQuaternionInterpolation,
    )

    assert GroupedSpringConfig is SpringConfig
    assert GroupedSpringQuaternionInterpolation is SpringQuaternionInterpolation
    assert "SpringConfig" in interpolatepy.__all__
    assert "SpringQuaternionInterpolation" in interpolatepy.__all__


def _same_orientation(first: Quaternion, second: Quaternion, atol: float = 1e-9) -> bool:
    return bool(np.isclose(abs(first.dot_product(second)), 1.0, atol=atol))


def _curved_keyframes() -> tuple[list[float], list[Quaternion]]:
    return (
        [0.0, 1.0, 2.0, 3.0],
        [
            Quaternion.identity(),
            Quaternion.from_euler_angles(0.9, 0.1, 0.2),
            Quaternion.from_euler_angles(0.2, 1.1, 0.5),
            Quaternion.from_euler_angles(-0.4, 0.3, 1.4),
        ],
    )


def test_spring_preserves_keyframes_and_unit_sphere() -> None:
    times, keyframes = _curved_keyframes()
    spring = SpringQuaternionInterpolation(
        times,
        keyframes,
        SpringConfig(num_samples=41, iterations=150),
    )

    assert len(spring.samples) == 41
    assert np.allclose(spring.sample_times[spring.keyframe_indices], times)
    assert all(_same_orientation(spring.evaluate(t), expected) for t, expected in zip(times, keyframes))
    assert all(np.isclose(sample.norm(), 1.0, atol=1e-12) for sample in spring.samples)


def test_spring_reduces_discrete_curvature_energy() -> None:
    times, keyframes = _curved_keyframes()
    spring = SpringQuaternionInterpolation(
        times,
        keyframes,
        SpringConfig(num_samples=41, iterations=200),
    )

    assert spring.final_energy < spring.initial_energy
    assert spring.iterations_run > 0
    assert np.all(np.diff(spring.energy_history) < 0.0)
    assert spring.energy_history is spring.stage_energy_history[-1]
    assert all(np.all(np.diff(history) < 0.0) for history in spring.stage_energy_history)


def test_spring_uses_paper_shaped_multistep_schedule() -> None:
    assert refinement_sample_counts(6, 550, 3) == (22, 110, 550)
    assert iteration_budgets(450, 3) == (200, 150, 100)

    times, keyframes = _curved_keyframes()
    spring = SpringQuaternionInterpolation(
        times,
        keyframes,
        SpringConfig(num_samples=101, iterations=300),
    )

    assert spring.refinement_sample_counts == (7, 20, 101)
    assert isinstance(spring.refinement_sample_counts, tuple)
    assert isinstance(spring.stage_energy_history, tuple)
    assert all(isinstance(history, tuple) for history in spring.stage_energy_history)
    assert len(spring.stage_energy_history) == 3
    assert spring.iterations_run <= 300


def test_spring_can_retain_one_stage_optimization() -> None:
    times, keyframes = _curved_keyframes()
    spring = SpringQuaternionInterpolation(
        times,
        keyframes,
        SpringConfig(num_samples=31, iterations=30, refinement_levels=1),
    )

    assert spring.refinement_sample_counts == (31,)
    assert len(spring.stage_energy_history) == 1
    assert spring.energy_history == spring.stage_energy_history[0]


def test_refinement_grids_are_nested_and_copy_preceding_samples() -> None:
    keyframe_indices = np.array([0, 25, 60, 100])
    levels = nested_level_indices(101, keyframe_indices, (7, 20, 101))
    assert set(levels[0]).issubset(levels[1])
    assert set(levels[1]).issubset(levels[2])
    assert set(keyframe_indices).issubset(levels[0])

    random = np.random.default_rng(7)
    source_frames = random.normal(size=(len(levels[0]), 4))
    source_frames /= np.linalg.norm(source_frames, axis=1)[:, None]
    refined = SpringQuaternionInterpolation._refine_initial_curve(levels[0], source_frames, levels[1])
    copied_positions = np.searchsorted(levels[1], levels[0])
    assert np.array_equal(refined[copied_positions], source_frames)


def test_spring_analytic_gradient_matches_centered_difference() -> None:
    random = np.random.default_rng(42)
    frames = random.normal(size=(7, 4))
    frames /= np.linalg.norm(frames, axis=1)[:, None]
    weights = np.array([1.0, 1.0, 1.2, 1.0, 1.0])
    _, gradient = curvature_energy_gradient(frames, weights, norm_penalty=100.0)
    delta = 1e-6

    for index in ((0, 2), (2, 1), (3, 3), (6, 0)):
        above = frames.copy()
        below = frames.copy()
        above[index] += delta
        below[index] -= delta
        energy_above, _ = curvature_energy_gradient(above, weights, norm_penalty=100.0)
        energy_below, _ = curvature_energy_gradient(below, weights, norm_penalty=100.0)
        numerical = (energy_above - energy_below) / (2.0 * delta)
        assert np.isclose(gradient[index], numerical, rtol=1e-7, atol=1e-7)


def test_keyframe_weight_scales_only_its_curvature_gradient_terms() -> None:
    random = np.random.default_rng(9)
    frames = random.normal(size=(8, 4))
    frames /= np.linalg.norm(frames, axis=1)[:, None]
    keyframe_index = 3
    curvature_index = keyframe_index - 1
    keyframe_weight = 1.7
    unweighted = np.ones(len(frames) - 2)
    weighted = unweighted.copy()
    weighted[curvature_index] = keyframe_weight

    _, base_gradient = curvature_energy_gradient(frames, unweighted, norm_penalty=100.0)
    _, weighted_gradient = curvature_energy_gradient(frames, weighted, norm_penalty=100.0)
    difference = weighted_gradient - base_gradient

    second_difference = frames[keyframe_index - 1] - 2.0 * frames[keyframe_index] + frames[keyframe_index + 1]
    center = frames[keyframe_index]
    projection_scale = np.dot(second_difference, center) / np.dot(center, center)
    curvature = second_difference - projection_scale * center
    extra_weight = keyframe_weight - 1.0

    expected_neighbor = 2.0 * extra_weight * curvature
    expected_center = (-4.0 - 2.0 * projection_scale) * extra_weight * curvature
    assert np.allclose(difference[keyframe_index - 1], expected_neighbor)
    assert np.allclose(difference[keyframe_index], expected_center)
    assert np.allclose(difference[keyframe_index + 1], expected_neighbor)
    assert np.allclose(difference[: keyframe_index - 1], 0.0)
    assert np.allclose(difference[keyframe_index + 2 :], 0.0)


def test_spring_leaves_a_geodesic_unchanged() -> None:
    axis = np.array([0.0, 0.0, 1.0])
    keyframes = [Quaternion.from_angle_axis(angle, axis) for angle in (0.0, 0.8, 1.6)]
    spring = SpringQuaternionInterpolation(
        [0.0, 1.0, 2.0],
        keyframes,
        SpringConfig(num_samples=21, iterations=100),
    )

    for time in np.linspace(0.0, 2.0, 17):
        expected = keyframes[0].slerp(keyframes[-1], float(time / 2.0))
        assert _same_orientation(spring.evaluate(float(time)), expected, atol=1e-8)


def test_spring_is_invariant_to_keyframe_signs() -> None:
    times, keyframes = _curved_keyframes()
    flipped = [keyframes[0], -keyframes[1], keyframes[2], -keyframes[3]]
    config = SpringConfig(num_samples=31, iterations=100)
    original = SpringQuaternionInterpolation(times, keyframes, config)
    equivalent = SpringQuaternionInterpolation(times, flipped, config)

    for time in np.linspace(times[0], times[-1], 13):
        assert _same_orientation(original.evaluate(float(time)), equivalent.evaluate(float(time)))


def test_spring_distributes_frames_by_rotation_distance() -> None:
    axis = np.array([1.0, 0.0, 0.0])
    keyframes = [
        Quaternion.identity(),
        Quaternion.from_angle_axis(0.1, axis),
        Quaternion.from_angle_axis(1.5, axis),
    ]
    spring = SpringQuaternionInterpolation(
        [0.0, 1.0, 2.0],
        keyframes,
        SpringConfig(num_samples=21, iterations=0),
    )
    intervals = np.diff(spring.keyframe_indices)

    assert intervals[1] > intervals[0]
    assert int(np.sum(intervals)) == 20


def test_spring_conforms_to_quaternion_trajectory_protocol() -> None:
    times, keyframes = _curved_keyframes()
    spring = SpringQuaternionInterpolation(
        times,
        keyframes,
        SpringConfig(num_samples=21, iterations=20),
    )

    assert isinstance(spring, QuaternionTrajectory)
    assert spring.evaluate_velocity(1.5).shape == (3,)
    assert spring.evaluate_acceleration(1.5).shape == (3,)
    assert np.all(np.isfinite(spring.evaluate_velocity(1.5)))
    assert np.all(np.isfinite(spring.evaluate_acceleration(1.5)))

    trajectory_times, trajectory = spring.generate_trajectory(17)
    assert len(trajectory_times) == len(trajectory) == 17
    assert all(np.isclose(quaternion.norm(), 1.0, atol=1e-12) for quaternion in trajectory)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (SpringConfig(num_samples=2), "number of keyframes"),
        (SpringConfig(num_samples=10, iterations=0), None),
    ],
)
def test_spring_configured_sample_count_validation(config: SpringConfig, message: str | None) -> None:
    times, keyframes = _curved_keyframes()
    if message is None:
        spring = SpringQuaternionInterpolation(times, keyframes, config)
        assert len(spring.samples) == config.num_samples
    else:
        with pytest.raises(ValueError, match=message):
            SpringQuaternionInterpolation(times, keyframes, config)


@pytest.mark.parametrize(
    ("times", "keyframes", "message"),
    [
        ([0.0], [Quaternion.identity()], "At least 2"),
        ([0.0, 1.0], [Quaternion.identity()], "must match"),
        (
            [0.0, 0.0],
            [Quaternion.identity(), Quaternion.identity()],
            "strictly increasing",
        ),
        (
            [0.0, 1.0],
            [Quaternion.identity(), Quaternion(0.0, 0.0, 0.0, 0.0)],
            "finite and non-zero",
        ),
    ],
)
def test_spring_rejects_invalid_inputs(times: list[float], keyframes: list[Quaternion], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SpringQuaternionInterpolation(times, keyframes, SpringConfig(num_samples=8))


def test_spring_rejects_out_of_range_evaluation() -> None:
    spring = SpringQuaternionInterpolation(
        [0.0, 1.0],
        [Quaternion.identity(), Quaternion.from_euler_angles(0.2, 0.1, 0.3)],
        SpringConfig(num_samples=8, iterations=5),
    )

    with pytest.raises(ValueError, match="outside valid range"):
        spring.evaluate(-0.1)
    with pytest.raises(ValueError, match="at least 2"):
        spring.generate_trajectory(1)


@pytest.mark.parametrize(
    "arguments",
    [
        {"num_samples": 1},
        {"iterations": -1},
        {"refinement_levels": 0},
        {"step_size": 0.0},
        {"norm_penalty": 0.0},
        {"keyframe_curvature_weight": 0.0},
        {"tolerance": -1.0},
    ],
)
def test_spring_config_rejects_invalid_values(arguments: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="must be"):
        SpringConfig(**arguments)
