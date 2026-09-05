"""C++ adapter for SPRING quaternion interpolation."""

from __future__ import annotations

from typing import Any

import numpy as np

from interpolatepy._backend import get_cpp_module
from interpolatepy.quaternion.core import Quaternion as _PyQuaternion
from interpolatepy.quaternion.spring import SpringConfig as _PySpringConfig
from interpolatepy.quaternion.spring import (
    SpringQuaternionInterpolation as _PySpringQuaternionInterpolation,
)

_cpp = get_cpp_module()
_CppQuaternion = _cpp.quat.Quaternion
_CppSpringQuaternionInterpolation = _cpp.quat.SpringQuaternionInterpolation


def _py_to_cpp(q: _PyQuaternion) -> Any:
    return _CppQuaternion(q.w, q.x, q.y, q.z)


def _cpp_to_py(q: Any) -> _PyQuaternion:
    return _PyQuaternion(q.w, q.x, q.y, q.z)


def _checked_py_to_cpp(q: object, index: int) -> Any:
    if not isinstance(q, _PyQuaternion):
        msg = f"Element {index} is not a Quaternion instance"
        raise TypeError(msg)
    return _py_to_cpp(q)


class SpringQuaternionInterpolation(_CppSpringQuaternionInterpolation):  # type: ignore[valid-type, misc]
    """C++-backed SPRING interpolation with Python-compatible diagnostics."""

    def __init__(
        self,
        time_points: list[float] | np.ndarray,
        quaternions: list[_PyQuaternion],
        config: _PySpringConfig | None = None,
    ) -> None:
        active_config = config if config is not None else _PySpringConfig()
        time_values = np.asarray(time_points, dtype=np.float64)
        if time_values.ndim != 1:
            msg = "time_points must be one-dimensional"
            raise ValueError(msg)
        cpp_quaternions = [_checked_py_to_cpp(quaternion, index) for index, quaternion in enumerate(quaternions)]

        native_config = _cpp.quat.SpringConfig()
        native_config.num_samples = active_config.num_samples
        native_config.iterations = active_config.iterations
        native_config.refinement_levels = active_config.refinement_levels
        native_config.step_size = active_config.step_size
        native_config.norm_penalty = active_config.norm_penalty
        native_config.keyframe_curvature_weight = active_config.keyframe_curvature_weight
        native_config.tolerance = active_config.tolerance
        super().__init__(time_values, cpp_quaternions, native_config)
        self.config = active_config
        self.time_points = np.asarray(super().get_time_points(), dtype=np.float64)
        self.quaternions = [_cpp_to_py(quaternion) for quaternion in super().get_quaternions()]
        self._stage_energy_history = tuple(
            tuple(float(energy) for energy in history) for history in super().get_stage_energy_history()
        )

    @property
    def sample_times(self) -> np.ndarray:
        """Optimized frame times as a NumPy array."""
        return np.asarray(super().get_sample_times(), dtype=np.float64)

    @property
    def samples(self) -> list[_PyQuaternion]:
        """Optimized frames as Python quaternions."""
        return [_cpp_to_py(quaternion) for quaternion in super().get_samples()]

    @property
    def keyframe_indices(self) -> np.ndarray:
        """Indices of original keyframes in the final sample lattice."""
        return np.asarray(super().get_keyframe_indices(), dtype=np.int64)

    @property
    def refinement_sample_counts(self) -> tuple[int, ...]:
        """Sample counts used by the nested refinement stages."""
        return tuple(int(count) for count in super().get_refinement_sample_counts())

    @property
    def stage_energy_history(self) -> tuple[tuple[float, ...], ...]:
        """Immutable energy histories for every refinement stage."""
        return self._stage_energy_history

    @property
    def energy_history(self) -> tuple[float, ...]:
        """Energy history for the final refinement stage."""
        return self._stage_energy_history[-1]

    def evaluate(self, t: float) -> _PyQuaternion:
        """Evaluate the optimized trajectory at time ``t``."""
        return _cpp_to_py(super().evaluate(t))

    def generate_trajectory(
        self,
        num_points: int = 100,
    ) -> tuple[np.ndarray, list[_PyQuaternion]]:
        """Evaluate evenly spaced samples across the trajectory."""
        times, quaternions = super().generate_trajectory(num_points)
        return (
            np.asarray(times, dtype=np.float64),
            [_cpp_to_py(quaternion) for quaternion in quaternions],
        )

    @staticmethod
    def _refine_initial_curve(
        source_indices: np.ndarray,
        source_frames: np.ndarray,
        target_indices: np.ndarray,
    ) -> np.ndarray:
        """Retain the reference implementation's diagnostic helper."""
        return _PySpringQuaternionInterpolation._refine_initial_curve(  # noqa: SLF001
            source_indices,
            source_frames,
            target_indices,
        )

    def __len__(self) -> int:
        return len(self.samples)
