"""Native natural Riemannian cubic interpolation with Python value objects."""

from __future__ import annotations

from typing import Any

import numpy as np

from interpolatepy._backend import get_cpp_module
from interpolatepy.quaternion.core import Quaternion
from interpolatepy.quaternion.shooting import ShootingConfig


_cpp = get_cpp_module().quat
_CppShooting = _cpp.ShootingQuaternionInterpolation
_MIN_SAMPLES = 2


def _to_python(value: Any) -> Quaternion:
    return Quaternion(value.w, value.x, value.y, value.z)


class ShootingQuaternionInterpolation(_CppShooting):  # type: ignore[valid-type, misc]
    """C++ multiple shooting with sparse Newton and analytic sensitivities."""

    def __init__(
        self,
        time_points: list[float] | np.ndarray,
        quaternions: list[Quaternion],
        config: ShootingConfig | None = None,
    ) -> None:
        self.config = config if config is not None else ShootingConfig()
        times = np.asarray(time_points, dtype=np.float64)
        if times.ndim != 1:
            raise ValueError("time_points must be one-dimensional")
        native_quaternions = []
        for index, quaternion in enumerate(quaternions):
            if not isinstance(quaternion, Quaternion):
                raise TypeError(f"Element {index} is not a Quaternion instance")
            native_quaternions.append(_cpp.Quaternion(quaternion.w, quaternion.x, quaternion.y, quaternion.z))
        native_config = _cpp.ShootingConfig()
        native_config.tolerance = self.config.tolerance
        native_config.max_iterations = self.config.max_iterations
        native_config.integration_steps = self.config.integration_steps
        native_config.max_integration_steps = self.config.max_integration_steps
        super().__init__(times, native_quaternions, native_config)
        self.time_points = np.asarray(super().get_time_points(), dtype=np.float64)
        self.quaternions = [_to_python(quaternion) for quaternion in super().get_quaternions()]

    def evaluate(self, t: float) -> Quaternion:
        """Evaluate the continuous orientation curve."""
        return _to_python(super().evaluate(t))

    def generate_trajectory(self, num_points: int = 100) -> tuple[np.ndarray, list[Quaternion]]:
        """Sample the solved trajectory without rerunning optimization."""
        if isinstance(num_points, bool) or not isinstance(num_points, (int, np.integer)) or num_points < _MIN_SAMPLES:
            raise ValueError("num_points must be an integer of at least 2")
        times, values = super().generate_trajectory(num_points)
        return np.asarray(times, dtype=np.float64), [_to_python(value) for value in values]
