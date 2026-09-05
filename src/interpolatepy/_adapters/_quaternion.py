"""Adapters for the quaternion interpolation family.

The C++ ``evaluate()`` methods return C++ ``Quaternion`` objects, but user code
expects the Python ``Quaternion`` with its full API (dynamics methods, matrix
conversions, etc.).  These adapters convert on the boundary:

- Constructor: convert incoming Python Quaternions → C++ Quaternions
- ``evaluate()``: convert returned C++ Quaternions → Python Quaternions
"""

from __future__ import annotations

from typing import Any

import numpy as np

from interpolatepy._backend import get_cpp_module
from interpolatepy.quaternion.logarithmic import LogQuaternionInterpolation as _PyLogQuaternionInterpolation
from interpolatepy.quaternion.logarithmic import (
    ModifiedLogQuaternionInterpolation as _PyModifiedLogQuaternionInterpolation,
)
from interpolatepy.quaternion.core import Quaternion as _PyQuaternion
from interpolatepy.quaternion.spline import QuaternionSpline as _PyQuaternionSpline

_cpp = get_cpp_module()

_CppQuaternion = _cpp.quat.Quaternion
_CppQuaternionSpline = _cpp.quat.QuaternionSpline
_CppSquadC2 = _cpp.quat.SquadC2
_CppLogQuaternionInterpolation = _cpp.quat.LogQuaternionInterpolation
_CppModifiedLogQuaternionInterpolation = _cpp.quat.ModifiedLogQuaternionInterpolation


def _py_to_cpp(q: _PyQuaternion) -> Any:
    """Convert a Python Quaternion to a C++ Quaternion."""
    return _CppQuaternion(q.w, q.x, q.y, q.z)


def _cpp_to_py(q: Any) -> _PyQuaternion:
    """Convert a C++ Quaternion to a Python Quaternion."""
    return _PyQuaternion(q.w, q.x, q.y, q.z)


class QuaternionSpline(_CppQuaternionSpline):  # type: ignore[valid-type, misc]
    """C++-backed QuaternionSpline returning Python Quaternions."""

    def __init__(
        self,
        time_points: list[float],
        quaternions: list[_PyQuaternion],
        interpolation_method: str = "auto",
    ) -> None:
        cpp_quats = [_py_to_cpp(q) for q in quaternions]
        # Map string method names to C++ enum
        method_map = {
            "slerp": _cpp.quat.QuaternionSplineMethod.Slerp,
            "squad": _cpp.quat.QuaternionSplineMethod.Squad,
            "auto": _cpp.quat.QuaternionSplineMethod.Auto,
        }
        cpp_method = method_map.get(
            interpolation_method,
            _cpp.quat.QuaternionSplineMethod.Auto,
        )
        super().__init__(time_points, cpp_quats, cpp_method)
        self._py_quaternions = list(quaternions)
        self._method_str = interpolation_method
        self._python_impl = _PyQuaternionSpline(
            time_points,
            quaternions,
            interpolation_method,
        )
        self._use_python_evaluation = False

    def evaluate(self, t: float) -> _PyQuaternion:
        if not self.__dict__.get("_py_quaternions"):
            msg = "Cannot evaluate an empty QuaternionSpline"
            raise ValueError(msg)
        if self._use_python_evaluation:
            return self._python_impl.evaluate(t)
        return _cpp_to_py(super().evaluate(t))

    @property
    def interpolation_method(self) -> str:
        """Return the method string used for construction."""
        return getattr(self, "_method_str", "auto")

    @interpolation_method.setter
    def interpolation_method(self, value: str) -> None:
        implementation = self.__dict__.get("_python_impl")
        if implementation is not None and implementation.interpolation_method != value:
            implementation.set_interpolation_method(value)
            self._use_python_evaluation = True
        self._method_str = value

    @property
    def quat_data(self) -> Any:
        """Original quaternion waypoints."""
        implementation = self.__dict__.get("_python_impl")
        return {} if implementation is None else implementation.quat_data

    @quat_data.setter
    def quat_data(self, value: object) -> None:
        implementation = self.__dict__.get("_python_impl")
        if implementation is not None:
            implementation.quat_data = value
            self._use_python_evaluation = True

    def set_interpolation_method(self, method: str) -> None:
        """Change the interpolation method for subsequent evaluations."""
        self.interpolation_method = method

    def get_interpolation_method(self) -> str:
        """Return the active interpolation method name."""
        return self.interpolation_method

    def __getattr__(self, name: str) -> Any:
        """Delegate Python-only compatibility helpers to the reference object."""
        implementation = self.__dict__.get("_python_impl")
        if implementation is None or name.startswith("_"):
            raise AttributeError(name)
        return getattr(implementation, name)

    def get_time_range(self) -> tuple[float, float]:
        """Return (t_min, t_max)."""
        return (self.t_min, self.t_max)

    def __len__(self) -> int:
        return len(self.quat_data)

    def __str__(self) -> str:
        method = self._method_str
        count = len(self._py_quaternions)
        if count == 0:
            return f"QuaternionSpline(empty, method={method})"
        return (
            f"QuaternionSpline({count} points, "
            f"t=[{self.t_min:.3f}, {self.t_max:.3f}], method={method})"
        )

    def __repr__(self) -> str:
        return self.__str__()


class SquadC2(_CppSquadC2):  # type: ignore[valid-type, misc]
    """C++-backed SquadC2 returning Python Quaternions."""

    def __init__(
        self,
        time_points: list[float],
        quaternions: list[_PyQuaternion],
        normalize_quaternions: bool = True,
        validate_continuity: bool = True,
    ) -> None:
        cpp_quats = [_py_to_cpp(q) for q in quaternions]
        super().__init__(time_points, cpp_quats, normalize_quaternions, validate_continuity)
        self._n_original = len(quaternions)

    def evaluate(self, t: float) -> _PyQuaternion:
        return _cpp_to_py(super().evaluate(t))

    def get_time_range(self) -> tuple[float, float]:
        """Return (t_min, t_max)."""
        return (self.t_min, self.t_max)

    def __len__(self) -> int:
        return self._n_original

    def __str__(self) -> str:
        t_min, t_max = self.get_time_range()
        return (
            f"SquadC2({self._n_original} original waypoints, "
            f"t=[{t_min:.3f}, {t_max:.3f}])"
        )

    def __repr__(self) -> str:
        return self.__str__()


class LogQuaternionInterpolation(_CppLogQuaternionInterpolation):  # type: ignore[valid-type, misc]
    """C++-backed LogQuaternionInterpolation returning Python Quaternions."""

    def __init__(  # noqa: PLR0913
        self,
        time_points: list[float] | np.ndarray,
        quaternions: list[_PyQuaternion],
        degree: int = 3,
        initial_velocity: list | np.ndarray | None = None,
        final_velocity: list | np.ndarray | None = None,
        initial_acceleration: list | np.ndarray | None = None,
        final_acceleration: list | np.ndarray | None = None,
    ) -> None:
        cpp_quats = [_py_to_cpp(q) for q in quaternions]
        super().__init__(time_points, cpp_quats, degree, initial_velocity, final_velocity)
        self.time_points = np.asarray(time_points, dtype=np.float64)
        self.quaternions = list(quaternions)
        self.degree = degree
        self._initial_velocity = initial_velocity
        self._final_velocity = final_velocity
        self._initial_acceleration = initial_acceleration
        self._final_acceleration = final_acceleration
        self._python_impl: _PyLogQuaternionInterpolation | None = None
        self._use_python_evaluation = (
            initial_acceleration is not None or final_acceleration is not None
        )

    def _get_python_impl(self) -> _PyLogQuaternionInterpolation:
        if self._python_impl is None:
            self._python_impl = _PyLogQuaternionInterpolation(
                self.time_points,
                self.quaternions,
                self.degree,
                self._initial_velocity,
                self._final_velocity,
                self._initial_acceleration,
                self._final_acceleration,
            )
        return self._python_impl

    def __getattr__(self, name: str) -> Any:
        """Lazily provide Python-only diagnostics and helper methods."""
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._get_python_impl(), name)

    def evaluate(self, t: float) -> _PyQuaternion:
        if self._use_python_evaluation:
            return self._get_python_impl().evaluate(t)
        return _cpp_to_py(super().evaluate(t))

    def evaluate_velocity(self, t: float) -> np.ndarray:
        if self._use_python_evaluation:
            return self._get_python_impl().evaluate_velocity(t)
        return np.asarray(super().evaluate_velocity(t))

    def evaluate_acceleration(self, t: float) -> np.ndarray:
        if self._use_python_evaluation:
            return self._get_python_impl().evaluate_acceleration(t)
        return np.asarray(super().evaluate_acceleration(t))

    def generate_trajectory(self, num_points: int = 100) -> tuple[np.ndarray, list[_PyQuaternion]]:
        """Evaluate the native trajectory on evenly spaced time samples."""
        times = np.linspace(self.t_min, self.t_max, num_points)
        return times, [self.evaluate(float(t)) for t in times]


class ModifiedLogQuaternionInterpolation(_CppModifiedLogQuaternionInterpolation):  # type: ignore[valid-type, misc]
    """C++-backed ModifiedLogQuaternionInterpolation returning Python Quaternions."""

    def __init__(  # noqa: PLR0913
        self,
        time_points: list[float],
        quaternions: list[_PyQuaternion],
        degree: int = 3,
        normalize_axis: bool = True,
        initial_velocity: list | np.ndarray | None = None,
        final_velocity: list | np.ndarray | None = None,
        initial_acceleration: list | np.ndarray | None = None,
        final_acceleration: list | np.ndarray | None = None,
    ) -> None:
        cpp_quats = [_py_to_cpp(q) for q in quaternions]
        super().__init__(
            time_points, cpp_quats, degree, normalize_axis, initial_velocity, final_velocity
        )
        self.time_points = np.asarray(time_points, dtype=np.float64)
        self.quaternions = list(quaternions)
        self.degree = degree
        self._normalize_axis = normalize_axis
        self._initial_velocity = initial_velocity
        self._final_velocity = final_velocity
        self._initial_acceleration = initial_acceleration
        self._final_acceleration = final_acceleration
        self._python_impl: _PyModifiedLogQuaternionInterpolation | None = None
        self._use_python_evaluation = (
            initial_acceleration is not None or final_acceleration is not None
        )

    def _get_python_impl(self) -> _PyModifiedLogQuaternionInterpolation:
        if self._python_impl is None:
            self._python_impl = _PyModifiedLogQuaternionInterpolation(
                self.time_points,
                self.quaternions,
                self.degree,
                self._normalize_axis,
                self._initial_velocity,
                self._final_velocity,
                self._initial_acceleration,
                self._final_acceleration,
            )
        return self._python_impl

    def __getattr__(self, name: str) -> Any:
        """Lazily provide Python-only diagnostics and helper methods."""
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._get_python_impl(), name)

    def evaluate(self, t: float) -> _PyQuaternion:
        if self._use_python_evaluation:
            return self._get_python_impl().evaluate(t)
        return _cpp_to_py(super().evaluate(t))

    def evaluate_velocity(self, t: float) -> np.ndarray:
        if self._use_python_evaluation:
            return self._get_python_impl().evaluate_velocity(t)
        return np.asarray(super().evaluate_velocity(t))

    def evaluate_acceleration(self, t: float) -> np.ndarray:
        if self._use_python_evaluation:
            return self._get_python_impl().evaluate_acceleration(t)
        return np.asarray(super().evaluate_acceleration(t))

    def generate_trajectory(self, num_points: int = 100) -> tuple[np.ndarray, list[_PyQuaternion]]:
        """Evaluate the native trajectory on evenly spaced time samples."""
        times = np.linspace(self.t_min, self.t_max, num_points)
        return times, [self.evaluate(float(t)) for t in times]
