"""Adapters for the quaternion interpolation family.

The C++ ``evaluate()`` methods return C++ ``Quaternion`` objects, but user code
expects the Python ``Quaternion`` with its full API (dynamics methods, matrix
conversions, etc.).  These adapters convert on the boundary:

- Constructor: convert incoming Python Quaternions → C++ Quaternions
- ``evaluate()``: convert returned C++ Quaternions → Python Quaternions
"""

from __future__ import annotations

from typing import Any

from interpolatepy._backend import get_cpp_module
from interpolatepy.quat_core import Quaternion as _PyQuaternion

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

    def evaluate(self, t: float) -> _PyQuaternion:
        if not self._py_quaternions:
            msg = "Cannot evaluate an empty QuaternionSpline"
            raise ValueError(msg)
        return _cpp_to_py(super().evaluate(t))

    @property
    def interpolation_method(self) -> str:
        """Return the method string used for construction."""
        return getattr(self, "_method_str", "auto")

    @interpolation_method.setter
    def interpolation_method(self, value: str) -> None:
        self._method_str = value

    @property
    def quat_data(self) -> list[_PyQuaternion]:
        """Original quaternion waypoints."""
        return getattr(self, "_py_quaternions", [])

    @quat_data.setter
    def quat_data(self, value: object) -> None:
        self._py_quaternions = value  # type: ignore[assignment]

    def get_time_range(self) -> tuple[float, float]:
        """Return (t_min, t_max)."""
        return (self.t_min, self.t_max)

    def __len__(self) -> int:
        return len(self._py_quaternions)

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

    def __init__(
        self,
        time_points: list[float],
        quaternions: list[_PyQuaternion],
        degree: int = 3,
        initial_velocity: object = None,
        final_velocity: object = None,
    ) -> None:
        cpp_quats = [_py_to_cpp(q) for q in quaternions]
        super().__init__(time_points, cpp_quats, degree, initial_velocity, final_velocity)

    def evaluate(self, t: float) -> _PyQuaternion:
        return _cpp_to_py(super().evaluate(t))


class ModifiedLogQuaternionInterpolation(_CppModifiedLogQuaternionInterpolation):  # type: ignore[valid-type, misc]
    """C++-backed ModifiedLogQuaternionInterpolation returning Python Quaternions."""

    def __init__(  # noqa: PLR0913
        self,
        time_points: list[float],
        quaternions: list[_PyQuaternion],
        degree: int = 3,
        normalize_axis: bool = True,
        initial_velocity: object = None,
        final_velocity: object = None,
    ) -> None:
        cpp_quats = [_py_to_cpp(q) for q in quaternions]
        super().__init__(
            time_points, cpp_quats, degree, normalize_axis, initial_velocity, final_velocity
        )

    def evaluate(self, t: float) -> _PyQuaternion:
        return _cpp_to_py(super().evaluate(t))
