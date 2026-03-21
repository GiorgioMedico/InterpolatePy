"""Adapters for motion profile classes.  # noqa: PLC0415

``DoubleSTrajectory`` and ``ParabolicBlendTrajectory`` need adapters because
the C++ ``evaluate()`` returns a struct while the Python API exposes separate
``evaluate()``, ``evaluate_velocity()``, ``evaluate_acceleration()`` methods.

``TrapezoidalTrajectory`` and ``PolynomialTrajectory`` need adapters because
the Python API uses class methods returning callables, while the C++ API is
instance-based with ``evaluate()`` methods.

Deferred imports from ``interpolatepy.trapezoidal`` and ``interpolatepy.polynomials``
are intentional to avoid circular imports (``__init__.py`` → ``_adapters`` → module).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from interpolatepy.polynomials import BoundaryCondition as PyBoundaryCondition
    from interpolatepy.polynomials import TimeInterval as PyTimeInterval
    from interpolatepy.polynomials import TrajectoryParams as PolyTrajectoryParams
    from interpolatepy.trapezoidal import InterpolationParams
    from interpolatepy.trapezoidal import TrajectoryParams as TrapTrajectoryParams

from interpolatepy._backend import get_cpp_module

_cpp = get_cpp_module()

_CppDoubleSTrajectory = _cpp.motion.DoubleSTrajectory
_CppParabolicBlendTrajectory = _cpp.motion.ParabolicBlendTrajectory
_CppTrapezoidalTrajectory = _cpp.motion.TrapezoidalTrajectory
_CppPolynomialTrajectory = _cpp.motion.PolynomialTrajectory
_CppBoundaryCondition = _cpp.motion.BoundaryCondition
_CppTimeInterval = _cpp.motion.TimeInterval


class DoubleSTrajectory(_CppDoubleSTrajectory):  # type: ignore[valid-type, misc]
    """C++-backed DoubleSTrajectory matching the Python ScalarTrajectory protocol.

    The C++ class has a single ``evaluate(t) -> TrajectoryResult`` whereas the
    Python protocol requires separate position / velocity / acceleration
    methods.  This adapter splits the result accordingly.
    """

    def evaluate(self, t: float | np.ndarray) -> float | np.ndarray:
        if isinstance(t, np.ndarray):
            out = np.empty_like(t)
            for i, ti in enumerate(t.flat):
                out.flat[i] = super().evaluate(float(ti)).position
            return out
        return super().evaluate(t).position

    def evaluate_velocity(self, t: float | np.ndarray) -> float | np.ndarray:
        if isinstance(t, np.ndarray):
            out = np.empty_like(t)
            for i, ti in enumerate(t.flat):
                out.flat[i] = super().evaluate(float(ti)).velocity
            return out
        return super().evaluate(t).velocity

    def evaluate_acceleration(self, t: float | np.ndarray) -> float | np.ndarray:
        if isinstance(t, np.ndarray):
            out = np.empty_like(t)
            for i, ti in enumerate(t.flat):
                out.flat[i] = super().evaluate(float(ti)).acceleration
            return out
        return super().evaluate(t).acceleration

    def evaluate_jerk(self, t: float | np.ndarray) -> float | np.ndarray:
        if isinstance(t, np.ndarray):
            out = np.empty_like(t)
            for i, ti in enumerate(t.flat):
                out.flat[i] = super().evaluate(float(ti)).jerk
            return out
        return super().evaluate(t).jerk

    def evaluate_full(
        self, t: float | np.ndarray
    ) -> tuple[float | np.ndarray, ...]:
        """Evaluate position, velocity, acceleration, and jerk at time *t*."""
        if isinstance(t, np.ndarray):
            pos = np.empty_like(t)
            vel = np.empty_like(t)
            acc = np.empty_like(t)
            jrk = np.empty_like(t)
            for i, ti in enumerate(t.flat):
                r = super().evaluate(float(ti))
                pos.flat[i] = r.position
                vel.flat[i] = r.velocity
                acc.flat[i] = r.acceleration
                jrk.flat[i] = r.jerk
            return pos, vel, acc, jrk
        r = super().evaluate(t)
        return r.position, r.velocity, r.acceleration, r.jerk

    @property
    def T(self) -> float:  # noqa: N802
        """Total trajectory duration."""
        return self.duration


class ParabolicBlendTrajectory(_CppParabolicBlendTrajectory):  # type: ignore[valid-type, misc]
    """C++-backed ParabolicBlendTrajectory with evaluate protocol split."""

    def evaluate(self, t: float | np.ndarray) -> float | np.ndarray:
        if isinstance(t, np.ndarray):
            out = np.empty_like(t)
            for i, ti in enumerate(t.flat):
                out.flat[i] = super().evaluate(float(ti)).position
            return out
        return super().evaluate(t).position

    def evaluate_velocity(self, t: float | np.ndarray) -> float | np.ndarray:
        if isinstance(t, np.ndarray):
            out = np.empty_like(t)
            for i, ti in enumerate(t.flat):
                out.flat[i] = super().evaluate(float(ti)).velocity
            return out
        return super().evaluate(t).velocity

    def evaluate_acceleration(self, t: float | np.ndarray) -> float | np.ndarray:
        if isinstance(t, np.ndarray):
            out = np.empty_like(t)
            for i, ti in enumerate(t.flat):
                out.flat[i] = super().evaluate(float(ti)).acceleration
            return out
        return super().evaluate(t).acceleration


# ---------------------------------------------------------------------------
# TrapezoidalTrajectory adapter
# ---------------------------------------------------------------------------


def _to_cpp_bc(bc: PyBoundaryCondition) -> object:
    """Convert a Python BoundaryCondition to C++ BoundaryCondition."""
    cpp_bc = _CppBoundaryCondition()
    cpp_bc.position = bc.position
    cpp_bc.velocity = bc.velocity
    cpp_bc.acceleration = bc.acceleration
    cpp_bc.jerk = bc.jerk
    return cpp_bc


def _to_cpp_ti(ti: PyTimeInterval) -> object:
    """Convert a Python TimeInterval to C++ TimeInterval."""
    cpp_ti = _CppTimeInterval()
    cpp_ti.start = ti.start
    cpp_ti.end = ti.end
    return cpp_ti


class TrapezoidalTrajectory:
    """C++-backed TrapezoidalTrajectory matching the pure-Python class API.

    The pure-Python version exposes ``generate_trajectory()`` and
    ``interpolate_waypoints()`` class methods that return *callables*.
    The C++ class is instance-based with ``evaluate(t) -> TrajectoryResult``.
    This adapter bridges the two by wrapping C++ instances inside closures.
    """

    @staticmethod
    def generate_trajectory(
        params: TrapTrajectoryParams,
    ) -> tuple[Callable[[float], tuple[float, float, float]], float]:
        """Generate a single-segment trapezoidal trajectory.

        Parameters
        ----------
        params : TrajectoryParams
            Parameters including q0, q1, v0, v1, amax, vmax, duration.

        Returns
        -------
        tuple[Callable, float]
            Trajectory callable and total duration.
        """
        if params.amax is None:
            raise ValueError("Maximum acceleration (amax) must be provided")
        if params.duration is None and params.vmax is None:
            raise ValueError(
                "Either duration or maximum velocity (vmax) must be provided"
            )

        amax = abs(params.amax)

        if params.vmax is not None and params.duration is None:
            # Velocity-based: delegate to C++
            vmax = abs(params.vmax)
            cpp_traj = _CppTrapezoidalTrajectory(
                params.q0, params.q1, amax, vmax,
                params.v0, params.v1, params.t0,
            )
        elif params.duration is not None:
            # Duration-based: use C++ duration-based constructor (kwargs
            # needed to disambiguate from velocity-based overload)
            cpp_traj = _CppTrapezoidalTrajectory(
                q0=params.q0, q1=params.q1, amax=amax,
                v0=params.v0, v1=params.v1, t0=params.t0,
                duration=params.duration,
            )
        else:
            raise ValueError(
                "Invalid parameter combination. Provide either "
                "(amax, vmax) or (amax, duration)."
            )

        duration = cpp_traj.duration

        def trajectory(t: float) -> tuple[float, float, float]:
            r = cpp_traj.evaluate(t)
            return r.position, r.velocity, r.acceleration

        return trajectory, duration

    @staticmethod
    def calculate_heuristic_velocities(
        q_list: list[float],
        v0: float,
        vn: float,
        v_max: float | None = None,
        amax: float | None = None,
    ) -> list[float]:
        """Calculate intermediate velocities heuristically."""
        from interpolatepy.trapezoidal import (  # noqa: PLC0415
            TrapezoidalTrajectory as _PyTrap,
        )

        return _PyTrap.calculate_heuristic_velocities(
            q_list, v0, vn, v_max, amax
        )

    @classmethod
    def interpolate_waypoints(
        cls,
        params: InterpolationParams,
    ) -> tuple[Callable[[float], tuple[float, float, float]], float]:
        """Generate a multi-segment trajectory through waypoints.

        Parameters
        ----------
        params : InterpolationParams
            Parameters including points, velocities, times, amax, vmax.

        Returns
        -------
        tuple[Callable, float]
            Trajectory callable and total duration.
        """
        from interpolatepy.trapezoidal import (  # noqa: PLC0415
            TrapezoidalTrajectory as _PyTrap,
        )

        return _PyTrap.interpolate_waypoints(params)


# ---------------------------------------------------------------------------
# PolynomialTrajectory adapter
# ---------------------------------------------------------------------------

_ORDER_3 = 3
_ORDER_5 = 5
_ORDER_7 = 7


class PolynomialTrajectory:
    """C++-backed PolynomialTrajectory matching the pure-Python class API.

    The pure-Python version exposes ``order_3_trajectory()`` etc. class methods
    that return callables.  The C++ class takes ``order`` in its constructor
    and returns ``FullTrajectoryResult`` structs.  This adapter creates C++
    instances and wraps their ``evaluate()`` in closures.
    """

    VALID_ORDERS: ClassVar[tuple[int, ...]] = (3, 5, 7)

    @staticmethod
    def _make_callable(
        initial: PyBoundaryCondition,
        final: PyBoundaryCondition,
        time: PyTimeInterval,
        order: int,
    ) -> Callable[[float], tuple[float, float, float, float]]:
        """Create a C++ PolynomialTrajectory and wrap it in a callable."""
        cpp_traj = _CppPolynomialTrajectory(
            _to_cpp_bc(initial), _to_cpp_bc(final), _to_cpp_ti(time), order
        )

        def trajectory(t: float) -> tuple[float, float, float, float]:
            r = cpp_traj.evaluate(t)
            return r.position, r.velocity, r.acceleration, r.jerk

        return trajectory

    @classmethod
    def order_3_trajectory(
        cls,
        initial: PyBoundaryCondition,
        final: PyBoundaryCondition,
        time: PyTimeInterval,
    ) -> Callable[[float], tuple[float, float, float, float]]:
        """Generate a 3rd order polynomial trajectory."""
        return cls._make_callable(initial, final, time, 3)

    @classmethod
    def order_5_trajectory(
        cls,
        initial: PyBoundaryCondition,
        final: PyBoundaryCondition,
        time: PyTimeInterval,
    ) -> Callable[[float], tuple[float, float, float, float]]:
        """Generate a 5th order polynomial trajectory."""
        return cls._make_callable(initial, final, time, 5)

    @classmethod
    def order_7_trajectory(
        cls,
        initial: PyBoundaryCondition,
        final: PyBoundaryCondition,
        time: PyTimeInterval,
    ) -> Callable[[float], tuple[float, float, float, float]]:
        """Generate a 7th order polynomial trajectory."""
        return cls._make_callable(initial, final, time, 7)

    @staticmethod
    def heuristic_velocities(
        points: list[float], times: list[float]
    ) -> list[float]:
        """Compute heuristic intermediate velocities via C++."""
        return list(
            _CppPolynomialTrajectory.heuristic_velocities(points, times)
        )

    @classmethod
    def multipoint_trajectory(
        cls,
        params: PolyTrajectoryParams,
    ) -> Callable[[float], tuple[float, float, float, float]]:
        """Generate a multi-segment polynomial trajectory.

        Parameters
        ----------
        params : TrajectoryParams
            Parameters including points, times, order, velocities, etc.

        Returns
        -------
        Callable
            Function returning (position, velocity, acceleration, jerk).
        """
        n = len(params.points)

        if n != len(params.times):
            raise ValueError("Number of points and times must be the same")

        order = params.order
        if order not in cls.VALID_ORDERS:
            valid = ", ".join(str(o) for o in cls.VALID_ORDERS)
            raise ValueError(f"Order must be one of: {valid}")

        # Use C++ multipoint when only basic params are needed
        vel = params.velocities
        acc = params.accelerations
        jrk = params.jerks

        # For order 3 without custom vel/acc/jerk, use C++ static method
        if order == _ORDER_3 and vel is None and acc is None and jrk is None:
            cpp_segments = _CppPolynomialTrajectory.multipoint_trajectory(
                params.points, params.times, order, 0.0, 0.0
            )

            def _basic_trajectory(t: float) -> tuple[float, float, float, float]:
                r = _CppPolynomialTrajectory.evaluate_multipoint(
                    cpp_segments, t
                )
                return r.position, r.velocity, r.acceleration, r.jerk

            return _basic_trajectory

        # For higher orders or custom boundary conditions, build segments
        if vel is None:
            vel = cls.heuristic_velocities(params.points, params.times)
        if acc is None and order in {_ORDER_5, _ORDER_7}:
            acc = [0.0] * n
        if jrk is None and order == _ORDER_7:
            jrk = [0.0] * n

        from interpolatepy.polynomials import BoundaryCondition, TimeInterval  # noqa: PLC0415

        segments: list[
            tuple[
                Callable[[float], tuple[float, float, float, float]],
                float,
                float,
            ]
        ] = []

        for i in range(n - 1):
            time_interval = TimeInterval(
                params.times[i], params.times[i + 1]
            )
            if order == _ORDER_3:
                bc_i = BoundaryCondition(params.points[i], vel[i])
                bc_f = BoundaryCondition(
                    params.points[i + 1], vel[i + 1]
                )
            elif order == _ORDER_5:
                assert acc is not None
                bc_i = BoundaryCondition(
                    params.points[i], vel[i], acc[i]
                )
                bc_f = BoundaryCondition(
                    params.points[i + 1],
                    vel[i + 1],
                    acc[i + 1],
                )
            else:  # order == 7
                assert acc is not None
                assert jrk is not None
                bc_i = BoundaryCondition(
                    params.points[i],
                    vel[i],
                    acc[i],
                    jrk[i],
                )
                bc_f = BoundaryCondition(
                    params.points[i + 1],
                    vel[i + 1],
                    acc[i + 1],
                    jrk[i + 1],
                )

            seg_fn = cls._make_callable(bc_i, bc_f, time_interval, order)
            segments.append(
                (seg_fn, params.times[i], params.times[i + 1])
            )

        times_list = params.times

        def trajectory(t: float) -> tuple[float, float, float, float]:
            if t < times_list[0]:
                return segments[0][0](times_list[0])
            if t > times_list[-1]:
                return segments[-1][0](times_list[-1])
            left, right = 0, len(segments) - 1
            while left <= right:
                mid = (left + right) // 2
                if segments[mid][1] <= t <= segments[mid][2]:
                    return segments[mid][0](t)
                if t < segments[mid][1]:
                    right = mid - 1
                else:
                    left = mid + 1
            raise ValueError(f"No segment found for time {t}")

        return trajectory
