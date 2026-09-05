"""Native adapter for polynomial trajectories."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from interpolatepy._backend import get_cpp_module

if TYPE_CHECKING:
    from collections.abc import Callable

    from interpolatepy.motion.polynomial import BoundaryCondition
    from interpolatepy.motion.polynomial import TimeInterval
    from interpolatepy.motion.polynomial import TrajectoryParams

_cpp_motion = get_cpp_module().motion
_CppBoundaryCondition = _cpp_motion.BoundaryCondition
_CppPolynomialTrajectory = _cpp_motion.PolynomialTrajectory
_CppTimeInterval = _cpp_motion.TimeInterval

_ORDER_3 = 3
_ORDER_5 = 5
_ORDER_7 = 7


def _to_cpp_boundary_condition(boundary: BoundaryCondition) -> object:
    """Convert a Python boundary condition to its native equivalent."""
    cpp_boundary = _CppBoundaryCondition()
    cpp_boundary.position = boundary.position
    cpp_boundary.velocity = boundary.velocity
    cpp_boundary.acceleration = boundary.acceleration
    cpp_boundary.jerk = boundary.jerk
    return cpp_boundary


def _to_cpp_time_interval(time: TimeInterval) -> object:
    """Convert a Python time interval to its native equivalent."""
    cpp_time = _CppTimeInterval()
    cpp_time.start = time.start
    cpp_time.end = time.end
    return cpp_time


class PolynomialTrajectory:
    """C++-backed trajectory matching the pure-Python class API."""

    VALID_ORDERS: ClassVar[tuple[int, ...]] = (3, 5, 7)

    @staticmethod
    def _make_callable(
        initial: BoundaryCondition,
        final: BoundaryCondition,
        time: TimeInterval,
        order: int,
    ) -> Callable[[float], tuple[float, float, float, float]]:
        """Create a native polynomial trajectory wrapped in a callable."""
        cpp_traj = _CppPolynomialTrajectory(
            _to_cpp_boundary_condition(initial),
            _to_cpp_boundary_condition(final),
            _to_cpp_time_interval(time),
            order,
        )

        def trajectory(t: float) -> tuple[float, float, float, float]:
            result = cpp_traj.evaluate(t)
            return result.position, result.velocity, result.acceleration, result.jerk

        return trajectory

    @classmethod
    def order_3_trajectory(
        cls,
        initial: BoundaryCondition,
        final: BoundaryCondition,
        time: TimeInterval,
    ) -> Callable[[float], tuple[float, float, float, float]]:
        """Generate a third-order polynomial trajectory."""
        return cls._make_callable(initial, final, time, _ORDER_3)

    @classmethod
    def order_5_trajectory(
        cls,
        initial: BoundaryCondition,
        final: BoundaryCondition,
        time: TimeInterval,
    ) -> Callable[[float], tuple[float, float, float, float]]:
        """Generate a fifth-order polynomial trajectory."""
        return cls._make_callable(initial, final, time, _ORDER_5)

    @classmethod
    def order_7_trajectory(
        cls,
        initial: BoundaryCondition,
        final: BoundaryCondition,
        time: TimeInterval,
    ) -> Callable[[float], tuple[float, float, float, float]]:
        """Generate a seventh-order polynomial trajectory."""
        return cls._make_callable(initial, final, time, _ORDER_7)

    @staticmethod
    def heuristic_velocities(points: list[float], times: list[float]) -> list[float]:
        """Compute heuristic intermediate velocities via C++."""
        return list(_CppPolynomialTrajectory.heuristic_velocities(points, times))

    @classmethod
    def multipoint_trajectory(
        cls,
        params: TrajectoryParams,
    ) -> Callable[[float], tuple[float, float, float, float]]:
        """Generate a multi-segment polynomial trajectory."""
        point_count = len(params.points)
        if point_count != len(params.times):
            raise ValueError("Number of points and times must be the same")

        order = params.order
        if order not in cls.VALID_ORDERS:
            valid = ", ".join(str(valid_order) for valid_order in cls.VALID_ORDERS)
            raise ValueError(f"Order must be one of: {valid}")

        velocities = params.velocities
        accelerations = params.accelerations
        jerks = params.jerks
        if order == _ORDER_3 and velocities is None and accelerations is None and jerks is None:
            cpp_segments = _CppPolynomialTrajectory.multipoint_trajectory(params.points, params.times, order, 0.0, 0.0)

            def basic_trajectory(t: float) -> tuple[float, float, float, float]:
                result = _CppPolynomialTrajectory.evaluate_multipoint(cpp_segments, t)
                return result.position, result.velocity, result.acceleration, result.jerk

            return basic_trajectory

        if velocities is None:
            velocities = cls.heuristic_velocities(params.points, params.times)
        if accelerations is None and order in {_ORDER_5, _ORDER_7}:
            accelerations = [0.0] * point_count
        if jerks is None and order == _ORDER_7:
            jerks = [0.0] * point_count

        segments = cls._build_segments(params, velocities, accelerations, jerks)
        return _multipoint_evaluator(segments, params.times)

    @classmethod
    def _build_segments(
        cls,
        params: TrajectoryParams,
        velocities: list[float],
        accelerations: list[float] | None,
        jerks: list[float] | None,
    ) -> list[
        tuple[
            Callable[[float], tuple[float, float, float, float]],
            float,
            float,
        ]
    ]:
        """Build the native callable for each polynomial segment."""
        from interpolatepy.motion.polynomial import BoundaryCondition  # noqa: PLC0415
        from interpolatepy.motion.polynomial import TimeInterval  # noqa: PLC0415

        segments = []
        for index in range(len(params.points) - 1):
            time = TimeInterval(params.times[index], params.times[index + 1])
            if params.order == _ORDER_3:
                initial = BoundaryCondition(params.points[index], velocities[index])
                final = BoundaryCondition(params.points[index + 1], velocities[index + 1])
            elif params.order == _ORDER_5:
                assert accelerations is not None
                initial = BoundaryCondition(params.points[index], velocities[index], accelerations[index])
                final = BoundaryCondition(
                    params.points[index + 1],
                    velocities[index + 1],
                    accelerations[index + 1],
                )
            else:
                assert accelerations is not None
                assert jerks is not None
                initial = BoundaryCondition(
                    params.points[index],
                    velocities[index],
                    accelerations[index],
                    jerks[index],
                )
                final = BoundaryCondition(
                    params.points[index + 1],
                    velocities[index + 1],
                    accelerations[index + 1],
                    jerks[index + 1],
                )
            segments.append((cls._make_callable(initial, final, time, params.order), time.start, time.end))
        return segments


def _multipoint_evaluator(
    segments: list[
        tuple[
            Callable[[float], tuple[float, float, float, float]],
            float,
            float,
        ]
    ],
    times: list[float],
) -> Callable[[float], tuple[float, float, float, float]]:
    """Return an evaluator that selects the appropriate polynomial segment."""

    def trajectory(t: float) -> tuple[float, float, float, float]:
        if t < times[0]:
            return segments[0][0](times[0])
        if t > times[-1]:
            return segments[-1][0](times[-1])
        left, right = 0, len(segments) - 1
        while left <= right:
            middle = (left + right) // 2
            if segments[middle][1] <= t <= segments[middle][2]:
                return segments[middle][0](t)
            if t < segments[middle][1]:
                right = middle - 1
            else:
                left = middle + 1
        raise ValueError(f"No segment found for time {t}")

    return trajectory


__all__ = ["PolynomialTrajectory"]
