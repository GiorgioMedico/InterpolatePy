"""Native adapter for trapezoidal trajectories."""

from __future__ import annotations

from typing import TYPE_CHECKING

from interpolatepy._backend import get_cpp_module

if TYPE_CHECKING:
    from collections.abc import Callable

    from interpolatepy.motion.trapezoidal import InterpolationParams
    from interpolatepy.motion.trapezoidal import TrajectoryParams

_CppTrapezoidalTrajectory = get_cpp_module().motion.TrapezoidalTrajectory


class TrapezoidalTrajectory:
    """C++-backed trajectory matching the pure-Python class API."""

    @staticmethod
    def generate_trajectory(
        params: TrajectoryParams,
    ) -> tuple[Callable[[float], tuple[float, float, float]], float]:
        """Generate a single-segment trapezoidal trajectory."""
        if params.amax is None:
            raise ValueError("Maximum acceleration (amax) must be provided")
        if params.duration is None and params.vmax is None:
            raise ValueError("Either duration or maximum velocity (vmax) must be provided")

        amax = abs(params.amax)
        if params.vmax is not None and params.duration is None:
            cpp_traj = _CppTrapezoidalTrajectory(
                params.q0,
                params.q1,
                amax,
                abs(params.vmax),
                params.v0,
                params.v1,
                params.t0,
            )
        elif params.duration is not None:
            cpp_traj = _CppTrapezoidalTrajectory(
                q0=params.q0,
                q1=params.q1,
                amax=amax,
                v0=params.v0,
                v1=params.v1,
                t0=params.t0,
                duration=params.duration,
            )
        else:
            raise ValueError("Invalid parameter combination. Provide either (amax, vmax) or (amax, duration).")

        duration = cpp_traj.duration

        def trajectory(t: float) -> tuple[float, float, float]:
            result = cpp_traj.evaluate(t)
            return result.position, result.velocity, result.acceleration

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
        from interpolatepy.motion.trapezoidal import (  # noqa: PLC0415
            TrapezoidalTrajectory as PythonTrapezoidalTrajectory,
        )

        return PythonTrapezoidalTrajectory.calculate_heuristic_velocities(q_list, v0, vn, v_max, amax)

    @classmethod
    def interpolate_waypoints(
        cls,
        params: InterpolationParams,
    ) -> tuple[Callable[[float], tuple[float, float, float]], float]:
        """Generate a multi-segment trajectory through waypoints."""
        from interpolatepy.motion.trapezoidal import (  # noqa: PLC0415
            TrapezoidalTrajectory as PythonTrapezoidalTrajectory,
        )

        return PythonTrapezoidalTrajectory.interpolate_waypoints(params)


__all__ = ["TrapezoidalTrajectory"]
