"""Native adapter for the Double-S motion profile."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from interpolatepy._backend import get_cpp_module

if TYPE_CHECKING:
    from collections.abc import Callable

_CppDoubleSTrajectory = get_cpp_module().motion.DoubleSTrajectory


class DoubleSTrajectory(_CppDoubleSTrajectory):  # type: ignore[valid-type, misc]
    """C++-backed trajectory matching the Python scalar-trajectory protocol."""

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

    def evaluate_full(self, t: float | np.ndarray) -> tuple[float | np.ndarray, ...]:
        """Evaluate position, velocity, acceleration, and jerk at time *t*."""
        if isinstance(t, np.ndarray):
            pos = np.empty_like(t)
            vel = np.empty_like(t)
            acc = np.empty_like(t)
            jrk = np.empty_like(t)
            for i, ti in enumerate(t.flat):
                result = super().evaluate(float(ti))
                pos.flat[i] = result.position
                vel.flat[i] = result.velocity
                acc.flat[i] = result.acceleration
                jrk.flat[i] = result.jerk
            return pos, vel, acc, jrk
        result = super().evaluate(t)
        return result.position, result.velocity, result.acceleration, result.jerk

    def get_duration(self) -> float:
        """Return the total trajectory duration."""
        return self.duration

    def get_phase_durations(self) -> dict[str, float]:
        """Return the duration of each trajectory phase."""
        return dict(self.phase_durations())

    @staticmethod
    def create_trajectory(
        state_params: object,
        bounds: object,
    ) -> tuple[Callable[[float | np.ndarray], tuple[float | np.ndarray, ...]], float]:
        """Create a trajectory callable and return it with its duration."""
        planner = DoubleSTrajectory(state_params, bounds)

        def trajectory(t: float | np.ndarray) -> tuple[float | np.ndarray, ...]:
            return planner.evaluate_full(t)

        return trajectory, planner.get_duration()

    @property
    def T(self) -> float:  # noqa: N802
        """Total trajectory duration."""
        return self.duration


__all__ = ["DoubleSTrajectory"]
