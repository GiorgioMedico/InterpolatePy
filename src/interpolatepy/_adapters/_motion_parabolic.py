"""Native adapter for parabolic-blend trajectories."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from interpolatepy._backend import get_cpp_module

if TYPE_CHECKING:
    from collections.abc import Callable

_CppParabolicBlendTrajectory = get_cpp_module().motion.ParabolicBlendTrajectory


class ParabolicBlendTrajectory(_CppParabolicBlendTrajectory):  # type: ignore[valid-type, misc]
    """C++-backed parabolic blend with Python-compatible evaluation methods."""

    def __init__(
        self,
        q: list[float] | np.ndarray,
        t: list[float] | np.ndarray,
        dt_blend: list[float] | np.ndarray,
        dt: float = 0.01,
    ) -> None:
        self.q = np.asarray(q, dtype=float)
        self.t = np.asarray(t, dtype=float)
        self.dt_blend = np.asarray(dt_blend, dtype=float)
        self.dt = float(dt)
        super().__init__(self.q.tolist(), self.t.tolist(), self.dt_blend.tolist())

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

    def generate(
        self,
    ) -> tuple[Callable[[float | np.ndarray], tuple[float | np.ndarray, ...]], float]:
        """Return a Python-style trajectory callable and total duration."""

        def trajectory(t: float | np.ndarray) -> tuple[float | np.ndarray, ...]:
            return (
                self.evaluate(t),
                self.evaluate_velocity(t),
                self.evaluate_acceleration(t),
            )

        return trajectory, self.duration

    def plot(
        self,
        times: np.ndarray | None = None,
        pos: np.ndarray | None = None,
        vel: np.ndarray | None = None,
        acc: np.ndarray | None = None,
    ) -> None:
        """Plot position, velocity, and acceleration samples."""
        import matplotlib.pyplot as plt  # noqa: PLC0415

        if times is None or pos is None or vel is None or acc is None:
            times = np.arange(0.0, self.duration + self.dt, self.dt)
            pos = np.asarray(self.evaluate(times))
            vel = np.asarray(self.evaluate_velocity(times))
            acc = np.asarray(self.evaluate_acceleration(times))

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        ax1.plot(times, pos)
        ax1.set_ylabel("Position")
        ax2.plot(times, vel)
        ax2.set_ylabel("Velocity")
        ax3.plot(times, acc)
        ax3.set_ylabel("Acceleration")
        ax3.set_xlabel("Time")
        fig.tight_layout()


__all__ = ["ParabolicBlendTrajectory"]
