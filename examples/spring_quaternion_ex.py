"""Compare SPRING, multiple shooting, log-quaternion, piecewise SLERP, SQUAD, and SQUAD-C2.

The plots show the orientation path in Modified Rodrigues Parameters, physical
angular-speed magnitude, and the accumulated angular-acceleration energy.
The console table reports median construction and evaluation times. Timings
describe the active backend on the current machine; they are not universal
performance claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from itertools import pairwise
from statistics import median
from time import perf_counter
from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np

from interpolatepy import HAS_CPP
from interpolatepy import LogQuaternionInterpolation
from interpolatepy import ModifiedLogQuaternionInterpolation
from interpolatepy import Quaternion
from interpolatepy import QuaternionSpline
from interpolatepy import QuaternionTrajectory
from interpolatepy import ShootingQuaternionInterpolation
from interpolatepy import SpringConfig
from interpolatepy import SpringQuaternionInterpolation
from interpolatepy import SquadC2
from interpolatepy.visualization.quaternion import QuaternionTrajectoryVisualizer

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D
    from numpy.typing import NDArray


VISUAL_SAMPLES = 401
TIMING_SAMPLES = 1_000
DETAIL_METHODS = 3
COLORS = {
    "Piecewise SLERP": "tab:green",
    "SQUAD": "tab:red",
    "SQUAD-C2": "tab:blue",
    "SPRING": "tab:purple",
    "Multiple shooting": "tab:orange",
    "LQI": "tab:brown",
    "mLQI": "tab:olive",
}
TIMING_REPEATS = 5


@dataclass(frozen=True, slots=True)
class TimingResult:
    """Median wall-clock timings for one interpolation method."""

    construction_seconds: float
    evaluation_seconds: float
    evaluation_count: int

    @property
    def construction_ms(self) -> float:
        """Return median construction time in milliseconds."""
        return self.construction_seconds * 1_000.0

    @property
    def evaluation_ms(self) -> float:
        """Return median time for the complete evaluation batch in milliseconds."""
        return self.evaluation_seconds * 1_000.0

    @property
    def microseconds_per_evaluation(self) -> float:
        """Return median batch time normalized by its sample count."""
        return self.evaluation_seconds * 1_000_000.0 / self.evaluation_count


def create_waypoints() -> tuple[list[float], list[Quaternion]]:
    """Return deterministic keyframes with several changes in rotation direction."""
    times = [0.0, 0.8, 1.8, 3.0, 4.2, 5.0]
    quaternions = [
        Quaternion.identity(),
        Quaternion.from_euler_angles(0.8, 0.1, 0.2),
        Quaternion.from_euler_angles(1.2, 0.9, 0.1),
        Quaternion.from_euler_angles(0.1, 1.3, 1.0),
        Quaternion.from_euler_angles(-0.7, 0.3, 1.6),
        Quaternion.from_euler_angles(-0.2, -0.5, 2.0),
    ]
    return times, quaternions


def create_factories(
    times: list[float],
    quaternions: list[Quaternion],
) -> dict[str, Callable[[], QuaternionTrajectory]]:
    """Create fresh-construction callables for every compared method."""
    return {
        "Piecewise SLERP": lambda: QuaternionSpline(times, quaternions, Quaternion.SLERP),
        "SQUAD": lambda: QuaternionSpline(times, quaternions, Quaternion.SQUAD),
        "SQUAD-C2": lambda: SquadC2(times, quaternions),
        "Multiple shooting": lambda: ShootingQuaternionInterpolation(times, quaternions),
        "LQI": lambda: LogQuaternionInterpolation(times, quaternions),
        "mLQI": lambda: ModifiedLogQuaternionInterpolation(times, quaternions),
        "SPRING": lambda: SpringQuaternionInterpolation(
            times,
            quaternions,
            SpringConfig(num_samples=151, iterations=450, refinement_levels=3),
        ),
    }


def sample_methods(
    methods: dict[str, QuaternionTrajectory],
    evaluation_times: NDArray[np.float64],
) -> dict[str, list[Quaternion]]:
    """Evaluate every method at the same time values."""
    return {name: [method.evaluate(float(time)) for time in evaluation_times] for name, method in methods.items()}


def angular_speed(
    quaternions: list[Quaternion],
    evaluation_times: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return physical angular-speed magnitudes in radians per second.

    Derived from the sampled orientations rather than each method's own
    ``evaluate_velocity``: the log-quaternion methods return derivatives of
    their internal parametrization (r, or theta and the axis components),
    which are not body angular velocities and would not be comparable. The
    result matches their ``get_physical_kinematics`` to 1e-5 rad/s.
    """
    time_step = float(evaluation_times[1] - evaluation_times[0])
    speeds = []
    for previous, current in pairwise(quaternions):
        relative = previous.inverse() * current
        if relative.s_ < 0.0:
            relative = -relative
        speeds.append(2.0 * np.linalg.norm(relative.Log().v_) / time_step)
    return np.asarray([speeds[0], *speeds], dtype=np.float64)


def accumulated_acceleration_energy(
    quaternions: list[Quaternion],
    evaluation_times: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Accumulate the angular-acceleration energy integral of omega.

    The running total of ``|domega/dt|^2 dt`` separates the methods far more
    readably than the pointwise value, whose keyframe spikes are two orders of
    magnitude above the rest of the curve.
    """
    time_step = float(evaluation_times[1] - evaluation_times[0])
    velocities = []
    for previous, current in pairwise(quaternions):
        relative = previous.inverse() * current
        if relative.s_ < 0.0:
            relative = -relative
        velocities.append(2.0 * relative.Log().v_ / time_step)
    increments = np.sum(np.diff(np.asarray(velocities), axis=0) ** 2, axis=1) / time_step
    return evaluation_times[1:-1], np.cumsum(increments)


def _median_runtime(operation: Callable[[], object], repeats: int) -> float:
    """Warm an operation once and return its median elapsed time."""
    operation()
    samples = []
    for _ in range(repeats):
        started = perf_counter()
        operation()
        samples.append(perf_counter() - started)
    return median(samples)


def _evaluate_batch(
    method: QuaternionTrajectory,
    evaluation_times: NDArray[np.float64],
) -> list[Quaternion]:
    """Evaluate one method over the benchmark time array."""
    return [method.evaluate(float(time)) for time in evaluation_times]


def benchmark_methods(
    factories: dict[str, Callable[[], QuaternionTrajectory]],
    evaluation_times: NDArray[np.float64],
    repeats: int = TIMING_REPEATS,
) -> dict[str, TimingResult]:
    """Benchmark construction and a fixed scalar-evaluation batch separately."""
    results = {}
    for name, factory in factories.items():
        construction_seconds = _median_runtime(factory, repeats)
        method = factory()
        evaluate_batch = partial(_evaluate_batch, method, evaluation_times)
        evaluation_seconds = _median_runtime(evaluate_batch, repeats)
        results[name] = TimingResult(
            construction_seconds=construction_seconds,
            evaluation_seconds=evaluation_seconds,
            evaluation_count=len(evaluation_times),
        )
    return results


def print_timing_table(timings: dict[str, TimingResult]) -> None:
    """Print aligned median construction and evaluation timings."""
    backend = "native adapters where available" if HAS_CPP else "pure Python"
    print(f"Active backend: {backend}")
    print(f"Median of {TIMING_REPEATS} repeats; evaluation batch: {TIMING_SAMPLES} samples")
    print(f"{'Method':<18} {'construct (ms)':>15} {'batch (ms)':>13} {'us/sample':>12}")
    for name, result in timings.items():
        print(
            f"{name:<18} {result.construction_ms:15.3f} "
            f"{result.evaluation_ms:13.3f} {result.microseconds_per_evaluation:12.3f}"
        )


def plot_comparison(  # noqa: PLR0913
    waypoint_times: list[float],
    waypoints: list[Quaternion],
    evaluation_times: NDArray[np.float64],
    sampled: dict[str, list[Quaternion]],
    speeds: dict[str, NDArray[np.float64]],
    energies: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
    timings: dict[str, TimingResult],
) -> Figure:
    """Plot orientation paths, angular speeds, acceleration energy, and timings."""
    colors = COLORS
    figure = plt.figure(figsize=(16, 11), constrained_layout=True)
    path_axis = cast("Axes3D", figure.add_subplot(2, 2, 1, projection="3d"))
    speed_axis = figure.add_subplot(2, 2, 2)
    energy_axis = figure.add_subplot(2, 2, 3)
    timing_axis = figure.add_subplot(2, 2, 4)
    visualizer = QuaternionTrajectoryVisualizer()

    for name, trajectory in sampled.items():
        projected = visualizer.project_trajectory(trajectory)
        path_axis.plot(
            projected[:, 0],
            projected[:, 1],
            projected[:, 2],
            color=colors[name],
            linewidth=2,
            label=name,
        )
    projected_waypoints = visualizer.project_trajectory(waypoints)
    cast("Any", path_axis).scatter(
        projected_waypoints[:, 0],
        projected_waypoints[:, 1],
        projected_waypoints[:, 2],
        color="black",
        marker="D",
        s=40,
        label="Keyframes",
        zorder=10,
    )
    path_axis.set_title("Orientation path (MRP projection)")
    path_axis.set_xlabel("MRP X")
    path_axis.set_ylabel("MRP Y")
    path_axis.set_zlabel("MRP Z")

    for name, values in speeds.items():
        speed_axis.plot(evaluation_times, values, color=colors[name], label=name)
    for time in waypoint_times:
        speed_axis.axvline(time, color="black", linestyle=":", alpha=0.25)
    speed_axis.set_title("Physical angular speed")
    speed_axis.set_xlabel("Time (s)")
    speed_axis.set_ylabel("|omega| (rad/s)")
    speed_axis.grid(alpha=0.25)

    for name, (times, values) in energies.items():
        energy_axis.plot(times, values, color=colors[name], label=name)
    for time in waypoint_times:
        energy_axis.axvline(time, color="black", linestyle=":", alpha=0.25)
    energy_axis.set_title("Accumulated angular-acceleration energy")
    energy_axis.set_xlabel("Time (s)")
    energy_axis.set_ylabel("integral of |domega/dt|^2 dt")
    energy_axis.grid(alpha=0.25)

    names = list(timings)
    positions = np.arange(len(names), dtype=np.float64)
    width = 0.36
    timing_axis.bar(
        positions - width / 2,
        [timings[name].construction_ms for name in names],
        width,
        label="Construction",
    )
    timing_axis.bar(
        positions + width / 2,
        [timings[name].evaluation_ms for name in names],
        width,
        label=f"{TIMING_SAMPLES} evaluations",
    )
    timing_axis.set_yscale("log")
    timing_axis.set_xticks(positions, names, rotation=15, ha="right")
    timing_axis.set_ylabel("Median time (ms, logarithmic)")
    timing_axis.set_title("Timing on this machine")
    timing_axis.grid(axis="y", alpha=0.25)
    # One shared legend: three per-axes legends covered the curves they labeled.
    figure.legend(*path_axis.get_legend_handles_labels(), loc="outside upper center", ncol=len(colors) + 1)
    timing_axis.legend()

    return figure


def plot_smoothest(
    waypoint_times: list[float],
    evaluation_times: NDArray[np.float64],
    speeds: dict[str, NDArray[np.float64]],
    energies: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
    count: int = DETAIL_METHODS,
) -> Figure:
    """Plot the methods with the lowest acceleration energy on their own scale.

    The overview figure spans two orders of magnitude on this measure, which
    flattens the leaders into a single line at the bottom of the panel.
    """
    smoothest = sorted(energies, key=lambda name: energies[name][1][-1])[:count]
    figure, (speed_axis, energy_axis) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True, constrained_layout=True
    )
    for name in smoothest:
        speed_axis.plot(evaluation_times, speeds[name], color=COLORS[name], label=name)
        times, values = energies[name]
        energy_axis.plot(times, values, color=COLORS[name], label=name)
    for axis in (speed_axis, energy_axis):
        for time in waypoint_times:
            axis.axvline(time, color="black", linestyle=":", alpha=0.25)
        axis.grid(alpha=0.25)
    speed_axis.set_title(f"{count} smoothest methods: {', '.join(smoothest)}")
    speed_axis.set_ylabel("|omega| (rad/s)")
    speed_axis.legend()
    energy_axis.set_ylabel("integral of |domega/dt|^2 dt")
    energy_axis.set_xlabel("Time (s)")
    return figure


def main() -> None:
    """Run the visual and timing comparison."""
    waypoint_times, waypoints = create_waypoints()
    factories = create_factories(waypoint_times, waypoints)
    methods = {name: factory() for name, factory in factories.items()}

    visual_times = np.linspace(waypoint_times[0], waypoint_times[-1], VISUAL_SAMPLES)
    sampled = sample_methods(methods, visual_times)
    speeds = {name: angular_speed(trajectory, visual_times) for name, trajectory in sampled.items()}
    energies = {
        name: accumulated_acceleration_energy(trajectory, visual_times) for name, trajectory in sampled.items()
    }

    timing_times = np.linspace(waypoint_times[0], waypoint_times[-1], TIMING_SAMPLES)
    timings = benchmark_methods(factories, timing_times)
    print_timing_table(timings)

    spring = methods["SPRING"]
    if isinstance(spring, SpringQuaternionInterpolation):
        reduction = 1.0 - spring.final_energy / spring.initial_energy
        print(f"SPRING refinement samples: {spring.refinement_sample_counts}")
        print(f"SPRING curvature-energy reduction: {reduction:.1%}")

    plot_comparison(
        waypoint_times,
        waypoints,
        visual_times,
        sampled,
        speeds,
        energies,
        timings,
    )
    plot_smoothest(waypoint_times, visual_times, speeds, energies)
    plt.show()


if __name__ == "__main__":
    main()
