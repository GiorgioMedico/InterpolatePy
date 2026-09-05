"""Compare SPRING with piecewise SLERP, SQUAD, and SQUAD-C2.

The plots show the orientation path in Modified Rodrigues Parameters, physical
angular-speed magnitude, and a common discrete tangential-curvature measure.
The console table reports median construction and evaluation times. Timings
describe the active backend on the current machine; they are not universal
performance claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from statistics import median
from time import perf_counter
from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np

from interpolatepy import HAS_CPP
from interpolatepy import Quaternion
from interpolatepy import QuaternionSpline
from interpolatepy import QuaternionTrajectory
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
    method: QuaternionTrajectory,
    evaluation_times: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return physical angular-speed magnitudes in radians per second."""
    return np.asarray(
        [np.linalg.norm(method.evaluate_velocity(float(time))) for time in evaluation_times],
        dtype=np.float64,
    )


def discrete_tangential_curvature(
    quaternions: list[Quaternion],
    evaluation_times: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Measure the tangential component of centered quaternion second differences."""
    values = np.asarray(
        [[quaternion.w, quaternion.x, quaternion.y, quaternion.z] for quaternion in quaternions],
        dtype=np.float64,
    )
    for index in range(1, len(values)):
        if np.dot(values[index - 1], values[index]) < 0.0:
            values[index] *= -1.0

    time_step = float(evaluation_times[1] - evaluation_times[0])
    second_difference = (values[:-2] - 2.0 * values[1:-1] + values[2:]) / time_step**2
    centers = values[1:-1]
    radial_scale = np.einsum("ij,ij->i", second_difference, centers) / np.einsum("ij,ij->i", centers, centers)
    tangential = second_difference - radial_scale[:, None] * centers
    return evaluation_times[1:-1], np.linalg.norm(tangential, axis=1)


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
    print(f"Active backend: {backend}; SPRING is always Python")
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
    curvatures: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
    timings: dict[str, TimingResult],
) -> Figure:
    """Plot orientation paths, angular speeds, curvature, and measured timings."""
    colors = {
        "Piecewise SLERP": "tab:green",
        "SQUAD": "tab:red",
        "SQUAD-C2": "tab:blue",
        "SPRING": "tab:purple",
    }
    figure = plt.figure(figsize=(16, 11), constrained_layout=True)
    path_axis = cast("Axes3D", figure.add_subplot(2, 2, 1, projection="3d"))
    speed_axis = figure.add_subplot(2, 2, 2)
    curvature_axis = figure.add_subplot(2, 2, 3)
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
    path_axis.legend()

    for name, values in speeds.items():
        speed_axis.plot(evaluation_times, values, color=colors[name], label=name)
    for time in waypoint_times:
        speed_axis.axvline(time, color="black", linestyle=":", alpha=0.25)
    speed_axis.set_title("Physical angular speed")
    speed_axis.set_xlabel("Time (s)")
    speed_axis.set_ylabel("|omega| (rad/s)")
    speed_axis.grid(alpha=0.25)
    speed_axis.legend()

    for name, (times, values) in curvatures.items():
        curvature_axis.plot(times, values, color=colors[name], label=name)
    for time in waypoint_times:
        curvature_axis.axvline(time, color="black", linestyle=":", alpha=0.25)
    curvature_axis.set_title("Discrete tangential curvature")
    curvature_axis.set_xlabel("Time (s)")
    curvature_axis.set_ylabel("Magnitude (1/s^2)")
    curvature_axis.grid(alpha=0.25)
    curvature_axis.legend()

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
    timing_axis.legend()

    figure.suptitle("Quaternion interpolation comparison", fontsize=16)
    return figure


def main() -> None:
    """Run the visual and timing comparison."""
    waypoint_times, waypoints = create_waypoints()
    factories = create_factories(waypoint_times, waypoints)
    methods = {name: factory() for name, factory in factories.items()}

    visual_times = np.linspace(waypoint_times[0], waypoint_times[-1], VISUAL_SAMPLES)
    sampled = sample_methods(methods, visual_times)
    speeds = {name: angular_speed(method, visual_times) for name, method in methods.items()}
    curvatures = {name: discrete_tangential_curvature(trajectory, visual_times) for name, trajectory in sampled.items()}

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
        curvatures,
        timings,
    )
    plt.show()


if __name__ == "__main__":
    main()
