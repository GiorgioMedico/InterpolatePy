"""Compare multiple-shooting setup/sampling with coarse and dense SPRING.

Run with ``python examples/shooting_quaternion_benchmark.py``. Set
``INTERPOLATEPY_NO_CPP=1`` to measure the Python reference instead of C++.
These are different objectives, not an equal-error performance comparison:
shooting solves continuous natural cubic matching to tolerance 1e-8, whereas
SPRING relaxes sampled tangential curvature for at most 300 iterations.
"""

from __future__ import annotations

from functools import partial
from statistics import median
from time import perf_counter
from typing import TYPE_CHECKING

from interpolatepy import HAS_CPP
from interpolatepy import Quaternion
from interpolatepy import ShootingQuaternionInterpolation
from interpolatepy import SpringConfig
from interpolatepy import SpringQuaternionInterpolation

if TYPE_CHECKING:
    from collections.abc import Callable


REPEATS = 7
OUTPUT_SAMPLES = 1001


def main() -> None:
    """Print median warm construction and fixed-size output sampling timings."""
    times = [0.0, 0.8, 1.8, 3.0, 4.2, 5.0]
    keyframes = [
        Quaternion.identity(),
        Quaternion.from_euler_angles(0.8, 0.1, 0.2),
        Quaternion.from_euler_angles(1.2, 0.9, 0.1),
        Quaternion.from_euler_angles(0.1, 1.3, 1.0),
        Quaternion.from_euler_angles(-0.7, 0.3, 1.6),
        Quaternion.from_euler_angles(-0.2, -0.5, 2.0),
    ]
    factories: dict[str, Callable[[], ShootingQuaternionInterpolation | SpringQuaternionInterpolation]] = {
        "Multiple shooting": partial(ShootingQuaternionInterpolation, times, keyframes),
        "SPRING (101)": partial(
            SpringQuaternionInterpolation, times, keyframes, SpringConfig(num_samples=101, iterations=300)
        ),
        "SPRING (1001)": partial(
            SpringQuaternionInterpolation, times, keyframes, SpringConfig(num_samples=1001, iterations=300)
        ),
    }
    print(f"Backend: {'C++' if HAS_CPP else 'Python'}; {len(times)} keyframes; median of {REPEATS} warm runs")
    print(f"{'Method':<22} {'Build (ms)':>12} {f'Sample {OUTPUT_SAMPLES} (ms)':>20} {'Steps':>8}")
    for name, factory in factories.items():
        curve = factory()
        curve.generate_trajectory(OUTPUT_SAMPLES)
        setup, sampling = [], []
        for _ in range(REPEATS):
            start = perf_counter()
            curve = factory()
            setup.append(perf_counter() - start)
            start = perf_counter()
            curve.generate_trajectory(OUTPUT_SAMPLES)
            sampling.append(perf_counter() - start)
        print(f"{name:<22} {1000 * median(setup):12.3f} {1000 * median(sampling):20.3f} {curve.iterations_run:8d}")
        if isinstance(curve, ShootingQuaternionInterpolation):
            print(
                f"  {curve.num_variables} unknowns; residual {curve.residual_norm:.2e}; "
                f"RK4 grid {curve.integration_steps}"
            )
    print("Different objectives and stopping criteria; timings are not an equal-error comparison.")


if __name__ == "__main__":
    main()
