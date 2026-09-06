"""Compare solvers of the SAME discrete SPRING problem at a common tolerance.

Coarse solves, fixed anchors, sample lattice, weights and final iteration
limits are identical. A speedup is printed only if both solvers converge and
their orientation difference satisfies an explicit angular error limit.
Run with ``python examples/spring_solver_benchmark.py``; set
``INTERPOLATEPY_NO_CPP=1`` to test the Python reference.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from statistics import median
from time import perf_counter

import numpy as np

from interpolatepy import HAS_CPP
from interpolatepy import Quaternion
from interpolatepy import SpringConfig
from interpolatepy import SpringQuaternionInterpolation


def main() -> None:
    """Measure time, stationarity and angular agreement on identical inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1001)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--final-iterations", type=int, default=30000)
    parser.add_argument("--angle-tolerance-deg", type=float, default=0.005)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    times = [0.0, 1.0, 2.0, 3.0]
    keyframes = [
        Quaternion.identity(),
        Quaternion.from_euler_angles(0.9, 0.1, 0.2),
        Quaternion.from_euler_angles(0.2, 1.1, 0.5),
        Quaternion.from_euler_angles(-0.4, 0.3, 1.4),
    ]
    config = SpringConfig(
        num_samples=args.samples, iterations=100, final_iterations=args.final_iterations, tolerance=args.tolerance
    )
    print(f"Backend: {'C++' if HAS_CPP else 'Python'}; samples: {args.samples}; gradient tolerance: {args.tolerance:g}")
    print(f"{'Solver':<20} {'Median ms':>12} {'Final steps':>12} {'Gradient norm':>16} {'Converged':>11}")
    curves: list[SpringQuaternionInterpolation] = []
    timings: list[float] = []
    for solver in ("gradient_descent", "gauss_newton"):
        active = replace(config, solver=solver)
        curve = SpringQuaternionInterpolation(times, keyframes, active)
        durations = []
        for _ in range(args.repeats):
            start = perf_counter()
            curve = SpringQuaternionInterpolation(times, keyframes, active)
            durations.append(perf_counter() - start)
        elapsed = median(durations)
        curves.append(curve)
        timings.append(elapsed)
        print(
            f"{solver:<20} {1000 * elapsed:12.3f} {len(curve.energy_history) - 1:12d} "
            f"{curve.stage_gradient_norms[-1]:16.3e} {curve.converged!s:>11}"
        )
    reference, accelerated = curves
    if reference.stage_energy_history[:-1] != accelerated.stage_energy_history[:-1]:
        raise RuntimeError("Coarse anchors differ: this is not the same final problem")
    if reference.energy_history[0] != accelerated.energy_history[0]:
        raise RuntimeError("Final starting energies differ")
    errors = []
    for time in np.linspace(times[0], times[-1], 2001):
        difference = reference.evaluate(float(time)).inverse() * accelerated.evaluate(float(time))
        errors.append(2.0 * np.arctan2(np.linalg.norm(difference.v_), abs(difference.w)))
    maximum_error = float(np.degrees(max(errors)))
    print(f"Maximum orientation difference over 2001 times: {maximum_error:.6g} degrees")
    print(f"Final full energies: {reference.energy_history[-1]:.12g}, {accelerated.energy_history[-1]:.12g}")
    if all(curve.converged for curve in curves) and maximum_error <= args.angle_tolerance_deg:
        print(f"Speedup at the verified tolerances: {timings[0] / timings[1]:.2f}x")
    else:
        print("No equivalent-accuracy speedup claimed: convergence or angular tolerance was not reached.")


if __name__ == "__main__":
    main()
