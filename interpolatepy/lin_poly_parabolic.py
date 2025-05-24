from collections.abc import Callable  # noqa: EXE002

import matplotlib.pyplot as plt
import numpy as np


class ParabolicBlendTrajectory:
    """
    Class to generate trajectories composed of linear segments with parabolic blends at via points.

    The trajectory duration is extended:
        total_duration = t[-1] - t[0] + (dt_blend[0] + dt_blend[-1]) / 2

    Initial and final velocities are set to zero (q̇0,1 = q̇N,N+1 = 0).

    Parameters
    ----------
    q : list | np.ndarray
        Positions of via points (length N).
    t : list | np.ndarray
        Nominal times at which via points are reached (length N).
    dt_blend : list | np.ndarray
        Blend durations at via points (length N).
    dt : float, optional
        Sampling interval for plotting trajectory. Default is 0.01.
    """

    def __init__(
        self,
        q: list | np.ndarray,
        t: list | np.ndarray,
        dt_blend: list | np.ndarray,
        dt: float = 0.01,
    ) -> None:
        self.q = np.asarray(q, dtype=float)
        self.t = np.asarray(t, dtype=float)
        self.dt_blend = np.asarray(dt_blend, dtype=float)
        self.dt = dt

        if not (len(self.q) == len(self.t) == len(self.dt_blend)):
            raise ValueError("Lengths of q, t, and dt_blend must match.")

    def generate(self) -> tuple[Callable[[float], tuple[float, float, float]], float]:
        """
        Generate the parabolic blend trajectory function.

        Returns
        -------
        trajectory_function : callable
            Function that takes a time value and returns position, velocity, and acceleration.
        total_duration : float
            The total duration of the trajectory.
        """
        # Use lowercase for count variable per style
        n = len(self.q)

        # Vectorized computation of segment velocities with zero initial and final
        v_before = np.zeros(n)
        # Calculate differences in positions and times
        dq = np.diff(self.q)
        dt = np.diff(self.t)
        # Vectorized velocity calculation
        v_before[1:] = dq / dt

        # Shift velocities for v_after
        v_after = np.zeros(n)
        v_after[:-1] = v_before[1:]

        # Accelerations for parabolic blends
        a = (v_after - v_before) / self.dt_blend

        # Preallocate arrays for region data with structure of arrays (SoA)
        # Estimating 2*n-1 regions (initial blend + n-1 pairs of linear+blend)
        num_regions = 2 * n - 1
        reg_t0 = np.zeros(num_regions)
        reg_t1 = np.zeros(num_regions)
        reg_q0 = np.zeros(num_regions)
        reg_v0 = np.zeros(num_regions)
        reg_a = np.zeros(num_regions)

        # Initial blend
        reg_idx = 0
        t0 = self.t[0] - self.dt_blend[0] / 2
        t1 = self.t[0] + self.dt_blend[0] / 2
        reg_t0[reg_idx] = t0
        reg_t1[reg_idx] = t1
        reg_q0[reg_idx] = self.q[0]
        reg_v0[reg_idx] = v_before[0]
        reg_a[reg_idx] = a[0]
        reg_idx += 1

        # Build remaining regions efficiently
        for k in range(n - 1):
            # Constant-velocity segment
            t0_c = reg_t1[reg_idx - 1]
            t1_c = self.t[k + 1] - self.dt_blend[k + 1] / 2

            # Calculate position at start of constant velocity segment
            dt0 = t0_c - reg_t0[reg_idx - 1]
            q0_c = (
                reg_q0[reg_idx - 1]
                + reg_v0[reg_idx - 1] * dt0
                + 0.5 * reg_a[reg_idx - 1] * dt0**2
            )

            # Store constant velocity segment
            reg_t0[reg_idx] = t0_c
            reg_t1[reg_idx] = t1_c
            reg_q0[reg_idx] = q0_c
            reg_v0[reg_idx] = v_after[k]
            reg_a[reg_idx] = 0.0
            reg_idx += 1

            # Parabolic blend
            t0_b = t1_c
            t1_b = self.t[k + 1] + self.dt_blend[k + 1] / 2

            # Calculate position at start of blend
            dt0b = t0_b - reg_t0[reg_idx - 1]
            q0_b = reg_q0[reg_idx - 1] + reg_v0[reg_idx - 1] * dt0b

            # Store parabolic blend
            reg_t0[reg_idx] = t0_b
            reg_t1[reg_idx] = t1_b
            reg_q0[reg_idx] = q0_b
            reg_v0[reg_idx] = v_before[k + 1]
            reg_a[reg_idx] = a[k + 1]
            reg_idx += 1

        # Trim unused regions if we overestimated
        if reg_idx < num_regions:
            reg_t0 = reg_t0[:reg_idx]
            reg_t1 = reg_t1[:reg_idx]
            reg_q0 = reg_q0[:reg_idx]
            reg_v0 = reg_v0[:reg_idx]
            reg_a = reg_a[:reg_idx]

        # Determine overall duration
        t_start = reg_t0[0]
        t_end = reg_t1[-1]
        total_duration = t_end - t_start

        # Prepare binary search for region lookup
        region_boundaries = np.append(reg_t0[0], reg_t1)

        # Function to evaluate trajectory at any time t
        def trajectory_function(t: float) -> tuple[float, float, float]:
            """
            Evaluate the trajectory at time t.

            Parameters
            ----------
            t : float
                Time at which to evaluate the trajectory

            Returns
            -------
            tuple[float, float, float]
                Tuple containing position, velocity, and acceleration at time t
            """
            # Clip time to valid range
            t = np.clip(t, 0.0, total_duration)

            # Convert to absolute time
            t_abs = t + t_start

            # Find region using binary search
            region_idx = np.searchsorted(region_boundaries, t_abs, side="right") - 1
            region_idx = min(region_idx, len(reg_t0) - 1)

            # Calculate values
            u = t_abs - reg_t0[region_idx]
            pos = (
                reg_q0[region_idx]
                + reg_v0[region_idx] * u
                + 0.5 * reg_a[region_idx] * u**2
            )
            vel = reg_v0[region_idx] + reg_a[region_idx] * u
            acc = reg_a[region_idx]

            return pos, vel, acc

        return trajectory_function, total_duration

    def plot(
        self,
        times: np.ndarray | None = None,
        pos: np.ndarray | None = None,
        vel: np.ndarray | None = None,
        acc: np.ndarray | None = None,
    ) -> None:
        """
        Plot the trajectory's position, velocity, and acceleration.

        If trajectory data is not provided, it will be generated.

        Parameters
        ----------
        times : ndarray, optional
            Time samples; if None, generated from trajectory function.
        pos : ndarray, optional
            Position samples; if None, generated from trajectory function.
        vel : ndarray, optional
            Velocity samples; if None, generated from trajectory function.
        acc : ndarray, optional
            Acceleration samples; if None, generated from trajectory function.
        """
        if times is None or pos is None or vel is None or acc is None:
            traj_func, total_duration = self.generate()
            times = np.arange(0.0, total_duration + self.dt, self.dt)
            pos = np.zeros_like(times)
            vel = np.zeros_like(times)
            acc = np.zeros_like(times)

            for i, t in enumerate(times):
                pos[i], vel[i], acc[i] = traj_func(t)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        ax1.plot(times, pos)
        ax1.set_ylabel("Position")
        ax2.plot(times, vel)
        ax2.set_ylabel("Velocity")
        ax3.plot(times, acc)
        ax3.set_ylabel("Acceleration")
        ax3.set_xlabel("Time")
        fig.tight_layout()
        plt.show()
