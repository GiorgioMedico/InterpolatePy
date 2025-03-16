"""
Module for generating and managing trapezoidal velocity profiles for trajectory planning.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


# Constants
MIN_POINTS = 2  # Minimum number of points needed for interpolation
EPSILON = 1e-10  # Small value to prevent division by zero


@dataclass
class TrajectoryParams:
    """Parameters for trapezoidal trajectory generation."""

    q0: float
    q1: float
    t0: float = 0.0
    v0: float = 0.0
    v1: float = 0.0
    amax: float | None = None
    vmax: float | None = None
    duration: float | None = None


@dataclass
class CalculationParams:
    """Parameters for trajectory calculations."""

    q0: float
    q1: float
    v0: float
    v1: float
    amax: float


@dataclass
class InterpolationParams:
    """Parameters for multi-point interpolation."""

    points: list[float]
    v0: float = 0.0
    vn: float = 0.0
    inter_velocities: list[float] | None = None
    times: list[float] | None = None
    amax: float = 10.0
    vmax: float | None = None


class TrapezoidalTrajectory:
    """
    Generate trapezoidal velocity profiles for trajectory planning.

    This class provides methods to create trapezoidal velocity profiles for various
    trajectory planning scenarios, including single segment trajectories and
    multi-point interpolation.
    """

    @staticmethod
    def _calculate_duration_based_trajectory(
        params: CalculationParams, duration: float
    ) -> tuple[float, float, float]:
        """
        Calculate trajectory parameters for duration-based constraints.

        Parameters
        ----------
        params : CalculationParams
            Basic trajectory parameters
        duration : float
            Desired duration

        Returns
        -------
        tuple[float, float, float]
            Cruise velocity, acceleration time, deceleration time

        Raises
        ------
        ValueError
            If trajectory is not feasible with given parameters
        """
        q0 = params.q0
        q1 = params.q1
        v0 = params.v0
        v1 = params.v1
        amax = params.amax
        h = q1 - q0

        # Check feasibility using equation (3.14)
        if amax * h < abs(v0**2 - v1**2) / 2:
            raise ValueError("Trajectory not feasible. Try increasing amax or reducing velocities.")

        # Check minimum required acceleration (equation 3.15)
        term_under_sqrt = (
            4 * h**2 - 4 * h * (v0 + v1) * duration + 2 * (v0**2 + v1**2) * duration**2
        )

        # Ensure term under sqrt is non-negative to avoid numerical issues
        if term_under_sqrt < 0:
            if term_under_sqrt > -EPSILON:  # Very close to zero, likely numerical error
                term_under_sqrt = 0
            else:
                raise ValueError(
                    "Trajectory not feasible with given duration. Try increasing duration."
                )

        alim = (2 * h - duration * (v0 + v1) + np.sqrt(term_under_sqrt)) / max(duration**2, EPSILON)

        if amax < alim:
            # Adjust amax to minimum required
            amax = alim
            print(f"Warning: Using minimum required acceleration: {alim:.4f}")

        # Calculate constant velocity (vv) from equation in section 3.2.7
        sqrt_term = (
            amax**2 * duration**2 - 4 * amax * h + 2 * amax * (v0 + v1) * duration - (v0 - v1) ** 2
        )

        # Ensure sqrt term is non-negative
        if sqrt_term < 0:
            if sqrt_term > -EPSILON:  # Very close to zero, likely numerical error
                sqrt_term = 0
            else:
                raise ValueError(
                    "Numerical issue in trajectory calculation. "
                    "The parameters may lead to an invalid trajectory."
                )

        vv = 0.5 * (v0 + v1 + amax * duration - np.sqrt(sqrt_term))

        # Calculate acceleration and deceleration times with numerical stability
        ta = (vv - v0) / (amax + EPSILON)
        td = (vv - v1) / (amax + EPSILON)

        return vv, ta, td

    @staticmethod
    def _calculate_velocity_based_trajectory(
        params: CalculationParams, vmax: float
    ) -> tuple[float, float, float, float]:
        """
        Calculate trajectory parameters for velocity-based constraints.

        Parameters
        ----------
        params : CalculationParams
            Basic trajectory parameters
        vmax : float
            Maximum velocity

        Returns
        -------
        tuple[float, float, float, float]
            Cruise velocity, acceleration time, deceleration time, total duration

        Raises
        ------
        ValueError
            If trajectory is not feasible with given parameters
        """
        q0 = params.q0
        q1 = params.q1
        v0 = params.v0
        v1 = params.v1
        amax = params.amax
        h = q1 - q0

        # Determine if vmax is reached (Case 1 or Case 2)
        if h * amax > vmax**2 - (v0**2 + v1**2) / 2:
            # vmax is reached
            vv = vmax

            # Calculate acceleration and deceleration times with numerical stability
            ta = (vmax - v0) / (amax + EPSILON)
            td = (vmax - v1) / (amax + EPSILON)

            # Calculate total duration with numerical stability
            v0_vmax_ratio = v0 / max(vmax, EPSILON)
            v1_vmax_ratio = v1 / max(vmax, EPSILON)

            # Ensure ratios are within valid range to avoid numerical issues
            v0_vmax_ratio = np.clip(v0_vmax_ratio, -1.0 + EPSILON, 1.0 - EPSILON)
            v1_vmax_ratio = np.clip(v1_vmax_ratio, -1.0 + EPSILON, 1.0 - EPSILON)

            duration = (
                (h / max(vmax, EPSILON))
                + (vmax / (2 * amax + EPSILON)) * (1 - v0_vmax_ratio) ** 2
                + (vmax / (2 * amax + EPSILON)) * (1 - v1_vmax_ratio) ** 2
            )

        else:
            # vmax is not reached (triangular profile)
            # Ensure the term under sqrt is non-negative
            sqrt_term = h * amax + (v0**2 + v1**2) / 2
            if sqrt_term < 0:
                if sqrt_term > -EPSILON:  # Very close to zero, likely numerical error
                    sqrt_term = 0
                else:
                    raise ValueError(
                        "Invalid trajectory parameters. The calculation resulted in "
                        "a negative value under a square root."
                    )

            vlim = np.sqrt(sqrt_term)
            vv = vlim

            # Calculate acceleration and deceleration times with numerical stability
            ta = (vlim - v0) / (amax + EPSILON)
            td = (vlim - v1) / (amax + EPSILON)

            # Total duration
            duration = ta + td

        return vv, ta, td, duration

    @staticmethod
    def generate_trajectory(
        params: TrajectoryParams,
    ) -> tuple[Callable[[float], tuple[float, float, float]], float]:
        """
        Generate a trapezoidal trajectory with non-null initial and final velocities.

        Handles both positive and negative displacements according to section 3.4.2.
        Uses absolute values for amax and vmax and includes numerical stability enhancements.

        Parameters
        ----------
        params : TrajectoryParams
            Parameters for trajectory generation including initial and final positions,
            velocities, acceleration and velocity limits, and optional duration.

        Returns
        -------
        tuple[Callable[[float], tuple[float, float, float]], float]
            A tuple containing:
            - Function that computes position, velocity, and acceleration at time t
            - Duration of trajectory

        Raises
        ------
        ValueError
            If parameter combination is invalid or trajectory is not feasible
        """
        # Local variables for better readability
        q0 = params.q0
        q1 = params.q1
        t0 = params.t0
        v0 = params.v0
        v1 = params.v1
        amax = params.amax
        vmax = params.vmax
        t_duration = params.duration

        # Parameter validation
        if amax is None:
            raise ValueError("Maximum acceleration (amax) must be provided")

        if t_duration is None and vmax is None:
            raise ValueError("Either duration or maximum velocity (vmax) must be provided")

        # Ensure amax and vmax are positive using absolute values if provided
        amax = abs(amax)
        if vmax is not None:
            vmax = abs(vmax)

        # Calculate displacement
        h = q1 - q0

        # Handle negative displacement (q1 < q0) according to section 3.4.2
        invert_results = False
        if h < 0:
            invert_results = True
            # Transform initial and final positions/velocities with opposite signs
            q0, q1 = -q0, -q1
            v0, v1 = -v0, -v1

        # Recalculate displacement (should be positive now)
        h = q1 - q0

        # Create calculation parameters
        calc_params = CalculationParams(q0=q0, q1=q1, v0=v0, v1=v1, amax=amax)

        # Pre-declare variables used in the trajectory function
        ta = 0.0  # Acceleration time
        td = 0.0  # Deceleration time
        vv = 0.0  # Cruise velocity
        duration = 0.0  # Total duration

        # Determine which case to use based on provided parameters
        if t_duration is not None and vmax is None:
            # Case 1: Preassigned duration and acceleration
            vv, ta, td = TrapezoidalTrajectory._calculate_duration_based_trajectory(
                calc_params, t_duration
            )
            duration = t_duration

        elif vmax is not None and t_duration is None:
            # Case 2: Preassigned acceleration and velocity
            vv, ta, td, duration = TrapezoidalTrajectory._calculate_velocity_based_trajectory(
                calc_params, vmax
            )

        else:
            # This should not happen due to the parameter validation above
            raise ValueError(
                "Invalid parameter combination. Provide either (amax, duration) or (amax, vmax)."
            )

        t1 = t0 + duration

        # Define the trajectory function
        def trajectory_original(t: float) -> tuple[float, float, float]:
            """
            Evaluate position, velocity, and acceleration at time t.

            Parameters
            ----------
            t : float
                Time at which to evaluate the trajectory

            Returns
            -------
            tuple[float, float, float]
                Tuple containing position, velocity, and acceleration at time t
            """
            # Ensure t is within bounds
            t = np.clip(t, t0, t1)

            # Initialize variables
            position = 0.0
            velocity = 0.0
            acceleration = 0.0

            # Ensure ta and td are not too small to avoid numerical issues
            ta_safe = max(ta, EPSILON)
            td_safe = max(td, EPSILON)

            # Calculate trajectory for the given time
            if t0 <= t < t0 + ta_safe:
                # Acceleration phase
                dt = t - t0
                position = q0 + v0 * dt + (vv - v0) / (2 * ta_safe) * dt**2
                velocity = v0 + (vv - v0) / ta_safe * dt
                acceleration = (vv - v0) / ta_safe
            elif t0 + ta_safe <= t < t1 - td_safe:
                # Constant velocity phase
                position = q0 + v0 * ta_safe / 2 + vv * (t - t0 - ta_safe / 2)
                velocity = vv
                acceleration = 0
            elif t1 - td_safe <= t <= t1:
                # Deceleration phase
                dt = t1 - t
                position = q1 - v1 * dt - (vv - v1) / (2 * td_safe) * dt**2
                velocity = v1 + (vv - v1) / td_safe * dt
                acceleration = -(vv - v1) / td_safe

            return position, velocity, acceleration

        # If we had a negative displacement, invert the resulting profiles
        if invert_results:

            def trajectory(t: float) -> tuple[float, float, float]:
                pos, vel, acc = trajectory_original(t)
                # Invert signs to transform back according to equation (3.33)
                return -pos, -vel, -acc

        else:
            trajectory = trajectory_original

        return trajectory, duration

    @staticmethod
    def calculate_heuristic_velocities(
        q_list: list[float],
        v0: float,
        vn: float,
        v_max: float | None = None,
        amax: float | None = None,
    ) -> list[float]:
        """
        Calculate velocities based on height differences with multiple options for
        heuristic velocity calculation.

        Parameters
        ----------
        q_list : list[float]
            List of height values [q0, q1, ..., qn]
        v0 : float
            Initial velocity (assigned)
        vn : float
            Final velocity (assigned)
        v_max : float | None, optional
            Maximum velocity value (positive magnitude)
        amax : float | None, optional
            Maximum acceleration (needed if v_max is not provided)

        Returns
        -------
        list[float]
            Calculated velocities [v0, v1, ..., vn]

        Raises
        ------
        ValueError
            If neither v_max nor amax is provided
        """
        v0 = float(v0)
        vn = float(vn)
        # Calculate height differences h_k = q_k - q_(k-1)
        h_values = [float(q_list[k] - q_list[k - 1]) for k in range(1, len(q_list))]

        # If v_max is not provided, compute it heuristically
        if v_max is None:
            if amax is None:
                raise ValueError("Either v_max or amax must be provided")

            # Ensure amax is positive
            amax = abs(amax)

            # OPTION 1: Time-Based Approach
            # Estimate a reasonable total duration for the path and derive velocity
            total_distance = sum(abs(h) for h in h_values)
            estimated_duration = np.sqrt(2 * total_distance / amax)  # From acceleration equation
            v_max = total_distance / estimated_duration * 0.75  # 75% of average velocity

            # OPTION 2: Segment-Optimized Approach
            # Calculate optimal velocity for each segment based on its length
            segment_velocities = []
            for h in h_values:
                # Calculate velocity that allows comfortable acceleration/deceleration
                segment_length = abs(h)
                # Distance to accelerate from 0 to v and decelerate back to 0 is (v^2)/a
                # We want this to be less than the segment length, solving for v:
                v_segment = np.sqrt(amax * segment_length / 2)
                segment_velocities.append(v_segment)

            # Choose a velocity that works well for all segments
            v_max_segments = (
                min(segment_velocities) * 0.8
            )  # 80% of minimum optimal segment velocity

            # OPTION 3: Curvature-Based Approach
            # Look at changes in direction to determine velocity
            direction_changes = []
            for i in range(len(h_values) - 1):
                # Calculate angle between consecutive segments
                if h_values[i] * h_values[i + 1] < 0:  # Direction change
                    direction_changes.append(1.0)  # Full direction change
                else:
                    # Calculate relative change in slope
                    rel_change = abs(h_values[i + 1] - h_values[i]) / (
                        abs(h_values[i]) + abs(h_values[i + 1])
                    )
                    direction_changes.append(rel_change)

            # More direction changes or sharper changes suggest lower velocity
            avg_change = sum(direction_changes) / max(len(direction_changes), 1)
            v_max_curvature = np.sqrt(amax * total_distance / (len(h_values) + 5 * avg_change))

            # Choose the minimum of all approaches for safety
            v_max = min(v_max, v_max_segments, v_max_curvature)

        # Ensure v_max is valid
        if v_max is None or v_max <= 0:
            raise ValueError("Failed to calculate a valid maximum velocity")

        # Initialize velocities array with v0 as the first element
        velocities = [v0]

        # Calculate intermediate velocities (v1 to v_(n-1))
        for k in range(len(h_values) - 1):
            if np.sign(h_values[k]) != np.sign(h_values[k + 1]):
                velocities.append(0.0)
            else:
                velocities.append(float(np.sign(h_values[k]) * v_max))

        # Add the final velocity vn
        velocities.append(vn)

        return velocities

    @classmethod
    def interpolate_waypoints(
        cls, params: InterpolationParams
    ) -> tuple[Callable[[float], tuple[float, float, float]], float]:
        """
        Generate a trajectory through a sequence of points using trapezoidal velocity profiles.

        Supports both positive and negative displacements.

        Parameters
        ----------
        params : InterpolationParams
            Parameters for interpolation including points, velocities,
            times, and motion constraints.

        Returns
        -------
        tuple[Callable[[float], tuple[float, float, float]], float]
            A tuple containing:
            - Function that returns position, velocity, and acceleration at any time t
            - Total duration of the trajectory

        Raises
        ------
        ValueError
            If less than two points are provided or if invalid velocity counts are provided
        """
        # Ensure input is valid
        if len(params.points) < MIN_POINTS:
            raise ValueError("At least two points are required for interpolation")

        # Calculate intermediate velocities if not provided
        if params.inter_velocities is None:
            velocities = cls.calculate_heuristic_velocities(
                params.points, params.v0, params.vn, params.vmax, params.amax
            )
        elif len(params.inter_velocities) != len(params.points) - 2:
            raise ValueError(
                f"Expected {len(params.points) - 2} intermediate velocities, "
                f"got {len(params.inter_velocities)}"
            )
        else:
            # Use provided velocities
            velocities = [params.v0]
            velocities.extend(params.inter_velocities)
            velocities.append(params.vn)

        # If vmax was computed in the heuristic, use it for the trajectories
        vmax = params.vmax
        if vmax is None and params.inter_velocities is None:
            # Extract the computed vmax from the heuristic (maximum absolute velocity)
            computed_vmax = max(abs(v) for v in velocities)
            vmax = computed_vmax

        # Initialize containers for combined trajectory
        all_trajectories = []
        cumulative_time = 0.0
        segment_end_times = [0.0]  # Start with initial time

        # Generate individual segment trajectories
        for i in range(len(params.points) - 1):
            q0 = params.points[i]
            q1 = params.points[i + 1]
            v_start = velocities[i]
            v_end = velocities[i + 1]

            if params.times is None:
                # Calculate trajectory with velocity/acceleration constraints
                traj_params = TrajectoryParams(
                    q0=q0,
                    q1=q1,
                    t0=cumulative_time,
                    v0=v_start,
                    v1=v_end,
                    amax=params.amax,
                    vmax=vmax,
                )
                traj_func, segment_duration = cls.generate_trajectory(traj_params)
            else:
                # Use specified time for this segment
                segment_duration = params.times[i + 1] - params.times[i]
                traj_params = TrajectoryParams(
                    q0=q0,
                    q1=q1,
                    t0=cumulative_time,
                    v0=v_start,
                    v1=v_end,
                    amax=params.amax,
                    duration=segment_duration,
                )
                traj_func, _ = cls.generate_trajectory(traj_params)

            cumulative_time += segment_duration
            segment_end_times.append(cumulative_time)
            all_trajectories.append(traj_func)

        # Total duration of the trajectory
        total_duration = cumulative_time

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

            # Determine which segment this time belongs to
            segment_idx = np.searchsorted(segment_end_times, t, side="right") - 1

            if segment_idx < len(all_trajectories):
                position, velocity, acceleration = all_trajectories[segment_idx](t)
                return position, velocity, acceleration
            # If beyond the end, return final position with zero velocity and acceleration
            return params.points[-1], 0.0, 0.0

        return trajectory_function, total_duration
