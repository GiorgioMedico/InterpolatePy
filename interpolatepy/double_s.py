from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


# Constants to avoid magic numbers
EPSILON = 1e-6  # Small value to avoid division by zero
MIN_GAMMA = 0.01  # Lower bound for gamma in binary search
MAX_ITERATIONS = 50  # Maximum iterations for binary search


@dataclass
class TrajectoryBounds:
    """Bounds for trajectory planning.

    Parameters
    ----------
    v_bound : float
        Velocity bound (absolute value will be used for both min/max).
    a_bound : float
        Acceleration bound (absolute value will be used for both min/max).
    j_bound : float
        Jerk bound (absolute value will be used for both min/max).
    """

    v_bound: float
    a_bound: float
    j_bound: float

    def __post_init__(self) -> None:
        """Validate the bounds."""
        if not all(isinstance(x, int | float) for x in [self.v_bound, self.a_bound, self.j_bound]):
            raise TypeError("All bounds must be numeric values")

        if self.v_bound <= 0 or self.a_bound <= 0 or self.j_bound <= 0:
            raise ValueError("Bounds must be positive values")

        # Convert to absolute values
        self.v_bound = abs(self.v_bound)
        self.a_bound = abs(self.a_bound)
        self.j_bound = abs(self.j_bound)


class StateParams(NamedTuple):
    """Parameters representing position and velocity state.

    Parameters
    ----------
    q_0 : float
        Start position.
    q_1 : float
        End position.
    v_0 : float
        Velocity at start of trajectory.
    v_1 : float
        Velocity at end of trajectory.
    """

    q_0: float
    q_1: float
    v_0: float
    v_1: float


class DoubleSTrajectory:
    """
    Double S-Trajectory Planner Class

    This class implements a trajectory planning algorithm that generates smooth motion
    profiles with bounded velocity, acceleration, and jerk (double S-trajectory).
    """

    def __init__(self, state_params: StateParams, bounds: TrajectoryBounds) -> None:
        """
        Initialize the trajectory planner.

        Parameters:
        -----------
        state_params : StateParams
            Start and end states for trajectory planning
        bounds : TrajectoryBounds
            Velocity, acceleration, and jerk bounds for trajectory planning
        """
        # Input validation for state params
        if not all(
            isinstance(x, int | float)
            for x in [state_params.q_0, state_params.q_1, state_params.v_0, state_params.v_1]
        ):
            raise TypeError("All state parameters must be numeric values")

        # Store initial parameters
        self.state = state_params
        self.bounds = bounds

        # Check if initial or final velocities exceed bounds
        if abs(state_params.v_0) > bounds.v_bound or abs(state_params.v_1) > bounds.v_bound:
            raise ValueError(
                f"Initial or final velocities exceed the velocity bound of {bounds.v_bound}"
            )

        # Initialize trajectory parameters
        self.T = 0.0  # Total trajectory duration
        self.Ta = 0.0  # Acceleration phase duration
        self.Tv = 0.0  # Constant velocity phase duration
        self.Td = 0.0  # Deceleration phase duration
        self.Tj_1 = 0.0  # Jerk time in acceleration phase
        self.Tj_2 = 0.0  # Jerk time in deceleration phase
        self.a_lim_a = 0.0  # Acceleration limit during acceleration phase
        self.a_lim_d = 0.0  # Acceleration limit during deceleration phase
        self.v_lim = 0.0  # Velocity limit during constant velocity phase
        self.sigma = 1.0  # Direction coefficient

        # Transformed parameters
        self.q_0_transformed = 0.0
        self.q_1_transformed = 0.0
        self.v_0_transformed = 0.0
        self.v_1_transformed = 0.0
        self.v_max = 0.0
        self.v_min = 0.0
        self.a_max = 0.0
        self.a_min = 0.0
        self.j_max = 0.0
        self.j_min = 0.0

        # Plan the trajectory
        self._plan_trajectory()

    def _plan_trajectory(self) -> None:
        """
        Plan the trajectory based on the initialized parameters.
        This internal method computes all necessary trajectory parameters.
        """
        qd_0, qd_1 = self.state.q_0, self.state.q_1
        vd_0, vd_1 = self.state.v_0, self.state.v_1

        # BLOCK 1: Given the initial conditions compute direction and transform parameters
        # If positions are equal, handle specially
        if np.isclose(qd_1, qd_0):
            # For equal positions, create a minimal trajectory to match velocities if needed
            if np.isclose(vd_1, vd_0):
                # If velocities also match, return a static trajectory
                self.T = 0.0
                return

            # Different velocities but same position requires a special trajectory
            t_min = abs(vd_1 - vd_0) / self.bounds.a_bound
            self.T = max(t_min * 1.5, 0.1)  # Ensure minimum duration
            return

        # Normal case - different positions
        self.sigma = np.sign(qd_1 - qd_0)

        # Transform parameters based on direction
        self.q_0_transformed = self.sigma * qd_0
        self.q_1_transformed = self.sigma * qd_1
        self.v_0_transformed = self.sigma * vd_0
        self.v_1_transformed = self.sigma * vd_1

        # Set limits based on direction
        self.v_max = (self.sigma + 1) / 2 * self.bounds.v_bound + (self.sigma - 1) / 2 * (
            -self.bounds.v_bound
        )
        self.v_min = (self.sigma + 1) / 2 * (-self.bounds.v_bound) + (
            self.sigma - 1
        ) / 2 * self.bounds.v_bound
        self.a_max = (self.sigma + 1) / 2 * self.bounds.a_bound + (self.sigma - 1) / 2 * (
            -self.bounds.a_bound
        )
        self.a_min = (self.sigma + 1) / 2 * (-self.bounds.a_bound) + (
            self.sigma - 1
        ) / 2 * self.bounds.a_bound
        self.j_max = (self.sigma + 1) / 2 * self.bounds.j_bound + (self.sigma - 1) / 2 * (
            -self.bounds.j_bound
        )
        self.j_min = (self.sigma + 1) / 2 * (-self.bounds.j_bound) + (
            self.sigma - 1
        ) / 2 * self.bounds.j_bound

        # Compute time intervals assuming v_max and a_max are reached

        # Acceleration part
        if ((self.v_max - self.v_0_transformed) * self.j_max) < self.a_max**2:
            self.Tj_1 = np.sqrt(
                max((self.v_max - self.v_0_transformed) / self.j_max, 0)
            )  # Prevent negative sqrt
            self.Ta = 2 * self.Tj_1
        else:
            self.Tj_1 = self.a_max / self.j_max
            self.Ta = self.Tj_1 + (self.v_max - self.v_0_transformed) / self.a_max

        # Deceleration part
        if ((self.v_max - self.v_1_transformed) * self.j_max) < self.a_max**2:
            self.Tj_2 = np.sqrt(
                max((self.v_max - self.v_1_transformed) / self.j_max, 0)
            )  # Prevent negative sqrt
            self.Td = 2 * self.Tj_2
        else:
            self.Tj_2 = self.a_max / self.j_max
            self.Td = self.Tj_2 + (self.v_max - self.v_1_transformed) / self.a_max

        # Determine the time duration of the constant velocity phase
        if abs(self.v_max) < EPSILON:  # Avoid division by zero
            self.Tv = 0
        else:
            self.Tv = (
                (self.q_1_transformed - self.q_0_transformed) / self.v_max
                - self.Ta / 2 * (1 + self.v_0_transformed / self.v_max)
                - self.Td / 2 * (1 + self.v_1_transformed / self.v_max)
            )

        # Check if Tv < 0 (v_max is not reached)
        if self.Tv < 0:
            # Set Tv = 0 (to prevent computation errors)
            self.Tv = 0

            # Iterate to find appropriate acceleration constraints using binary search
            gamma_high = 1.0
            gamma_low = MIN_GAMMA
            gamma_mid = 0.5
            iteration = 0

            while iteration < MAX_ITERATIONS:
                iteration += 1
                gamma_mid = (gamma_high + gamma_low) / 2

                # Test with current gamma
                a_max_test = gamma_mid * self.bounds.a_bound
                a_min_test = -gamma_mid * self.bounds.a_bound

                # Recalculate time intervals
                tj = a_max_test / self.j_max
                delta = (
                    a_max_test**4 / self.j_max**2
                    + 2 * (self.v_0_transformed**2 + self.v_1_transformed**2)
                    + a_max_test
                    * (
                        4 * (self.q_1_transformed - self.q_0_transformed)
                        - 2
                        * a_max_test
                        / self.j_max
                        * (self.v_0_transformed + self.v_1_transformed)
                    )
                )

                # Check if delta is negative (no solution with current gamma)
                if delta < 0:
                    gamma_high = gamma_mid
                    continue

                ta = (a_max_test**2 / self.j_max - 2 * self.v_0_transformed + np.sqrt(delta)) / (
                    2 * a_max_test
                )
                td = (a_max_test**2 / self.j_max - 2 * self.v_1_transformed + np.sqrt(delta)) / (
                    2 * a_max_test
                )

                if ta < 0:
                    if (
                        abs(self.v_1_transformed + self.v_0_transformed) < EPSILON
                    ):  # Avoid division by zero
                        ta = 0
                        td = 0
                        tj_1 = 0
                        tj_2 = 0
                        break
                    ta = 0
                    td = (
                        2
                        * (self.q_1_transformed - self.q_0_transformed)
                        / (self.v_1_transformed + self.v_0_transformed)
                    )
                    tj_2_arg = self.j_max * (self.q_1_transformed - self.q_0_transformed) - np.sqrt(
                        self.j_max
                        * (
                            self.j_max * (self.q_1_transformed - self.q_0_transformed) ** 2
                            + (self.v_1_transformed + self.v_0_transformed) ** 2
                            * (self.v_1_transformed - self.v_0_transformed)
                        )
                    )
                    tj_2 = (
                        tj_2_arg / (self.j_max * (self.v_1_transformed + self.v_0_transformed))
                        if abs(tj_2_arg) > EPSILON
                        else 0
                    )
                elif td < 0:
                    if (
                        abs(self.v_1_transformed + self.v_0_transformed) < EPSILON
                    ):  # Avoid division by zero
                        ta = 0
                        td = 0
                        tj_1 = 0
                        tj_2 = 0
                        break
                    td = 0
                    ta = (
                        2
                        * (self.q_1_transformed - self.q_0_transformed)
                        / (self.v_1_transformed + self.v_0_transformed)
                    )
                    tj_1_arg = self.j_max * (self.q_1_transformed - self.q_0_transformed) - np.sqrt(
                        self.j_max
                        * (
                            self.j_max * (self.q_1_transformed - self.q_0_transformed) ** 2
                            - (self.v_1_transformed + self.v_0_transformed) ** 2
                            * (self.v_1_transformed - self.v_0_transformed)
                        )
                    )
                    tj_1 = (
                        tj_1_arg / (self.j_max * (self.v_1_transformed + self.v_0_transformed))
                        if abs(tj_1_arg) > EPSILON
                        else 0
                    )
                elif (ta > 2 * tj) and (td > 2 * tj):
                    # Valid solution found
                    self.a_max = a_max_test
                    self.a_min = a_min_test
                    self.Tj_1 = tj
                    self.Tj_2 = tj
                    self.Ta = ta
                    self.Td = td
                    break
                else:
                    # Need to reduce gamma further
                    gamma_high = gamma_mid
                    continue

                # Check if solution is valid
                if tj_1 >= 0 and tj_2 >= 0 and ta >= 0 and td >= 0:
                    self.a_max = a_max_test
                    self.a_min = a_min_test
                    self.Tj_1 = tj_1
                    self.Tj_2 = tj_2
                    self.Ta = ta
                    self.Td = td
                    break
                gamma_high = gamma_mid

        # Compute trajectory parameters
        self.a_lim_a = self.j_max * self.Tj_1
        self.a_lim_d = -self.j_max * self.Tj_2

        # Ensure we don't divide by zero or have negative time periods
        self.Ta = max(self.Ta, 0)
        self.Td = max(self.Td, 0)
        self.Tv = max(self.Tv, 0)
        self.Tj_1 = max(self.Tj_1, 0)
        self.Tj_2 = max(self.Tj_2, 0)

        # Calculate v_lim safely
        if self.Ta <= self.Tj_1:
            self.v_lim = self.v_0_transformed + self.j_max * self.Ta**2 / 2
        else:
            self.v_lim = self.v_0_transformed + (self.Ta - self.Tj_1) * self.a_lim_a

        # Total trajectory time
        self.T = self.Ta + self.Tv + self.Td

        # Round final time to discrete ticks (in milliseconds)
        self.T = round(self.T * 1000) / 1000

    def evaluate(self, t: float | np.ndarray) -> tuple[float | np.ndarray, ...]:
        """
        Evaluate the double-S trajectory at time t.

        Parameters:
        -----------
        t : float or ndarray
            Time(s) at which to evaluate the trajectory

        Returns:
        --------
        q : float or ndarray
            Position at time t
        qp : float or ndarray
            Velocity at time t
        qpp : float or ndarray
            Acceleration at time t
        qppp : float or ndarray
            Jerk at time t
        """
        # Handle array input
        if isinstance(t, list | np.ndarray):
            # Preallocate arrays for efficiency
            q = np.zeros_like(t, dtype=float)
            qp = np.zeros_like(t, dtype=float)
            qpp = np.zeros_like(t, dtype=float)
            qppp = np.zeros_like(t, dtype=float)

            # Compute for each time point
            for i, t_i in enumerate(t):
                q[i], qp[i], qpp[i], qppp[i] = self.evaluate(float(t_i))
            return q, qp, qpp, qppp

        # Special case for equal positions with equal velocities
        if np.isclose(self.state.q_0, self.state.q_1) and np.isclose(
            self.state.v_0, self.state.v_1
        ):
            return self.state.q_0, self.state.v_0, 0.0, 0.0

        # Special case for equal positions with different velocities
        if np.isclose(self.state.q_0, self.state.q_1) and not np.isclose(
            self.state.v_0, self.state.v_1
        ):
            t_norm = min(t / self.T, 1.0)
            qp = self.state.v_0 + t_norm * (self.state.v_1 - self.state.v_0)

            phase = 2 * np.pi * t_norm
            amplitude = (self.state.v_1 - self.state.v_0) * self.T / (2 * np.pi)
            q = self.state.q_0 + amplitude * np.sin(phase)

            qpp = (self.state.v_1 - self.state.v_0) / self.T + amplitude * (
                2 * np.pi / self.T
            ) * np.cos(phase)
            qppp = -amplitude * (2 * np.pi / self.T) ** 2 * np.sin(phase)

            return q, qp, qpp, qppp

        # Ensure t is within bounds [0, T]
        t = np.clip(t, 0, self.T)

        # Handle zero or near-zero duration trajectory
        if self.T < EPSILON:
            return self.state.q_1, self.state.v_1, 0, 0

        # Use transformed coordinates for calculation
        q_0 = self.q_0_transformed
        q_1 = self.q_1_transformed

        # ACCELERATION PHASE
        if t <= self.Tj_1 and self.Tj_1 > 0:
            # t in [0, Tj_1]
            q = q_0 + self.v_0_transformed * t + self.j_max * t**3 / 6
            qp = self.v_0_transformed + self.j_max * (t**2) / 2
            qpp = self.j_max * t
            qppp = self.j_max

        elif t <= (self.Ta - self.Tj_1) and self.Ta > self.Tj_1:
            # t in [Tj_1, Ta - Tj_1]
            q = (
                q_0
                + self.v_0_transformed * t
                + self.a_lim_a / 6 * (3 * t**2 - 3 * self.Tj_1 * t + self.Tj_1**2)
            )
            qp = self.v_0_transformed + self.a_lim_a * (t - self.Tj_1 / 2)
            qpp = self.a_lim_a
            qppp = 0

        elif t <= self.Ta and self.Ta > 0:
            # t in [Ta-Tj_1, Ta]
            q = (
                q_0
                + (self.v_lim + self.v_0_transformed) * self.Ta / 2
                - self.v_lim * (self.Ta - t)
                - self.j_min * (self.Ta - t) ** 3 / 6
            )
            qp = self.v_lim + self.j_min * (self.Ta - t) ** 2 / 2
            qpp = -self.j_min * (self.Ta - t)
            qppp = self.j_min

        # CONSTANT VELOCITY PHASE
        elif t <= (self.Ta + self.Tv) and self.Tv > 0:
            # t in [Ta, Ta+Tv]
            q = q_0 + (self.v_lim + self.v_0_transformed) * self.Ta / 2 + self.v_lim * (t - self.Ta)
            qp = self.v_lim
            qpp = 0
            qppp = 0

        # DECELERATION PHASE
        elif t <= (self.Ta + self.Tv + self.Tj_2) and self.Tj_2 > 0:
            # t in [Ta+Tv, Ta+Tv+Tj_2]
            q = (
                q_1
                - (self.v_lim + self.v_1_transformed) * self.Td / 2
                + self.v_lim * (t - self.T + self.Td)
                - self.j_max * (t - self.T + self.Td) ** 3 / 6
            )
            qp = self.v_lim - self.j_max * (t - self.T + self.Td) ** 2 / 2
            qpp = -self.j_max * (t - self.T + self.Td)
            qppp = -self.j_max

        elif t <= (self.Ta + self.Tv + (self.Td - self.Tj_2)) and self.Td > self.Tj_2:
            # t in [Ta+Tv+Tj_2, Ta+Tv+(Td-Tj_2)]
            q = (
                q_1
                - (self.v_lim + self.v_1_transformed) * self.Td / 2
                + self.v_lim * (t - self.T + self.Td)
                + self.a_lim_d
                / 6
                * (
                    3 * (t - self.T + self.Td) ** 2
                    - 3 * self.Tj_2 * (t - self.T + self.Td)
                    + self.Tj_2**2
                )
            )
            qp = self.v_lim + self.a_lim_d * (t - self.T + self.Td - self.Tj_2 / 2)
            qpp = self.a_lim_d
            qppp = 0

        elif t <= self.T and self.Td > 0:
            # t in [Ta+Tv+(Td-Tj_2), T]
            q = q_1 - self.v_1_transformed * (self.T - t) - self.j_max * (self.T - t) ** 3 / 6
            qp = self.v_1_transformed + self.j_max * (self.T - t) ** 2 / 2
            qpp = -self.j_max * (self.T - t)
            qppp = self.j_max

        else:
            # After end of trajectory or for empty phases
            q = q_1
            qp = self.v_1_transformed
            qpp = 0
            qppp = 0

        # Transform back using sigma
        q = self.sigma * q
        qp = self.sigma * qp
        qpp = self.sigma * qpp
        qppp = self.sigma * qppp

        return q, qp, qpp, qppp

    def get_duration(self) -> float:
        """
        Returns the total duration of the trajectory.

        Returns:
        --------
        T : float
            Total trajectory duration in seconds
        """
        return self.T

    def get_phase_durations(self) -> dict[str, float]:
        """
        Returns the durations of each phase in the trajectory.

        Returns:
        --------
        phases : dict
            Dictionary containing the durations of each phase
        """
        return {
            "total": self.T,
            "acceleration": self.Ta,
            "constant_velocity": self.Tv,
            "deceleration": self.Td,
            "jerk_acceleration": self.Tj_1,
            "jerk_deceleration": self.Tj_2,
        }

    @staticmethod
    def create_trajectory(
        state_params: StateParams, bounds: TrajectoryBounds
    ) -> tuple[Callable[[float | np.ndarray], tuple[float | np.ndarray, ...]], float]:
        """
        Static factory method to create a trajectory function and return its duration.
        This method provides an interface similar to the original function-based API.

        Parameters:
        -----------
        state_params : StateParams
            Start and end states for trajectory planning
        bounds : TrajectoryBounds
            Velocity, acceleration, and jerk bounds for trajectory planning

        Returns:
        --------
        trajectory_function : Callable
            A function that takes time t as input and returns position, velocity,
            acceleration, and jerk at that time
        T : float
            Total trajectory duration
        """
        planner = DoubleSTrajectory(state_params, bounds)

        def trajectory(t: float | np.ndarray) -> tuple[float | np.ndarray, ...]:
            return planner.evaluate(t)

        return trajectory, planner.get_duration()
