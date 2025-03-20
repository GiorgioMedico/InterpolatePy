from dataclasses import dataclass
from typing import NamedTuple
from typing import TypeAlias

import numpy as np
import numpy.typing as npt


# Type aliases for cleaner annotations
ArrayLike: TypeAlias = list[float] | npt.NDArray[np.float64]
TrajectoryOutput: TypeAlias = tuple[
    float | npt.NDArray[np.float64],
    float | npt.NDArray[np.float64],
    float | npt.NDArray[np.float64],
    float | npt.NDArray[np.float64],
]

# Constants to avoid magic numbers
EPSILON = 1e-6


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
    Double S-Trajectory Planner Class.

    This class implements a trajectory planner that generates smooth motion profiles
    with bounded jerk, acceleration, and velocity (Double S-curve trajectory).

    Parameters
    ----------
    state : StateParams
        Object containing start/end positions and velocities.
    bounds : TrajectoryBounds
        Object containing bounds for velocity, acceleration, and jerk.

    Raises
    ------
    TypeError
        If any parameter is not a numeric value.
    ValueError
        If any bound is not positive or if initial/final velocities exceed bounds.

    Notes
    -----
    The trajectory is planned during initialization and can be evaluated at any
    time point using the evaluate() method or by calling the instance directly.
    """

    def __init__(
        self,
        state: StateParams,
        bounds: TrajectoryBounds,
    ) -> None:
        # Input validation for state parameters
        if not all(isinstance(x, int | float) for x in state):
            raise TypeError("All state parameters must be numeric values")

        # Store original parameters
        self.q_0: float = float(state.q_0)
        self.q_1: float = float(state.q_1)
        self.v_0: float = float(state.v_0)
        self.v_1: float = float(state.v_1)

        # Store bounds
        self.v_bound: float = bounds.v_bound
        self.a_bound: float = bounds.a_bound
        self.j_bound: float = bounds.j_bound

        # Check if initial or final velocities exceed bounds
        if abs(self.v_0) > self.v_bound or abs(self.v_1) > self.v_bound:
            raise ValueError(
                f"Initial or final velocities exceed the velocity bound of {self.v_bound}"
            )

        # Initialize attributes for trajectory parameters
        self.t_total: float = 0.0
        self.t_accel: float = 0.0
        self.t_const: float = 0.0
        self.t_decel: float = 0.0
        self.t_jerk1: float = 0.0
        self.t_jerk2: float = 0.0
        self.v_lim: float = 0.0
        self.a_lim_a: float = 0.0
        self.a_lim_d: float = 0.0
        self.sigma: float = 1.0  # Direction sign

        # Compute the trajectory parameters
        self._plan_trajectory()

    def _plan_trajectory(self) -> None:
        """
        Plan the double S trajectory by computing all necessary parameters.

        This method is called during initialization to compute the trajectory parameters.
        """
        # If positions are equal, handle specially
        if np.isclose(self.q_1, self.q_0):
            self._handle_equal_positions()
            return

        # Normal case - different positions
        self._handle_different_positions()

    def _handle_equal_positions(self) -> None:
        """Handle the special case where start and end positions are equal."""
        if np.isclose(self.v_1, self.v_0):
            # If velocities also match, set minimal trajectory parameters
            self.t_total = 0.0
            self.t_accel = 0.0
            self.t_const = 0.0
            self.t_decel = 0.0
            self.t_jerk1 = 0.0
            self.t_jerk2 = 0.0
            self.v_lim = self.v_0
            self.a_lim_a = 0.0
            self.a_lim_d = 0.0
            return

        # Different velocities but same position
        t_min = abs(self.v_1 - self.v_0) / self.a_bound
        self.t_total = max(t_min * 1.5, 0.1)  # Ensure minimum duration

        # Set other parameters accordingly
        self.t_accel = self.t_total / 2.0
        self.t_const = 0.0
        self.t_decel = self.t_total / 2.0
        self.t_jerk1 = self.t_accel / 4.0
        self.t_jerk2 = self.t_decel / 4.0
        self.v_lim = (self.v_0 + self.v_1) / 2.0
        self.a_lim_a = (self.v_1 - self.v_0) / (self.t_total / 2.0)
        self.a_lim_d = -self.a_lim_a

    def _handle_different_positions(self) -> None:
        """Handle the normal case where start and end positions differ."""
        # Determine direction
        self.sigma = np.sign(self.q_1 - self.q_0)

        # Create transformed state parameters
        transformed_state = StateParams(
            q_0=self.sigma * self.q_0,
            q_1=self.sigma * self.q_1,
            v_0=self.sigma * self.v_0,
            v_1=self.sigma * self.v_1,
        )

        # Apply sigma to jerk bound
        j_max = self.j_bound if self.sigma > 0 else -self.j_bound
        v_max = self.v_bound if self.sigma > 0 else -self.v_bound
        a_max = self.a_bound if self.sigma > 0 else -self.a_bound

        # Compute time intervals assuming v_max and a_max are reached
        q_0, q_1, v_0, v_1 = transformed_state

        # Acceleration part
        if ((v_max - v_0) * j_max) < a_max**2:
            t_jerk1 = np.sqrt(max((v_max - v_0) / j_max, 0))
            t_accel = 2 * t_jerk1
        else:
            t_jerk1 = a_max / j_max
            t_accel = t_jerk1 + (v_max - v_0) / a_max

        # Deceleration part
        if ((v_max - v_1) * j_max) < a_max**2:
            t_jerk2 = np.sqrt(max((v_max - v_1) / j_max, 0))
            t_decel = 2 * t_jerk2
        else:
            t_jerk2 = a_max / j_max
            t_decel = t_jerk2 + (v_max - v_1) / a_max

        # Determine the time duration of the constant velocity phase
        if abs(v_max) < EPSILON:  # Avoid division by zero
            t_const = 0.0
        else:
            t_const = (
                (q_1 - q_0) / v_max
                - t_accel / 2 * (1 + v_0 / v_max)
                - t_decel / 2 * (1 + v_1 / v_max)
            )

        # Check if t_const < 0 (v_max is not reached)
        if t_const < 0:
            # Set t_const = 0 and adjust parameters
            t_const = 0.0
            a_max, _, t_accel, t_decel, t_jerk1, t_jerk2 = self._adjust_for_unreachable_velocity(
                transformed_state, j_max
            )

        # Store results
        self.t_jerk1 = max(t_jerk1, 0.0)
        self.t_jerk2 = max(t_jerk2, 0.0)
        self.t_accel = max(t_accel, 0.0)
        self.t_decel = max(t_decel, 0.0)
        self.t_const = max(t_const, 0.0)
        self.a_lim_a = j_max * self.t_jerk1
        self.a_lim_d = -j_max * self.t_jerk2

        # Calculate v_lim safely
        if self.t_accel <= self.t_jerk1:
            self.v_lim = v_0 + j_max * self.t_accel**2 / 2
        else:
            self.v_lim = v_0 + (self.t_accel - self.t_jerk1) * self.a_lim_a

        # Total trajectory time
        self.t_total = self.t_accel + self.t_const + self.t_decel

        # Round final time to discrete ticks (in milliseconds)
        self.t_total = round(self.t_total * 1000) / 1000

    def _adjust_for_unreachable_velocity(
        self,
        state: StateParams,
        j_max: float,
    ) -> tuple[float, float, float, float, float, float]:
        """
        Adjust parameters when maximum velocity cannot be reached.

        Parameters
        ----------
        state : StateParams
            Transformed state parameters.
        j_max : float
            Transformed maximum jerk.

        Returns
        -------
        tuple[float, float, float, float, float, float]
            Adjusted parameters (a_max, a_min, t_accel, t_decel, t_jerk1, t_jerk2).
        """
        # Extract state parameters
        q_0, q_1, v_0, v_1 = state

        # Iterate to find appropriate acceleration constraints using binary search
        gamma_high = 1.0
        gamma_low = 0.01  # Lower bound for gamma
        gamma_mid = 0.5
        max_iterations = 50

        a_max = self.a_bound
        a_min = -self.a_bound
        t_accel = 0.0
        t_decel = 0.0
        t_jerk1 = 0.0
        t_jerk2 = 0.0

        for _ in range(max_iterations):
            gamma_mid = (gamma_high + gamma_low) / 2

            # Test with current gamma
            a_max_test = gamma_mid * self.a_bound
            a_min_test = -gamma_mid * self.a_bound

            # Recalculate time intervals
            t_jerk = a_max_test / j_max
            delta = (
                a_max_test**4 / j_max**2
                + 2 * (v_0**2 + v_1**2)
                + a_max_test * (4 * (q_1 - q_0) - 2 * a_max_test / j_max * (v_0 + v_1))
            )

            # Check if delta is negative (no solution with current gamma)
            if delta < 0:
                gamma_high = gamma_mid
                continue

            t_accel = (a_max_test**2 / j_max - 2 * v_0 + np.sqrt(delta)) / (2 * a_max_test)
            t_decel = (a_max_test**2 / j_max - 2 * v_1 + np.sqrt(delta)) / (2 * a_max_test)

            if t_accel < 0:
                if abs(v_1 + v_0) < EPSILON:  # Avoid division by zero
                    t_accel = 0.0
                    t_decel = 0.0
                    t_jerk1 = 0.0
                    t_jerk2 = 0.0
                    break
                t_accel = 0.0
                t_decel = 2.0 * (q_1 - q_0) / (v_1 + v_0)
                t_jerk2_arg = j_max * (q_1 - q_0) - np.sqrt(
                    j_max * (j_max * (q_1 - q_0) ** 2 + (v_1 + v_0) ** 2 * (v_1 - v_0))
                )
                t_jerk2 = float(
                    t_jerk2_arg / (j_max * (v_1 + v_0)) if abs(t_jerk2_arg) > EPSILON else 0.0
                )
            elif t_decel < 0:
                if abs(v_1 + v_0) < EPSILON:  # Avoid division by zero
                    t_accel = 0.0
                    t_decel = 0.0
                    t_jerk1 = 0.0
                    t_jerk2 = 0.0
                    break
                t_decel = 0.0
                t_accel = 2.0 * (q_1 - q_0) / (v_1 + v_0)
                t_jerk1_arg = j_max * (q_1 - q_0) - np.sqrt(
                    j_max * (j_max * (q_1 - q_0) ** 2 - (v_1 + v_0) ** 2 * (v_1 - v_0))
                )
                t_jerk1 = float(
                    t_jerk1_arg / (j_max * (v_1 + v_0)) if abs(t_jerk1_arg) > EPSILON else 0.0
                )
            elif (t_accel > 2 * t_jerk) and (t_decel > 2 * t_jerk):
                # Valid solution found
                a_max = a_max_test
                a_min = a_min_test
                t_jerk1 = t_jerk
                t_jerk2 = t_jerk
                break
            else:
                # Need to reduce gamma further
                gamma_high = gamma_mid
                continue

            # Check if solution is valid
            if t_jerk1 >= 0 and t_jerk2 >= 0 and t_accel >= 0 and t_decel >= 0:
                a_max = a_max_test
                a_min = a_min_test
                break
            gamma_high = gamma_mid

        return a_max, a_min, t_accel, t_decel, t_jerk1, t_jerk2

    def evaluate(self, t: float | ArrayLike) -> TrajectoryOutput:
        """
        Evaluate the double-S trajectory at given time(s).

        Parameters
        ----------
        t : float or array_like
            Time(s) at which to evaluate the trajectory.

        Returns
        -------
        tuple
            Contains position, velocity, acceleration, and jerk at given time(s).
            Each element is a float (scalar input) or ndarray (array input).
        """
        return self(t)

    def __call__(self, t: float | ArrayLike) -> TrajectoryOutput:
        """
        Callable interface to evaluate the double-S trajectory at given time(s).

        Parameters
        ----------
        t : float or array_like
            Time(s) at which to evaluate the trajectory.

        Returns
        -------
        tuple
            Contains position, velocity, acceleration, and jerk at given time(s).
            Each element is a float (scalar input) or ndarray (array input).
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
                q[i], qp[i], qpp[i], qppp[i] = self._evaluate_scalar(float(t_i))
            return q, qp, qpp, qppp

        # Handle scalar input
        return self._evaluate_scalar(float(t))

    def _evaluate_scalar(self, t: float) -> tuple[float, float, float, float]:
        """
        Evaluate the double-S trajectory at a single time point.

        Parameters
        ----------
        t : float
            Time at which to evaluate the trajectory.

        Returns
        -------
        tuple
            Contains position, velocity, acceleration, and jerk (all float).
        """
        # Ensure t is within bounds [0, t_total]
        t = np.clip(t, 0, self.t_total)

        # Handle zero or near-zero duration trajectory
        if self.t_total < EPSILON:
            return self.q_1, self.v_1, 0.0, 0.0

        # If positions are equal, handle specially
        if np.isclose(self.q_1, self.q_0) and np.isclose(self.v_1, self.v_0):
            return self.q_0, self.v_0, 0.0, 0.0

        # Transform using sigma for calculation
        q_0 = self.sigma * self.q_0
        q_1 = self.sigma * self.q_1
        v_0 = self.sigma * self.v_0
        v_1 = self.sigma * self.v_1
        j_max = self.j_bound
        j_min = -self.j_bound

        # Compute trajectory segments
        if np.isclose(self.q_1, self.q_0) and not np.isclose(self.v_1, self.v_0):
            # Special case for equal positions but different velocities
            t_norm = min(t / self.t_total, 1.0)
            qp = v_0 + t_norm * (v_1 - v_0)

            phase = 2 * np.pi * t_norm
            amplitude = (v_1 - v_0) * self.t_total / (2 * np.pi)
            q = q_0 + amplitude * np.sin(phase)

            qpp = (v_1 - v_0) / self.t_total + amplitude * (2 * np.pi / self.t_total) * np.cos(
                phase
            )
            qppp = -amplitude * (2 * np.pi / self.t_total) ** 2 * np.sin(phase)
        # Standard trajectory segments
        # ACCELERATION PHASE
        elif t <= self.t_jerk1 and self.t_jerk1 > 0:
            # t in [0, t_jerk1]
            q = q_0 + v_0 * t + j_max * t**3 / 6
            qp = v_0 + j_max * (t**2) / 2
            qpp = j_max * t
            qppp = j_max

        elif t <= (self.t_accel - self.t_jerk1) and self.t_accel > self.t_jerk1:
            # t in [t_jerk1, t_accel - t_jerk1]
            q = (
                q_0
                + v_0 * t
                + self.a_lim_a / 6 * (3 * t**2 - 3 * self.t_jerk1 * t + self.t_jerk1**2)
            )
            qp = v_0 + self.a_lim_a * (t - self.t_jerk1 / 2)
            qpp = self.a_lim_a
            qppp = 0

        elif t <= self.t_accel and self.t_accel > 0:
            # t in [t_accel-t_jerk1, t_accel]
            q = (
                q_0
                + (self.v_lim + v_0) * self.t_accel / 2
                - self.v_lim * (self.t_accel - t)
                - j_min * (self.t_accel - t) ** 3 / 6
            )
            qp = self.v_lim + j_min * (self.t_accel - t) ** 2 / 2
            qpp = -j_min * (self.t_accel - t)
            qppp = j_min

        # CONSTANT VELOCITY PHASE
        elif t <= (self.t_accel + self.t_const) and self.t_const > 0:
            # t in [t_accel, t_accel+t_const]
            q = q_0 + (self.v_lim + v_0) * self.t_accel / 2 + self.v_lim * (t - self.t_accel)
            qp = self.v_lim
            qpp = 0
            qppp = 0

        # DECELERATION PHASE
        elif t <= (self.t_accel + self.t_const + self.t_jerk2) and self.t_jerk2 > 0:
            # t in [t_accel+t_const, t_accel+t_const+t_jerk2]
            q = (
                q_1
                - (self.v_lim + v_1) * self.t_decel / 2
                + self.v_lim * (t - self.t_total + self.t_decel)
                - j_max * (t - self.t_total + self.t_decel) ** 3 / 6
            )
            qp = self.v_lim - j_max * (t - self.t_total + self.t_decel) ** 2 / 2
            qpp = -j_max * (t - self.t_total + self.t_decel)
            qppp = -j_max

        elif (
            t <= (self.t_accel + self.t_const + (self.t_decel - self.t_jerk2))
            and self.t_decel > self.t_jerk2
        ):
            # t in [t_accel+t_const+t_jerk2, t_accel+t_const+(t_decel-t_jerk2)]
            q = (
                q_1
                - (self.v_lim + v_1) * self.t_decel / 2
                + self.v_lim * (t - self.t_total + self.t_decel)
                + self.a_lim_d
                / 6
                * (
                    3 * (t - self.t_total + self.t_decel) ** 2
                    - 3 * self.t_jerk2 * (t - self.t_total + self.t_decel)
                    + self.t_jerk2**2
                )
            )
            qp = self.v_lim + self.a_lim_d * (t - self.t_total + self.t_decel - self.t_jerk2 / 2)
            qpp = self.a_lim_d
            qppp = 0

        elif t <= self.t_total and self.t_decel > 0:
            # t in [t_accel+t_const+(t_decel-t_jerk2), t_total]
            q = q_1 - v_1 * (self.t_total - t) - j_max * (self.t_total - t) ** 3 / 6
            qp = v_1 + j_max * (self.t_total - t) ** 2 / 2
            qpp = -j_max * (self.t_total - t)
            qppp = j_max

        else:
            # After end of trajectory or for empty phases
            q = q_1
            qp = v_1
            qpp = 0
            qppp = 0

        # Transform back using sigma
        q = self.sigma * q
        qp = self.sigma * qp
        qpp = self.sigma * qpp
        qppp = self.sigma * qppp

        return q, qp, qpp, qppp

    @property
    def duration(self) -> float:
        """
        Get the total duration of the trajectory.

        Returns
        -------
        float
            Total trajectory duration in seconds.
        """
        return self.t_total

    def get_time_intervals(self) -> dict[str, float]:
        """
        Get all time intervals of the trajectory.

        Returns
        -------
        dict
            Dictionary containing all time intervals of the trajectory.
        """
        return {
            "total": self.t_total,
            "accel": self.t_accel,
            "const": self.t_const,
            "decel": self.t_decel,
            "jerk1": self.t_jerk1,
            "jerk2": self.t_jerk2,
        }
