from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

import numpy as np


# Define constants for polynomial orders
ORDER_3 = 3
ORDER_5 = 5
ORDER_7 = 7


@dataclass
class BoundaryCondition:
    """Class for storing boundary conditions for trajectory generation."""

    position: float
    velocity: float
    acceleration: float = 0.0
    jerk: float = 0.0


@dataclass
class TimeInterval:
    """Class for storing time interval for trajectory generation."""

    start: float
    end: float


@dataclass
class TrajectoryParams:
    """Class for storing parameters for multipoint trajectory generation."""

    points: list[float]
    times: list[float]
    velocities: list[float] | None = None
    accelerations: list[float] | None = None
    jerks: list[float] | None = None
    order: int = ORDER_3


class PolynomialTrajectory:
    """
    A class for generating polynomial trajectories with specified boundary conditions.

    This class provides methods to create polynomial trajectories of different orders
    (3rd, 5th, and 7th) with specified boundary conditions such as position, velocity,
    acceleration, and jerk. It also supports creating trajectories through multiple points.

    Methods
    -------
    order_3_trajectory
        Generate a 3rd order polynomial trajectory with position and velocity constraints
    order_5_trajectory
        Generate a 5th order polynomial trajectory with position, velocity, and
        acceleration constraints
    order_7_trajectory
        Generate a 7th order polynomial trajectory with position, velocity, acceleration,
        and jerk constraints
    heuristic_velocities
        Compute intermediate velocities for a sequence of points
    multipoint_trajectory
        Generate a trajectory through a sequence of points with specified times
    """

    # Define the valid polynomial orders as class variables
    VALID_ORDERS: ClassVar[tuple[int, ...]] = (ORDER_3, ORDER_5, ORDER_7)

    @staticmethod
    def order_3_trajectory(
        initial: BoundaryCondition,
        final: BoundaryCondition,
        time: TimeInterval,
    ) -> Callable[[float], tuple[float, float, float, float]]:
        """
        Generate a 3rd order polynomial trajectory with specified boundary conditions.

        Parameters
        ----------
        initial : BoundaryCondition
            Initial boundary conditions (position, velocity)
        final : BoundaryCondition
            Final boundary conditions (position, velocity)
        time : TimeInterval
            Time interval for the trajectory

        Returns
        -------
        Callable[[float], tuple[float, float, float, float]]
            Function that computes position, velocity, acceleration, and jerk at time t
        """
        t_diff = time.end - time.start
        h = final.position - initial.position

        # Coefficients as defined in equation (2.2)
        a0 = initial.position
        a1 = initial.velocity
        a2 = (3 * h - (2 * initial.velocity + final.velocity) * t_diff) / (t_diff**2)
        a3 = (-2 * h + (initial.velocity + final.velocity) * t_diff) / (t_diff**3)

        def trajectory(t: float) -> tuple[float, float, float, float]:
            # Ensure t is within bounds
            t = np.clip(t, time.start, time.end)

            # Time relative to t_start
            tau = t - time.start

            # Position
            q = a0 + a1 * tau + a2 * tau**2 + a3 * tau**3

            # Velocity
            qd = a1 + 2 * a2 * tau + 3 * a3 * tau**2

            # Acceleration
            qdd = 2 * a2 + 6 * a3 * tau

            # Jerk
            qddd = 6 * a3

            return q, qd, qdd, qddd

        return trajectory

    @staticmethod
    def order_5_trajectory(
        initial: BoundaryCondition,
        final: BoundaryCondition,
        time: TimeInterval,
    ) -> Callable[[float], tuple[float, float, float, float]]:
        """
        Generate a 5th order polynomial trajectory with specified boundary conditions.

        Parameters
        ----------
        initial : BoundaryCondition
            Initial boundary conditions (position, velocity, acceleration)
        final : BoundaryCondition
            Final boundary conditions (position, velocity, acceleration)
        time : TimeInterval
            Time interval for the trajectory

        Returns
        -------
        Callable[[float], tuple[float, float, float, float]]
            Function that computes position, velocity, acceleration, and jerk at time t
        """
        t_diff = time.end - time.start
        h = final.position - initial.position

        # Coefficients as defined in equation (2.5)
        a0 = initial.position
        a1 = initial.velocity
        a2 = initial.acceleration / 2
        a3 = (1 / (2 * t_diff**3)) * (
            20 * h
            - (8 * final.velocity + 12 * initial.velocity) * t_diff
            - (3 * initial.acceleration - final.acceleration) * t_diff**2
        )
        a4 = (1 / (2 * t_diff**4)) * (
            -30 * h
            + (14 * final.velocity + 16 * initial.velocity) * t_diff
            + (3 * initial.acceleration - 2 * final.acceleration) * t_diff**2
        )
        a5 = (1 / (2 * t_diff**5)) * (
            12 * h
            - 6 * (final.velocity + initial.velocity) * t_diff
            + (final.acceleration - initial.acceleration) * t_diff**2
        )

        def trajectory(t: float) -> tuple[float, float, float, float]:
            # Ensure t is within bounds
            t = np.clip(t, time.start, time.end)

            # Time relative to t_start
            tau = t - time.start

            # Position
            q = a0 + a1 * tau + a2 * tau**2 + a3 * tau**3 + a4 * tau**4 + a5 * tau**5

            # Velocity
            qd = a1 + 2 * a2 * tau + 3 * a3 * tau**2 + 4 * a4 * tau**3 + 5 * a5 * tau**4

            # Acceleration
            qdd = 2 * a2 + 6 * a3 * tau + 12 * a4 * tau**2 + 20 * a5 * tau**3

            # Jerk
            qddd = 6 * a3 + 24 * a4 * tau + 60 * a5 * tau**2

            return q, qd, qdd, qddd

        return trajectory

    @staticmethod
    def order_7_trajectory(
        initial: BoundaryCondition,
        final: BoundaryCondition,
        time: TimeInterval,
    ) -> Callable[[float], tuple[float, float, float, float]]:
        """
        Generate a 7th order polynomial trajectory with specified boundary conditions.

        Parameters
        ----------
        initial : BoundaryCondition
            Initial boundary conditions (position, velocity, acceleration, jerk)
        final : BoundaryCondition
            Final boundary conditions (position, velocity, acceleration, jerk)
        time : TimeInterval
            Time interval for the trajectory

        Returns
        -------
        Callable[[float], tuple[float, float, float, float]]
            Function that computes position, velocity, acceleration, and jerk at time t
        """
        t_diff = time.end - time.start
        h = final.position - initial.position

        # Coefficients for 7th order polynomial
        a0 = initial.position
        a1 = initial.velocity
        a2 = initial.acceleration / 2
        a3 = initial.jerk / 6
        a4 = (
            210 * h
            - t_diff
            * (
                (30 * initial.acceleration - 15 * final.acceleration) * t_diff
                + (4 * initial.jerk + final.jerk) * t_diff**2
                + 120 * initial.velocity
                + 90 * final.velocity
            )
        ) / (6 * t_diff**4)
        a5 = (
            -168 * h
            + t_diff
            * (
                (20 * initial.acceleration - 14 * final.acceleration) * t_diff
                + (2 * initial.jerk + final.jerk) * t_diff**2
                + 90 * initial.velocity
                + 78 * final.velocity
            )
        ) / (2 * t_diff**5)
        a6 = (
            420 * h
            - t_diff
            * (
                (45 * initial.acceleration - 39 * final.acceleration) * t_diff
                + (4 * initial.jerk + 3 * final.jerk) * t_diff**2
                + 216 * initial.velocity
                + 204 * final.velocity
            )
        ) / (6 * t_diff**6)
        a7 = (
            -120 * h
            + t_diff
            * (
                (12 * initial.acceleration - 12 * final.acceleration) * t_diff
                + (initial.jerk + final.jerk) * t_diff**2
                + 60 * initial.velocity
                + 60 * final.velocity
            )
        ) / (6 * t_diff**7)

        def trajectory(t: float) -> tuple[float, float, float, float]:
            # Ensure t is within bounds
            t = np.clip(t, time.start, time.end)

            # Time relative to t_start
            tau = t - time.start

            # Position
            q = (
                a0
                + a1 * tau
                + a2 * tau**2
                + a3 * tau**3
                + a4 * tau**4
                + a5 * tau**5
                + a6 * tau**6
                + a7 * tau**7
            )

            # Velocity
            qd = (
                a1
                + 2 * a2 * tau
                + 3 * a3 * tau**2
                + 4 * a4 * tau**3
                + 5 * a5 * tau**4
                + 6 * a6 * tau**5
                + 7 * a7 * tau**6
            )

            # Acceleration
            qdd = (
                2 * a2
                + 6 * a3 * tau
                + 12 * a4 * tau**2
                + 20 * a5 * tau**3
                + 30 * a6 * tau**4
                + 42 * a7 * tau**5
            )

            # Jerk
            qddd = 6 * a3 + 24 * a4 * tau + 60 * a5 * tau**2 + 120 * a6 * tau**3 + 210 * a7 * tau**4

            return q, qd, qdd, qddd

        return trajectory

    @staticmethod
    def heuristic_velocities(points: list[float], times: list[float]) -> list[float]:
        """
        Compute intermediate velocities for a sequence of points using the heuristic rule.

        The heuristic rule sets the velocity at each intermediate point to the average of
        the slopes of the adjacent segments, unless the slopes have different signs, in which
        case the velocity is set to zero.

        Parameters
        ----------
        points : list[float]
            List of position points [q0, q1, ..., qn]
        times : list[float]
            List of time points [t0, t1, ..., tn]

        Returns
        -------
        list[float]
            List of velocities [v0, v1, ..., vn]
        """
        n = len(points)
        velocities = [0.0] * n  # Initialize with zeros

        # Compute the slopes between consecutive points
        slopes = [(points[i] - points[i - 1]) / (times[i] - times[i - 1]) for i in range(1, n)]

        # First and last velocities are set to 0 by default
        velocities[0] = 0.0
        velocities[n - 1] = 0.0

        # Compute intermediate velocities using the heuristic rule
        for i in range(1, n - 1):
            if np.sign(slopes[i - 1]) != np.sign(slopes[i]):
                velocities[i] = 0.0
            else:
                velocities[i] = 0.5 * (slopes[i - 1] + slopes[i])

        return velocities

    @classmethod
    def multipoint_trajectory(
        cls: type["PolynomialTrajectory"],
        params: TrajectoryParams,
    ) -> Callable[[float], tuple[float, float, float, float]]:
        """
        Generate a trajectory through a sequence of points with specified times.

        Parameters
        ----------
        params : TrajectoryParams
            Parameters for trajectory generation including points, times, and optional
            velocities, accelerations, jerks, and polynomial order.

        Returns
        -------
        Callable[[float], tuple[float, float, float, float]]
            Function that computes trajectory at time t

        Raises
        ------
        ValueError
            If number of points and times are not the same, or if order is not
            one of the valid polynomial orders.
        """
        n = len(params.points)

        if n != len(params.times):
            raise ValueError("Number of points and times must be the same")

        if params.order not in cls.VALID_ORDERS:
            valid_orders_str = ", ".join(str(order) for order in cls.VALID_ORDERS)
            raise ValueError(f"Order must be one of: {valid_orders_str}")

        # If velocities are not provided, compute using heuristic rule
        vel = params.velocities
        if vel is None:
            vel = cls.heuristic_velocities(params.points, params.times)

        # If accelerations are not provided, set to zeros
        acc = params.accelerations
        if acc is None and params.order in {ORDER_5, ORDER_7}:
            acc = [0.0] * n

        # If jerks are not provided, set to zeros
        jrk = params.jerks
        if jrk is None and params.order == ORDER_7:
            jrk = [0.0] * n

        # Create a list of segment trajectories
        segments = []

        for i in range(n - 1):
            # Create time interval for this segment
            time_interval = TimeInterval(params.times[i], params.times[i + 1])

            if params.order == ORDER_3:
                # 3rd order trajectory
                initial = BoundaryCondition(params.points[i], vel[i])
                final = BoundaryCondition(params.points[i + 1], vel[i + 1])
                segment = cls.order_3_trajectory(initial, final, time_interval)
            elif params.order == ORDER_5 and acc is not None:
                # 5th order trajectory
                initial = BoundaryCondition(params.points[i], vel[i], acc[i])
                final = BoundaryCondition(params.points[i + 1], vel[i + 1], acc[i + 1])
                segment = cls.order_5_trajectory(initial, final, time_interval)
            elif params.order == ORDER_7 and acc is not None and jrk is not None:
                # 7th order trajectory
                initial = BoundaryCondition(params.points[i], vel[i], acc[i], jrk[i])
                final = BoundaryCondition(params.points[i + 1], vel[i + 1], acc[i + 1], jrk[i + 1])
                segment = cls.order_7_trajectory(initial, final, time_interval)

            segments.append((segment, params.times[i], params.times[i + 1]))

        def trajectory(t: float) -> tuple[float, float, float, float]:
            # Handle boundary cases first for efficiency
            if t < params.times[0]:
                return segments[0][0](params.times[0])
            if t > params.times[-1]:
                return segments[-1][0](params.times[-1])

            # Binary search to find the appropriate segment
            left, right = 0, len(segments) - 1

            while left <= right:
                mid = (left + right) // 2
                t_start, t_end = segments[mid][1], segments[mid][2]

                if t_start <= t <= t_end:
                    return segments[mid][0](t)
                if t < t_start:
                    right = mid - 1
                else:  # t > t_end
                    left = mid + 1

            raise ValueError(f"No segment found for time {t}")

        return trajectory
