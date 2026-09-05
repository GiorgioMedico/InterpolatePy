"""
Polynomial trajectory generation for smooth motion profiles.

This module provides polynomial-based trajectory planning algorithms that generate
smooth motion profiles with continuous derivatives up to the jerk level. The implementation
supports 3rd, 5th, and 7th order polynomials with customizable boundary conditions.

The mathematical foundations follow classical polynomial trajectory planning techniques
used in robotics and control systems, ensuring continuity of position, velocity,
acceleration, and optionally jerk at waypoints.
"""

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
    """
    Boundary conditions for polynomial trajectory generation.
    Parameters
    ----------
    position : float
        Position constraint.
    velocity : float
        Velocity constraint.
    acceleration : float, optional
        Acceleration constraint. Default is 0.0.
    jerk : float, optional
        Jerk constraint. Default is 0.0.
    Notes
    -----
    Higher-order polynomial trajectories require more boundary conditions:
    - 3rd order: position and velocity
    - 5th order: position, velocity, and acceleration
    - 7th order: position, velocity, acceleration, and jerk
    """

    position: float
    velocity: float
    acceleration: float = 0.0
    jerk: float = 0.0


@dataclass
class TimeInterval:
    """
    Time interval for trajectory generation.

    Parameters
    ----------
    start : float
        Start time of the trajectory segment.
    end : float
        End time of the trajectory segment.
    """

    start: float
    end: float


@dataclass
class TrajectoryParams:
    """
    Parameters for multipoint polynomial trajectory generation.

    Parameters
    ----------
    points : list[float]
        List of position waypoints to interpolate through.
    times : list[float]
        List of time points corresponding to each waypoint.
    velocities : list[float], optional
        Velocity constraints at each waypoint. If None, velocities are computed
        using heuristic rules.
    accelerations : list[float], optional
        Acceleration constraints at each waypoint. Required for 5th and 7th order
        polynomials. If None, zero accelerations are assumed.
    jerks : list[float], optional
        Jerk constraints at each waypoint. Required for 7th order polynomials.
        If None, zero jerks are assumed.
    order : int, optional
        Polynomial order (3, 5, or 7). Default is 3.
    """

    points: list[float]
    times: list[float]
    velocities: list[float] | None = None
    accelerations: list[float] | None = None
    jerks: list[float] | None = None
    order: int = ORDER_3


class PolynomialTrajectory:
    """
    Generate smooth polynomial trajectories with specified boundary conditions.

    This class provides methods to create polynomial trajectories of different orders
    (3rd, 5th, and 7th) with specified boundary conditions such as position, velocity,
    acceleration, and jerk. The polynomials ensure smooth motion profiles with continuous
    derivatives up to the jerk level, making them ideal for robotics and control applications.

    Methods
    -------
    order_3_trajectory(initial, final, time)
        Generate a 3rd order polynomial trajectory with position and velocity constraints.
    order_5_trajectory(initial, final, time)
        Generate a 5th order polynomial trajectory with position, velocity, and
        acceleration constraints.
    order_7_trajectory(initial, final, time)
        Generate a 7th order polynomial trajectory with position, velocity, acceleration,
        and jerk constraints.
    heuristic_velocities(points, times)
        Compute intermediate velocities for a sequence of points using heuristic rules.
    multipoint_trajectory(params)
        Generate a trajectory through a sequence of points with specified times.

    Notes
    -----
    The polynomial trajectories are defined as:

    3rd Order: q(t) = a₀ + a₁τ + a₂τ² + a₃τ³
    5th Order: q(t) = a₀ + a₁τ + a₂τ² + a₃τ³ + a₄τ⁴ + a₅τ⁵
    7th Order: q(t) = a₀ + a₁τ + ... + a₇τ⁷

    Where τ = t - t_start is the normalized time within each segment.

    The coefficients are computed to satisfy the specified boundary conditions:
    - 3rd order requires position and velocity at both endpoints (4 constraints)
    - 5th order requires position, velocity, and acceleration (6 constraints)
    - 7th order requires position, velocity, acceleration, and jerk (8 constraints)

    Examples
    --------
    >>> import numpy as np
    >>> from interpolatepy import PolynomialTrajectory, BoundaryCondition, TimeInterval
    >>>
    >>> # Create a 5th order polynomial trajectory
    >>> initial = BoundaryCondition(position=0, velocity=0, acceleration=0)
    >>> final = BoundaryCondition(position=10, velocity=0, acceleration=0)
    >>> time_interval = TimeInterval(start=0, end=2.0)
    >>>
    >>> trajectory_func = PolynomialTrajectory.order_5_trajectory(
    ...     initial, final, time_interval
    ... )
    >>>
    >>> # Evaluate trajectory at various times
    >>> for t in np.linspace(0, 2, 5):
    ...     pos, vel, acc, jerk = trajectory_func(t)
    ...     print(f"t={t:.1f}: pos={pos:.2f}, vel={vel:.2f}, acc={acc:.2f}")
    >>>
    >>> # Multi-point trajectory example
    >>> from interpolatepy import TrajectoryParams
    >>> params = TrajectoryParams(
    ...     points=[0, 5, 3, 8],
    ...     times=[0, 1, 2, 3],
    ...     order=5
    ... )
    >>> multi_traj = PolynomialTrajectory.multipoint_trajectory(params)
    >>>
    >>> # Evaluate at any time
    >>> pos, vel, acc, jerk = multi_traj(1.5)
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
            Initial boundary conditions (position, velocity).
        final : BoundaryCondition
            Final boundary conditions (position, velocity).
        time : TimeInterval
            Time interval for the trajectory.

        Returns
        -------
        Callable[[float], tuple[float, float, float, float]]
            Function that computes position, velocity, acceleration, and jerk at time t.

        Notes
        -----
        The 3rd order polynomial is defined as:
        q(τ) = a₀ + a₁τ + a₂τ² + a₃τ³

        Where the coefficients are determined by the boundary conditions:
        - q(0) = q₀, q̇(0) = v₀
        - q(T) = q₁, q̇(T) = v₁

        The coefficient formulas (equation 2.2) are:
        - a₀ = q₀
        - a₁ = v₀
        - a₂ = (3h - (2v₀ + v₁)T) / T²
        - a₃ = (-2h + (v₀ + v₁)T) / T³

        Where h = q₁ - q₀ and T = t_end - t_start.

        Examples
        --------
        >>> # Simple point-to-point motion
        >>> initial = BoundaryCondition(position=0, velocity=1)
        >>> final = BoundaryCondition(position=5, velocity=0)
        >>> time_interval = TimeInterval(start=0, end=2.0)
        >>> traj = PolynomialTrajectory.order_3_trajectory(initial, final, time_interval)
        >>>
        >>> # Evaluate at midpoint
        >>> pos, vel, acc, jerk = traj(1.0)
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
            Initial boundary conditions (position, velocity, acceleration).
        final : BoundaryCondition
            Final boundary conditions (position, velocity, acceleration).
        time : TimeInterval
            Time interval for the trajectory.

        Returns
        -------
        Callable[[float], tuple[float, float, float, float]]
            Function that computes position, velocity, acceleration, and jerk at time t.

        Notes
        -----
        The 5th order polynomial provides smooth acceleration profiles and is defined as:
        q(τ) = a₀ + a₁τ + a₂τ² + a₃τ³ + a₄τ⁴ + a₅τ⁵

        The six boundary conditions are:
        - q(0) = q₀, q̇(0) = v₀, q̈(0) = a₀
        - q(T) = q₁, q̇(T) = v₁, q̈(T) = a₁

        The coefficients (equation 2.5) ensure continuous position, velocity, and
        acceleration, making this ideal for applications requiring smooth acceleration
        profiles such as robotic manipulators.

        Examples
        --------
        >>> # Trajectory with zero initial and final accelerations
        >>> initial = BoundaryCondition(position=0, velocity=0, acceleration=0)
        >>> final = BoundaryCondition(position=10, velocity=2, acceleration=0)
        >>> time_interval = TimeInterval(start=0, end=3.0)
        >>> traj = PolynomialTrajectory.order_5_trajectory(initial, final, time_interval)
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
            q = a0 + a1 * tau + a2 * tau**2 + a3 * tau**3 + a4 * tau**4 + a5 * tau**5 + a6 * tau**6 + a7 * tau**7

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
            qdd = 2 * a2 + 6 * a3 * tau + 12 * a4 * tau**2 + 20 * a5 * tau**3 + 30 * a6 * tau**4 + 42 * a7 * tau**5

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
        case the velocity is set to zero. This prevents oscillatory behavior near direction
        changes.

        Parameters
        ----------
        points : list[float]
            List of position points [q₀, q₁, ..., qₙ].
        times : list[float]
            List of time points [t₀, t₁, ..., tₙ].

        Returns
        -------
        list[float]
            List of velocities [v₀, v₁, ..., vₙ].

        Notes
        -----
        The heuristic rule is defined as:

        For intermediate points i = 1, 2, ..., n-1:
        - If sign(sᵢ₋₁) ≠ sign(sᵢ): vᵢ = 0
        - Otherwise: vᵢ = (sᵢ₋₁ + sᵢ) / 2

        Where sᵢ = (qᵢ₊₁ - qᵢ) / (tᵢ₊₁ - tᵢ) is the slope of segment i.

        The boundary velocities v₀ and vₙ are set to zero by default.

        Examples
        --------
        >>> points = [0, 2, 1, 3]
        >>> times = [0, 1, 2, 3]
        >>> velocities = PolynomialTrajectory.heuristic_velocities(points, times)
        >>> print(velocities)  # [0.0, 2.0, 0.0, 0.0]
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
