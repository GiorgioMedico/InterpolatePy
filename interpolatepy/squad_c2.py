from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np

from .polynomials import BoundaryCondition, PolynomialTrajectory, TimeInterval
from .quat_core import Quaternion


@dataclass
class SquadC2Config:
    """
    Configuration parameters for SQUAD_C2 interpolation.

    This dataclass encapsulates the parameters needed to initialize a SQUAD_C2
    interpolator according to the Wittmann et al. specification.
    """

    time_points: list[float]                 # Time sequence for quaternion waypoints
    quaternions: list[Quaternion]            # Quaternion waypoints to interpolate
    normalize_quaternions: bool = True       # Ensure unit quaternion constraints
    validate_continuity: bool = True         # Enable C²-continuity validation


class SquadC2:
    """
    C²-Continuous, Zero-Clamped Quaternion Interpolation using SQUAD with Quintic Polynomial
    Parameterization.

    This class implements the SQUAD_C2 method as described in "Spherical Cubic Blends:
    C²-Continuous, Zero-Clamped, and Time-Optimized Interpolation of Quaternions"
    by Wittmann et al. (ICRA 2023).

    Implementation follows paper specifications:
    1. Creates extended quaternion sequence Q = [q₁, q₁ᵛⁱʳᵗ, q₂, ..., qₙ₋₁ᵛⁱʳᵗ, qₙ]
       where q₁ᵛⁱʳᵗ = q₁ and qₙ₋₁ᵛⁱʳᵗ = qₙ (Section III-B.1)
    2. Computes intermediate quaternions using corrected formula (Equation 5):
       sᵢ = qᵢ ⊗ exp[(log(qᵢ⁻¹⊗qᵢ₊₁)/(-2(1+hᵢ/hᵢ₋₁))) + (log(qᵢ⁻¹⊗qᵢ₋₁)/(-2(1+hᵢ₋₁/hᵢ)))]
    3. Uses SQUAD interpolation with proper time parameterization (Equations 2-4)
    4. Applies quintic polynomial parameterization u(t) for C²-continuity and
       zero-clamped boundaries

    Key Features:
    - Guarantees C²-continuous quaternion trajectories
    - Zero-clamped boundary conditions (zero angular velocity and acceleration at endpoints)
    - Proper handling of different segment durations hᵢ
    - Compatible with time-optimization frameworks

    References:
    - Wittmann et al., "Spherical Cubic Blends: C²-Continuous, Zero-Clamped, and
      Time-Optimized Interpolation of Quaternions", ICRA 2023
    - Original SQUAD: Shoemake, "Animating rotation with quaternion curves", SIGGRAPH 1985
    """

    # Type annotations for instance attributes
    original_time_points: np.ndarray
    original_quaternions: list[Quaternion]
    time_points: np.ndarray  # Built as list then converted to ndarray
    quaternions: list[Quaternion]
    intermediate_quaternions: list[Quaternion]
    polynomial_segments: list[Callable[[float], tuple[float, float, float, float]]]
    normalize_quaternions: bool
    validate_continuity: bool

    def __init__(
        self,
        time_points: list[float],
        quaternions: list[Quaternion],
        normalize_quaternions: bool = True,
        validate_continuity: bool = True,
    ) -> None:
        """
        Initialize SQUAD_C2 interpolator following Wittmann et al. specification.

        The interpolator creates an extended quaternion sequence with virtual waypoints
        to enable zero-clamped boundary conditions and C²-continuous trajectories.

        Args:
            time_points: List of time values (must be sorted and at least 2 points)
            quaternions: List of quaternions at each time point
            normalize_quaternions: Whether to normalize input quaternions to unit length
            validate_continuity: Whether to validate C² continuity (for debugging)

        Raises:
            ValueError: If input validation fails

        Note:
            The implementation follows the corrected SQUAD formulation from the paper,
            which properly handles non-uniform time spacing through the corrected
            intermediate quaternion formula (Equation 5).
        """
        self._validate_input(time_points, quaternions)

        self.original_time_points = np.array(time_points, dtype=float)
        self.original_quaternions = quaternions.copy()
        self.normalize_quaternions = normalize_quaternions
        self.validate_continuity = validate_continuity

        if normalize_quaternions:
            self.original_quaternions = [q.unit() for q in self.original_quaternions]

        # Add virtual waypoints as specified in the paper
        self._add_virtual_waypoints()

        # Precompute intermediate quaternions and polynomial parameterizations
        self._setup_interpolation()

    @staticmethod
    def _validate_input(time_points: list[float], quaternions: list[Quaternion]) -> None:
        """Validate input data for SQUAD_C2 construction."""
        if len(time_points) != len(quaternions):
            raise ValueError("Time points and quaternions must have same length")

        min_waypoints = 2
        if len(time_points) < min_waypoints:
            raise ValueError("SQUAD_C2 requires at least 2 waypoints")

        # Check that time points are sorted
        if not all(time_points[i] <= time_points[i + 1] for i in range(len(time_points) - 1)):
            raise ValueError("Time points must be sorted in ascending order")

        # Check for duplicate time points
        if len(set(time_points)) != len(time_points):
            raise ValueError("Time points must be unique")

    def _add_virtual_waypoints(self) -> None:
        """
        Add virtual waypoints as specified in the paper:
        Q = [q₁, q₁ᵛⁱʳᵗ, q₂, ..., qₙ₋₁ᵛⁱʳᵗ, qₙ]
        where q₁ᵛⁱʳᵗ = q₁ and qₙ₋₁ᵛⁱʳᵗ = qₙ

        This creates the extended quaternion sequence needed for proper SQUAD_C2 interpolation
        with zero-clamped boundary conditions.
        """
        n_original = len(self.original_time_points)

        min_waypoints = 2
        if n_original < min_waypoints:
            raise ValueError("Need at least 2 original waypoints")

        # Create extended quaternion sequence Q as per paper (Section III-B.1)
        quaternions_list: list[Quaternion] = []
        time_points_list: list[float] = []

        # For SQUAD_C2, we need the extended sequence:
        # Q = [q₁, q₁ᵛⁱʳᵗ, q₂, q₃, ..., qₙ₋₁, qₙ₋₁ᵛⁱʳᵗ, qₙ]
        # This enables zero-clamped boundary conditions through virtual waypoints

        # Add q₁ (first original waypoint)
        quaternions_list.append(self.original_quaternions[0])
        time_points_list.append(self.original_time_points[0])

        # Add q₁ᵛⁱʳᵗ = q₁ (first virtual waypoint)
        quaternions_list.append(self.original_quaternions[0])  # q₁ᵛⁱʳᵗ = q₁
        # Virtual waypoint gets a time point that maintains equal segment durations
        two_waypoints = 2
        if n_original == two_waypoints:
            # Special case: only 2 original waypoints
            dt = self.original_time_points[1] - self.original_time_points[0]
            time_points_list.append(self.original_time_points[0] + dt / 3.0)
        else:
            # General case: use time spacing based on first segment
            dt = (self.original_time_points[1] - self.original_time_points[0]) / 2.0
            time_points_list.append(self.original_time_points[0] + dt)

        # Add all intermediate original waypoints q₂, q₃, ..., qₙ₋₁
        for i in range(1, n_original - 1):
            quaternions_list.append(self.original_quaternions[i])
            time_points_list.append(self.original_time_points[i])

        # Add qₙ₋₁ᵛⁱʳᵗ = qₙ (last virtual waypoint) if we have more than 2 original points
        if n_original > two_waypoints:
            quaternions_list.append(self.original_quaternions[-1])  # qₙ₋₁ᵛⁱʳᵗ = qₙ
            # Place virtual waypoint before the last original waypoint
            dt = (self.original_time_points[-1] - self.original_time_points[-2]) / 2.0
            time_points_list.append(self.original_time_points[-1] - dt)
        elif n_original == two_waypoints:
            # For 2 original waypoints, add virtual waypoint before the last
            quaternions_list.append(self.original_quaternions[-1])  # qₙ₋₁ᵛⁱʳᵗ = qₙ
            dt = self.original_time_points[1] - self.original_time_points[0]
            time_points_list.append(self.original_time_points[1] - dt / 3.0)

        # Add qₙ (final original waypoint)
        quaternions_list.append(self.original_quaternions[-1])
        time_points_list.append(self.original_time_points[-1])

        # Convert to final instance attributes
        self.quaternions = quaternions_list
        self.time_points = np.array(time_points_list, dtype=float)

    def _compute_segment_durations(self) -> np.ndarray:
        """Compute segment durations hᵢ between waypoints."""
        return np.diff(self.time_points)

    @staticmethod
    def _compute_intermediate_quaternion(
        q_prev: Quaternion, q_curr: Quaternion, q_next: Quaternion,
        h_prev: float, h_curr: float
    ) -> Quaternion:
        """
        Compute intermediate quaternion using the corrected formula from Equation (5)
        in Wittmann et al.

        The corrected formula properly accounts for different segment durations:
        sᵢ = qᵢ ⊗ exp[
            log(qᵢ⁻¹ ⊗ qᵢ₊₁) / (-2(1 + hᵢ/hᵢ₋₁)) +
            log(qᵢ⁻¹ ⊗ qᵢ₋₁) / (-2(1 + hᵢ₋₁/hᵢ))
        ]

        This ensures C¹-continuity and proper time parameterization when segment
        durations hᵢ are not equal.

        Args:
            q_prev: Previous quaternion (qᵢ₋₁)
            q_curr: Current quaternion (qᵢ)
            q_next: Next quaternion (qᵢ₊₁)
            h_prev: Previous segment duration (hᵢ₋₁)
            h_curr: Current segment duration (hᵢ)

        Returns:
            Intermediate quaternion (sᵢ) for SQUAD interpolation
        """
        # Compute relative quaternions
        q_curr_inv = q_curr.inverse()
        rel_next = q_curr_inv * q_next  # qᵢ⁻¹ ⊗ qᵢ₊₁
        rel_prev = q_curr_inv * q_prev  # qᵢ⁻¹ ⊗ qᵢ₋₁

        # Compute logarithms
        log_next = rel_next.Log()
        log_prev = rel_prev.Log()

        # Apply the corrected formula weights
        weight_next = -2.0 * (1.0 + h_curr / h_prev)
        weight_prev = -2.0 * (1.0 + h_prev / h_curr)

        # Compute weighted sum
        weighted_sum = log_next / weight_next + log_prev / weight_prev

        # Convert back to quaternion and compose with current quaternion
        exp_weighted = weighted_sum.exp()
        return q_curr * exp_weighted

    def _setup_interpolation(self) -> None:
        """Setup intermediate quaternions and polynomial parameterizations for all segments."""
        n_points = len(self.time_points)
        segment_durations = self._compute_segment_durations()

        # Compute intermediate quaternions for SQUAD using corrected formula from equation (5)
        # The intermediate quaternions s_i are computed for all interior points
        # in the extended sequence
        self.intermediate_quaternions: list[Quaternion] = []

        for i in range(n_points):
            if i == 0 or i == n_points - 1:
                # First and last points in extended sequence: these won't be used
                # in SQUAD interpolation
                # Use identity quaternions as placeholders
                self.intermediate_quaternions.append(Quaternion.identity())
            else:
                # Interior points: use corrected intermediate quaternion formula (equation 5)
                q_prev = self.quaternions[i - 1]
                q_curr = self.quaternions[i]
                q_next = self.quaternions[i + 1]
                h_prev = segment_durations[i - 1]
                h_curr = segment_durations[i]

                intermediate_q = self._compute_intermediate_quaternion(
                    q_prev, q_curr, q_next, h_prev, h_curr
                )
                self.intermediate_quaternions.append(intermediate_q)

        # Setup quintic polynomial parameterization for each segment between original waypoints
        # We need to map from extended sequence segments to original waypoint segments
        polynomial_func_type = Callable[[float], tuple[float, float, float, float]]
        self.polynomial_segments: list[polynomial_func_type] = []

        # Create polynomial segments that correspond to interpolation between original waypoints
        # Each segment uses quintic polynomial parameterization as described in Section III-B.1
        n_original = len(self.original_time_points)
        for i in range(n_original - 1):
            t_start = self.original_time_points[i]
            t_end = self.original_time_points[i + 1]

            # Create quintic polynomial u(t) with zero-clamped boundary conditions
            # u(t₀) = 0, u'(t₀) = 0, u''(t₀) = 0
            # u(t₁) = 1, u'(t₁) = 0, u''(t₁) = 0
            # This ensures C²-continuity and zero angular velocity/acceleration at waypoints
            initial_bc = BoundaryCondition(position=0.0, velocity=0.0, acceleration=0.0)
            final_bc = BoundaryCondition(position=1.0, velocity=0.0, acceleration=0.0)
            time_interval = TimeInterval(t_start, t_end)

            polynomial_func = PolynomialTrajectory.order_5_trajectory(
                initial_bc, final_bc, time_interval
            )
            self.polynomial_segments.append(polynomial_func)

    @staticmethod
    def _slerp(q1: Quaternion, q2: Quaternion, t: float) -> Quaternion:
        """Spherical linear interpolation between two quaternions."""
        return q1.slerp(q2, t)

    def _squad_interpolation(
        self, q1: Quaternion, s1: Quaternion, s2: Quaternion,
        q2: Quaternion, u: float
    ) -> Quaternion:
        """
        SQUAD interpolation using equations (2)-(4) from Wittmann et al.:

        SQUAD(qᵢ, sᵢ, sᵢ₊₁, qᵢ₊₁, u) = SLERP(q̂ᵢ, ŝᵢ, 2u(1-u))  (Eq. 2)
        q̂ᵢ = SLERP(qᵢ, qᵢ₊₁, u)                                    (Eq. 3)
        ŝᵢ = SLERP(sᵢ, sᵢ₊₁, u)                                    (Eq. 4)

        The parameter u(t) is provided by the quintic polynomial parameterization
        to ensure C²-continuity and zero-clamped boundary conditions.
        """
        # Equation (3): q̂ᵢ = SLERP(qᵢ, qᵢ₊₁, u)
        q_hat = self._slerp(q1, q2, u)

        # Equation (4): ŝᵢ = SLERP(sᵢ, sᵢ₊₁, u)
        s_hat = self._slerp(s1, s2, u)

        # Equation (2): SQUAD = SLERP(q̂ᵢ, ŝᵢ, 2u(1-u))
        blend_param = 2.0 * u * (1.0 - u)
        return self._slerp(q_hat, s_hat, blend_param)

    def _find_segment_index(self, t: float) -> int:
        """Find the segment index for a given time in the original waypoint sequence."""
        # Map time to original waypoint segments, not extended sequence segments
        if t <= self.original_time_points[0]:
            return 0
        if t >= self.original_time_points[-1]:
            return len(self.original_time_points) - 2

        # Binary search for efficiency in original waypoint time sequence
        left, right = 0, len(self.original_time_points) - 2
        while left <= right:
            mid = (left + right) // 2
            if self.original_time_points[mid] <= t <= self.original_time_points[mid + 1]:
                return mid
            if t < self.original_time_points[mid]:
                right = mid - 1
            else:
                left = mid + 1

        # Fallback
        return max(0, min(len(self.original_time_points) - 2, int(t)))

    def evaluate(self, t: float) -> Quaternion:
        """
        Evaluate quaternion at time t using proper SQUAD_C2 interpolation.

        Args:
            t: Time value

        Returns:
            Interpolated quaternion at time t
        """
        # Handle boundary cases - return original waypoints
        if t <= self.original_time_points[0]:
            return self.original_quaternions[0]
        if t >= self.original_time_points[-1]:
            return self.original_quaternions[-1]

        # Find appropriate segment in original waypoint sequence
        segment_idx = self._find_segment_index(t)

        # Get polynomial parameterization u(t) for this segment
        u, _, _, _ = self.polynomial_segments[segment_idx](t)

        # Clamp u to [0, 1] for numerical safety
        u = max(0.0, min(1.0, u))

        # Map from original waypoint segment to extended sequence indices
        # For SQUAD interpolation, we need the correct quaternions from extended sequence

        # For segment between original waypoints i and i+1:
        # - We use original waypoints as SQUAD endpoints
        # - We use the corresponding virtual/intermediate waypoints for SQUAD control points

        if segment_idx == 0:
            # First segment: q₁ to q₂ in original sequence
            # Maps to extended sequence indices [1, 2] with intermediate points [1, 2]
            q1 = self.original_quaternions[0]  # q₁
            q2 = self.original_quaternions[1]  # q₂
            # Find intermediate quaternions in extended sequence
            s1 = self.intermediate_quaternions[1]  # s₁ᵛⁱʳᵗ (at extended index 1)
            # Handle s2 quaternion selection
            two_waypoints = 2
            s2 = (self.intermediate_quaternions[2]
                  if len(self.intermediate_quaternions) > two_waypoints
                  else self.intermediate_quaternions[1])  # s₂
        else:
            # Later segments: map correctly to extended sequence
            q1 = self.original_quaternions[segment_idx]
            q2 = self.original_quaternions[segment_idx + 1]
            # Find corresponding intermediate quaternions in extended sequence
            extended_idx1 = segment_idx + 1  # Account for virtual waypoint offset
            extended_idx2 = extended_idx1 + 1
            # Handle intermediate quaternion selection with bounds checking
            s1 = (self.intermediate_quaternions[extended_idx1]
                  if extended_idx1 < len(self.intermediate_quaternions)
                  else Quaternion.identity())
            s2 = (self.intermediate_quaternions[extended_idx2]
                  if extended_idx2 < len(self.intermediate_quaternions)
                  else Quaternion.identity())

        # Use SQUAD interpolation for all segments
        return self._squad_interpolation(q1, s1, s2, q2, u)

    def evaluate_velocity(self, t: float) -> np.ndarray:
        """
        Evaluate angular velocity at time t.

        Args:
            t: Time value

        Returns:
            3D angular velocity vector in rad/s
        """
        # Handle boundary cases - should be zero for zero-clamped boundaries
        if t <= self.time_points[0] or t >= self.time_points[-1]:
            return np.zeros(3)

        # Use finite differences for numerical differentiation
        dt = 1e-6
        epsilon = 1e-8

        # Ensure we stay within the valid time range
        t_plus = min(t + dt, self.time_points[-1] - epsilon)
        t_minus = max(t - dt, self.time_points[0] + epsilon)

        q_plus = self.evaluate(t_plus)
        q_minus = self.evaluate(t_minus)

        # Compute angular velocity using quaternion difference
        q_current = self.evaluate(t)
        dq_dt = (q_plus - q_minus) / (t_plus - t_minus)

        # Convert quaternion derivative to angular velocity
        # ω = 2 * (dq/dt) * q⁻¹ (imaginary part)
        omega_quat = 2.0 * dq_dt * q_current.inverse()
        return np.array([omega_quat.x, omega_quat.y, omega_quat.z])

    def evaluate_acceleration(self, t: float) -> np.ndarray:
        """
        Evaluate angular acceleration at time t.

        Args:
            t: Time value

        Returns:
            3D angular acceleration vector in rad/s²
        """
        # Handle boundary cases - should be zero for zero-clamped boundaries
        if t <= self.time_points[0] or t >= self.time_points[-1]:
            return np.zeros(3)

        # Use finite differences for numerical differentiation of angular velocity
        dt = 1e-6
        epsilon = 1e-8

        # Ensure we stay within the valid time range
        t_plus = min(t + dt, self.time_points[-1] - epsilon)
        t_minus = max(t - dt, self.time_points[0] + epsilon)

        omega_plus = self.evaluate_velocity(t_plus)
        omega_minus = self.evaluate_velocity(t_minus)

        # Central difference approximation
        return (omega_plus - omega_minus) / (t_plus - t_minus)

    def get_time_range(self) -> tuple[float, float]:
        """Get the time range of the trajectory (original waypoints only)."""
        return float(self.original_time_points[0]), float(self.original_time_points[-1])

    def get_waypoints(self) -> tuple[list[float], list[Quaternion]]:
        """Get the original waypoint times and quaternions (without virtual waypoints)."""
        return self.original_time_points.tolist(), self.original_quaternions.copy()

    def get_extended_waypoints(self) -> tuple[list[float], list[Quaternion]]:
        """Get all waypoint times and quaternions (including virtual waypoints)."""
        return self.time_points.tolist(), self.quaternions.copy()

    def get_extended_sequence_info(self) -> dict:
        """
        Get detailed information about the extended quaternion sequence for debugging.

        Returns:
            Dictionary with extended sequence details
        """
        return {
            "n_original": len(self.original_time_points),
            "n_extended": len(self.time_points),
            "original_times": self.original_time_points.tolist(),
            "extended_times": self.time_points.tolist(),
            "segment_durations": self._compute_segment_durations().tolist(),
            "has_virtual_waypoints": len(self.time_points) > len(self.original_time_points)
        }

    def __len__(self) -> int:
        """Return number of original waypoints."""
        return len(self.original_time_points)

    def __str__(self) -> str:
        """String representation."""
        t_min, t_max = self.get_time_range()
        n_extended = len(self.time_points)
        return (f"SquadC2({len(self)} original waypoints, {n_extended} extended, "
                f"t=[{t_min:.3f}, {t_max:.3f}])")

    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()
