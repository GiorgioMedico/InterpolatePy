from collections import OrderedDict

import numpy as np

from .quat_core import Quaternion

# Constants
INTERPOLATION_OUT_OF_RANGE_ERROR = -3


class QuaternionSpline:
    """
    Quaternion spline interpolator for smooth trajectory planning.

    Supports both SLERP and SQUAD interpolation methods with automatic
    method selection based on the number of control points.

    Example usage:
        # Create quaternions
        q1 = Quaternion.identity()
        q2 = Quaternion.from_euler_angles(0.1, 0.2, 0.3)
        q3 = Quaternion.from_euler_angles(0.2, 0.4, 0.6)
        q4 = Quaternion.identity()

        # Create spline with SLERP interpolation
        spline_slerp = QuaternionSpline([0, 1, 2, 3], [q1, q2, q3, q4], Quaternion.SLERP)

        # Create spline with SQUAD interpolation
        spline_squad = QuaternionSpline([0, 1, 2, 3], [q1, q2, q3, q4], Quaternion.SQUAD)

        # Interpolate at specific time
        result, status = spline_slerp.interpolate_at_time(1.5)

        # Change interpolation method
        spline_slerp.set_interpolation_method(Quaternion.SQUAD)

        # Force specific interpolation regardless of setting
        slerp_result, _ = spline_squad.interpolate_slerp(1.5)
        squad_result, _ = spline_slerp.interpolate_squad(1.5)
    """

    def __init__(
        self,
        time_points: list[float],
        quaternions: list[Quaternion],
        interpolation_method: str = Quaternion.AUTO,
    ) -> None:
        """
        Initialize quaternion spline interpolator.

        Args:
            time_points: List of time values (must be sorted)
            quaternions: List of quaternions at each time point
            interpolation_method: Interpolation method - "slerp", "squad", or "auto"
        """
        if interpolation_method not in {Quaternion.SLERP, Quaternion.SQUAD, Quaternion.AUTO}:
            raise ValueError(f"Invalid interpolation method: {interpolation_method}")

        self._setup_spline(time_points, quaternions, interpolation_method)

    def _setup_spline(
        self,
        time_points: list[float],
        quaternions: list[Quaternion],
        interpolation_method: str = Quaternion.AUTO,
    ) -> None:
        """
        Setup this quaternion as a spline interpolator.

        Args:
            time_points: List of time values (must be sorted)
            quaternions: List of quaternions at each time point
            interpolation_method: Interpolation method to use
        """
        self._validate_input_data(time_points, quaternions)

        # Store interpolation method
        self.interpolation_method = interpolation_method

        # Create ordered dictionary
        sorted_data = sorted(zip(time_points, quaternions))
        self.quat_data = OrderedDict(sorted_data)

        # Only precompute intermediate quaternions if using Squad interpolation
        if interpolation_method in {Quaternion.SQUAD, Quaternion.AUTO}:
            self._compute_intermediate_quaternions()
        else:
            self.intermediate_quaternions = {}

    @staticmethod
    def _validate_input_data(time_points: list[float], quaternions: list[Quaternion]) -> None:
        """Validate input data for spline construction."""
        if len(time_points) != len(quaternions):
            raise ValueError("Time points and quaternions must have same length")
        if len(time_points) < Quaternion.MIN_INTERPOLATION_POINTS:
            raise ValueError("Need at least 2 points for interpolation")

    def _compute_intermediate_quaternions(self) -> None:
        """Precompute intermediate quaternions for smooth Squad interpolation"""
        times = list(self.quat_data.keys())
        self.intermediate_quaternions = {}

        if len(times) < Quaternion.MIN_SQUAD_POINTS:
            # Not enough points for Squad, will use Slerp
            return

        for i in range(1, len(times) - 1):
            t_prev, t_curr, t_next = times[i - 1], times[i], times[i + 1]
            q_prev = self.quat_data[t_prev]
            q_curr = self.quat_data[t_curr]
            q_next = self.quat_data[t_next]

            self.intermediate_quaternions[t_curr] = Quaternion.compute_intermediate_quaternion(
                q_prev, q_curr, q_next
            )

    def interpolate_at_time(self, t: float) -> tuple[Quaternion, int]:  # noqa: PLR0911
        """
        Quaternion interpolation at given time.

        Returns: (interpolated_quaternion, status_code)
        """
        if not self.quat_data:
            return Quaternion.identity(), -1

        times = list(self.quat_data.keys())

        if t <= times[0]:
            return self.quat_data[times[0]], 0
        if t >= times[-1]:
            return self.quat_data[times[-1]], 0

        # Find surrounding time points
        for i in range(len(times) - 1):
            if times[i] <= t <= times[i + 1]:
                t0, t1 = times[i], times[i + 1]
                q0, q1 = self.quat_data[t0], self.quat_data[t1]

                # Normalized parameter
                dt = (t - t0) / (t1 - t0)

                # Choose interpolation method based on user preference
                if self.interpolation_method == Quaternion.SLERP:
                    return Quaternion.Slerp(q0, q1, dt), 0
                if self.interpolation_method == Quaternion.SQUAD:
                    # Check if we have enough points for Squad
                    if len(times) < Quaternion.MIN_SQUAD_POINTS:
                        print(
                            "Warning: Not enough points for SQUAD interpolation, "
                            "falling back to SLERP"
                        )
                        return Quaternion.Slerp(q0, q1, dt), 0

                    # For boundary segments in SQUAD, we still need special handling
                    if i == 0 or i == len(times) - 2:
                        # Use Slerp for first and last segments in SQUAD mode
                        return Quaternion.Slerp(q0, q1, dt), 0
                    # Squad interpolation for interior segments
                    a = self.intermediate_quaternions[t0]
                    b = self.intermediate_quaternions[t1]
                    return Quaternion.Squad(q0, a, b, q1, dt), 0
                # AUTO mode
                # Use original logic for automatic selection
                if i == 0 or i == len(times) - 2 or len(times) < Quaternion.MIN_SQUAD_POINTS:
                    return Quaternion.Slerp(q0, q1, dt), 0
                # Squad interpolation
                a = self.intermediate_quaternions[t0]
                b = self.intermediate_quaternions[t1]
                return Quaternion.Squad(q0, a, b, q1, dt), 0

        print("QuaternionSpline::interpolate_at_time: t not in range.")
        return Quaternion.identity(), -3  # NOT_IN_RANGE

    def interpolate_with_velocity(self, t: float) -> tuple[Quaternion, np.ndarray, int]:
        """
        Quaternion interpolation with angular velocity.

        Returns: (interpolated_quaternion, angular_velocity, status_code)
        """
        q, status = self.interpolate_at_time(t)
        if status != 0:
            return q, np.zeros(3), status

        # Compute derivative using finite differences
        dt = 1e-6
        times = list(self.quat_data.keys())

        if t + dt <= times[-1]:
            q_next, _ = self.interpolate_at_time(t + dt)
            dq = (q_next - q) * (1.0 / dt)
        elif t - dt >= times[0]:
            q_prev, _ = self.interpolate_at_time(t - dt)
            dq = (q - q_prev) * (1.0 / dt)
        else:
            return q, np.zeros(3), 0

        # Convert to angular velocity
        w = Quaternion.Omega(q, dq)

        return q, w, 0

    def get_time_range(self) -> tuple[float, float]:
        """Get the time range of the spline"""
        if not self.quat_data:
            return 0.0, 0.0
        times = list(self.quat_data.keys())
        return times[0], times[-1]

    def is_empty(self) -> bool:
        """Check if this spline has no data"""
        return len(self.quat_data) == 0

    def set_interpolation_method(self, method: str) -> None:
        """
        Set the interpolation method for this spline.

        Args:
            method: "slerp", "squad", or "auto"
        """
        if method not in {Quaternion.SLERP, Quaternion.SQUAD, Quaternion.AUTO}:
            raise ValueError(f"Invalid interpolation method: {method}")

        old_method = self.interpolation_method
        self.interpolation_method = method

        # Recompute intermediate quaternions if switching to/from Squad methods
        if method in {Quaternion.SQUAD, Quaternion.AUTO} and old_method not in {
            Quaternion.SQUAD,
            Quaternion.AUTO,
        }:
            self._compute_intermediate_quaternions()
        elif method == Quaternion.SLERP and old_method in {Quaternion.SQUAD, Quaternion.AUTO}:
            self.intermediate_quaternions = {}

    def get_interpolation_method(self) -> str:
        """Get the current interpolation method"""
        return self.interpolation_method

    def interpolate_slerp(self, t: float) -> tuple[Quaternion, int]:
        """
        Force SLERP interpolation at given time, regardless of current method setting.

        Returns: (interpolated_quaternion, status_code)
        """
        if not self.quat_data:
            return Quaternion.identity(), -1

        times = list(self.quat_data.keys())

        if t <= times[0]:
            return self.quat_data[times[0]], 0
        if t >= times[-1]:
            return self.quat_data[times[-1]], 0

        # Find surrounding time points
        for i in range(len(times) - 1):
            if times[i] <= t <= times[i + 1]:
                t0, t1 = times[i], times[i + 1]
                q0, q1 = self.quat_data[t0], self.quat_data[t1]

                # Normalized parameter
                dt = (t - t0) / (t1 - t0)
                return Quaternion.Slerp(q0, q1, dt), 0

        print("QuaternionSpline::interpolate_slerp: t not in range.")
        return Quaternion.identity(), -3

    def interpolate_squad(self, t: float) -> tuple[Quaternion, int]:  # noqa: PLR0911
        """
        Force SQUAD interpolation at given time, regardless of current method setting.

        Returns: (interpolated_quaternion, status_code)
        """
        if not self.quat_data:
            return Quaternion.identity(), -1

        times = list(self.quat_data.keys())

        if len(times) < Quaternion.MIN_SQUAD_POINTS:
            print("Error: Not enough points for SQUAD interpolation (need at least 4)")
            return Quaternion.identity(), -2

        if t <= times[0]:
            return self.quat_data[times[0]], 0
        if t >= times[-1]:
            return self.quat_data[times[-1]], 0

        # Find surrounding time points
        for i in range(len(times) - 1):
            if times[i] <= t <= times[i + 1]:
                t0, t1 = times[i], times[i + 1]
                q0, q1 = self.quat_data[t0], self.quat_data[t1]

                # Normalized parameter
                dt = (t - t0) / (t1 - t0)

                # For boundary segments, use linear blending to boundary quaternions
                if i == 0 or i == len(times) - 2:
                    return Quaternion.Slerp(q0, q1, dt), 0
                # Squad interpolation for interior segments
                a = self.intermediate_quaternions[t0]
                b = self.intermediate_quaternions[t1]
                return Quaternion.Squad(q0, a, b, q1, dt), 0

        print("QuaternionSpline::interpolate_squad: t not in range.")
        return Quaternion.identity(), -3

    def get_quaternion_at_time(self, t: float) -> Quaternion:
        """
        Get quaternion at specific time, raising exception on error.

        Returns: Interpolated quaternion
        Raises: ValueError if time is out of range or spline is empty
        """
        result, status = self.interpolate_at_time(t)
        if status != 0:
            if status == -1:
                raise ValueError("Spline is empty")
            if status == INTERPOLATION_OUT_OF_RANGE_ERROR:
                raise ValueError(f"Time {t} is out of range")
            raise ValueError(f"Interpolation failed with status {status}")
        return result

    def get_time_points(self) -> list[float]:
        """Get all time points in the spline"""
        return list(self.quat_data.keys())

    def get_quaternions(self) -> list[Quaternion]:
        """Get all quaternions in the spline"""
        return list(self.quat_data.values())

    def __len__(self) -> int:
        """Return number of quaternion waypoints"""
        return len(self.quat_data)

    def __str__(self) -> str:
        """String representation"""
        method = self.interpolation_method
        count = len(self.quat_data)
        if count == 0:
            return f"QuaternionSpline(empty, method={method})"

        t_min, t_max = self.get_time_range()
        return f"QuaternionSpline({count} points, t=[{t_min:.3f}, {t_max:.3f}], method={method})"

    def __repr__(self) -> str:
        """Detailed string representation"""
        return self.__str__()
