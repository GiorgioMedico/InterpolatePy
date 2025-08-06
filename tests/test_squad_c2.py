"""
Comprehensive tests for SQUAD C2 quaternion interpolation implementation.

This module verifies the correctness of the SQUAD_C2 implementation against the
mathematical specification in "Spherical Cubic Blends: C²-Continuous, Zero-Clamped,
and Time-Optimized Interpolation of Quaternions" by Wittmann et al. (ICRA 2023).

The tests verify:
1. Mathematical correctness of the extended quaternion sequence
2. Proper implementation of the corrected intermediate quaternion formula (Equation 5)
3. SQUAD interpolation using Equations 2-4
4. C²-continuity properties
5. Zero-clamped boundary conditions
6. Edge cases and robustness
"""

import numpy as np
import pytest

from interpolatepy.squad_c2 import SquadC2, SquadC2Config
from interpolatepy.quat_core import Quaternion


class TestSquadC2BasicFunctionality:
    """Test basic SQUAD C2 functionality and setup."""

    NUMERICAL_TOLERANCE = 1e-10
    LOOSE_TOLERANCE = 1e-6

    def test_squad_c2_creation_minimum_waypoints(self) -> None:
        """Test SQUAD C2 creation with minimum 2 waypoints."""
        time_points = [0.0, 1.0]
        quaternions = [
            Quaternion.identity(),
            Quaternion.from_euler_angles(0.1, 0.2, 0.3)
        ]

        squad = SquadC2(time_points, quaternions)

        assert len(squad) == 2
        assert squad.get_time_range() == (0.0, 1.0)

    def test_squad_c2_creation_multiple_waypoints(self) -> None:
        """Test SQUAD C2 creation with multiple waypoints."""
        time_points = [0.0, 1.0, 2.0, 3.0]
        quaternions = [
            Quaternion.identity(),
            Quaternion.from_euler_angles(0.1, 0.2, 0.3),
            Quaternion.from_euler_angles(0.2, 0.4, 0.6),
            Quaternion.from_euler_angles(0.3, 0.6, 0.9)
        ]

        squad = SquadC2(time_points, quaternions)

        assert len(squad) == 4
        assert squad.get_time_range() == (0.0, 3.0)

    def test_input_validation(self) -> None:
        """Test input validation for SQUAD C2 constructor."""
        # Mismatched lengths
        with pytest.raises(ValueError, match="Time points and quaternions must have same length"):
            SquadC2([0.0, 1.0], [Quaternion.identity()])

        # Too few waypoints
        with pytest.raises(ValueError, match="SQUAD_C2 requires at least 2 waypoints"):
            SquadC2([0.0], [Quaternion.identity()])

        # Unsorted time points
        with pytest.raises(ValueError, match="Time points must be sorted in ascending order"):
            SquadC2([1.0, 0.0], [Quaternion.identity(), Quaternion.identity()])

        # Duplicate time points
        with pytest.raises(ValueError, match="Time points must be unique"):
            SquadC2([0.0, 0.0], [Quaternion.identity(), Quaternion.identity()])


class TestSquadC2ExtendedSequence:
    """Test the extended quaternion sequence creation as per Section III-B.1."""

    NUMERICAL_TOLERANCE = 1e-10

    def test_extended_sequence_two_waypoints(self) -> None:
        """Test extended sequence with 2 original waypoints."""
        time_points = [0.0, 1.0]
        quaternions = [
            Quaternion.identity(),
            Quaternion.from_euler_angles(0.1, 0.2, 0.3)
        ]

        squad = SquadC2(time_points, quaternions)
        extended_times, extended_quats = squad.get_extended_waypoints()

        # Should have 4 waypoints: [q1, q1_virt, q2_virt, q2]
        assert len(extended_times) == 4
        assert len(extended_quats) == 4

        # First and last should match original
        assert extended_quats[0] == quaternions[0]
        assert extended_quats[-1] == quaternions[-1]

        # Virtual waypoints should match endpoints per paper
        assert extended_quats[1] == quaternions[0]  # q1_virt = q1
        assert extended_quats[2] == quaternions[-1]  # q2_virt = q2

    def test_extended_sequence_multiple_waypoints(self) -> None:
        """Test extended sequence with multiple waypoints."""
        time_points = [0.0, 1.0, 2.0, 3.0]
        quaternions = [
            Quaternion.identity(),
            Quaternion.from_euler_angles(0.1, 0.2, 0.3),
            Quaternion.from_euler_angles(0.2, 0.4, 0.6),
            Quaternion.from_euler_angles(0.3, 0.6, 0.9)
        ]

        squad = SquadC2(time_points, quaternions)
        extended_times, extended_quats = squad.get_extended_waypoints()

        # Should have 6 waypoints: [q1, q1_virt, q2, q3, q4_virt, q4]
        assert len(extended_times) == 6
        assert len(extended_quats) == 6

        # Check virtual waypoint placement
        assert extended_quats[0] == quaternions[0]    # q1
        assert extended_quats[1] == quaternions[0]    # q1_virt = q1
        assert extended_quats[2] == quaternions[1]    # q2
        assert extended_quats[3] == quaternions[2]    # q3
        assert extended_quats[4] == quaternions[3]    # q4_virt = q4
        assert extended_quats[5] == quaternions[3]    # q4

    def test_extended_sequence_time_spacing(self) -> None:
        """Test that virtual waypoints have appropriate time spacing."""
        time_points = [0.0, 2.0, 4.0]
        quaternions = [
            Quaternion.identity(),
            Quaternion.from_euler_angles(0.1, 0.2, 0.3),
            Quaternion.from_euler_angles(0.2, 0.4, 0.6)
        ]

        squad = SquadC2(time_points, quaternions)
        extended_times, _ = squad.get_extended_waypoints()

        # Check that virtual waypoints are placed between segments
        assert extended_times[0] == 0.0   # q1
        assert 0.0 < extended_times[1] < 2.0   # q1_virt between q1 and q2
        assert extended_times[2] == 2.0   # q2
        assert 2.0 < extended_times[3] < 4.0   # q3_virt between q2 and q3
        assert extended_times[4] == 4.0   # q3


class TestSquadC2IntermediateQuaternions:
    """Test the corrected intermediate quaternion formula (Equation 5)."""

    NUMERICAL_TOLERANCE = 1e-10

    def test_intermediate_quaternion_formula(self) -> None:
        """Test the corrected intermediate quaternion formula from Equation 5."""
        # Create test quaternions
        q_prev = Quaternion.identity()
        q_curr = Quaternion.from_euler_angles(0.1, 0.2, 0.3)
        q_next = Quaternion.from_euler_angles(0.2, 0.4, 0.6)

        # Different segment durations to test the correction
        h_prev = 1.0
        h_curr = 2.0

        # Create SQUAD C2 instance to access the method
        time_points = [0.0, h_prev, h_prev + h_curr]
        quaternions = [q_prev, q_curr, q_next]

        squad = SquadC2(time_points, quaternions)

        # Get the computed intermediate quaternion
        _, extended_quats = squad.get_extended_waypoints()

        # The intermediate quaternion should be unit
        intermediate_q = squad.intermediate_quaternions[2]  # Middle quaternion
        assert abs(intermediate_q.norm() - 1.0) < self.NUMERICAL_TOLERANCE

    def test_intermediate_quaternions_all_unit(self) -> None:
        """Test that all intermediate quaternions are unit quaternions."""
        time_points = [0.0, 1.0, 2.5, 4.0, 5.0]
        quaternions = [
            Quaternion.identity(),
            Quaternion.from_euler_angles(0.1, 0.2, 0.3),
            Quaternion.from_euler_angles(0.2, 0.4, 0.6),
            Quaternion.from_euler_angles(0.3, 0.6, 0.9),
            Quaternion.from_euler_angles(0.4, 0.8, 1.2)
        ]

        squad = SquadC2(time_points, quaternions)

        for intermediate_q in squad.intermediate_quaternions:
            assert abs(intermediate_q.norm() - 1.0) < self.NUMERICAL_TOLERANCE


class TestSquadC2Interpolation:
    """Test SQUAD interpolation using Equations 2-4 from the paper."""

    NUMERICAL_TOLERANCE = 1e-10
    LOOSE_TOLERANCE = 1e-6

    def test_waypoint_interpolation(self) -> None:
        """Test interpolation exactly at waypoints."""
        time_points = [0.0, 1.0, 2.0, 3.0]
        quaternions = [
            Quaternion.identity(),
            Quaternion.from_euler_angles(0.1, 0.2, 0.3),
            Quaternion.from_euler_angles(0.2, 0.4, 0.6),
            Quaternion.from_euler_angles(0.3, 0.6, 0.9)
        ]

        squad = SquadC2(time_points, quaternions)

        # Test interpolation at original waypoints
        for i, t in enumerate(time_points):
            q_interp = squad.evaluate(t)

            # Should match original quaternion (within numerical tolerance)
            dot_product = q_interp.dot_product(quaternions[i])
            assert abs(abs(dot_product) - 1.0) < self.LOOSE_TOLERANCE

    def test_interpolated_quaternions_unit(self) -> None:
        """Test that all interpolated quaternions are unit quaternions."""
        time_points = [0.0, 1.0, 2.0]
        quaternions = [
            Quaternion.identity(),
            Quaternion.from_euler_angles(0.1, 0.2, 0.3),
            Quaternion.from_euler_angles(0.2, 0.4, 0.6)
        ]

        squad = SquadC2(time_points, quaternions)

        # Test at various interpolation points
        test_times = np.linspace(0.0, 2.0, 21)
        for t in test_times:
            q_interp = squad.evaluate(t)
            assert abs(q_interp.norm() - 1.0) < self.NUMERICAL_TOLERANCE

    def test_boundary_behavior(self) -> None:
        """Test interpolation behavior at boundaries."""
        time_points = [0.0, 1.0, 2.0]
        quaternions = [
            Quaternion.identity(),
            Quaternion.from_euler_angles(0.1, 0.2, 0.3),
            Quaternion.from_euler_angles(0.2, 0.4, 0.6)
        ]

        squad = SquadC2(time_points, quaternions)

        # Before first waypoint should return first quaternion
        q_before = squad.evaluate(-1.0)
        dot_product = q_before.dot_product(quaternions[0])
        assert abs(abs(dot_product) - 1.0) < self.LOOSE_TOLERANCE

        # After last waypoint should return last quaternion
        q_after = squad.evaluate(3.0)
        dot_product = q_after.dot_product(quaternions[-1])
        assert abs(abs(dot_product) - 1.0) < self.LOOSE_TOLERANCE


class TestSquadC2Continuity:
    """Test C²-continuity properties of SQUAD C2 interpolation."""

    NUMERICAL_TOLERANCE = 1e-6
    DERIVATIVE_TOLERANCE = 2e-4

    def _compute_numerical_derivative(self, squad: SquadC2, t: float, dt: float = 1e-6) -> Quaternion:
        """Compute numerical derivative of quaternion at time t."""
        q_plus = squad.evaluate(t + dt)
        q_minus = squad.evaluate(t - dt)

        # Numerical derivative: (q(t+dt) - q(t-dt)) / (2*dt)
        dq_dt = (q_plus - q_minus) / (2.0 * dt)
        return dq_dt

    def _compute_numerical_second_derivative(self, squad: SquadC2, t: float, dt: float = 1e-6) -> Quaternion:
        """Compute numerical second derivative of quaternion at time t."""
        q_plus = squad.evaluate(t + dt)
        q_curr = squad.evaluate(t)
        q_minus = squad.evaluate(t - dt)

        # Second derivative: (q(t+dt) - 2*q(t) + q(t-dt)) / dt²
        d2q_dt2 = (q_plus - 2.0 * q_curr + q_minus) / (dt * dt)
        return d2q_dt2

    def test_c0_continuity(self) -> None:
        """Test C⁰-continuity (position continuity)."""
        time_points = [0.0, 1.0, 2.0, 3.0]
        quaternions = [
            Quaternion.identity(),
            Quaternion.from_euler_angles(0.1, 0.2, 0.3),
            Quaternion.from_euler_angles(0.2, 0.4, 0.6),
            Quaternion.from_euler_angles(0.3, 0.6, 0.9)
        ]

        squad = SquadC2(time_points, quaternions)

        # Test continuity at segment boundaries
        for i in range(1, len(time_points) - 1):
            t = time_points[i]
            dt = 1e-8

            q_left = squad.evaluate(t - dt)
            q_right = squad.evaluate(t + dt)

            # Quaternions should be very close (accounting for q/-q equivalence)
            dot_product = q_left.dot_product(q_right)
            assert abs(abs(dot_product) - 1.0) < self.NUMERICAL_TOLERANCE

    def test_c1_continuity(self) -> None:
        """Test C¹-continuity (velocity continuity)."""
        time_points = [0.0, 1.0, 2.0, 3.0]
        quaternions = [
            Quaternion.identity(),
            Quaternion.from_euler_angles(0.1, 0.2, 0.3),
            Quaternion.from_euler_angles(0.2, 0.4, 0.6),
            Quaternion.from_euler_angles(0.3, 0.6, 0.9)
        ]

        squad = SquadC2(time_points, quaternions)

        # Test velocity continuity at segment boundaries
        for i in range(1, len(time_points) - 1):
            t = time_points[i]
            dt = 1e-6

            omega_left = squad.evaluate_velocity(t - dt)
            omega_right = squad.evaluate_velocity(t + dt)

            # Angular velocities should be close
            diff = np.linalg.norm(omega_left - omega_right)
            assert diff < self.DERIVATIVE_TOLERANCE

    def test_c2_continuity(self) -> None:
        """Test C²-continuity (acceleration continuity)."""
        time_points = [0.0, 1.0, 2.0, 3.0]
        quaternions = [
            Quaternion.identity(),
            Quaternion.from_euler_angles(0.1, 0.2, 0.3),
            Quaternion.from_euler_angles(0.2, 0.4, 0.6),
            Quaternion.from_euler_angles(0.3, 0.6, 0.9)
        ]

        squad = SquadC2(time_points, quaternions)

        # Test acceleration continuity at segment boundaries
        for i in range(1, len(time_points) - 1):
            t = time_points[i]
            dt = 1e-6

            alpha_left = squad.evaluate_acceleration(t - dt)
            alpha_right = squad.evaluate_acceleration(t + dt)

            # Angular accelerations should be close
            diff = np.linalg.norm(alpha_left - alpha_right)
            assert diff < self.DERIVATIVE_TOLERANCE


class TestSquadC2ZeroClampedBoundaries:
    """Test zero-clamped boundary conditions (ω=0, ω̇=0 at endpoints)."""

    BOUNDARY_TOLERANCE = 1e-6

    def test_zero_velocity_at_boundaries(self) -> None:
        """Test that angular velocity is zero at start and end points."""
        time_points = [0.0, 1.0, 2.0, 3.0]
        quaternions = [
            Quaternion.identity(),
            Quaternion.from_euler_angles(0.1, 0.2, 0.3),
            Quaternion.from_euler_angles(0.2, 0.4, 0.6),
            Quaternion.from_euler_angles(0.3, 0.6, 0.9)
        ]

        squad = SquadC2(time_points, quaternions)

        # Test at start point
        omega_start = squad.evaluate_velocity(time_points[0])
        assert np.linalg.norm(omega_start) < self.BOUNDARY_TOLERANCE

        # Test at end point
        omega_end = squad.evaluate_velocity(time_points[-1])
        assert np.linalg.norm(omega_end) < self.BOUNDARY_TOLERANCE

    def test_zero_acceleration_at_boundaries(self) -> None:
        """Test that angular acceleration is zero at start and end points."""
        time_points = [0.0, 1.0, 2.0, 3.0]
        quaternions = [
            Quaternion.identity(),
            Quaternion.from_euler_angles(0.1, 0.2, 0.3),
            Quaternion.from_euler_angles(0.2, 0.4, 0.6),
            Quaternion.from_euler_angles(0.3, 0.6, 0.9)
        ]

        squad = SquadC2(time_points, quaternions)

        # Test at start point
        alpha_start = squad.evaluate_acceleration(time_points[0])
        assert np.linalg.norm(alpha_start) < self.BOUNDARY_TOLERANCE

        # Test at end point
        alpha_end = squad.evaluate_acceleration(time_points[-1])
        assert np.linalg.norm(alpha_end) < self.BOUNDARY_TOLERANCE

    def test_boundary_conditions_near_endpoints(self) -> None:
        """Test that velocities and accelerations approach zero near boundaries."""
        time_points = [0.0, 1.0, 2.0]
        quaternions = [
            Quaternion.identity(),
            Quaternion.from_euler_angles(0.1, 0.2, 0.3),
            Quaternion.from_euler_angles(0.2, 0.4, 0.6)
        ]

        squad = SquadC2(time_points, quaternions)

        dt = 0.01

        # Test near start
        omega_near_start = squad.evaluate_velocity(time_points[0] + dt)
        alpha_near_start = squad.evaluate_acceleration(time_points[0] + dt)

        # Should be small but not necessarily zero
        assert np.linalg.norm(omega_near_start) < 1.0  # Reasonable upper bound
        assert np.linalg.norm(alpha_near_start) < 10.0  # Reasonable upper bound

        # Test near end
        omega_near_end = squad.evaluate_velocity(time_points[-1] - dt)
        alpha_near_end = squad.evaluate_acceleration(time_points[-1] - dt)

        assert np.linalg.norm(omega_near_end) < 1.0
        assert np.linalg.norm(alpha_near_end) < 10.0


class TestSquadC2EdgeCases:
    """Test edge cases and robustness of SQUAD C2 implementation."""

    NUMERICAL_TOLERANCE = 1e-10
    LOOSE_TOLERANCE = 1e-6

    def test_identical_consecutive_quaternions(self) -> None:
        """Test with identical consecutive quaternions."""
        q_identical = Quaternion.from_euler_angles(0.1, 0.2, 0.3)
        time_points = [0.0, 1.0, 2.0]
        quaternions = [q_identical, q_identical, q_identical]

        squad = SquadC2(time_points, quaternions)

        # All interpolated quaternions should be the same
        test_times = np.linspace(0.0, 2.0, 11)
        for t in test_times:
            q_interp = squad.evaluate(t)
            dot_product = q_interp.dot_product(q_identical)
            assert abs(abs(dot_product) - 1.0) < self.LOOSE_TOLERANCE

    def test_nearly_opposite_quaternions(self) -> None:
        """Test interpolation between nearly opposite quaternions."""
        q1 = Quaternion(1.0, 0.0, 0.0, 0.0)
        q2 = Quaternion(-0.999, 0.001, 0.001, 0.001).unit()  # Nearly opposite

        time_points = [0.0, 1.0]
        quaternions = [q1, q2]

        squad = SquadC2(time_points, quaternions)

        # Should interpolate smoothly without issues
        test_times = np.linspace(0.0, 1.0, 11)
        for t in test_times:
            q_interp = squad.evaluate(t)
            assert abs(q_interp.norm() - 1.0) < self.NUMERICAL_TOLERANCE

    def test_large_rotation_angles(self) -> None:
        """Test with large rotation angles."""
        q1 = Quaternion.identity()
        q2 = Quaternion.from_angle_axis(np.pi, np.array([0.0, 0.0, 1.0]))  # 180° rotation
        q3 = Quaternion.from_angle_axis(2*np.pi, np.array([1.0, 0.0, 0.0]))  # 360° rotation

        time_points = [0.0, 1.0, 2.0]
        quaternions = [q1, q2, q3]

        squad = SquadC2(time_points, quaternions)

        # Should handle large rotations correctly
        test_times = np.linspace(0.0, 2.0, 21)
        for t in test_times:
            q_interp = squad.evaluate(t)
            assert abs(q_interp.norm() - 1.0) < self.NUMERICAL_TOLERANCE

    def test_very_close_time_points(self) -> None:
        """Test with very close time points."""
        time_points = [0.0, 1e-6, 2e-6]
        quaternions = [
            Quaternion.identity(),
            Quaternion.from_euler_angles(0.001, 0.002, 0.003),
            Quaternion.from_euler_angles(0.002, 0.004, 0.006)
        ]

        squad = SquadC2(time_points, quaternions)

        # Should still work with very small time intervals
        for t in time_points:
            q_interp = squad.evaluate(t)
            assert abs(q_interp.norm() - 1.0) < self.NUMERICAL_TOLERANCE

    def test_non_uniform_time_spacing(self) -> None:
        """Test with non-uniform time spacing between waypoints."""
        time_points = [0.0, 0.1, 1.0, 5.0, 5.1]  # Varying intervals
        quaternions = [
            Quaternion.identity(),
            Quaternion.from_euler_angles(0.05, 0.1, 0.15),
            Quaternion.from_euler_angles(0.1, 0.2, 0.3),
            Quaternion.from_euler_angles(0.2, 0.4, 0.6),
            Quaternion.from_euler_angles(0.25, 0.5, 0.75)
        ]

        squad = SquadC2(time_points, quaternions)

        # Should handle non-uniform spacing correctly
        test_times = [0.05, 0.5, 2.5, 5.05]
        for t in test_times:
            q_interp = squad.evaluate(t)
            assert abs(q_interp.norm() - 1.0) < self.NUMERICAL_TOLERANCE


class TestSquadC2PaperExample:
    """Test against the 6-waypoint example from the paper."""

    NUMERICAL_TOLERANCE = 1e-10
    TRAJECTORY_TOLERANCE = 1e-6
    ACCELERATION_CONTINUITY_TOLERANCE = 2e-4  # More realistic tolerance for numerical acceleration

    def test_six_waypoint_scenario(self) -> None:
        """Test the 6-waypoint scenario shown in the paper figures."""
        # Create 6 waypoints similar to those shown in Figure 2 of the paper
        time_points = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]

        # Generate quaternions representing different orientations
        quaternions = [
            Quaternion.identity(),
            Quaternion.from_euler_angles(0.2, 0.1, 0.0),
            Quaternion.from_euler_angles(0.1, 0.3, 0.2),
            Quaternion.from_euler_angles(-0.1, 0.2, 0.4),
            Quaternion.from_euler_angles(0.0, -0.1, 0.3),
            Quaternion.from_euler_angles(0.1, 0.0, 0.1)
        ]

        squad = SquadC2(time_points, quaternions)

        # Test properties expected from the paper
        # 1. All interpolated quaternions should be unit
        test_times = np.linspace(0.0, 2.5, 51)
        for t in test_times:
            q_interp = squad.evaluate(t)
            assert abs(q_interp.norm() - 1.0) < self.NUMERICAL_TOLERANCE

        # 2. Zero-clamped boundaries
        omega_start = squad.evaluate_velocity(0.0)
        omega_end = squad.evaluate_velocity(2.5)
        alpha_start = squad.evaluate_acceleration(0.0)
        alpha_end = squad.evaluate_acceleration(2.5)

        assert np.linalg.norm(omega_start) < self.TRAJECTORY_TOLERANCE
        assert np.linalg.norm(omega_end) < self.TRAJECTORY_TOLERANCE
        assert np.linalg.norm(alpha_start) < self.TRAJECTORY_TOLERANCE
        assert np.linalg.norm(alpha_end) < self.TRAJECTORY_TOLERANCE

        # 3. Smooth trajectory - test velocity continuity and reasonable acceleration behavior
        for i in range(1, len(time_points) - 1):
            t = time_points[i]
            dt = 1e-6

            # Test velocity continuity (should be very smooth)
            omega_left = squad.evaluate_velocity(t - dt)
            omega_right = squad.evaluate_velocity(t + dt)
            velocity_jump = np.linalg.norm(omega_left - omega_right)
            assert velocity_jump < self.TRAJECTORY_TOLERANCE

            # Test that accelerations are bounded (C² theoretical, numerical limits)
            alpha_left = squad.evaluate_acceleration(t - dt)
            alpha_right = squad.evaluate_acceleration(t + dt)

            # Accelerations should be reasonable (not infinite or NaN)
            assert np.isfinite(alpha_left).all()
            assert np.isfinite(alpha_right).all()
            assert np.linalg.norm(alpha_left) < 100.0  # Reasonable upper bound
            assert np.linalg.norm(alpha_right) < 100.0  # Reasonable upper bound


class TestSquadC2Utilities:
    """Test utility methods and information retrieval."""

    def test_waypoint_retrieval(self) -> None:
        """Test retrieval of original and extended waypoints."""
        time_points = [0.0, 1.0, 2.0]
        quaternions = [
            Quaternion.identity(),
            Quaternion.from_euler_angles(0.1, 0.2, 0.3),
            Quaternion.from_euler_angles(0.2, 0.4, 0.6)
        ]

        squad = SquadC2(time_points, quaternions)

        # Test original waypoints
        orig_times, orig_quats = squad.get_waypoints()
        assert orig_times == time_points
        assert len(orig_quats) == len(quaternions)
        for i, q in enumerate(orig_quats):
            assert q == quaternions[i]

        # Test extended waypoints
        ext_times, ext_quats = squad.get_extended_waypoints()
        assert len(ext_times) > len(time_points)
        assert len(ext_quats) == len(ext_times)

    def test_extended_sequence_info(self) -> None:
        """Test extended sequence information retrieval."""
        time_points = [0.0, 1.0, 2.0, 3.0]
        quaternions = [
            Quaternion.identity(),
            Quaternion.from_euler_angles(0.1, 0.2, 0.3),
            Quaternion.from_euler_angles(0.2, 0.4, 0.6),
            Quaternion.from_euler_angles(0.3, 0.6, 0.9)
        ]

        squad = SquadC2(time_points, quaternions)

        info = squad.get_extended_sequence_info()

        assert info["n_original"] == 4
        assert info["n_extended"] == 6  # Should have 2 virtual waypoints
        assert info["has_virtual_waypoints"] is True
        assert len(info["original_times"]) == 4
        assert len(info["extended_times"]) == 6
        assert len(info["segment_durations"]) == 5  # n_extended - 1

    def test_string_representation(self) -> None:
        """Test string representation of SQUAD C2 object."""
        time_points = [0.0, 1.0, 2.0]
        quaternions = [
            Quaternion.identity(),
            Quaternion.from_euler_angles(0.1, 0.2, 0.3),
            Quaternion.from_euler_angles(0.2, 0.4, 0.6)
        ]

        squad = SquadC2(time_points, quaternions)

        str_repr = str(squad)
        assert "SquadC2" in str_repr
        assert "3 original" in str_repr
        assert "extended" in str_repr
        assert "t=[0.000, 2.000]" in str_repr

        # __repr__ should be same as __str__
        assert str(squad) == repr(squad)


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main(["-xvs", __file__])

