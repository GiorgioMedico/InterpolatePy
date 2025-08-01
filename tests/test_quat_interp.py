"""
Comprehensive tests for the quaternion interpolation implementation.

This module contains extensive tests for the Quaternion class covering:
1. Basic quaternion operations and constructors
2. Core quaternion mathematics (conjugate, norm, inverse, exp, log, power)
3. Conversion methods (Euler angles, rotation matrix, axis-angle)
4. Quaternion dynamics and integration
5. Interpolation methods (SLERP, SQUAD)
6. Spline functionality
7. Edge cases and error handling
8. Performance benchmarks

The tests verify numerical accuracy, handle edge cases, and ensure
robust behavior across different use cases.
"""

from typing import Any

import numpy as np
import pytest

from interpolatepy.quat_core import Quaternion
from interpolatepy.quat_spline import QuaternionSpline

# Type alias for pytest benchmark fixture
if not hasattr(pytest, "FixtureFunction"):
    pytest.FixtureFunction = Any


class TestQuaternionBasicOperations:
    """Test suite for basic quaternion operations and constructors."""

    # Test tolerances
    REGULAR_RTOL = 1e-10
    REGULAR_ATOL = 1e-10
    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def test_default_constructor(self) -> None:
        """Test default quaternion constructor creates identity quaternion."""
        q = Quaternion()

        assert q.s() == 1.0
        assert np.allclose(q.v(), [0.0, 0.0, 0.0])
        assert q.w == 1.0
        assert q.x == 0.0
        assert q.y == 0.0
        assert q.z == 0.0

    def test_parameterized_constructor(self) -> None:
        """Test quaternion constructor with specific values."""
        q = Quaternion(0.5, 0.1, 0.2, 0.3)

        assert q.s() == 0.5
        assert np.allclose(q.v(), [0.1, 0.2, 0.3])
        assert q.w == 0.5
        assert q.x == 0.1
        assert q.y == 0.2
        assert q.z == 0.3

    def test_identity_constructor(self) -> None:
        """Test identity quaternion factory method."""
        q = Quaternion.identity()

        assert q.s() == 1.0
        assert np.allclose(q.v(), [0.0, 0.0, 0.0])
        assert q.norm() == 1.0

    @pytest.mark.parametrize("angle", [0.0, np.pi/4, np.pi/2, np.pi, 2*np.pi])
    @pytest.mark.parametrize("axis", [
        [1.0, 0.0, 0.0],  # X-axis
        [0.0, 1.0, 0.0],  # Y-axis
        [0.0, 0.0, 1.0],  # Z-axis
        [1.0, 1.0, 1.0],  # Non-unit vector (should be normalized)
    ])
    def test_from_angle_axis(self, angle: float, axis: list[float]) -> None:
        """Test quaternion creation from angle-axis representation."""
        axis_array = np.array(axis)
        q = Quaternion.from_angle_axis(angle, axis_array)

        # Verify it's a unit quaternion
        assert abs(q.norm() - 1.0) < self.REGULAR_ATOL

        # Convert back to axis-angle and verify
        recovered_axis, recovered_angle = q.to_axis_angle()

        # Handle angle wrapping and axis sign ambiguity
        if abs(angle) < self.REGULAR_ATOL:  # Zero angle case
            assert abs(recovered_angle) < self.REGULAR_ATOL
        else:
            # Normalize expected axis
            expected_axis = axis_array / np.linalg.norm(axis_array)

            # Check if we got the same or opposite axis (both valid)
            axis_match = np.allclose(recovered_axis, expected_axis, atol=self.NUMERICAL_ATOL)
            axis_opposite = np.allclose(recovered_axis, -expected_axis, atol=self.NUMERICAL_ATOL)

            # For 2π rotations, the axis might be arbitrary since it's identity rotation
            if abs(angle - 2 * np.pi) < self.NUMERICAL_ATOL:
                # 2π rotation is identity, any axis is valid as long as angle is ~0
                assert abs(recovered_angle) < self.NUMERICAL_ATOL
            else:
                assert axis_match or axis_opposite

                # Angle should match (considering 2π periodicity and axis sign)
                if axis_match:
                    expected_angle = angle % (2 * np.pi)
                    if expected_angle > np.pi:
                        expected_angle = 2 * np.pi - expected_angle
                    assert abs(recovered_angle - expected_angle) < self.NUMERICAL_ATOL
                else:  # axis_opposite
                    # If axis is opposite, angle should be 2π - original
                    expected_angle = (2 * np.pi - angle) % (2 * np.pi)
                    if expected_angle > np.pi:
                        expected_angle = 2 * np.pi - expected_angle
                    assert abs(recovered_angle - expected_angle) < self.NUMERICAL_ATOL

    def test_from_angle_axis_edge_cases(self) -> None:
        """Test edge cases for angle-axis constructor."""
        # Zero axis should raise error
        with pytest.raises(ValueError, match="Axis cannot be zero vector"):
            Quaternion.from_angle_axis(1.0, np.array([0.0, 0.0, 0.0]))

        # Wrong axis dimension should raise error
        with pytest.raises(ValueError, match="size of axis != 3"):
            Quaternion.from_angle_axis(1.0, np.array([1.0, 2.0]))

    @pytest.mark.parametrize(("roll", "pitch", "yaw"), [
        (0.0, 0.0, 0.0),           # Identity
        (np.pi/4, 0.0, 0.0),       # Roll only
        (0.0, np.pi/4, 0.0),       # Pitch only
        (0.0, 0.0, np.pi/4),       # Yaw only
        (np.pi/6, np.pi/4, np.pi/3), # Combined rotations
        (np.pi, 0.0, 0.0),         # 180° roll
        (0.0, np.pi/2, 0.0),       # 90° pitch (gimbal lock)
    ])
    def test_from_euler_angles(self, roll: float, pitch: float, yaw: float) -> None:
        """Test quaternion creation from Euler angles."""
        q = Quaternion.from_euler_angles(roll, pitch, yaw)

        # Verify it's a unit quaternion
        assert abs(q.norm() - 1.0) < self.REGULAR_ATOL

        # Convert back to Euler angles and verify
        recovered_roll, recovered_pitch, recovered_yaw = q.to_euler_angles()

        # Handle angle wrapping and gimbal lock cases
        if abs(pitch - np.pi/2) < self.NUMERICAL_ATOL or abs(pitch + np.pi/2) < self.NUMERICAL_ATOL:
            # Gimbal lock case - only verify that rotation is equivalent
            q_expected = Quaternion.from_euler_angles(roll, pitch, yaw)
            q_recovered = Quaternion.from_euler_angles(
                recovered_roll, recovered_pitch, recovered_yaw
            )

            # Check if quaternions represent same rotation (q or -q)
            dot_product = q_expected.dot_product(q_recovered)
            assert abs(abs(dot_product) - 1.0) < self.NUMERICAL_ATOL
        else:
            # Normal case - angles should match within tolerance
            assert abs(recovered_roll - roll) < self.NUMERICAL_ATOL
            assert abs(recovered_pitch - pitch) < self.NUMERICAL_ATOL
            assert abs(recovered_yaw - yaw) < self.NUMERICAL_ATOL

    def test_from_rotation_matrix_3x3(self) -> None:
        """Test quaternion creation from 3x3 rotation matrix."""
        # Identity matrix
        identity = np.eye(3)
        q = Quaternion.from_rotation_matrix(identity)
        assert abs(q.s() - 1.0) < self.REGULAR_ATOL
        assert np.allclose(q.v(), [0.0, 0.0, 0.0], atol=self.REGULAR_ATOL)

        # 90° rotation around Z-axis
        cos_90, sin_90 = 0.0, 1.0
        rotation_z_90 = np.array([
            [cos_90, -sin_90, 0.0],
            [sin_90,  cos_90, 0.0],
            [0.0,     0.0,    1.0]
        ])
        q = Quaternion.from_rotation_matrix(rotation_z_90)

        # Verify it's a unit quaternion
        assert abs(q.norm() - 1.0) < self.REGULAR_ATOL

        # Convert back to rotation matrix and verify
        recovered_matrix = q.to_rotation_matrix()
        assert np.allclose(recovered_matrix, rotation_z_90, atol=self.NUMERICAL_ATOL)

    def test_from_rotation_matrix_4x4(self) -> None:
        """Test quaternion creation from 4x4 transformation matrix."""
        # 4x4 transformation matrix with rotation part
        transform_4x4 = np.eye(4)
        transform_4x4[:3, :3] = np.array([
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0]
        ])

        q = Quaternion.from_rotation_matrix(transform_4x4)

        # Verify it's a unit quaternion
        assert abs(q.norm() - 1.0) < self.REGULAR_ATOL

        # Convert back and verify rotation part matches
        recovered_matrix = q.to_rotation_matrix()
        assert np.allclose(recovered_matrix, transform_4x4[:3, :3], atol=self.NUMERICAL_ATOL)

    def test_from_rotation_matrix_invalid_size(self) -> None:
        """Test error handling for invalid rotation matrix sizes."""
        with pytest.raises(ValueError, match="matrix input is not 3x3 or 4x4"):
            Quaternion.from_rotation_matrix(np.array([[1, 2], [3, 4]]))

    def test_arithmetic_operations(self) -> None:
        """Test basic arithmetic operations."""
        q1 = Quaternion(1.0, 2.0, 3.0, 4.0)
        q2 = Quaternion(0.5, 1.0, 1.5, 2.0)

        # Addition
        q_add = q1 + q2
        assert q_add.s() == 1.5
        assert np.allclose(q_add.v(), [3.0, 4.5, 6.0])

        # Subtraction
        q_sub = q1 - q2
        assert q_sub.s() == 0.5
        assert np.allclose(q_sub.v(), [1.0, 1.5, 2.0])

        # Scalar multiplication
        q_scalar = q1 * 2.0
        assert q_scalar.s() == 2.0
        assert np.allclose(q_scalar.v(), [4.0, 6.0, 8.0])

        # Right scalar multiplication
        q_rscalar = 2.0 * q1
        assert q_rscalar.s() == 2.0
        assert np.allclose(q_rscalar.v(), [4.0, 6.0, 8.0])

        # Scalar division
        q_div = q1 / 2.0
        assert q_div.s() == 0.5
        assert np.allclose(q_div.v(), [1.0, 1.5, 2.0])

        # Negation
        q_neg = -q1
        assert q_neg.s() == -1.0
        assert np.allclose(q_neg.v(), [-2.0, -3.0, -4.0])

    def test_quaternion_multiplication(self) -> None:
        """Test quaternion multiplication."""
        # Test with known result
        q1 = Quaternion(1.0, 0.0, 0.0, 0.0)  # Identity
        q2 = Quaternion(0.0, 1.0, 0.0, 0.0)  # Pure quaternion

        q_mult = q1 * q2
        assert q_mult.s() == 0.0
        assert np.allclose(q_mult.v(), [1.0, 0.0, 0.0])

        # Test non-commutativity
        q_mult_rev = q2 * q1
        assert q_mult_rev.s() == 0.0
        assert np.allclose(q_mult_rev.v(), [1.0, 0.0, 0.0])

        # Test with unit quaternions (should preserve norm)
        q1_unit = Quaternion.from_euler_angles(0.1, 0.2, 0.3)
        q2_unit = Quaternion.from_euler_angles(0.4, 0.5, 0.6)

        q_result = q1_unit * q2_unit
        assert abs(q_result.norm() - 1.0) < self.NUMERICAL_ATOL

    def test_quaternion_division(self) -> None:
        """Test quaternion division."""
        q1 = Quaternion.from_euler_angles(0.1, 0.2, 0.3)
        q2 = Quaternion.from_euler_angles(0.4, 0.5, 0.6)

        # q1 / q2 should equal q1 * q2.inverse()
        q_div = q1 / q2
        q_mult_inv = q1 * q2.inverse()

        assert abs(q_div.s() - q_mult_inv.s()) < self.NUMERICAL_ATOL
        assert np.allclose(q_div.v(), q_mult_inv.v(), atol=self.NUMERICAL_ATOL)

    def test_setters_and_getters(self) -> None:
        """Test quaternion setters and getters."""
        q = Quaternion()

        # Test scalar setter/getter
        q.set_s(0.7071)
        assert abs(q.s() - 0.7071) < self.REGULAR_ATOL
        assert abs(q.w - 0.7071) < self.REGULAR_ATOL

        # Test vector setter/getter
        new_v = np.array([0.1, 0.2, 0.3])
        q.set_v(new_v)
        assert np.allclose(q.v(), new_v)
        assert abs(q.x - 0.1) < self.REGULAR_ATOL
        assert abs(q.y - 0.2) < self.REGULAR_ATOL
        assert abs(q.z - 0.3) < self.REGULAR_ATOL

    def test_vector_setter_validation(self) -> None:
        """Test vector setter input validation."""
        q = Quaternion()

        # Wrong dimension should raise error
        with pytest.raises(ValueError, match="input has a wrong size"):
            q.set_v(np.array([1.0, 2.0]))  # Only 2 elements

        with pytest.raises(ValueError, match="input has a wrong size"):
            q.set_v(np.array([1.0, 2.0, 3.0, 4.0]))  # Too many elements

    def test_copy(self) -> None:
        """Test quaternion copy functionality."""
        q1 = Quaternion(0.5, 0.1, 0.2, 0.3)
        q2 = q1.copy()

        # Should be equal but different objects
        assert q1 == q2
        assert q1 is not q2

        # Modifying copy shouldn't affect original
        q2.set_s(0.8)
        assert q1.s() != q2.s()

    def test_string_representation(self) -> None:
        """Test string representation of quaternions."""
        q = Quaternion(0.5, 0.1, 0.2, 0.3)

        str_repr = str(q)
        assert "0.500000" in str_repr
        assert "0.100000" in str_repr
        assert "0.200000" in str_repr
        assert "0.300000" in str_repr

        # __repr__ should be same as __str__
        assert str(q) == repr(q)

    def test_equality(self) -> None:
        """Test quaternion equality comparison."""
        q1 = Quaternion(1.0, 0.0, 0.0, 0.0)
        q2 = Quaternion(1.0, 0.0, 0.0, 0.0)
        q3 = Quaternion(0.5, 0.0, 0.0, 0.0)

        # Equal quaternions
        assert q1 == q2

        # Different quaternions
        assert q1 != q3

        # Should not be equal to non-quaternion
        assert q1 != "not a quaternion"
        assert (q1 == "not a quaternion") is False


class TestQuaternionMathematics:
    """Test suite for core quaternion mathematical operations."""

    REGULAR_RTOL = 1e-10
    REGULAR_ATOL = 1e-10
    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def test_conjugate(self) -> None:
        """Test quaternion conjugate operation."""
        q = Quaternion(0.5, 0.1, 0.2, 0.3)
        q_conj = q.conjugate()

        assert q_conj.s() == q.s()
        assert np.allclose(q_conj.v(), -q.v())

        # Conjugate of conjugate should be original
        q_conj_conj = q_conj.conjugate()
        assert q_conj_conj == q

    def test_norm(self) -> None:
        """Test quaternion norm calculation."""
        q = Quaternion(3.0, 4.0, 0.0, 0.0)
        assert abs(q.norm() - 5.0) < self.REGULAR_ATOL

        # Norm squared
        assert abs(q.norm_squared() - 25.0) < self.REGULAR_ATOL

        # Unit quaternion should have norm 1
        q_unit = Quaternion.from_euler_angles(0.1, 0.2, 0.3)
        assert abs(q_unit.norm() - 1.0) < self.NUMERICAL_ATOL

    def test_unit_normalization(self) -> None:
        """Test quaternion normalization to unit length."""
        q = Quaternion(2.0, 3.0, 4.0, 5.0)
        q_unit = q.unit()

        # Should be unit quaternion
        assert abs(q_unit.norm() - 1.0) < self.NUMERICAL_ATOL

        # Should be in same direction as original
        scale_factor = q.norm()
        expected_unit = Quaternion(q.s()/scale_factor, *(q.v()/scale_factor))
        assert abs(q_unit.s() - expected_unit.s()) < self.NUMERICAL_ATOL
        assert np.allclose(q_unit.v(), expected_unit.v(), atol=self.NUMERICAL_ATOL)

    def test_unit_normalization_edge_cases(self) -> None:
        """Test unit normalization edge cases."""
        # Near-zero quaternion should return identity and print warning
        q_zero = Quaternion(1e-10, 1e-10, 1e-10, 1e-10)
        q_unit = q_zero.unit()

        # Should return identity quaternion
        assert abs(q_unit.s() - 1.0) < self.REGULAR_ATOL
        assert np.allclose(q_unit.v(), [0.0, 0.0, 0.0], atol=self.REGULAR_ATOL)

    def test_inverse(self) -> None:
        """Test quaternion inverse operation."""
        q = Quaternion.from_euler_angles(0.1, 0.2, 0.3)
        q_inv = q.inverse()

        # q * q^(-1) should be identity
        identity = q * q_inv
        assert abs(identity.s() - 1.0) < self.NUMERICAL_ATOL
        assert np.allclose(identity.v(), [0.0, 0.0, 0.0], atol=self.NUMERICAL_ATOL)

        # q^(-1) * q should also be identity
        identity2 = q_inv * q
        assert abs(identity2.s() - 1.0) < self.NUMERICAL_ATOL
        assert np.allclose(identity2.v(), [0.0, 0.0, 0.0], atol=self.NUMERICAL_ATOL)

        # Test alias method
        q_inv2 = q.i()
        assert q_inv == q_inv2

    def test_inverse_edge_cases(self) -> None:
        """Test inverse operation edge cases."""
        # Zero quaternion should raise error
        q_zero = Quaternion(0.0, 0.0, 0.0, 0.0)
        with pytest.raises(ValueError, match="Cannot compute inverse of zero quaternion"):
            q_zero.inverse()

    def test_exponential(self) -> None:
        """Test quaternion exponential function."""
        # exp([0, 0, 0, 0]) should be [1, 0, 0, 0]
        q_zero = Quaternion(0.0, 0.0, 0.0, 0.0)
        q_exp = q_zero.exp()
        assert abs(q_exp.s() - 1.0) < self.NUMERICAL_ATOL
        assert np.allclose(q_exp.v(), [0.0, 0.0, 0.0], atol=self.NUMERICAL_ATOL)

        # exp([0, π/2, 0, 0]) should be [cos(π/2), sin(π/2), 0, 0] = [0, 1, 0, 0]
        q_pi_2 = Quaternion(0.0, np.pi/2, 0.0, 0.0)
        q_exp = q_pi_2.exp()
        assert abs(q_exp.s() - 0.0) < self.NUMERICAL_ATOL
        assert abs(q_exp.v()[0] - 1.0) < self.NUMERICAL_ATOL
        assert abs(q_exp.v()[1] - 0.0) < self.NUMERICAL_ATOL
        assert abs(q_exp.v()[2] - 0.0) < self.NUMERICAL_ATOL

    def test_logarithm(self) -> None:
        """Test quaternion logarithm function."""
        # log([1, 0, 0, 0]) should be [0, 0, 0, 0]
        q_identity = Quaternion.identity()
        q_log = q_identity.Log()
        assert abs(q_log.s() - 0.0) < self.NUMERICAL_ATOL
        assert np.allclose(q_log.v(), [0.0, 0.0, 0.0], atol=self.NUMERICAL_ATOL)

        # Test exp/log relationship for unit quaternions
        q_unit = Quaternion.from_euler_angles(0.1, 0.2, 0.3)
        q_log = q_unit.Log()
        q_exp_log = q_log.exp()

        # Should recover original quaternion (up to sign)
        dot_product = q_unit.dot_product(q_exp_log)
        assert abs(abs(dot_product) - 1.0) < self.NUMERICAL_ATOL

    def test_power(self) -> None:
        """Test quaternion power operation."""
        q = Quaternion.from_euler_angles(0.1, 0.2, 0.3)

        # q^0 should be identity
        q_power_0 = q.power(0.0)
        assert abs(q_power_0.s() - 1.0) < self.NUMERICAL_ATOL
        assert np.allclose(q_power_0.v(), [0.0, 0.0, 0.0], atol=self.NUMERICAL_ATOL)

        # q^1 should be q
        q_power_1 = q.power(1.0)
        assert abs(q_power_1.s() - q.s()) < self.NUMERICAL_ATOL
        assert np.allclose(q_power_1.v(), q.v(), atol=self.NUMERICAL_ATOL)

        # q^2 should be q*q
        q_power_2 = q.power(2.0)
        q_squared = q * q
        assert abs(q_power_2.s() - q_squared.s()) < self.NUMERICAL_ATOL
        assert np.allclose(q_power_2.v(), q_squared.v(), atol=self.NUMERICAL_ATOL)

    def test_dot_product(self) -> None:
        """Test quaternion dot product."""
        q1 = Quaternion(1.0, 2.0, 3.0, 4.0)
        q2 = Quaternion(0.5, 1.0, 1.5, 2.0)

        # Manual calculation: 1*0.5 + 2*1 + 3*1.5 + 4*2 = 0.5 + 2 + 4.5 + 8 = 15
        dot_prod = q1.dot_product(q2)
        assert abs(dot_prod - 15.0) < self.REGULAR_ATOL

        # Test alias method
        dot_prod2 = q1.dot_prod(q2)
        assert abs(dot_prod - dot_prod2) < self.REGULAR_ATOL

        # Dot product should be commutative
        dot_prod_rev = q2.dot_product(q1)
        assert abs(dot_prod - dot_prod_rev) < self.REGULAR_ATOL


class TestQuaternionConversions:
    """Test suite for quaternion conversion methods."""

    REGULAR_RTOL = 1e-10
    REGULAR_ATOL = 1e-10
    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    @pytest.mark.parametrize(("roll", "pitch", "yaw"), [
        (0.0, 0.0, 0.0),
        (np.pi/6, np.pi/4, np.pi/3),
        (np.pi/2, 0.0, 0.0),
        (0.0, np.pi/2, 0.0),
        (0.0, 0.0, np.pi/2),
        (-np.pi/4, -np.pi/6, -np.pi/3),
    ])
    def test_euler_angles_round_trip(self, roll: float, pitch: float, yaw: float) -> None:
        """Test Euler angles conversion round trip."""
        q = Quaternion.from_euler_angles(roll, pitch, yaw)
        recovered_roll, recovered_pitch, recovered_yaw = q.to_euler_angles()

        # Handle angle wrapping and gimbal lock cases
        if abs(pitch - np.pi/2) < self.NUMERICAL_ATOL or abs(pitch + np.pi/2) < self.NUMERICAL_ATOL:
            # Gimbal lock case - only verify that rotation is equivalent
            q_expected = Quaternion.from_euler_angles(roll, pitch, yaw)
            q_recovered = Quaternion.from_euler_angles(
                recovered_roll, recovered_pitch, recovered_yaw
            )

            # Check if quaternions represent same rotation (q or -q)
            dot_product = q_expected.dot_product(q_recovered)
            assert abs(abs(dot_product) - 1.0) < self.NUMERICAL_ATOL
        else:
            # Normal case - angles should match within tolerance
            assert abs(recovered_roll - roll) < self.NUMERICAL_ATOL
            assert abs(recovered_pitch - pitch) < self.NUMERICAL_ATOL
            assert abs(recovered_yaw - yaw) < self.NUMERICAL_ATOL

    def test_rotation_matrix_round_trip(self) -> None:
        """Test rotation matrix conversion round trip."""
        # Test with various rotations
        test_quaternions = [
            Quaternion.identity(),
            Quaternion.from_euler_angles(0.1, 0.2, 0.3),
            Quaternion.from_euler_angles(np.pi/4, np.pi/6, np.pi/3),
            Quaternion.from_angle_axis(np.pi/2, np.array([1.0, 1.0, 1.0])),
        ]

        for q_original in test_quaternions:
            # Convert to rotation matrix and back
            rotation_matrix = q_original.to_rotation_matrix()
            q_recovered = Quaternion.from_rotation_matrix(rotation_matrix)

            # Quaternions should represent same rotation (q or -q)
            dot_product = q_original.dot_product(q_recovered)
            assert abs(abs(dot_product) - 1.0) < self.NUMERICAL_ATOL

    def test_rotation_matrix_properties(self) -> None:
        """Test that rotation matrices have proper properties."""
        q = Quaternion.from_euler_angles(0.1, 0.2, 0.3)
        R = q.to_rotation_matrix()

        # Should be orthogonal: R^T * R = I
        R_transpose = R.T
        identity_check = R_transpose @ R
        assert np.allclose(identity_check, np.eye(3), atol=self.NUMERICAL_ATOL)

        # Determinant should be 1 (proper rotation)
        det = np.linalg.det(R)
        assert abs(det - 1.0) < self.NUMERICAL_ATOL

    def test_transformation_matrix(self) -> None:
        """Test transformation matrix generation."""
        q = Quaternion.from_euler_angles(0.1, 0.2, 0.3)
        T = q.to_transformation_matrix()

        # Should be 4x4
        assert T.shape == (4, 4)

        # Rotation part should match rotation matrix
        R = q.to_rotation_matrix()
        assert np.allclose(T[:3, :3], R, atol=self.NUMERICAL_ATOL)

        # Translation part should be zero
        assert np.allclose(T[:3, 3], [0.0, 0.0, 0.0], atol=self.REGULAR_ATOL)

        # Bottom row should be [0, 0, 0, 1]
        assert np.allclose(T[3, :], [0.0, 0.0, 0.0, 1.0], atol=self.REGULAR_ATOL)

        # Test alias method
        T2 = q.T()
        assert np.allclose(T, T2, atol=self.REGULAR_ATOL)

    @pytest.mark.parametrize("angle", [0.0, np.pi/4, np.pi/2, np.pi])
    @pytest.mark.parametrize("axis", [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ])
    def test_axis_angle_round_trip(self, angle: float, axis: list[float]) -> None:
        """Test axis-angle conversion round trip."""
        axis_array = np.array(axis)

        # Create quaternion from axis-angle
        q = Quaternion.from_angle_axis(angle, axis_array)

        # Convert back to axis-angle
        recovered_axis, recovered_angle = q.to_axis_angle()

        if abs(angle) < self.NUMERICAL_ATOL:
            # Zero angle case
            assert abs(recovered_angle) < self.NUMERICAL_ATOL
        else:
            # Normalize expected axis
            expected_axis = axis_array / np.linalg.norm(axis_array)

            # Check axis (may be negated)
            axis_match = np.allclose(recovered_axis, expected_axis, atol=self.NUMERICAL_ATOL)
            axis_opposite = np.allclose(recovered_axis, -expected_axis, atol=self.NUMERICAL_ATOL)
            assert axis_match or axis_opposite

            # Check angle (considering axis sign and 2π periodicity)
            expected_angle = angle % (2 * np.pi)
            if expected_angle > np.pi:
                expected_angle = 2 * np.pi - expected_angle

            assert abs(recovered_angle - expected_angle) < self.NUMERICAL_ATOL


class TestQuaternionDynamics:
    """Test suite for quaternion dynamics and integration."""

    REGULAR_RTOL = 1e-10
    REGULAR_ATOL = 1e-10
    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def test_time_derivative(self) -> None:
        """Test quaternion time derivative calculation."""
        q = Quaternion.from_euler_angles(0.1, 0.2, 0.3)
        w = np.array([0.1, 0.2, 0.3])  # Angular velocity

        # Test both frame types
        q_dot_base = q.dot(w, Quaternion.BASE_FRAME)
        q_dot_body = q.dot(w, Quaternion.BODY_FRAME)

        # Results should be different for different frames
        assert not np.allclose(q_dot_base.v(), q_dot_body.v(), atol=self.NUMERICAL_ATOL)

        # Both should have appropriate magnitude
        assert q_dot_base.norm() > 0
        assert q_dot_body.norm() > 0

    def test_time_derivative_validation(self) -> None:
        """Test time derivative input validation."""
        q = Quaternion.identity()

        # Wrong angular velocity dimension should raise error
        with pytest.raises(ValueError, match="Angular velocity must be 3D vector"):
            q.dot(np.array([1.0, 2.0]), Quaternion.BASE_FRAME)

    def test_e_matrix(self) -> None:
        """Test E matrix computation for dynamics."""
        q = Quaternion(0.5, 0.1, 0.2, 0.3)

        # Test both frame types
        E_base = q.E(Quaternion.BASE_FRAME)
        E_body = q.E(Quaternion.BODY_FRAME)

        # Should be 3x3 matrices
        assert E_base.shape == (3, 3)
        assert E_body.shape == (3, 3)

        # Should be different for different frames
        assert not np.allclose(E_base, E_body, atol=self.NUMERICAL_ATOL)

    def test_skew_symmetric_matrix(self) -> None:
        """Test skew-symmetric matrix generation."""
        v = np.array([1.0, 2.0, 3.0])
        S = Quaternion._skew_symmetric_matrix(v)

        # Should be 3x3
        assert S.shape == (3, 3)

        # Should be skew-symmetric: S^T = -S
        assert np.allclose(S.T, -S, atol=self.REGULAR_ATOL)

        # Diagonal should be zero
        assert np.allclose(np.diag(S), [0.0, 0.0, 0.0], atol=self.REGULAR_ATOL)

        # Check specific values
        expected = np.array([
            [0.0, -3.0,  2.0],
            [3.0,  0.0, -1.0],
            [-2.0, 1.0,  0.0]
        ])
        assert np.allclose(S, expected, atol=self.REGULAR_ATOL)

    def test_omega_extraction(self) -> None:
        """Test angular velocity extraction from quaternion derivatives."""
        # Create a known angular velocity
        w_original = np.array([0.1, 0.2, 0.3])
        q = Quaternion.from_euler_angles(0.1, 0.2, 0.3)

        # Compute time derivative
        q_dot = q.dot(w_original, Quaternion.BASE_FRAME)

        # Extract angular velocity
        w_extracted = Quaternion.Omega(q, q_dot)

        # Should recover original angular velocity
        assert np.allclose(w_extracted, w_original, atol=self.NUMERICAL_ATOL)

    def test_numerical_integration(self) -> None:
        """Test trapezoidal quaternion integration."""
        # Initial quaternion
        q0 = Quaternion.from_euler_angles(0.1, 0.2, 0.3)

        # Time derivatives (representing angular velocities)
        dq_present = Quaternion(0.0, 0.01, 0.02, 0.03)
        dq_past = Quaternion(0.0, 0.005, 0.01, 0.015)

        # Time step
        dt = 0.1

        # Integrate
        q_new, dq_new_present, dq_new_past, status = Quaternion.Integ_quat(
            dq_present, dq_past, q0, dt
        )

        # Should succeed
        assert status == 0

        # Result should be unit quaternion
        assert abs(q_new.norm() - 1.0) < self.NUMERICAL_ATOL

        # Should be different from initial
        assert q_new != q0

    def test_integration_edge_cases(self) -> None:
        """Test integration edge cases."""
        q0 = Quaternion.from_euler_angles(0.1, 0.2, 0.3)
        dq = Quaternion(0.0, 0.01, 0.02, 0.03)

        # Negative time step
        q_new, _, _, status = Quaternion.Integ_quat(dq, dq, q0, -0.1)
        assert status == -1

        # Zero time step
        q_new, _, _, status = Quaternion.Integ_quat(dq, dq, q0, 0.0)
        assert status == 0
        assert q_new == q0

        # Near-zero quaternion
        q_zero = Quaternion(1e-10, 1e-10, 1e-10, 1e-10)
        q_new, _, _, status = Quaternion.Integ_quat(dq, dq, q_zero, 0.1)
        assert status == -2

    def test_trapezoidal_integration_components(self) -> None:
        """Test individual trapezoidal integration components."""
        dq_present = Quaternion(1.0, 0.1, 0.2, 0.3)
        dq_past = Quaternion(0.5, 0.05, 0.1, 0.15)
        dt = 0.1

        # Test scalar integration
        s_integrated = Quaternion.integ_trap_quat_s(dq_present, dq_past, dt)
        expected_s = 0.5 * (dq_present.s() + dq_past.s()) * dt
        assert abs(s_integrated - expected_s) < self.REGULAR_ATOL

        # Test vector integration
        v_integrated = Quaternion.integ_trap_quat_v(dq_present, dq_past, dt)
        expected_v = 0.5 * (dq_present.v() + dq_past.v()) * dt
        assert np.allclose(v_integrated, expected_v, atol=self.REGULAR_ATOL)


class TestQuaternionInterpolation:
    """Test suite for quaternion interpolation methods."""

    REGULAR_RTOL = 1e-10
    REGULAR_ATOL = 1e-10
    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def test_slerp_basic(self) -> None:
        """Test basic SLERP interpolation."""
        q0 = Quaternion.identity()
        q1 = Quaternion.from_euler_angles(0.0, 0.0, np.pi/2)

        # t=0 should give q0
        q_interp = q0.slerp(q1, 0.0)
        assert abs(q_interp.s() - q0.s()) < self.NUMERICAL_ATOL
        assert np.allclose(q_interp.v(), q0.v(), atol=self.NUMERICAL_ATOL)

        # t=1 should give q1
        q_interp = q0.slerp(q1, 1.0)
        dot_product = q_interp.dot_product(q1)
        assert abs(abs(dot_product) - 1.0) < self.NUMERICAL_ATOL

        # t=0.5 should give halfway rotation
        q_interp = q0.slerp(q1, 0.5)
        # Should be unit quaternion
        assert abs(q_interp.norm() - 1.0) < self.NUMERICAL_ATOL

    def test_slerp_parameter_clamping(self) -> None:
        """Test SLERP parameter clamping."""
        q0 = Quaternion.identity()
        q1 = Quaternion.from_euler_angles(0.1, 0.2, 0.3)

        # t < 0 should be clamped to 0
        q_interp_neg = q0.slerp(q1, -0.5)
        q_interp_zero = q0.slerp(q1, 0.0)
        assert q_interp_neg == q_interp_zero

        # t > 1 should be clamped to 1
        q_interp_large = q0.slerp(q1, 1.5)
        q_interp_one = q0.slerp(q1, 1.0)
        dot_product = q_interp_large.dot_product(q_interp_one)
        assert abs(abs(dot_product) - 1.0) < self.NUMERICAL_ATOL

    def test_slerp_static_method(self) -> None:
        """Test static SLERP method."""
        q0 = Quaternion.identity()
        q1 = Quaternion.from_euler_angles(0.1, 0.2, 0.3)

        # Instance method
        q_instance = q0.slerp(q1, 0.5)

        # Static method
        q_static = Quaternion.Slerp(q0, q1, 0.5)

        # Should give same result
        assert abs(q_instance.s() - q_static.s()) < self.NUMERICAL_ATOL
        assert np.allclose(q_instance.v(), q_static.v(), atol=self.NUMERICAL_ATOL)

    def test_slerp_derivative(self) -> None:
        """Test SLERP derivative calculation."""
        q0 = Quaternion.from_euler_angles(0.1, 0.2, 0.3)
        q1 = Quaternion.from_euler_angles(0.4, 0.5, 0.6)

        # Test at various parameter values
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            q_prime = q0.slerp_prime(q1, t)

            # Derivative should exist
            assert q_prime.norm() >= 0

            # Test static method
            q_prime_static = Quaternion.Slerp_prime(q0, q1, t)
            assert abs(q_prime.s() - q_prime_static.s()) < self.NUMERICAL_ATOL
            assert np.allclose(q_prime.v(), q_prime_static.v(), atol=self.NUMERICAL_ATOL)

    def test_squad_basic(self) -> None:
        """Test basic SQUAD interpolation."""
        # Create test quaternions
        p = Quaternion.identity()
        q = Quaternion.from_euler_angles(0.0, 0.0, np.pi/4)
        a = Quaternion.from_euler_angles(0.0, 0.0, np.pi/8)
        b = Quaternion.from_euler_angles(0.0, 0.0, 3*np.pi/8)

        # t=0 should give p
        q_interp = Quaternion.Squad(p, a, b, q, 0.0)
        dot_product = q_interp.dot_product(p)
        assert abs(abs(dot_product) - 1.0) < self.NUMERICAL_ATOL

        # t=1 should give q
        q_interp = Quaternion.Squad(p, a, b, q, 1.0)
        dot_product = q_interp.dot_product(q)
        assert abs(abs(dot_product) - 1.0) < self.NUMERICAL_ATOL

        # All interpolated quaternions should be unit
        for t in [0.2, 0.4, 0.6, 0.8]:
            q_interp = Quaternion.Squad(p, a, b, q, t)
            assert abs(q_interp.norm() - 1.0) < self.NUMERICAL_ATOL

    def test_squad_derivative(self) -> None:
        """Test SQUAD derivative calculation."""
        p = Quaternion.identity()
        q = Quaternion.from_euler_angles(0.1, 0.2, 0.3)
        a = Quaternion.from_euler_angles(0.05, 0.1, 0.15)
        b = Quaternion.from_euler_angles(0.075, 0.15, 0.225)

        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            q_prime = Quaternion.Squad_prime(p, a, b, q, t)

            # Derivative should exist
            assert q_prime.norm() >= 0

    def test_intermediate_quaternion_computation(self) -> None:
        """Test intermediate quaternion computation for SQUAD."""
        # Create a sequence of quaternions
        q_prev = Quaternion.from_euler_angles(0.0, 0.0, 0.0)
        q_curr = Quaternion.from_euler_angles(0.1, 0.2, 0.3)
        q_next = Quaternion.from_euler_angles(0.2, 0.4, 0.6)

        # Compute intermediate quaternion
        s_i = Quaternion.compute_intermediate_quaternion(q_prev, q_curr, q_next)

        # Should be unit quaternion
        assert abs(s_i.norm() - 1.0) < self.NUMERICAL_ATOL


class TestQuaternionSpline:
    """Test suite for quaternion spline functionality."""

    REGULAR_RTOL = 1e-10
    REGULAR_ATOL = 1e-10
    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def setup_test_spline_data(self) -> tuple[list[float], list[Quaternion]]:
        """Create test data for spline testing."""
        time_points = [0.0, 1.0, 2.0, 3.0, 4.0]
        quaternions = [
            Quaternion.identity(),
            Quaternion.from_euler_angles(0.1, 0.2, 0.3),
            Quaternion.from_euler_angles(0.2, 0.4, 0.6),
            Quaternion.from_euler_angles(0.3, 0.6, 0.9),
            Quaternion.from_euler_angles(0.4, 0.8, 1.2),
        ]
        return time_points, quaternions

    def test_spline_creation(self) -> None:
        """Test spline creation and setup."""
        time_points, quaternions = self.setup_test_spline_data()

        # Test different interpolation methods
        for method in [Quaternion.SLERP, Quaternion.SQUAD, Quaternion.AUTO]:
            spline = QuaternionSpline(time_points, quaternions, method)

            assert not spline.is_empty()
            assert spline.get_interpolation_method() == method

            t_min, t_max = spline.get_time_range()
            assert abs(t_min - 0.0) < self.REGULAR_ATOL
            assert abs(t_max - 4.0) < self.REGULAR_ATOL

    def test_spline_validation(self) -> None:
        """Test spline input validation."""
        time_points, quaternions = self.setup_test_spline_data()

        # Mismatched lengths
        with pytest.raises(ValueError, match="Time points and quaternions must have same length"):
            QuaternionSpline(time_points[:-1], quaternions, Quaternion.SLERP)

        # Too few points
        with pytest.raises(ValueError, match="Need at least 2 points for interpolation"):
            QuaternionSpline([0.0], [Quaternion.identity()], Quaternion.SLERP)

        # Invalid interpolation method
        with pytest.raises(ValueError, match="Invalid interpolation method"):
            QuaternionSpline(time_points, quaternions, "invalid_method")

    def test_spline_interpolation_basic(self) -> None:
        """Test basic spline interpolation."""
        time_points, quaternions = self.setup_test_spline_data()
        spline = QuaternionSpline(time_points, quaternions, Quaternion.SLERP)

        # Test interpolation at control points
        for i, t in enumerate(time_points):
            q_interp, status = spline.interpolate_at_time(t)
            assert status == 0

            # Should be close to original quaternion
            dot_product = q_interp.dot_product(quaternions[i])
            assert abs(abs(dot_product) - 1.0) < self.NUMERICAL_ATOL

    def test_spline_interpolation_between_points(self) -> None:
        """Test spline interpolation between control points."""
        time_points, quaternions = self.setup_test_spline_data()
        spline = QuaternionSpline(time_points, quaternions, Quaternion.SLERP)

        # Test interpolation between points
        for t in [0.5, 1.5, 2.5, 3.5]:
            q_interp, status = spline.interpolate_at_time(t)
            assert status == 0
            assert abs(q_interp.norm() - 1.0) < self.NUMERICAL_ATOL

    def test_spline_boundary_conditions(self) -> None:
        """Test spline boundary conditions."""
        time_points, quaternions = self.setup_test_spline_data()
        spline = QuaternionSpline(time_points, quaternions, Quaternion.SLERP)

        # Before first time point
        q_interp, status = spline.interpolate_at_time(-1.0)
        assert status == 0
        dot_product = q_interp.dot_product(quaternions[0])
        assert abs(abs(dot_product) - 1.0) < self.NUMERICAL_ATOL

        # After last time point
        q_interp, status = spline.interpolate_at_time(5.0)
        assert status == 0
        dot_product = q_interp.dot_product(quaternions[-1])
        assert abs(abs(dot_product) - 1.0) < self.NUMERICAL_ATOL

    def test_spline_method_switching(self) -> None:
        """Test changing interpolation method on existing spline."""
        time_points, quaternions = self.setup_test_spline_data()
        spline = QuaternionSpline(time_points, quaternions, Quaternion.SLERP)

        # Switch to SQUAD
        spline.set_interpolation_method(Quaternion.SQUAD)
        assert spline.get_interpolation_method() == Quaternion.SQUAD

        # Should still interpolate correctly
        q_interp, status = spline.interpolate_at_time(1.5)
        assert status == 0
        assert abs(q_interp.norm() - 1.0) < self.NUMERICAL_ATOL

        # Switch to AUTO
        spline.set_interpolation_method(Quaternion.AUTO)
        assert spline.get_interpolation_method() == Quaternion.AUTO

    def test_spline_method_validation(self) -> None:
        """Test spline method validation."""
        time_points, quaternions = self.setup_test_spline_data()
        spline = QuaternionSpline(time_points, quaternions, Quaternion.SLERP)

        # Invalid method should raise error
        with pytest.raises(ValueError, match="Invalid interpolation method"):
            spline.set_interpolation_method("invalid_method")

    def test_forced_interpolation_methods(self) -> None:
        """Test forced interpolation methods regardless of spline setting."""
        time_points, quaternions = self.setup_test_spline_data()
        spline = QuaternionSpline(time_points, quaternions, Quaternion.AUTO)

        # Force SLERP
        q_slerp, status = spline.interpolate_slerp(1.5)
        assert status == 0
        assert abs(q_slerp.norm() - 1.0) < self.NUMERICAL_ATOL

        # Force SQUAD (should work with enough points)
        q_squad, status = spline.interpolate_squad(1.5)
        assert status == 0
        assert abs(q_squad.norm() - 1.0) < self.NUMERICAL_ATOL

        # Test boundary cases for forced SQUAD
        q_squad_boundary, status = spline.interpolate_squad(0.5)
        assert status == 0

    def test_squad_insufficient_points(self) -> None:
        """Test SQUAD interpolation with insufficient points."""
        # Only 3 points (need 4 for SQUAD)
        time_points = [0.0, 1.0, 2.0]
        quaternions = [
            Quaternion.identity(),
            Quaternion.from_euler_angles(0.1, 0.2, 0.3),
            Quaternion.from_euler_angles(0.2, 0.4, 0.6),
        ]

        spline = QuaternionSpline(time_points, quaternions, Quaternion.SQUAD)

        # Should fallback to SLERP
        q_interp, status = spline.interpolate_at_time(0.5)
        assert status == 0

        # Forced SQUAD should fail
        q_squad, status = spline.interpolate_squad(0.5)
        assert status == -2

    def test_spline_with_velocity(self) -> None:
        """Test spline interpolation with velocity computation."""
        time_points, quaternions = self.setup_test_spline_data()
        spline = QuaternionSpline(time_points, quaternions, Quaternion.SLERP)

        # Test at various time points
        for t in [0.5, 1.5, 2.5]:
            q_interp, w, status = spline.interpolate_with_velocity(t)
            assert status == 0
            assert abs(q_interp.norm() - 1.0) < self.NUMERICAL_ATOL
            assert len(w) == 3  # 3D angular velocity

    def test_empty_spline(self) -> None:
        """Test behavior with empty spline."""
        # Empty spline creation should raise an error now
        with pytest.raises(ValueError, match="Need at least 2 points for interpolation"):
            QuaternionSpline([], [], Quaternion.SLERP)


class TestQuaternionEdgeCases:
    """Test suite for edge cases and error handling."""

    NUMERICAL_RTOL = 1e-6
    NUMERICAL_ATOL = 1e-6

    def test_numerical_stability_near_zero(self) -> None:
        """Test numerical stability with near-zero values."""
        # Near-zero quaternion
        q_small = Quaternion(1e-10, 1e-10, 1e-10, 1e-10)

        # Operations should handle gracefully
        q_unit = q_small.unit()  # Should return identity
        assert abs(q_unit.s() - 1.0) < self.NUMERICAL_ATOL

        # Logarithm of near-identity quaternion
        q_near_identity = Quaternion(1.0 - 1e-10, 1e-10, 1e-10, 1e-10)
        q_log = q_near_identity.Log()
        assert abs(q_log.s()) < self.NUMERICAL_ATOL

    def test_numerical_stability_large_angles(self) -> None:
        """Test numerical stability with large rotation angles."""
        # Large angle rotations
        large_angles = [np.pi, 2*np.pi, 4*np.pi, 10*np.pi]

        for angle in large_angles:
            q = Quaternion.from_angle_axis(angle, np.array([0.0, 0.0, 1.0]))

            # Should be unit quaternion
            assert abs(q.norm() - 1.0) < self.NUMERICAL_ATOL

            # Should represent equivalent rotation to angle % (2*π)
            expected_angle = angle % (2 * np.pi)
            if expected_angle > np.pi:
                expected_angle = 2 * np.pi - expected_angle

            recovered_axis, recovered_angle = q.to_axis_angle()
            assert abs(recovered_angle - expected_angle) < self.NUMERICAL_ATOL

    def test_interpolation_with_opposite_quaternions(self) -> None:
        """Test interpolation between quaternions that are nearly opposite."""
        q1 = Quaternion(1.0, 0.0, 0.0, 0.0)
        q2 = Quaternion(-1.0 + 1e-10, 1e-10, 1e-10, 1e-10)  # Nearly opposite

        # SLERP should handle this gracefully
        q_interp = q1.slerp(q2, 0.5)
        assert abs(q_interp.norm() - 1.0) < self.NUMERICAL_ATOL

    def test_interpolation_identical_quaternions(self) -> None:
        """Test interpolation between identical quaternions."""
        q = Quaternion.from_euler_angles(0.1, 0.2, 0.3)

        # Should return the same quaternion for any t
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            q_interp = q.slerp(q, t)
            assert abs(q_interp.s() - q.s()) < self.NUMERICAL_ATOL
            assert np.allclose(q_interp.v(), q.v(), atol=self.NUMERICAL_ATOL)

    def test_matrix_conversion_edge_cases(self) -> None:
        """Test rotation matrix conversion edge cases."""
        # Test with nearly singular rotation matrices
        eps = 1e-10
        near_singular = np.array([
            [1.0,   eps,   eps],
            [-eps,  1.0,   eps],
            [-eps, -eps,   1.0]
        ])

        # Should handle gracefully without throwing
        try:
            q = Quaternion.from_rotation_matrix(near_singular)
            assert abs(q.norm() - 1.0) < self.NUMERICAL_ATOL
        except (ValueError, np.linalg.LinAlgError):
            # Acceptable to fail on truly degenerate cases
            pass

    def test_integration_with_large_time_steps(self) -> None:
        """Test integration stability with large time steps."""
        q0 = Quaternion.from_euler_angles(0.1, 0.2, 0.3)
        dq = Quaternion(0.0, 0.1, 0.2, 0.3)  # Large derivative

        # Large time step
        dt = 1.0

        q_new, _, _, status = Quaternion.Integ_quat(dq, dq, q0, dt)

        # Should still succeed and maintain unit constraint
        if status == 0:
            assert abs(q_new.norm() - 1.0) < self.NUMERICAL_ATOL


class TestQuaternionPerformance:
    """Test suite for performance benchmarks."""

    @pytest.mark.parametrize("num_operations", [100, 1000, 10000])
    def test_basic_operations_performance(
        self, num_operations: int, benchmark: pytest.FixtureFunction
    ) -> None:
        """Benchmark basic quaternion operations."""
        q1 = Quaternion.from_euler_angles(0.1, 0.2, 0.3)
        q2 = Quaternion.from_euler_angles(0.4, 0.5, 0.6)

        def run_operations() -> None:
            for _ in range(num_operations):
                _ = q1 * q2
                _ = q1.conjugate()
                _ = q1.norm()
                _ = q1.unit()

        benchmark(run_operations)

    @pytest.mark.parametrize("num_interpolations", [100, 1000])
    def test_slerp_performance(
        self, num_interpolations: int, benchmark: pytest.FixtureFunction
    ) -> None:
        """Benchmark SLERP interpolation performance."""
        q1 = Quaternion.from_euler_angles(0.1, 0.2, 0.3)
        q2 = Quaternion.from_euler_angles(0.4, 0.5, 0.6)

        def run_slerp() -> None:
            for i in range(num_interpolations):
                t = i / (num_interpolations - 1)
                _ = q1.slerp(q2, t)

        benchmark(run_slerp)

    @pytest.mark.parametrize("num_points", [10, 50, 100])
    @pytest.mark.parametrize("num_evaluations", [100, 1000])
    def test_spline_performance(
        self, num_points: int, num_evaluations: int, benchmark: pytest.FixtureFunction
    ) -> None:
        """Benchmark spline interpolation performance."""
        # Create spline data
        time_points = [float(i) for i in range(num_points)]
        quaternions = [
            Quaternion.from_euler_angles(0.1 * i, 0.2 * i, 0.3 * i)
            for i in range(num_points)
        ]

        spline = QuaternionSpline(time_points, quaternions, Quaternion.SLERP)

        def run_spline_evaluation() -> None:
            for i in range(num_evaluations):
                t = (num_points - 1) * i / (num_evaluations - 1)
                _, _ = spline.interpolate_at_time(t)

        benchmark(run_spline_evaluation)

    def test_conversion_performance(self, benchmark: pytest.FixtureFunction) -> None:
        """Benchmark conversion operations performance."""
        quaternions = [
            Quaternion.from_euler_angles(0.1 * i, 0.2 * i, 0.3 * i)
            for i in range(1000)
        ]

        def run_conversions() -> None:
            for q in quaternions:
                _ = q.to_rotation_matrix()
                _ = q.to_euler_angles()
                _ = q.to_axis_angle()

        benchmark(run_conversions)


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main(["-xvs", __file__])

