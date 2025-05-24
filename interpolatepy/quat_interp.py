import os
from collections import OrderedDict
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


class Quaternion:  # noqa: PLR0904
    """
    Complete Quaternion class implementing all ROBOOP library functions.

    This class implements all quaternion operations from the ROBOOP library including:
    - Basic quaternion arithmetic and operations
    - Quaternion dynamics (time derivatives, integration)
    - Spherical linear interpolation (Slerp)
    - Spherical cubic interpolation (Squad)
    - Quaternion splines with file I/O
    - 3D visualization capabilities

    A quaternion is represented as q = [s, v] where:
    - s is the scalar part (w component)
    - v is the vector part [v1, v2, v3] (x, y, z components)
    """

    EPSILON = 1e-7
    BASE_FRAME = 0
    BODY_FRAME = 1

    # Constants for magic numbers
    VECTOR_DIM = 3
    ROTATION_MATRIX_ELEMENTS = 9
    QUATERNION_ELEMENTS = 4
    MIN_SQUAD_POINTS = 4
    MIN_INTERPOLATION_POINTS = 2
    INTEGRATION_TIME_TOLERANCE = 0.01

    def __init__(
        self, s: float = 1.0, v1: float = 0.0, v2: float = 0.0, v3: float = 0.0
    ) -> None:
        """
        Initialize quaternion with scalar and vector components.
        Default constructor creates identity quaternion [1, 0, 0, 0].

        Args:
            s: Scalar part (w component)
            v1: X component of vector part
            v2: Y component of vector part
            v3: Z component of vector part
        """
        self.s_ = float(s)
        self.v_ = np.array([float(v1), float(v2), float(v3)])

    # ==================== CONSTRUCTORS AND FACTORY METHODS ====================

    @classmethod
    def identity(cls) -> "Quaternion":
        """Create identity quaternion [1, 0, 0, 0]"""
        return cls(1.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_angle_axis(cls, angle: float, axis: np.ndarray) -> "Quaternion":
        """
        Create quaternion from rotation angle and axis.
        Implements ROBOOP constructor: Quaternion(const Real angle, const ColumnVector & axis)

        Args:
            angle: Rotation angle in radians
            axis: 3D rotation axis vector

        Returns:
            Quaternion representing the rotation
        """
        if len(axis) != cls.VECTOR_DIM:
            raise ValueError("Quaternion::Quaternion, size of axis != 3")

        # Make sure axis is a unit vector
        norm_axis = np.linalg.norm(axis)
        if norm_axis == 0:
            raise ValueError("Axis cannot be zero vector")

        if abs(norm_axis - 1.0) > cls.EPSILON:
            print("Quaternion::Quaternion(angle, axis), axis is not unit")
            print("Make the axis unit.")
            axis /= norm_axis

        half_angle = angle / 2.0
        s = np.cos(half_angle)
        v = np.sin(half_angle) * axis

        return cls(s, v[0], v[1], v[2])

    @classmethod
    def from_rotation_matrix(cls, rotation_matrix: np.ndarray) -> "Quaternion":
        """
        Create quaternion from 3x3 or 4x4 rotation matrix.
        Implements ROBOOP constructor: Quaternion(const Matrix & R)
        """
        if rotation_matrix.shape not in {(3, 3), (4, 4)}:
            raise ValueError("Quaternion::Quaternion: matrix input is not 3x3 or 4x4")

        # Extract 3x3 rotation part
        if rotation_matrix.shape == (4, 4):
            rotation_matrix = rotation_matrix[:3, :3]

        # Follow ROBOOP algorithm exactly
        tmp = abs(
            rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2] + 1
        )
        s = 0.5 * np.sqrt(tmp)

        if s > cls.EPSILON:
            # Standard case
            v1 = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / (4 * s)
            v2 = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / (4 * s)
            v3 = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / (4 * s)
        else:
            # |s| <= 1/2, use alternative method
            s_i_next = [1, 2, 0]  # Equivalent to static int s_iNext[3] = { 2, 3, 1 };
            i = 0
            if rotation_matrix[1, 1] > rotation_matrix[0, 0]:
                i = 1
            if rotation_matrix[2, 2] > rotation_matrix[i, i]:
                i = 2

            j = s_i_next[i]
            k = s_i_next[j]

            f_root = np.sqrt(
                rotation_matrix[i, i]
                - rotation_matrix[j, j]
                - rotation_matrix[k, k]
                + 1.0
            )

            v = np.zeros(3)
            v[i] = 0.5 * f_root
            f_root = 0.5 / f_root
            s = (rotation_matrix[k, j] - rotation_matrix[j, k]) * f_root
            v[j] = (rotation_matrix[j, i] + rotation_matrix[i, j]) * f_root
            v[k] = (rotation_matrix[k, i] + rotation_matrix[i, k]) * f_root

            v1, v2, v3 = v[0], v[1], v[2]

        return cls(s, v1, v2, v3)

    @classmethod
    def from_euler_angles(cls, roll: float, pitch: float, yaw: float) -> "Quaternion":
        """Create quaternion from Euler angles (roll, pitch, yaw) in radians"""
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)

        s = cr * cp * cy + sr * sp * sy
        v1 = sr * cp * cy - cr * sp * sy
        v2 = cr * sp * cy + sr * cp * sy
        v3 = cr * cp * sy - sr * sp * cy

        return cls(s, v1, v2, v3)

    # ==================== BASIC ARITHMETIC OPERATIONS ====================

    def __add__(self, other: "Quaternion") -> "Quaternion":
        """
        Quaternion addition: q1 + q2 = [s1+s2, v1+v2]
        Implements ROBOOP: Quaternion operator+(const Quaternion & rhs)const
        """
        return Quaternion(self.s_ + other.s_, *(self.v_ + other.v_))

    def __sub__(self, other: "Quaternion") -> "Quaternion":
        """
        Quaternion subtraction: q1 - q2 = [s1-s2, v1-v2]
        Implements ROBOOP: Quaternion operator-(const Quaternion & rhs)const
        """
        return Quaternion(self.s_ - other.s_, *(self.v_ - other.v_))

    def __mul__(self, other: Union["Quaternion", float]) -> "Quaternion":
        """
        Quaternion multiplication or scalar multiplication.
        Implements ROBOOP: Quaternion operator*(const Quaternion & rhs)const
        """
        if isinstance(other, Quaternion):
            # Quaternion multiplication: q = q1q2 = [s1s2 - v1·v2, v1 cross v2 + s1v2 + s2v1]
            s = self.s_ * other.s_ - np.dot(self.v_, other.v_)
            v = self.s_ * other.v_ + other.s_ * self.v_ + np.cross(self.v_, other.v_)
            return Quaternion(s, v[0], v[1], v[2])
        # Scalar multiplication
        return Quaternion(self.s_ * other, *(self.v_ * other))

    def __rmul__(self, scalar: float) -> "Quaternion":
        """
        Right scalar multiplication: c * q
        Implements ROBOOP: operator*(const Real c, const Quaternion & q)
        """
        return Quaternion(self.s_ * scalar, *(self.v_ * scalar))

    def __truediv__(self, other: Union["Quaternion", float]) -> "Quaternion":
        """
        Quaternion division or scalar division.
        Implements ROBOOP: Quaternion operator/(const Quaternion & rhs)const
        """
        if isinstance(other, Quaternion):
            return self * other.i()
        return Quaternion(self.s_ / other, *(self.v_ / other))

    def __neg__(self) -> "Quaternion":
        """Quaternion negation: -q = [-s, -v]"""
        return Quaternion(-self.s_, *(-self.v_))

    # ==================== QUATERNION OPERATIONS ====================

    def conjugate(self) -> "Quaternion":
        """
        Quaternion conjugate: q* = [s, -v]
        Implements ROBOOP: Quaternion conjugate()const
        """
        return Quaternion(self.s_, *(-self.v_))

    def norm(self) -> float:
        """
        Quaternion norm: ||q|| = sqrt(s² + v·v)
        Implements ROBOOP: Real norm()const
        """
        return np.sqrt(self.s_**2 + np.dot(self.v_, self.v_))

    def norm_squared(self) -> float:
        """Squared quaternion norm: ||q||² = s² + v·v"""
        return self.s_**2 + np.dot(self.v_, self.v_)

    def unit(self) -> "Quaternion":
        """
        Normalize quaternion to unit length.
        Implements ROBOOP: Quaternion & unit() - returns new quaternion instead of
        modifying in place
        """
        tmp = self.norm()
        if tmp > self.EPSILON:
            return Quaternion(self.s_ / tmp, *(self.v_ / tmp))
        return Quaternion(self.s_, *self.v_)

    def i(self) -> "Quaternion":
        """
        Quaternion inverse: q^(-1) = q*/||q||²
        Implements ROBOOP: Quaternion i()const
        """
        return self.conjugate() / self.norm_squared()

    def inverse(self) -> "Quaternion":
        """Alias for i() - quaternion inverse"""
        return self.i()

    def exp(self) -> "Quaternion":
        """
        Quaternion exponential.
        Implements ROBOOP: Quaternion exp() const

        For q = [0, θv], exp(q) = [cos(θ), v*sin(θ)]
        """
        theta = np.linalg.norm(self.v_)
        sin_theta = np.sin(theta)

        s = np.cos(theta)
        v = (
            self.v_ * sin_theta / theta
            if abs(sin_theta) > self.EPSILON
            else self.v_.copy()
        )

        return Quaternion(s, v[0], v[1], v[2])

    def Log(self) -> "Quaternion":  # noqa: N802
        """
        Quaternion logarithm for unit quaternions.
        Implements ROBOOP: Quaternion Log()const

        For unit q = [cos(θ), v*sin(θ)], log(q) = [0, v*θ]
        """
        theta = np.acos(min(1.0, abs(self.s_)))
        sin_theta = np.sin(theta)

        s = 0.0
        v = (
            self.v_ / sin_theta * theta
            if abs(sin_theta) > self.EPSILON
            else self.v_.copy()
        )

        return Quaternion(s, v[0], v[1], v[2])

    def power(self, t: float) -> "Quaternion":
        """
        Quaternion power: q^t = exp(t * log(q))
        Implements ROBOOP: Quaternion power(const Real t) const
        """
        return (self.Log() * t).exp()

    def dot_prod(self, other: "Quaternion") -> float:
        """
        Quaternion dot product: q1·q2 = s1*s2 + v1·v2
        Implements ROBOOP: Real dot_prod(const Quaternion & q)const
        """
        return self.s_ * other.s_ + np.dot(self.v_, other.v_)

    def dot_product(self, other: "Quaternion") -> float:
        """Alias for dot_prod"""
        return self.dot_prod(other)

    # ==================== DYNAMICS FUNCTIONS ====================

    def dot(self, w: np.ndarray, sign: int) -> "Quaternion":
        """
        Quaternion time derivative.
        Implements ROBOOP: Quaternion dot(const ColumnVector & w, const short sign)const

        The quaternion time derivative (quaternion propagation equation):
        ṡ = -½v^T*w₀
        v̇ = ½E(s,v)*w₀

        Where E = sI - S(v) for BASE_FRAME (sign=0)
              E = sI + S(v) for BODY_FRAME (sign=1)

        Args:
            w: Angular velocity vector (3D)
            sign: Frame type (BASE_FRAME=0 or BODY_FRAME=1)
        """
        if len(w) != self.VECTOR_DIM:
            raise ValueError("Angular velocity must be 3D vector")

        # Compute scalar time derivative: ṡ = -½v^T*w
        s_dot = -0.5 * np.dot(self.v_, w)

        # Compute vector time derivative: v̇ = ½E*w
        e_matrix = self.E(sign)
        v_dot = 0.5 * e_matrix @ w

        return Quaternion(s_dot, v_dot[0], v_dot[1], v_dot[2])

    def E(self, sign: int) -> np.ndarray:  # noqa: N802
        """
        Matrix E for quaternion dynamics.
        Implements ROBOOP: ReturnMatrix E(const short sign)const

        E = sI - S(v) for BASE_FRAME (sign=0)
        E = sI + S(v) for BODY_FRAME (sign=1)

        where S(v) is the skew-symmetric matrix of v
        """
        identity_matrix = np.eye(3)
        skew_v = self._skew_symmetric_matrix(self.v_)

        if sign == self.BODY_FRAME:
            return self.s_ * identity_matrix + skew_v
        # BASE_FRAME
        return self.s_ * identity_matrix - skew_v

    @staticmethod
    def _skew_symmetric_matrix(v: np.ndarray) -> np.ndarray:
        """
        Create skew-symmetric matrix from vector.
        S(v) = [[ 0,  -v3,  v2],
                [ v3,  0,  -v1],
                [-v2,  v1,  0 ]]
        """
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    @staticmethod
    def Omega(q: "Quaternion", q_dot: "Quaternion") -> np.ndarray:  # noqa: N802
        """
        Return angular velocity from quaternion and its time derivative.
        Implements ROBOOP: ReturnMatrix Omega(const Quaternion & q, const Quaternion & q_dot)

        Solves: q̇ = ½E*ω for ω
        """
        matrix_a = 0.5 * q.E(Quaternion.BASE_FRAME)
        vector_b = q_dot.v_

        # Solve A*w = B for w using least squares (more robust than inverse)
        try:
            w = (
                np.linalg.solve(matrix_a, vector_b)
                if np.linalg.det(matrix_a) != 0
                else np.linalg.pinv(matrix_a) @ vector_b
            )
        except np.linalg.LinAlgError:
            w = np.zeros(3)

        return w

    # ==================== NUMERICAL INTEGRATION ====================

    @staticmethod
    def Integ_quat(  # noqa: N802
        dquat_present: "Quaternion",
        dquat_past: "Quaternion",
        quat: "Quaternion",
        dt: float,
    ) -> tuple["Quaternion", "Quaternion", "Quaternion", int]:
        """
        Trapezoidal quaternion integration.
        Implements ROBOOP: short Integ_quat(Quaternion & dquat_present, Quaternion & dquat_past,
                                           Quaternion & quat, const Real dt)

        Returns: (updated_quat, updated_dquat_present, updated_dquat_past, status)
        """
        if dt < 0:
            print("Integ_Trap(quat1, quat2, dt): dt < 0. dt is set to 0.")
            return quat, dquat_present, dquat_past, -1

        # Quaternion algebraic constraint (commented out in original)
        # K_lambda = 0.5 * (1 - quat.norm_squared())
        # dquat_present.s_ += K_lambda * quat.s_
        # dquat_present.v_ += K_lambda * quat.v_

        # Integrate using trapezoidal rule
        s_integrated = Quaternion.integ_trap_quat_s(dquat_present, dquat_past, dt)
        v_integrated = Quaternion.integ_trap_quat_v(dquat_present, dquat_past, dt)

        # Update quaternion
        new_quat = Quaternion(quat.s_ + s_integrated, *(quat.v_ + v_integrated))

        # Update past derivative
        new_dquat_past = Quaternion(dquat_present.s_, *dquat_present.v_)

        # Normalize quaternion to maintain unit constraint
        new_quat = new_quat.unit()

        return new_quat, dquat_present, new_dquat_past, 0

    @staticmethod
    def integ_trap_quat_s(
        present: "Quaternion", past: "Quaternion", dt: float
    ) -> float:
        """
        Trapezoidal quaternion scalar part integration.
        Implements ROBOOP: Real integ_trap_quat_s(const Quaternion & present, Quaternion & past,
                                                  const Real dt)
        """
        return 0.5 * (present.s_ + past.s_) * dt

    @staticmethod
    def integ_trap_quat_v(
        present: "Quaternion", past: "Quaternion", dt: float
    ) -> np.ndarray:
        """
        Trapezoidal quaternion vector part integration.
        Implements ROBOOP: ReturnMatrix integ_trap_quat_v(const Quaternion & present,
                                                          Quaternion & past, const Real dt)
        """
        return 0.5 * (present.v_ + past.v_) * dt

    # ==================== CONVERSION METHODS ====================

    def R(self) -> np.ndarray:  # noqa: N802
        """
        Rotation matrix from unit quaternion.
        Implements ROBOOP: ReturnMatrix R()const

        R = (s² - v·v)I + 2vv^T + 2s*S(v)
        """
        s, v = self.s_, self.v_
        v1, v2, v3 = v[0], v[1], v[2]

        return np.array(
            [
                [
                    s**2 + v1**2 - v2**2 - v3**2,
                    2 * v1 * v2 - 2 * s * v3,
                    2 * v1 * v3 + 2 * s * v2,
                ],
                [
                    2 * v1 * v2 + 2 * s * v3,
                    s**2 - v1**2 + v2**2 - v3**2,
                    2 * v2 * v3 - 2 * s * v1,
                ],
                [
                    2 * v1 * v3 - 2 * s * v2,
                    2 * v2 * v3 + 2 * s * v1,
                    s**2 - v1**2 - v2**2 + v3**2,
                ],
            ]
        )

    def to_rotation_matrix(self) -> np.ndarray:
        """Alias for R() method"""
        return self.R()

    def T(self) -> np.ndarray:  # noqa: N802
        """
        Transformation matrix from quaternion.
        Implements ROBOOP: ReturnMatrix T()const
        """
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = self.R()
        return transformation_matrix

    def to_transformation_matrix(self) -> np.ndarray:
        """Alias for T() method"""
        return self.T()

    def to_euler_angles(self) -> tuple[float, float, float]:
        """Convert quaternion to Euler angles (roll, pitch, yaw) in radians"""
        s, v1, v2, v3 = self.s_, self.v_[0], self.v_[1], self.v_[2]

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (s * v1 + v2 * v3)
        cosr_cosp = 1 - 2 * (v1 * v1 + v2 * v2)
        roll = np.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (s * v2 - v3 * v1)
        pitch = np.copysign(np.pi / 2, sinp) if abs(sinp) >= 1 else np.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (s * v3 + v1 * v2)
        cosy_cosp = 1 - 2 * (v2 * v2 + v3 * v3)
        yaw = np.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def to_axis_angle(self) -> tuple[np.ndarray, float]:
        """Convert quaternion to axis-angle representation"""
        if abs(self.s_) >= 1.0:
            return np.array([1.0, 0.0, 0.0]), 0.0

        angle = 2.0 * np.acos(abs(self.s_))
        sin_half_angle = np.sqrt(1.0 - self.s_**2)

        if sin_half_angle < self.EPSILON:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = self.v_ / sin_half_angle

        return axis, angle

    # ==================== INTERPOLATION METHODS ====================

    def slerp(self, other: "Quaternion", t: float) -> "Quaternion":
        """
        Spherical Linear Interpolation (Slerp).
        Implements ROBOOP: Quaternion Slerp(const Quaternion & q0, const Quaternion & q1,
                                           const Real t)
        """
        if t < 0 or t > 1:
            print("Slerp(q0, q1, t): t < 0 or t > 1. t is set to 0.")
            t = max(0, min(1, t))

        if t == 0:
            return Quaternion(self.s_, *self.v_)
        if t == 1:
            return Quaternion(other.s_, *other.v_)

        # Choose sign to avoid extra spinning
        if self.dot_prod(other) >= 0:
            return self * (self.i() * other).power(t)
        return self * (self.i() * (-other)).power(t)

    @staticmethod
    def Slerp(q0: "Quaternion", q1: "Quaternion", t: float) -> "Quaternion":  # noqa: N802
        """Static version of slerp - implements ROBOOP interface exactly"""
        return q0.slerp(q1, t)

    def slerp_prime(self, other: "Quaternion", t: float) -> "Quaternion":
        """
        Spherical Linear Interpolation derivative.
        Implements ROBOOP: Quaternion Slerp_prime(const Quaternion & q0, const Quaternion & q1,
                                                  const Real t)
        """
        if t < 0 or t > 1:
            print("Slerp_prime(q0, q1, t): t < 0 or t > 1. t is set to 0.")
            t = max(0, min(1, t))

        q_rel = self.i() * other if self.dot_prod(other) >= 0 else self.i() * (-other)

        return self.slerp(other, t) * q_rel.Log()

    @staticmethod
    def Slerp_prime(q0: "Quaternion", q1: "Quaternion", t: float) -> "Quaternion":  # noqa: N802
        """Static version of slerp_prime"""
        return q0.slerp_prime(q1, t)

    @staticmethod
    def Squad(  # noqa: N802
        p: "Quaternion", a: "Quaternion", b: "Quaternion", q: "Quaternion", t: float
    ) -> "Quaternion":
        """
        Spherical Cubic Interpolation (Squad).
        Implements ROBOOP: Quaternion Squad(const Quaternion & p, const Quaternion & a,
                                           const Quaternion & b, const Quaternion & q, const Real t)
        """
        if t < 0 or t > 1:
            print("Squad(p,a,b,q, t): t < 0 or t > 1. t is set to 0.")
            t = max(0, min(1, t))

        return Quaternion.Slerp(
            Quaternion.Slerp(p, q, t), Quaternion.Slerp(a, b, t), 2 * t * (1 - t)
        )

    @staticmethod
    def Squad_prime(  # noqa: N802
        p: "Quaternion", a: "Quaternion", b: "Quaternion", q: "Quaternion", t: float
    ) -> "Quaternion":
        """
        Spherical Cubic Interpolation derivative.
        Implements ROBOOP: Quaternion Squad_prime(const Quaternion & p, const Quaternion & a,
                                                 const Quaternion & b, const Quaternion & q,
                                                 const Real t)
        """
        if t < 0 or t > 1:
            print("Squad_prime(p,a,b,q, t): t < 0 or t > 1. t is set to 0.")
            t = max(0, min(1, t))

        u_interp = Quaternion.Slerp(p, q, t)
        v_interp = Quaternion.Slerp(a, b, t)
        w_interp = u_interp.i() * v_interp
        u_prime = u_interp * (p.i() * q).Log()
        v_prime = v_interp * (a.i() * b).Log()
        w_prime = u_interp.i() * v_prime - u_interp.power(-2) * u_prime * v_interp

        return u_interp * (
            w_interp.power(2 * t * (1 - t)) * w_interp.Log() * (2 - 4 * t)
            + w_interp.power(2 * t * (1 - t) - 1) * w_prime * 2 * t * (1 - t)
        ) + u_prime * w_interp.power(2 * t * (1 - t))

    @staticmethod
    def compute_intermediate_quaternion(
        q_prev: "Quaternion", q_curr: "Quaternion", q_next: "Quaternion"
    ) -> "Quaternion":
        """
        Compute intermediate quaternion for Squad interpolation.
        s_i = q_i * exp(-(log(q_i^(-1) * q_{i+1}) + log(q_i^(-1) * q_{i-1})) / 4)
        """
        log_next = (q_curr.i() * q_next).Log()
        log_prev = (q_curr.i() * q_prev).Log()
        return q_curr * ((log_next + log_prev) * (-0.25)).exp()

    # ==================== PROPERTIES AND UTILITIES ====================

    def s(self) -> float:
        """Get scalar part - implements ROBOOP interface"""
        return self.s_

    def v(self) -> np.ndarray:
        """Get vector part - implements ROBOOP interface"""
        return self.v_.copy()

    def set_s(self, s: float) -> None:
        """Set scalar part - implements ROBOOP interface"""
        self.s_ = float(s)

    def _validate_vector_dimension(self, v: np.ndarray) -> None:
        """Validate that vector has correct dimension."""
        if len(v) != self.VECTOR_DIM:
            raise ValueError("Quaternion::set_v: input has a wrong size.")

    def set_v(self, v: np.ndarray) -> None:
        """Set vector part - implements ROBOOP interface"""
        self._validate_vector_dimension(v)
        self.v_ = np.array(v, dtype=float)

    @property
    def w(self) -> float:
        """Alias for scalar part (w component)"""
        return self.s_

    @property
    def x(self) -> float:
        """Get x component of vector part"""
        return self.v_[0]

    @property
    def y(self) -> float:
        """Get y component of vector part"""
        return self.v_[1]

    @property
    def z(self) -> float:
        """Get z component of vector part"""
        return self.v_[2]

    def copy(self) -> "Quaternion":
        """Create a copy of this quaternion"""
        return Quaternion(self.s_, *self.v_)

    def __str__(self) -> str:
        """String representation"""
        return (
            f"Quaternion(w={self.s_:.6f}, x={self.v_[0]:.6f}, "
            f"y={self.v_[1]:.6f}, z={self.v_[2]:.6f})"
        )

    def __repr__(self) -> str:
        """Detailed string representation"""
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        """Equality comparison with epsilon tolerance"""
        if not isinstance(other, Quaternion):
            return NotImplemented
        return abs(self.s_ - other.s_) < self.EPSILON and np.allclose(
            self.v_, other.v_, atol=self.EPSILON
        )

    # ==================== VISUALIZATION METHODS ====================

    def plot_on_sphere(
        self,
        ax: plt.Axes | None = None,
        color: str = "red",
        size: int = 100,
        label: str = "",
    ) -> plt.Axes:
        """Plot quaternion as point on unit sphere"""
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

        q_norm = self.unit()
        ax.scatter(
            q_norm.v_[0],
            q_norm.v_[1],
            q_norm.v_[2],
            color=color,
            s=size,
            label=label,
            alpha=0.8,
        )
        return ax

    @staticmethod
    def create_sphere_wireframe(ax: plt.Axes, alpha: float = 0.1) -> None:
        """Add wireframe sphere to the plot"""
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_wireframe(x, y, z, alpha=alpha, color="gray")


class QuaternionSpline:
    """
    Quaternion spline interpolation with file I/O capabilities.
    Implements ROBOOP's Spl_Quaternion class functionality.
    """

    def __init__(
        self,
        time_points: list[float] | None = None,
        quaternions: list[Quaternion] | None = None,
        filename: str | None = None,
    ) -> None:
        """
        Initialize quaternion spline.

        Args:
            time_points: List of time values (must be sorted)
            quaternions: List of quaternions at each time point
            filename: File containing quaternion spline data
        """
        self.quat_data: dict[float, Quaternion] = {}
        self.intermediate_quaternions: dict[float, Quaternion] = {}

        if filename is not None:
            self._load_from_file(filename)
        elif time_points is not None and quaternions is not None:
            QuaternionSpline._validate_input_data(time_points, quaternions)
            # Create ordered dictionary
            sorted_data = sorted(zip(time_points, quaternions))
            self.quat_data = OrderedDict(sorted_data)
        else:
            raise ValueError(
                "Must provide either (time_points, quaternions) or filename"
            )

        # Precompute intermediate quaternions for Squad
        self._compute_intermediate_quaternions()

    @staticmethod
    def _validate_input_data(
        time_points: list[float], quaternions: list[Quaternion]
    ) -> None:
        """Validate input data for spline construction."""
        if len(time_points) != len(quaternions):
            raise ValueError("Time points and quaternions must have same length")
        if len(time_points) < Quaternion.MIN_INTERPOLATION_POINTS:
            raise ValueError("Need at least 2 points for interpolation")

    @staticmethod
    def _validate_rotation_matrix_values(values: list[float]) -> None:
        """Validate rotation matrix values."""
        if len(values) != Quaternion.ROTATION_MATRIX_ELEMENTS:
            msg = (
                f"Rotation matrix requires {Quaternion.ROTATION_MATRIX_ELEMENTS} "
                f"elements, got {len(values)}"
            )
            raise ValueError(msg)

    @staticmethod
    def _validate_quaternion_values(values: list[float]) -> None:
        """Validate quaternion values."""
        if len(values) != Quaternion.QUATERNION_ELEMENTS:
            msg = (
                f"Quaternion requires {Quaternion.QUATERNION_ELEMENTS} "
                f"elements, got {len(values)}"
            )
            raise ValueError(msg)

    @staticmethod
    def _validate_type_line(type_line: str) -> None:
        """Validate type line from file."""
        if type_line not in {"R", "Q"}:
            raise ValueError(f"Unknown type '{type_line}', expected 'R' or 'Q'")

    def _load_from_file(self, filename: str) -> None:
        """
        Load quaternion spline data from file.
        Implements ROBOOP: Spl_Quaternion(const string & filename)

        File format examples:
        # Comments start with #
        0
        R
        1.000000 0.000000 0.000000 0.000000 -1.000000 0.000000 0.000000 -0.000000 -1.000000
        0.5
        Q
        0.999943 0.008696 -0.006149 0.058659
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"Spl_Quaternion::Spl_Quaternion: can not open file {filename}"
            )

        self.quat_data = OrderedDict()

        with open(filename, encoding="utf-8") as file:
            lines = file.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Skip comments and empty lines
            if line.startswith("#") or not line:
                i += 1
                continue

            try:
                # Parse time
                time = float(line)
                i += 1

                if i >= len(lines):
                    break

                # Parse type (R for rotation matrix, Q for quaternion)
                type_line = lines[i].strip().upper()
                i += 1

                if i >= len(lines):
                    break

                # Parse data
                data_line = lines[i].strip()
                values = list(map(float, data_line.split()))
                i += 1

                if type_line == "R":
                    # Rotation matrix (9 elements)
                    QuaternionSpline._validate_rotation_matrix_values(values)
                    rotation_matrix = np.array(values).reshape(3, 3)
                    q = Quaternion.from_rotation_matrix(rotation_matrix)
                elif type_line == "Q":
                    # Quaternion (4 elements: w, x, y, z)
                    QuaternionSpline._validate_quaternion_values(values)
                    q = Quaternion(values[0], values[1], values[2], values[3])
                else:
                    QuaternionSpline._validate_type_line(type_line)

                self.quat_data[time] = q

            except (ValueError, IndexError) as e:
                print(
                    f"Spl_Quaternion::Spl_Quaternion: format of input file {filename} is incorrect"
                )
                print(f"Error at line {i}: {e}")
                raise

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

            self.intermediate_quaternions[t_curr] = (
                Quaternion.compute_intermediate_quaternion(q_prev, q_curr, q_next)
            )

    def quat(self, t: float) -> tuple[Quaternion, int]:
        """
        Quaternion interpolation.
        Implements ROBOOP: short quat(const Real t, Quaternion & s)

        Returns: (interpolated_quaternion, status_code)
        """
        times = list(self.quat_data.keys())

        if not times:
            return Quaternion.identity(), -1

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

                # Use Slerp for first and last segments, Squad for interior
                if (
                    i == 0
                    or i == len(times) - 2
                    or len(times) < Quaternion.MIN_SQUAD_POINTS
                ):
                    return Quaternion.Slerp(q0, q1, dt), 0
                # Squad interpolation
                a = self.intermediate_quaternions[t0]
                b = self.intermediate_quaternions[t1]
                return Quaternion.Squad(q0, a, b, q1, dt), 0

        print("Spl_Quaternion::quat: t not in range.")
        return Quaternion.identity(), -3  # NOT_IN_RANGE

    def quat_w(self, t: float) -> tuple[Quaternion, np.ndarray, int]:
        """
        Quaternion interpolation and angular velocity.
        Implements ROBOOP: short quat_w(const Real t, Quaternion & s, ColumnVector & w)

        Returns: (interpolated_quaternion, angular_velocity, status_code)
        """
        q, status = self.quat(t)
        if status != 0:
            return q, np.zeros(3), status

        # Compute derivative using finite differences
        dt = 1e-6
        times = list(self.quat_data.keys())

        if t + dt <= times[-1]:
            q_next, _ = self.quat(t + dt)
            dq = (q_next - q) * (1.0 / dt)
        elif t - dt >= times[0]:
            q_prev, _ = self.quat(t - dt)
            dq = (q - q_prev) * (1.0 / dt)
        else:
            return q, np.zeros(3), 0

        # Convert to angular velocity
        w = Quaternion.Omega(q, dq)

        return q, w, 0

    def get_time_range(self) -> tuple[float, float]:
        """Get the time range of the spline"""
        times = list(self.quat_data.keys())
        return times[0], times[-1]

    def save_to_file(self, filename: str, format_type: str = "Q") -> None:
        """
        Save quaternion spline to file.

        Args:
            filename: Output file name
            format_type: 'Q' for quaternion format, 'R' for rotation matrix format
        """
        with open(filename, "w", encoding="utf-8") as f:
            f.write("# Quaternion spline data\n")
            f.write("# Format: time, type, data\n")
            f.write("#\n")

            for time, quat in self.quat_data.items():
                f.write(f"{time}\n")
                f.write(f"{format_type}\n")

                if format_type.upper() == "Q":
                    f.write(f"{quat.s_} {quat.v_[0]} {quat.v_[1]} {quat.v_[2]}\n")
                elif format_type.upper() == "R":
                    rotation_matrix = quat.R()
                    f.write(
                        " ".join(
                            [
                                f"{rotation_matrix[i, j]}"
                                for i in range(3)
                                for j in range(3)
                            ]
                        )
                        + "\n"
                    )

    def plot_spline(
        self,
        num_samples: int = 100,
        ax: plt.Axes | None = None,
        show_control_points: bool = True,
        show_sphere: bool = True,
        title: str = "Quaternion Spline",
    ) -> plt.Figure:
        """Visualize the quaternion spline"""
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.figure

        # Sample the spline
        times = list(self.quat_data.keys())
        t_start, t_end = times[0], times[-1]
        t_samples = np.linspace(t_start, t_end, num_samples)

        path_points = []
        for t in t_samples:
            q, _ = self.quat(t)
            path_points.append(q.unit().v_)

        path_points_array = np.array(path_points)

        # Plot spline curve
        ax.plot(
            path_points_array[:, 0],
            path_points_array[:, 1],
            path_points_array[:, 2],
            "b-",
            linewidth=3,
            alpha=0.8,
            label="Spline curve",
        )

        # Plot control points
        if show_control_points:
            control_points = np.array([q.unit().v_ for q in self.quat_data.values()])
            ax.scatter(
                control_points[:, 0],
                control_points[:, 1],
                control_points[:, 2],
                color="red",
                s=150,
                alpha=0.9,
                label="Control points",
                zorder=5,
            )

            # Add time labels
            for t, q in self.quat_data.items():
                pos = q.unit().v_
                ax.text(pos[0], pos[1], pos[2], f"t={t:.2f}", fontsize=8)

        if show_sphere:
            Quaternion.create_sphere_wireframe(ax)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)
        ax.legend()
        ax.set_box_aspect([1, 1, 1])

        return fig
