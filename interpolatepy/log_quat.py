"""
Logarithmic Quaternion Interpolation.

Implements two methods from Parker et al. (2023) for smooth quaternion
trajectory generation via B-spline interpolation in axis-angle space:

- :class:`LogQuaternionInterpolation` (LQI): interpolates the rotation
  vector r = θ * n̂ as a single 3D B-spline.
- :class:`ModifiedLogQuaternionInterpolation` (mLQI): interpolates θ and
  the unit axis (X, Y, Z) as separate B-splines for better numerical
  stability.
"""

from __future__ import annotations

import warnings

import numpy as np

from .b_spline_interpolate import BSplineInterpolator
from .quat_core import Quaternion


_EPSILON = 1e-10
_DEFAULT_DEGREE = 3
_VALID_DEGREES = {3, 4, 5}
_MIN_QUATERNIONS = 2


def _validate_inputs(
    time_points: np.ndarray,
    quaternions: list[Quaternion],
    degree: int,
) -> None:
    if len(time_points) != len(quaternions):
        raise ValueError("Number of time points must match number of quaternions")
    if len(quaternions) < _MIN_QUATERNIONS:
        raise ValueError("At least 2 quaternions are required for interpolation")
    if degree not in _VALID_DEGREES:
        raise ValueError(f"Degree must be 3, 4, or 5, got {degree}")
    if len(quaternions) < degree + 1:
        raise ValueError(
            f"Not enough quaternions for degree {degree} B-spline interpolation. "
            f"Need at least {degree + 1} quaternions, got {len(quaternions)}"
        )
    if not np.all(np.diff(time_points) > 0):
        raise ValueError("Time points must be strictly increasing")

    for i, q in enumerate(quaternions):
        if not isinstance(q, Quaternion):
            raise TypeError(f"Element {i} is not a Quaternion instance")
        norm = q.norm()
        if abs(norm - 1.0) > _EPSILON:
            warnings.warn(
                f"Quaternion {i} is not unit (norm={norm:.6f}), normalizing.",
                UserWarning,
                stacklevel=3,
            )
            quaternions[i] = q.unit()


def _extract_axis_angle_raw(q: Quaternion) -> tuple[np.ndarray, float]:
    """
    Axis-angle extraction without canonicalising the sign of the scalar
    part, so the angle ranges over [0, 2π] instead of [0, π]. Downstream
    unwrap/branch-tracking steps need the original branch to detect
    rotations past 180°.
    """
    s = q.s_
    sin_half = np.sqrt(max(0.0, 1.0 - s * s))
    axis = np.array([1.0, 0.0, 0.0]) if sin_half < _EPSILON else q.v_ / sin_half
    angle = 2.0 * np.arccos(np.clip(s, -1.0, 1.0))
    return axis, angle


def _canonicalize_double_cover(quaternions: list[Quaternion]) -> None:
    """Flip each q so it stays in the same hemisphere as its predecessor."""
    for i in range(1, len(quaternions)):
        prev = quaternions[i - 1]
        if prev.dot_product(-quaternions[i]) > prev.dot_product(quaternions[i]):
            quaternions[i] = -quaternions[i]


def _omega_alpha(  # noqa: PLR0913
    theta: float,
    theta_dot: float,
    theta_ddot: float,
    u: np.ndarray,
    u_dot: np.ndarray,
    u_ddot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Left-Jacobian / dexp expansion of (θ, u, u̇, ü) into physical angular
    velocity and acceleration. Assumes u is a unit vector and u̇, ü are
    tangent to the unit sphere; the formula is exact in that case.
    """
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    one_minus_cos = 1.0 - cos_t
    one_plus_cos = 1.0 + cos_t

    cross_u_udot = np.cross(u, u_dot)
    cross_u_uddot = np.cross(u, u_ddot)

    omega = u * theta_dot + u_dot * sin_t + cross_u_udot * one_minus_cos
    alpha = (
        u * theta_ddot
        + u_dot * (theta_dot * one_plus_cos)
        + u_ddot * sin_t
        + cross_u_udot * (theta_dot * sin_t)
        + cross_u_uddot * one_minus_cos
    )
    return omega, alpha


class LogQuaternionInterpolation:
    """
    Logarithmic Quaternion Interpolation (LQI) using axis-angle representation.

    Transforms quaternions to axis-angle space r = θ*n̂ and interpolates the
    3D vector with a B-spline (Parker et al. 2023). Algorithm 1 from the
    paper resolves quaternion double-cover and axis-angle discontinuities so
    the recovered r(t) is continuous and the interpolation is C².

    Parameters
    ----------
    time_points : array_like
        Time values corresponding to each quaternion (must be strictly increasing).
    quaternions : array_like
        List of unit quaternions to interpolate between.
    degree : int, optional
        Degree of the B-spline (3, 4, or 5). Default is 3 (cubic).
    initial_velocity, final_velocity : array_like, optional
        Initial/final angular velocity constraints (3D vectors).
    initial_acceleration, final_acceleration : array_like, optional
        Initial/final angular acceleration constraints (3D vectors).
    """

    EPSILON = _EPSILON
    DEFAULT_DEGREE = _DEFAULT_DEGREE

    def __init__(  # noqa: PLR0913
        self,
        time_points: list | np.ndarray,
        quaternions: list[Quaternion],
        degree: int = _DEFAULT_DEGREE,
        initial_velocity: list | np.ndarray | None = None,
        final_velocity: list | np.ndarray | None = None,
        initial_acceleration: list | np.ndarray | None = None,
        final_acceleration: list | np.ndarray | None = None,
    ) -> None:
        self.time_points = np.array(time_points, dtype=np.float64)
        self.quaternions = list(quaternions)
        self.degree = degree

        _validate_inputs(self.time_points, self.quaternions, degree)

        axis_angle_vectors = self._recover_continuous_axis_angle()

        self.bspline_interpolator = BSplineInterpolator(
            degree=degree,
            points=axis_angle_vectors,
            times=self.time_points,
            initial_velocity=initial_velocity,
            final_velocity=final_velocity,
            initial_acceleration=initial_acceleration,
            final_acceleration=final_acceleration,
        )

        self.t_min = self.time_points[0]
        self.t_max = self.time_points[-1]

    def _check_time(self, t: float) -> float:
        if t < self.t_min - _EPSILON or t > self.t_max + _EPSILON:
            raise ValueError(f"Time {t} outside valid range [{self.t_min}, {self.t_max}]")
        return float(np.clip(t, self.t_min, self.t_max))

    def _recover_continuous_axis_angle(self) -> np.ndarray:
        """
        Algorithm 1 from Parker et al. (2023): produce a continuous
        axis-angle series by resolving double-cover, flipping axes to keep
        them continuous, and unwrapping the angle around ±2π.
        """
        n = len(self.quaternions)
        axes: list[np.ndarray] = []
        angles: list[float] = []
        for q in self.quaternions:
            axis, angle = _extract_axis_angle_raw(q)
            axes.append(axis)
            angles.append(angle)

        for i in range(1, n):
            q_neg = -self.quaternions[i]
            if self.quaternions[i - 1].dot_product(q_neg) > self.quaternions[i - 1].dot_product(
                self.quaternions[i]
            ):
                self.quaternions[i] = q_neg
                axes[i], angles[i] = _extract_axis_angle_raw(q_neg)

            # r = θ*n̂ is invariant under (θ, n̂) → (-θ, -n̂); the flip lets
            # the subsequent unwrap see a continuous angle sequence.
            if np.linalg.norm(axes[i - 1] - axes[i]) > np.linalg.norm(axes[i - 1] + axes[i]):
                angles[i] = -angles[i]
                axes[i] = -axes[i]

        unwrapped = np.unwrap(angles)
        axes_arr = np.array(axes)
        r = unwrapped[:, None] * axes_arr
        # Zero out vectors whose magnitude collapsed below EPSILON so the
        # fallback axis [1,0,0] doesn't bleed in as a spurious direction.
        r[np.abs(unwrapped) < _EPSILON] = 0.0
        return r

    def evaluate(self, t: float) -> Quaternion:
        """Evaluate the interpolated quaternion at time ``t``."""
        t = self._check_time(t)
        if abs(t - self.t_min) <= _EPSILON:
            return self.quaternions[0].copy()
        if abs(t - self.t_max) <= _EPSILON:
            return self.quaternions[-1].copy()

        r = self.bspline_interpolator.evaluate(t)
        theta = float(np.linalg.norm(r))
        if theta < _EPSILON:
            return Quaternion.identity()
        return Quaternion.from_angle_axis(theta, r / theta)

    def evaluate_velocity(self, t: float) -> np.ndarray:
        """Time-derivative of the rotation vector r(t) (3D)."""
        t = self._check_time(t)
        return self.bspline_interpolator.evaluate_derivative(t, order=1)

    def evaluate_acceleration(self, t: float) -> np.ndarray:
        """Second time-derivative of the rotation vector r(t) (3D)."""
        t = self._check_time(t)
        return self.bspline_interpolator.evaluate_derivative(t, order=2)

    def generate_trajectory(self, num_points: int = 100) -> tuple[np.ndarray, list[Quaternion]]:
        """Evaluate the trajectory at ``num_points`` evenly spaced times."""
        time_values = np.linspace(self.t_min, self.t_max, num_points)
        return time_values, [self.evaluate(t) for t in time_values]

    def get_physical_kinematics(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Physical 3D angular velocity (omega) and acceleration (alpha) at time ``t``.

        The spline interpolates r(t) = theta(t) * n_hat(t). As r -> 0 the
        left Jacobian J_l(r) -> I, so omega -> r_dot and alpha -> r_ddot;
        this limit is used directly when |r| < EPSILON to avoid dividing by
        a near-zero magnitude.
        """
        t = self._check_time(t)
        r = self.bspline_interpolator.evaluate(t)
        r_dot = self.bspline_interpolator.evaluate_derivative(t, order=1)
        r_ddot = self.bspline_interpolator.evaluate_derivative(t, order=2)

        theta = float(np.linalg.norm(r))
        if theta < _EPSILON:
            return r_dot.copy(), r_ddot.copy()

        # Decompose r = θ * u; u̇, ü built this way are automatically
        # tangent to the unit sphere.
        u = r / theta
        theta_dot = float(np.dot(r, r_dot)) / theta
        theta_ddot = (
            float(np.dot(r_dot, r_dot)) + float(np.dot(r, r_ddot)) - theta_dot * theta_dot
        ) / theta
        u_dot = (r_dot - theta_dot * u) / theta
        u_ddot = (r_ddot - theta_ddot * u) / theta - 2.0 * theta_dot * u_dot / theta

        return _omega_alpha(theta, theta_dot, theta_ddot, u, u_dot, u_ddot)


class ModifiedLogQuaternionInterpolation:
    """
    Modified Logarithmic Quaternion Interpolation (mLQI).

    Interpolates quaternions as (θ, X, Y, Z) with X²+Y²+Z²=1, using separate
    B-splines for the scalar angle and the unit-axis components for better
    numerical stability (Parker et al. 2023). Provides C² continuity.

    Parameters
    ----------
    time_points : array_like
        Time values corresponding to each quaternion (must be strictly increasing).
    quaternions : array_like
        List of unit quaternions to interpolate between.
    degree : int, optional
        Degree of the B-spline (3, 4, or 5). Default is 3 (cubic).
    normalize_axis : bool, optional
        If True (default), the spline-evaluated (X, Y, Z) is renormalised
        before reconstructing the quaternion. Setting this to False is only
        appropriate when the spline preserves unit norm exactly.
    initial_velocity, final_velocity : array_like, optional
        Initial/final boundary constraints as 4D vectors [θ̇, Ẋ, Ẏ, Ż].
    initial_acceleration, final_acceleration : array_like, optional
        Initial/final boundary constraints as 4D vectors.
    """

    EPSILON = _EPSILON
    DEFAULT_DEGREE = _DEFAULT_DEGREE

    def __init__(  # noqa: PLR0913
        self,
        time_points: list | np.ndarray,
        quaternions: list[Quaternion],
        degree: int = _DEFAULT_DEGREE,
        normalize_axis: bool = True,
        initial_velocity: list | np.ndarray | None = None,
        final_velocity: list | np.ndarray | None = None,
        initial_acceleration: list | np.ndarray | None = None,
        final_acceleration: list | np.ndarray | None = None,
    ) -> None:
        self.time_points = np.array(time_points, dtype=np.float64)
        self.quaternions = list(quaternions)
        self.degree = degree
        self.normalize_axis = normalize_axis

        _validate_inputs(self.time_points, self.quaternions, degree)
        _canonicalize_double_cover(self.quaternions)

        theta_values, xyz_values = self._transform_to_theta_xyz_space()

        theta_iv, xyz_iv = self._split_4d(initial_velocity)
        theta_fv, xyz_fv = self._split_4d(final_velocity)
        theta_ia, xyz_ia = self._split_4d(initial_acceleration)
        theta_fa, xyz_fa = self._split_4d(final_acceleration)

        self.theta_interpolator = BSplineInterpolator(
            degree=degree,
            points=theta_values.reshape(-1, 1),
            times=self.time_points,
            initial_velocity=theta_iv,
            final_velocity=theta_fv,
            initial_acceleration=theta_ia,
            final_acceleration=theta_fa,
        )

        self.xyz_interpolator = BSplineInterpolator(
            degree=degree,
            points=xyz_values,
            times=self.time_points,
            initial_velocity=xyz_iv,
            final_velocity=xyz_fv,
            initial_acceleration=xyz_ia,
            final_acceleration=xyz_fa,
        )

        self.t_min = self.time_points[0]
        self.t_max = self.time_points[-1]

    @staticmethod
    def _split_4d(
        constraint: list | np.ndarray | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if constraint is None:
            return None, None
        arr = np.asarray(constraint)
        return arr[:1], arr[1:4]

    def _check_time(self, t: float) -> float:
        if t < self.t_min - _EPSILON or t > self.t_max + _EPSILON:
            raise ValueError(f"Time {t} outside valid range [{self.t_min}, {self.t_max}]")
        return float(np.clip(t, self.t_min, self.t_max))

    def _transform_to_theta_xyz_space(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform quaternions to (θ, X, Y, Z) with axis/angle unrolling so
        consecutive samples stay on the same branch.

        The axis is flipped (and the angle replaced by 2π - angle) whenever
        it would otherwise reverse direction relative to the previous sample;
        the angle is then unwrapped modulo 2π so it can accumulate beyond a
        full turn.
        """
        theta_values: list[float] = []
        xyz_values: list[np.ndarray] = []
        prev_axis: np.ndarray | None = None
        for q in self.quaternions:
            axis, angle = _extract_axis_angle_raw(q)
            if prev_axis is not None and np.dot(axis, prev_axis) < 0:
                axis = -axis
                angle = 2.0 * np.pi - angle
            theta_values.append(angle)
            xyz_values.append(axis)
            prev_axis = axis
        return np.unwrap(theta_values), np.array(xyz_values)

    def evaluate(self, t: float) -> Quaternion:
        """Evaluate the interpolated quaternion at time ``t``."""
        t = self._check_time(t)
        if abs(t - self.t_min) <= _EPSILON:
            return self.quaternions[0].copy()
        if abs(t - self.t_max) <= _EPSILON:
            return self.quaternions[-1].copy()

        theta = self.theta_interpolator.evaluate(t)[0]
        xyz = self.xyz_interpolator.evaluate(t)
        if self.normalize_axis:
            norm_xyz = np.linalg.norm(xyz)
            xyz = xyz / norm_xyz if norm_xyz > _EPSILON else np.array([1.0, 0.0, 0.0])
        if abs(theta) < _EPSILON:
            return Quaternion.identity()

        cos_half = np.cos(theta / 2.0)
        sin_half = np.sin(theta / 2.0)
        return Quaternion(cos_half, sin_half * xyz[0], sin_half * xyz[1], sin_half * xyz[2])

    def evaluate_velocity(self, t: float) -> np.ndarray:
        """Derivative of (θ, X, Y, Z) at time ``t`` (4D vector)."""
        t = self._check_time(t)
        theta_dot = self.theta_interpolator.evaluate_derivative(t, order=1)[0]
        xyz_dot = self.xyz_interpolator.evaluate_derivative(t, order=1)
        return np.array([theta_dot, xyz_dot[0], xyz_dot[1], xyz_dot[2]])

    def evaluate_acceleration(self, t: float) -> np.ndarray:
        """Second derivative of (θ, X, Y, Z) at time ``t`` (4D vector)."""
        t = self._check_time(t)
        theta_ddot = self.theta_interpolator.evaluate_derivative(t, order=2)[0]
        xyz_ddot = self.xyz_interpolator.evaluate_derivative(t, order=2)
        return np.array([theta_ddot, xyz_ddot[0], xyz_ddot[1], xyz_ddot[2]])

    def generate_trajectory(self, num_points: int = 100) -> tuple[np.ndarray, list[Quaternion]]:
        """Evaluate the trajectory at ``num_points`` evenly spaced times."""
        time_values = np.linspace(self.t_min, self.t_max, num_points)
        return time_values, [self.evaluate(t) for t in time_values]

    def get_physical_kinematics(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Physical 3D angular velocity (omega) and acceleration (alpha) at time ``t``.

        With ``normalize_axis=True`` the raw xyz-spline derivatives are
        projected tangent to the unit sphere so radial drift in the spline
        does not leak into omega/alpha. With ``normalize_axis=False`` the
        raw derivatives are used as-is and the result is only accurate when
        the spline already preserves unit norm.
        """
        t = self._check_time(t)
        theta = self.theta_interpolator.evaluate(t)[0]
        theta_dot = self.theta_interpolator.evaluate_derivative(t, order=1)[0]
        theta_ddot = self.theta_interpolator.evaluate_derivative(t, order=2)[0]

        u_raw = self.xyz_interpolator.evaluate(t)
        u_dot_raw = self.xyz_interpolator.evaluate_derivative(t, order=1)
        u_ddot_raw = self.xyz_interpolator.evaluate_derivative(t, order=2)

        if self.normalize_axis:
            r = float(np.linalg.norm(u_raw))
            if r < _EPSILON:
                u = np.array([1.0, 0.0, 0.0])
                u_dot = np.zeros(3)
                u_ddot = np.zeros(3)
            else:
                # Map (u_raw, u̇_raw, ü_raw) ∈ R³ to (u, u̇, ü) on the unit
                # sphere by removing the radial component.
                u = u_raw / r
                r_dot = float(np.dot(u, u_dot_raw))
                r_ddot = (
                    float(np.dot(u_dot_raw, u_dot_raw))
                    + float(np.dot(u_raw, u_ddot_raw))
                    - r_dot * r_dot
                ) / r
                u_dot = (u_dot_raw - r_dot * u) / r
                u_ddot = (u_ddot_raw - 2.0 * r_dot * u_dot - r_ddot * u) / r
        else:
            u = u_raw
            u_dot = u_dot_raw
            u_ddot = u_ddot_raw

        return _omega_alpha(theta, theta_dot, theta_ddot, u, u_dot, u_ddot)
