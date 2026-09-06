"""Multiple shooting for natural Riemannian cubics on unit quaternions.

Each segment uses local time in [0, 1] and nine unknowns (v, a, c).
The pure-quaternion Lie bracket is [v, a] = 2 cross(v, a).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


_SMALL_ANGLE = 1e-7


def skew(values: np.ndarray) -> np.ndarray:
    """Return a batch of cross-product matrices."""
    result = np.zeros((*values.shape[:-1], 3, 3))
    result[..., 0, 1] = -values[..., 2]
    result[..., 0, 2] = values[..., 1]
    result[..., 1, 0] = values[..., 2]
    result[..., 1, 2] = -values[..., 0]
    result[..., 2, 0] = -values[..., 1]
    result[..., 2, 1] = values[..., 0]
    return result


def left_matrix(quaternions: np.ndarray) -> np.ndarray:
    """Matrices for left quaternion multiplication, in scalar-first order."""
    result = np.zeros((len(quaternions), 4, 4))
    result[:, 0, 0] = quaternions[:, 0]
    result[:, 0, 1:] = -quaternions[:, 1:]
    result[:, 1:, 0] = quaternions[:, 1:]
    result[:, 1:, 1:] = quaternions[:, :1, None] * np.eye(3) + skew(quaternions[:, 1:])
    return result


def rotation_residual(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quaternion logarithms and their analytic derivatives, modulo sign."""
    signs = np.where(values[:, 0] < 0.0, -1.0, 1.0)
    canonical = values * signs[:, None]
    scalar, vector = canonical[:, 0], canonical[:, 1:]
    radius = np.linalg.norm(vector, axis=1)
    squared_norm = scalar**2 + radius**2
    small = radius < _SMALL_ANGLE
    factor = np.empty(len(values))
    coefficient = np.empty(len(values))
    factor[small] = 1.0 / scalar[small] - radius[small] ** 2 / (3.0 * scalar[small] ** 3)
    coefficient[small] = -2.0 / (3.0 * scalar[small] ** 3)
    factor[~small] = np.arctan2(radius[~small], scalar[~small]) / radius[~small]
    coefficient[~small] = (scalar[~small] / squared_norm[~small] - factor[~small]) / radius[~small] ** 2
    derivative = np.empty((len(values), 3, 4))
    derivative[:, :, 0] = -vector / squared_norm[:, None]
    derivative[:, :, 1:] = (
        factor[:, None, None] * np.eye(3) + coefficient[:, None, None] * vector[:, :, None] * vector[:, None, :]
    )
    return factor[:, None] * vector, derivative * signs[:, None, None]


def _rhs(
    state: np.ndarray, constants: np.ndarray, sensitivity: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Evaluate the cubic ODE and its variational equations."""
    quaternion, velocity, acceleration = state[:, :4], state[:, 4:7], state[:, 7:10]
    result = np.empty_like(state)
    result[:, 0] = -np.sum(quaternion[:, 1:] * velocity, axis=1)
    result[:, 1:4] = quaternion[:, :1] * velocity + np.cross(quaternion[:, 1:], velocity)
    result[:, 4:7] = acceleration
    result[:, 7:10] = constants - 2.0 * np.cross(velocity, acceleration)
    if sensitivity is None:
        return result, None

    right = np.zeros((len(state), 4, 4))
    right[:, 0, 1:] = -velocity
    right[:, 1:, 0] = velocity
    right[:, 1:, 1:] = -skew(velocity)
    product = left_matrix(quaternion)[:, :, 1:]
    derivative = np.empty_like(sensitivity)
    derivative[:, :4] = right @ sensitivity[:, :4] + product @ sensitivity[:, 4:7]
    derivative[:, 4:7] = sensitivity[:, 7:10]
    derivative[:, 7:10] = 2.0 * skew(acceleration) @ sensitivity[:, 4:7] - 2.0 * skew(velocity) @ sensitivity[:, 7:10]
    derivative[:, 7:10, 6:9] += np.eye(3)
    return result, derivative


def _advance_sensitivity(
    sensitivity: np.ndarray | None, derivative: np.ndarray | None, step: float
) -> np.ndarray | None:
    if sensitivity is None:
        return None
    assert derivative is not None
    return sensitivity + step * derivative


def rk4_step(
    state: np.ndarray,
    constants: np.ndarray,
    step: float,
    sensitivity: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Advance states and optional sensitivities; accumulate squared acceleration."""
    k1, d1 = _rhs(state, constants, sensitivity)
    y2 = state + 0.5 * step * k1
    k2, d2 = _rhs(y2, constants, _advance_sensitivity(sensitivity, d1, 0.5 * step))
    y3 = state + 0.5 * step * k2
    k3, d3 = _rhs(y3, constants, _advance_sensitivity(sensitivity, d2, 0.5 * step))
    y4 = state + step * k3
    k4, d4 = _rhs(y4, constants, _advance_sensitivity(sensitivity, d3, step))
    result = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    norm = np.linalg.norm(result[:, :4], axis=1)
    result[:, :4] /= norm[:, None]
    if sensitivity is not None:
        assert d1 is not None
        assert d2 is not None
        assert d3 is not None
        assert d4 is not None
        updated = sensitivity + (step / 6.0) * (d1 + 2.0 * d2 + 2.0 * d3 + d4)
        quaternion = result[:, :4]
        radial = np.einsum("mi,mij->mj", quaternion, updated[:, :4])
        updated[:, :4] = (updated[:, :4] - quaternion[:, :, None] * radial[:, None, :]) / norm[:, None, None]
        sensitivity = updated
    energy = (step / 6.0) * (
        np.sum(state[:, 7:10] ** 2, axis=1)
        + 2.0 * np.sum(y2[:, 7:10] ** 2, axis=1)
        + 2.0 * np.sum(y3[:, 7:10] ** 2, axis=1)
        + np.sum(y4[:, 7:10] ** 2, axis=1)
    )
    return result, sensitivity, energy


def integrate(
    parameters: np.ndarray,
    starts: np.ndarray,
    steps: int,
    *,
    jacobian: bool = False,
    store_nodes: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Integrate every shooting segment together on its local unit interval."""
    state = np.column_stack((starts, parameters[:, :6]))
    sensitivity = None
    if jacobian:
        sensitivity = np.zeros((len(starts), 10, 9))
        sensitivity[:, 4:10, :6] = np.eye(6)
    nodes = [state.copy()] if store_nodes else []
    energy = np.zeros(len(starts))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(steps):
            state, sensitivity, increment = rk4_step(state, parameters[:, 6:9], 1.0 / steps, sensitivity)
            energy += increment
            if not np.all(np.isfinite(state)):
                raise RuntimeError("Multiple shooting integration diverged")
            if store_nodes:
                nodes.append(state.copy())
    return state, sensitivity, np.asarray(nodes), energy


def matching_system(
    parameters: np.ndarray,
    quaternions: np.ndarray,
    durations: np.ndarray,
    steps: int,
    *,
    jacobian: bool = False,
) -> tuple[np.ndarray, csc_matrix | None]:
    """Assemble orientation, velocity, acceleration and natural boundary residuals."""
    count = len(parameters)
    ends, sensitivity, _, _ = integrate(parameters, quaternions[:-1], steps, jacobian=jacobian)
    conjugates = quaternions[1:].copy()
    conjugates[:, 1:] *= -1.0
    rotations = left_matrix(conjugates)
    errors = np.einsum("mij,mj->mi", rotations, ends[:, :4])
    orientation, log_derivative = rotation_residual(errors)
    # Harmonic-mean matching weights, without multiplying physical durations
    # (which can underflow/overflow even when both spacings are finite).
    scale = np.maximum(durations[:-1], durations[1:])
    left_duration, right_duration = durations[:-1] / scale, durations[1:] / scale
    total = left_duration + right_duration
    left_scale, right_scale = 2.0 * right_duration / total, 2.0 * left_duration / total
    velocity = left_scale[:, None] * ends[:-1, 4:7] - right_scale[:, None] * parameters[1:, :3]
    acceleration = left_scale[:, None] ** 2 * ends[:-1, 7:10] - right_scale[:, None] ** 2 * parameters[1:, 3:6]
    residual = np.concatenate((
        orientation.ravel(),
        velocity.ravel(),
        acceleration.ravel(),
        parameters[0, 3:6],
        ends[-1, 7:10],
    ))
    if not np.all(np.isfinite(residual)):
        raise RuntimeError("Multiple shooting matching residual is not finite")
    if sensitivity is None:
        return residual, None

    matrix = lil_matrix((9 * count, 9 * count))
    orientation_derivative = log_derivative @ rotations @ sensitivity[:, :4]
    for index in range(count):
        columns = slice(9 * index, 9 * (index + 1))
        matrix[3 * index : 3 * index + 3, columns] = orientation_derivative[index]
        if index + 1 < count:
            row = 3 * count + 3 * index
            matrix[row : row + 3, columns] = left_scale[index] * sensitivity[index, 4:7]
            matrix[row : row + 3, 9 * (index + 1) : 9 * (index + 1) + 3] = -right_scale[index] * np.eye(3)
            row += 3 * (count - 1)
            matrix[row : row + 3, columns] = left_scale[index] ** 2 * sensitivity[index, 7:10]
            matrix[row : row + 3, 9 * (index + 1) + 3 : 9 * (index + 1) + 6] = -(right_scale[index] ** 2) * np.eye(3)
    matrix[-6:-3, 3:6] = np.eye(3)
    matrix[-3:, -9:] = sensitivity[-1, 7:10]
    return residual, matrix.tocsc()


def newton_solve(  # noqa: PLR0913
    parameters: np.ndarray,
    quaternions: np.ndarray,
    durations: np.ndarray,
    steps: int,
    tolerance: float,
    iterations: int,
) -> tuple[np.ndarray, int]:
    """Solve the sparse square matching system with damped Newton steps."""
    for iteration in range(iterations + 1):
        residual, jacobian = matching_system(parameters, quaternions, durations, steps, jacobian=True)
        if np.max(np.abs(residual)) <= tolerance:
            return parameters, iteration
        if iteration == iterations:
            break
        direction = spsolve(jacobian, -residual).reshape(-1, 9)
        if not np.all(np.isfinite(direction)):
            raise RuntimeError("Multiple shooting Jacobian is singular")
        step = 1.0
        for _ in range(20):
            candidate = parameters + step * direction
            try:
                trial, _ = matching_system(candidate, quaternions, durations, steps)
            except RuntimeError:
                step *= 0.5
                continue
            if np.linalg.norm(trial) < (1.0 - 1e-4 * step) * np.linalg.norm(residual):
                parameters = candidate
                break
            step *= 0.5
        else:
            raise RuntimeError("Multiple shooting line search failed to reduce the matching residual")
    raise RuntimeError("Multiple shooting did not converge within max_iterations")
