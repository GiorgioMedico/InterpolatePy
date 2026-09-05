"""Linear-system construction for exact B-spline interpolation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import solve

if TYPE_CHECKING:
    from .interpolation import BSplineInterpolator

_ILL_CONDITIONED = 1e12
_REGULARIZATION = 1e-10
_QUARTIC_DEGREE = 4
_QUINTIC_DEGREE = 5


@dataclass
class _InterpolationSystem:
    matrix: np.ndarray
    right_hand_side: np.ndarray
    row: int

    @property
    def row_count(self) -> int:
        return self.matrix.shape[0]


def compute_control_points(
    interpolator: BSplineInterpolator,
    degree: int,
    points: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Build and solve the interpolation and boundary-condition system."""
    segment_count = len(points) - 1
    additional_conditions = degree if degree % 2 == 0 else degree - 1
    control_point_count = len(points) + degree if degree % 2 == 0 else len(points) + degree - 1
    system = _InterpolationSystem(
        matrix=np.zeros((len(points) + additional_conditions, control_point_count)),
        right_hand_side=np.zeros((len(points) + additional_conditions, points.shape[1])),
        row=segment_count + 1,
    )
    _add_interpolation_rows(interpolator, system, degree, points, times)
    if interpolator.cyclic:
        _add_cyclic_rows(interpolator, system, degree, times, additional_conditions)
    else:
        _add_endpoint_rows(interpolator, system, degree, times)
    return _solve_system(interpolator, system, degree, points)


def _add_interpolation_rows(
    interpolator: BSplineInterpolator,
    system: _InterpolationSystem,
    degree: int,
    points: np.ndarray,
    times: np.ndarray,
) -> None:
    """Add rows that force the curve through every interpolation point."""
    for row, time in enumerate(times):
        span = interpolator.temp_spline.find_knot_span(time)
        basis_values = interpolator.temp_spline.basis_functions(time, span)
        for offset in range(degree + 1):
            column = span - degree + offset
            if 0 <= column < system.matrix.shape[1]:
                system.matrix[row, column] = basis_values[offset]
        system.right_hand_side[row] = points[row]


def _add_cyclic_rows(
    interpolator: BSplineInterpolator,
    system: _InterpolationSystem,
    degree: int,
    times: np.ndarray,
    additional_conditions: int,
) -> None:
    """Match start and end derivatives for a cyclic spline."""
    start_time, end_time = times[0], times[-1]
    start_span = interpolator.temp_spline.find_knot_span(start_time)
    end_span = interpolator.temp_spline.find_knot_span(end_time)
    for derivative_order in range(1, additional_conditions + 1):
        start_derivatives = interpolator.temp_spline.basis_function_derivatives(
            start_time, start_span, derivative_order
        )
        end_derivatives = interpolator.temp_spline.basis_function_derivatives(end_time, end_span, derivative_order)
        for offset in range(degree + 1):
            start_column = start_span - degree + offset
            if 0 <= start_column < system.matrix.shape[1]:
                system.matrix[system.row, start_column] = start_derivatives[derivative_order, offset]
            end_column = end_span - degree + offset
            if 0 <= end_column < system.matrix.shape[1]:
                system.matrix[system.row, end_column] = -end_derivatives[derivative_order, offset]
        system.row += 1
        if system.row >= system.row_count:
            break


def _add_endpoint_rows(
    interpolator: BSplineInterpolator,
    system: _InterpolationSystem,
    degree: int,
    times: np.ndarray,
) -> None:
    """Add explicit endpoint constraints followed by natural conditions."""
    explicit_constraints = (
        (times[0], 1, interpolator.initial_velocity),
        (times[-1], 1, interpolator.final_velocity),
        (times[0], 2, interpolator.initial_acceleration),
        (times[-1], 2, interpolator.final_acceleration),
    )
    for time, derivative_order, value in explicit_constraints:
        if value is not None and system.row < system.row_count:
            _add_derivative_row(
                interpolator,
                system,
                time,
                derivative_order,
                value,
            )

    pinned_accelerations = (
        interpolator.initial_acceleration is not None,
        interpolator.final_acceleration is not None,
    )
    candidates = [(2, endpoint) for endpoint in (0, 1) if not pinned_accelerations[endpoint]]
    candidates += [(order, endpoint) for order in range(3, degree) for endpoint in (0, 1)]
    remaining_rows = system.row_count - system.row
    for derivative_order, endpoint in candidates[:remaining_rows]:
        time = times[0] if endpoint == 0 else times[-1]
        _add_derivative_row(interpolator, system, time, derivative_order)


def _add_derivative_row(
    interpolator: BSplineInterpolator,
    system: _InterpolationSystem,
    time: float,
    derivative_order: int,
    value: list | np.ndarray | None = None,
) -> None:
    """Add one endpoint derivative constraint to the current system row."""
    degree = interpolator.temp_spline.degree
    span = interpolator.temp_spline.find_knot_span(time)
    derivatives = interpolator.temp_spline.basis_function_derivatives(time, span, derivative_order)
    for offset in range(degree + 1):
        column = span - degree + offset
        if 0 <= column < system.matrix.shape[1]:
            system.matrix[system.row, column] = derivatives[derivative_order, offset]
    if value is not None:
        system.right_hand_side[system.row] = value
    system.row += 1


def _solve_system(
    interpolator: BSplineInterpolator,
    system: _InterpolationSystem,
    degree: int,
    points: np.ndarray,
) -> np.ndarray:
    """Validate conditioning and solve for each control-point coordinate."""
    if np.linalg.matrix_rank(system.matrix) < min(system.matrix.shape):
        raise ValueError(
            "Linear system is rank-deficient. This typically occurs when there "
            "are too few points for the specified degree and constraints. "
            "Add more points or reduce the polynomial degree."
        )

    try:
        condition_number = np.linalg.cond(system.matrix)
        if condition_number > _ILL_CONDITIONED:
            print(f"Warning: The linear system is ill-conditioned (condition number: {condition_number:.2e})")
            print("This may lead to numerical inaccuracies in the spline interpolation.")
            print("Consider adding more points, using a lower degree, or adjusting the time distribution.")
            system.matrix += _REGULARIZATION * np.eye(system.matrix.shape[0], system.matrix.shape[1])
            print(f"Adding regularization (epsilon={_REGULARIZATION}) to improve numerical stability.")

        control_points = np.zeros((system.matrix.shape[1], points.shape[1]))
        for dimension in range(points.shape[1]):
            control_points[:, dimension] = solve(system.matrix, system.right_hand_side[:, dimension])
        return control_points  # noqa: TRY300
    except np.linalg.LinAlgError as error:
        raise ValueError(_solve_error_message(interpolator, degree, len(points), error)) from error


def _solve_error_message(
    interpolator: BSplineInterpolator,
    degree: int,
    point_count: int,
    error: np.linalg.LinAlgError,
) -> str:
    """Build an actionable message for an unsolvable interpolation system."""
    message = f"Failed to solve for control points: {error}\n"
    message += "This is likely due to an ill-posed interpolation problem.\n"
    message += (
        f"For degree {degree} B-splines, you should have at least {degree + 2} points (you provided {point_count}).\n"
    )
    if interpolator.cyclic:
        message += "When using cyclic conditions, you may need even more points.\n"
    if any(
        constraint is not None
        for constraint in (
            interpolator.initial_velocity,
            interpolator.final_velocity,
            interpolator.initial_acceleration,
            interpolator.final_acceleration,
        )
    ):
        message += "When specifying velocity or acceleration constraints, you may need more points.\n"
    if degree in {_QUARTIC_DEGREE, _QUINTIC_DEGREE}:
        message += f"Consider using a lower degree (e.g., degree=3) with {point_count} points.\n"
    return message


__all__ = ["compute_control_points"]
