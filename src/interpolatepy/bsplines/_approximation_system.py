"""Weighted least-squares system for B-spline approximation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .core import BSpline

if TYPE_CHECKING:
    from .approximation import ApproximationBSpline


@dataclass(frozen=True)
class ApproximationProblem:
    """Inputs needed to solve an endpoint-constrained approximation."""

    points: np.ndarray
    degree: int
    knots: np.ndarray
    parameters: np.ndarray
    control_point_count: int
    weights: np.ndarray


def approximate_control_points(
    approximation: ApproximationBSpline,
    problem: ApproximationProblem,
) -> np.ndarray:
    """Solve the endpoint-constrained weighted least-squares problem."""
    last_point = len(problem.points) - 1
    last_control = problem.control_point_count - 1
    control_points = np.zeros((problem.control_point_count, problem.points.shape[1]))
    control_points[0] = problem.points[0]
    control_points[last_control] = problem.points[last_point]

    if approximation.debug:
        print("\nCONTROL POINTS CALCULATION:")
        print(f"  n = {last_point} (number of points minus 1)")
        print(f"  m = {last_control} (number of control points minus 1)")
        print("  Fixed control points:")
        print(f"    P_0 = {problem.points[0]}")
        print(f"    P_{last_control} = {problem.points[last_point]}")

    if last_control <= 1:
        if approximation.debug:
            print("  Only two control points needed, returning interpolated curve.")
        return control_points

    basis_matrix, residual_matrix = _build_matrices(approximation, problem, last_point, last_control)
    internal_control_points = _solve_weighted_system(approximation, basis_matrix, residual_matrix, problem.weights)
    control_points[1:last_control] = internal_control_points

    if approximation.debug:
        print("\n  Calculated internal control points:")
        for index in range(1, last_control):
            print(f"    P_{index} = {control_points[index]}")
    return control_points


def _build_matrices(
    approximation: ApproximationBSpline,
    problem: ApproximationProblem,
    last_point: int,
    last_control: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the basis and endpoint-adjusted residual matrices."""
    basis_matrix = np.zeros((last_point - 1, last_control - 1))
    residual_matrix = np.zeros((last_point - 1, problem.points.shape[1]))
    temporary_spline = BSpline(
        problem.degree,
        problem.knots,
        np.zeros((problem.control_point_count, problem.points.shape[1])),
    )

    if approximation.debug:
        print(f"  B matrix shape: {basis_matrix.shape}")
        print(f"  R matrix shape: {residual_matrix.shape}")

    for point_index in range(1, last_point):
        parameter = problem.parameters[point_index]
        all_basis = _basis_row(
            temporary_spline,
            parameter,
            problem.degree,
            last_control,
        )
        basis_matrix[point_index - 1] = all_basis[1:last_control]
        residual_matrix[point_index - 1] = (
            problem.points[point_index]
            - all_basis[0] * problem.points[0]
            - all_basis[last_control] * problem.points[last_point]
        )
        if approximation.debug:
            print(f"\n  Processing point {point_index} at parameter u = {parameter:.6f}")
            print(f"    All basis values: {all_basis}")
            print(f"    Residual row: {residual_matrix[point_index - 1]}")

    if approximation.debug:
        print("\n  B matrix:")
        print(basis_matrix)
        print("\n  R matrix:")
        print(residual_matrix)
    return basis_matrix, residual_matrix


def _basis_row(
    spline: BSpline,
    parameter: float,
    degree: int,
    last_control: int,
) -> np.ndarray:
    """Expand the locally nonzero basis functions into a full matrix row."""
    all_basis = np.zeros(last_control + 1)
    span = spline.find_knot_span(parameter)
    basis_values = spline.basis_functions(parameter, span)
    for offset in range(degree + 1):
        index = span - degree + offset
        if 0 <= index <= last_control:
            all_basis[index] = basis_values[offset]
    return all_basis


def _solve_weighted_system(
    approximation: ApproximationBSpline,
    basis_matrix: np.ndarray,
    residual_matrix: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Solve the weighted normal equations, falling back to a pseudoinverse."""
    weight_matrix = np.diag(weights)
    weighted_transpose = basis_matrix.T @ weight_matrix
    normal_matrix = weighted_transpose @ basis_matrix
    normal_right_hand_side = weighted_transpose @ residual_matrix

    if approximation.debug:
        print("\n  Weights:")
        print(f"    {weights}")
        print("\n  Calculating pseudo-inverse solution:")
        print(f"    B^T W shape: {weighted_transpose.shape}")
        print(f"    B^T W B shape: {normal_matrix.shape}")
        print(f"    B^T W R shape: {normal_right_hand_side.shape}")

    try:
        result = np.linalg.solve(normal_matrix, normal_right_hand_side)
        if approximation.debug:
            print("    Used np.linalg.solve (direct solution)")
        return result  # noqa: TRY300
    except np.linalg.LinAlgError:
        if approximation.debug:
            print("    Used np.linalg.pinv (matrix was singular or poorly conditioned)")
        return np.linalg.pinv(normal_matrix) @ normal_right_hand_side


__all__ = ["ApproximationProblem", "approximate_control_points"]
