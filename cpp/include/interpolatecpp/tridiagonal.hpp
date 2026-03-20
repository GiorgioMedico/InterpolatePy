#pragma once

#include <Eigen/Core>
#include <stdexcept>

namespace interpolatecpp {

/// Solve a tridiagonal system Ax = d using the Thomas algorithm.
///
/// Takes vectors by value (creates working copies, matching Python's np.array() copy).
/// O(n) complexity instead of O(n^3) for general matrix solvers.
///
/// @param lower_diagonal  Lower diagonal (first element unused)
/// @param main_diagonal   Main diagonal
/// @param upper_diagonal  Upper diagonal (last element unused)
/// @param right_hand_side Right-hand side vector
/// @return Solution vector x
/// @throws std::invalid_argument if a pivot is zero
inline Eigen::VectorXd solve_tridiagonal(Eigen::VectorXd lower_diagonal,
                                         Eigen::VectorXd main_diagonal,
                                         Eigen::VectorXd upper_diagonal,
                                         Eigen::VectorXd right_hand_side) {
    const auto n = right_hand_side.size();

    if (main_diagonal(0) == 0.0) {
        throw std::invalid_argument(
            "Pivot cannot be zero. The system cannot be solved with this method.");
    }

    // Forward elimination
    for (Eigen::Index k = 1; k < n; ++k) {
        double m = lower_diagonal(k) / main_diagonal(k - 1);
        main_diagonal(k) -= m * upper_diagonal(k - 1);
        right_hand_side(k) -= m * right_hand_side(k - 1);
    }

    // Back substitution
    Eigen::VectorXd x(n);
    x(n - 1) = right_hand_side(n - 1) / main_diagonal(n - 1);
    for (Eigen::Index k = n - 2; k >= 0; --k) {
        x(k) = (right_hand_side(k) - upper_diagonal(k) * x(k + 1)) / main_diagonal(k);
    }

    return x;
}

}  // namespace interpolatecpp
