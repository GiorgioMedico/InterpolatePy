#include <interpolatecpp/bspline/cubic_bspline_interpolation.hpp>
#include <interpolatecpp/tridiagonal.hpp>

#include <cmath>
#include <stdexcept>

namespace interpolatecpp::bspline {

Eigen::VectorXd CubicBSplineInterpolation::calculate_parameters(
    const Eigen::MatrixXd& points, Parameterization method) {
    const int n = static_cast<int>(points.rows()) - 1;
    Eigen::VectorXd u_bars = Eigen::VectorXd::Zero(n + 1);
    u_bars[0] = 0.0;
    u_bars[n] = 1.0;

    switch (method) {
        case Parameterization::EquallySpaced: {
            for (int k = 1; k < n; ++k) {
                u_bars[k] = static_cast<double>(k) / static_cast<double>(n);
            }
            break;
        }
        case Parameterization::ChordLength: {
            double total_length = 0.0;
            for (int k = 1; k <= n; ++k) {
                total_length += (points.row(k) - points.row(k - 1)).norm();
            }
            for (int k = 1; k < n; ++k) {
                u_bars[k] = u_bars[k - 1] +
                             (points.row(k) - points.row(k - 1)).norm() / total_length;
            }
            break;
        }
        case Parameterization::Centripetal: {
            constexpr double mu = 0.5;
            double total_length = 0.0;
            for (int k = 1; k <= n; ++k) {
                total_length += std::pow((points.row(k) - points.row(k - 1)).norm(), mu);
            }
            for (int k = 1; k < n; ++k) {
                u_bars[k] =
                    u_bars[k - 1] +
                    std::pow((points.row(k) - points.row(k - 1)).norm(), mu) / total_length;
            }
            break;
        }
    }

    return u_bars;
}

Eigen::VectorXd CubicBSplineInterpolation::calculate_knot_vector(
    const Eigen::VectorXd& u_bars, int n_points) {
    const int n = n_points - 1;
    Eigen::VectorXd knots = Eigen::VectorXd::Zero(n + 7);

    // First 3 knots = u_bars[0]
    knots[0] = u_bars[0];
    knots[1] = u_bars[0];
    knots[2] = u_bars[0];

    // Middle knots: u_j+3 = u_bars[j] for j = 0, ..., n
    for (int j = 0; j <= n; ++j) {
        knots[j + 3] = u_bars[j];
    }

    // Last 3 knots = u_bars[n]
    knots[n + 4] = u_bars[n];
    knots[n + 5] = u_bars[n];
    knots[n + 6] = u_bars[n];

    return knots;
}

Eigen::MatrixXd CubicBSplineInterpolation::calculate_control_points(
    const Eigen::VectorXd& knots) const {
    const int n = static_cast<int>(interpolation_points_.rows()) - 1;
    const int dim = static_cast<int>(interpolation_points_.cols());

    Eigen::MatrixXd cp = Eigen::MatrixXd::Zero(n + 3, dim);

    // Direct control points from endpoints and derivatives (eq. 8.16)
    cp.row(0) = interpolation_points_.row(0);
    cp.row(1) = interpolation_points_.row(0) + (knots[4] / 3.0) * v0_.transpose();
    cp.row(n + 1) =
        interpolation_points_.row(n) - ((1.0 - knots[n + 2]) / 3.0) * vn_.transpose();
    cp.row(n + 2) = interpolation_points_.row(n);

    if (n < kMinPointsForTridiagonal) {
        return cp;
    }

    // Solve tridiagonal system for remaining control points
    const int sys_size = n - 1;
    Eigen::VectorXd lower_diag = Eigen::VectorXd::Zero(sys_size);
    Eigen::VectorXd main_diag = Eigen::VectorXd::Zero(sys_size);
    Eigen::VectorXd upper_diag = Eigen::VectorXd::Zero(sys_size);
    Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(sys_size, dim);

    // Create temporary BSpline for basis function calculation
    Eigen::MatrixXd temp_ctrl = Eigen::MatrixXd::Zero(n + 3, dim);
    BSpline temp_bs(3, std::span<const double>(knots.data(), knots.size()), temp_ctrl);

    for (int i = 0; i < sys_size; ++i) {
        const int k = i + 1;  // Point index (1 to n-1)
        const double u_bar = u_bars_[k];
        const int span = temp_bs.find_knot_span(u_bar);
        const Eigen::VectorXd basis_vals = temp_bs.basis_functions(u_bar, span);

        const double b3_k = basis_vals[0];
        const double b3_k1 = basis_vals[1];
        const double b3_k2 = basis_vals[2];

        if (k == 1) {
            main_diag[0] = b3_k1;
            upper_diag[0] = b3_k2;
            rhs.row(0) = interpolation_points_.row(k) - b3_k * cp.row(1);
        } else if (k == n - 1) {
            lower_diag[k - 2] = b3_k;
            main_diag[k - 1] = b3_k1;
            rhs.row(k - 1) = interpolation_points_.row(k) - b3_k2 * cp.row(n + 1);
        } else {
            lower_diag[k - 2] = b3_k;
            main_diag[k - 1] = b3_k1;
            upper_diag[k - 1] = b3_k2;
            rhs.row(k - 1) = interpolation_points_.row(k);
        }
    }

    // Solve per dimension using tridiagonal solver
    for (int d = 0; d < dim; ++d) {
        Eigen::VectorXd l_diag = Eigen::VectorXd::Zero(sys_size);
        if (sys_size > 1) {
            l_diag.tail(sys_size - 1) = lower_diag.head(sys_size - 1);
        }

        Eigen::VectorXd solution =
            interpolatecpp::solve_tridiagonal(l_diag, main_diag, upper_diag, rhs.col(d));
        cp.block(2, d, sys_size, 1) = solution;
    }

    return cp;
}

CubicBSplineInterpolation::CubicBSplineInterpolation(
    const Eigen::MatrixXd& points, const std::optional<Eigen::VectorXd>& v0,
    const std::optional<Eigen::VectorXd>& vn, Parameterization method,
    bool auto_derivatives)
    : BSpline(DeferInit{})
{
    // Ensure 2D points
    Eigen::MatrixXd pts = points;
    if (pts.cols() == 0) {
        throw std::invalid_argument("Points array is empty");
    }

    const int n_points = static_cast<int>(pts.rows());
    const int dim = static_cast<int>(pts.cols());
    const int n = n_points - 1;

    interpolation_points_ = pts;
    u_bars_ = calculate_parameters(pts, method);

    // Process v0
    if (v0.has_value()) {
        if (v0->size() != dim) {
            throw std::invalid_argument("v0 must be a vector of dimension " +
                                        std::to_string(dim));
        }
        v0_ = v0.value();
    } else if (auto_derivatives && n > 0) {
        double u_diff = u_bars_[1] - u_bars_[0];
        if (std::abs(u_diff) > kParamDiffThreshold) {
            v0_ = (pts.row(1) - pts.row(0)).transpose() / u_diff;
        } else {
            v0_ = Eigen::VectorXd::Zero(dim);
        }
    } else {
        v0_ = Eigen::VectorXd::Zero(dim);
    }

    // Process vn
    if (vn.has_value()) {
        if (vn->size() != dim) {
            throw std::invalid_argument("vn must be a vector of dimension " +
                                        std::to_string(dim));
        }
        vn_ = vn.value();
    } else if (auto_derivatives && n > 0) {
        double u_diff = u_bars_[n] - u_bars_[n - 1];
        if (std::abs(u_diff) > kParamDiffThreshold) {
            vn_ = (pts.row(n) - pts.row(n - 1)).transpose() / u_diff;
        } else {
            vn_ = Eigen::VectorXd::Zero(dim);
        }
    } else {
        vn_ = Eigen::VectorXd::Zero(dim);
    }

    // Calculate knots and control points
    Eigen::VectorXd new_knots = calculate_knot_vector(u_bars_, n_points);
    Eigen::MatrixXd new_cp = calculate_control_points(new_knots);

    // Reinitialize base class with computed values
    degree_ = 3;
    knots_ = new_knots;
    control_points_ = new_cp;
    u_min_ = knots_[3];
    u_max_ = knots_[static_cast<int>(knots_.size()) - 4];
    dimension_ = dim;
}

}  // namespace interpolatecpp::bspline
