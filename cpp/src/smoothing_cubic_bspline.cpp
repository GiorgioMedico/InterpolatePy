#include <interpolatecpp/bspline/smoothing_cubic_bspline.hpp>

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace interpolatecpp::bspline {

Eigen::VectorXd SmoothingCubicBSpline::calculate_parameters(
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
            double total = 0.0;
            for (int k = 1; k <= n; ++k) {
                total += (points.row(k) - points.row(k - 1)).norm();
            }
            double accum = 0.0;
            for (int k = 1; k < n; ++k) {
                accum += (points.row(k) - points.row(k - 1)).norm();
                u_bars[k] = accum / total;
            }
            break;
        }
        case Parameterization::Centripetal: {
            constexpr double mu = 0.5;
            double total = 0.0;
            for (int k = 1; k <= n; ++k) {
                total += std::pow((points.row(k) - points.row(k - 1)).norm(), mu);
            }
            double accum = 0.0;
            for (int k = 1; k < n; ++k) {
                accum += std::pow((points.row(k) - points.row(k - 1)).norm(), mu);
                u_bars[k] = accum / total;
            }
            break;
        }
    }

    return u_bars;
}

Eigen::VectorXd SmoothingCubicBSpline::calculate_knot_vector() const {
    const int n = static_cast<int>(approximation_points_.rows()) - 1;
    Eigen::VectorXd kv = Eigen::VectorXd::Zero(n + 7);

    kv[0] = u_bars_[0];
    kv[1] = u_bars_[0];
    kv[2] = u_bars_[0];

    for (int j = 0; j <= n; ++j) {
        kv[j + 3] = u_bars_[j];
    }

    kv[n + 4] = u_bars_[n];
    kv[n + 5] = u_bars_[n];
    kv[n + 6] = u_bars_[n];

    return kv;
}

Eigen::MatrixXd SmoothingCubicBSpline::construct_b_matrix() const {
    const int n = static_cast<int>(approximation_points_.rows()) - 1;
    const int m = n + 2;  // Last control point index (total m+1 = n+3)

    Eigen::MatrixXd b_mat = Eigen::MatrixXd::Zero(n + 1, m + 1);

    for (int k = 0; k <= n; ++k) {
        const double u = u_bars_[k];
        const int span = find_knot_span(u);
        const Eigen::VectorXd basis_vals = basis_functions(u, span);

        for (int j = 0; j <= degree_; ++j) {
            b_mat(k, span - degree_ + j) = basis_vals[j];
        }
    }

    return b_mat;
}

Eigen::MatrixXd SmoothingCubicBSpline::construct_a_matrix() const {
    const int n = static_cast<int>(approximation_points_.rows()) - 1;
    const int size = n + 1;

    Eigen::MatrixXd a_mat = Eigen::MatrixXd::Zero(size, size);

    for (int i = 0; i < size; ++i) {
        int idx3 = i + 3;
        int idx2 = i + 2;
        int idx4 = i + 4;

        if (idx4 < static_cast<int>(knots_.size())) {
            double ui3_i2 = knots_[idx3] - knots_[idx2];
            double ui4_i3 = knots_[idx4] - knots_[idx3];

            a_mat(i, i) = 2.0 * (ui3_i2 + ui4_i3);

            if (i < size - 1) {
                a_mat(i, i + 1) = ui4_i3;
            }
            if (i > 0) {
                a_mat(i, i - 1) = ui3_i2;
            }
        }
    }

    return a_mat;
}

Eigen::MatrixXd SmoothingCubicBSpline::construct_c_matrix() const {
    const int n = static_cast<int>(approximation_points_.rows()) - 1;
    const int size_r = n + 1;
    const int size_p = n + 3;

    Eigen::MatrixXd c_mat = Eigen::MatrixXd::Zero(size_r, size_p);

    for (int k = 0; k < size_r; ++k) {
        int k4 = k + 4;
        int k2 = k + 2;
        int k1 = k + 1;
        int k5 = k + 5;

        if (k4 < static_cast<int>(knots_.size()) &&
            k5 < static_cast<int>(knots_.size())) {
            double uk4_k2 = knots_[k4] - knots_[k2];
            double uk4_k1 = knots_[k4] - knots_[k1];
            double uk5_k2 = knots_[k5] - knots_[k2];

            if (std::abs(uk4_k2) > kEpsilon && std::abs(uk4_k1) > kEpsilon &&
                std::abs(uk5_k2) > kEpsilon) {
                if (k < size_p) {
                    c_mat(k, k) = 6.0 / (uk4_k2 * uk4_k1);
                }
                if (k + 1 < size_p) {
                    c_mat(k, k + 1) = -6.0 / uk4_k2 * (1.0 / uk4_k1 + 1.0 / uk5_k2);
                }
                if (k + 2 < size_p) {
                    c_mat(k, k + 2) = 6.0 / (uk4_k2 * uk5_k2);
                }
            }
        }
    }

    return c_mat;
}

Eigen::MatrixXd SmoothingCubicBSpline::calculate_control_points_impl() const {
    if (enforce_endpoints_) {
        return calculate_control_points_with_endpoints();
    }

    const int n = static_cast<int>(approximation_points_.rows()) - 1;
    const int dim = static_cast<int>(approximation_points_.cols());

    Eigen::MatrixXd b_mat = construct_b_matrix();
    Eigen::MatrixXd w_mat = weights_.asDiagonal();
    Eigen::MatrixXd a_mat = construct_a_matrix();
    Eigen::MatrixXd c_mat = construct_c_matrix();

    Eigen::MatrixXd ctac = c_mat.transpose() * a_mat * c_mat;
    Eigen::MatrixXd left = b_mat.transpose() * w_mat * b_mat + lambda_param_ * ctac;
    Eigen::MatrixXd right = b_mat.transpose() * w_mat * approximation_points_;

    Eigen::MatrixXd cp = Eigen::MatrixXd::Zero(n + 3, dim);

    Eigen::FullPivLU<Eigen::MatrixXd> lu(left);
    if (lu.isInvertible()) {
        for (int d = 0; d < dim; ++d) {
            cp.col(d) = lu.solve(right.col(d));
        }
    } else {
        for (int d = 0; d < dim; ++d) {
            cp.col(d) = left.completeOrthogonalDecomposition().solve(right.col(d));
        }
    }

    return cp;
}

Eigen::MatrixXd SmoothingCubicBSpline::calculate_control_points_with_endpoints() const {
    const int n = static_cast<int>(approximation_points_.rows()) - 1;
    const int dim = static_cast<int>(approximation_points_.cols());

    Eigen::MatrixXd cp = Eigen::MatrixXd::Zero(n + 3, dim);

    cp.row(0) = approximation_points_.row(0);
    cp.row(n + 2) = approximation_points_.row(n);

    double u4 = knots_[4];
    cp.row(1) = cp.row(0) + (u4 / 3.0) * v0_.transpose();

    double un2 = knots_[n + 2];
    cp.row(n + 1) = cp.row(n + 2) - ((1.0 - un2) / 3.0) * vn_.transpose();

    if (n <= 0) {
        return cp;
    }

    // Reduced system for p_2, ..., p_n
    const int sys_size = n - 1;
    Eigen::MatrixXd q_reduced = Eigen::MatrixXd::Zero(sys_size, dim);
    for (int k = 1; k < n; ++k) {
        q_reduced.row(k - 1) = approximation_points_.row(k);
    }

    Eigen::MatrixXd b_reduced = Eigen::MatrixXd::Zero(sys_size, sys_size);
    for (int k = 1; k < n; ++k) {
        const double u = u_bars_[k];
        const int span = find_knot_span(u);
        const Eigen::VectorXd basis_vals = basis_functions(u, span);

        // Subtract fixed control point contributions
        if (span - degree_ <= 0) {
            q_reduced.row(k - 1) -= basis_vals[0] * cp.row(0);
        }
        if (span - degree_ + 1 <= 1) {
            q_reduced.row(k - 1) -= basis_vals[1] * cp.row(1);
        }
        if (span >= n) {
            int basis_idx = n + 1 - (span - degree_);
            if (basis_idx >= 0 && basis_idx < basis_vals.size()) {
                q_reduced.row(k - 1) -= basis_vals[basis_idx] * cp.row(n + 1);
            }
        }
        if (span >= n + 1) {
            int basis_idx = n + 2 - (span - degree_);
            if (basis_idx >= 0 && basis_idx < basis_vals.size()) {
                q_reduced.row(k - 1) -= basis_vals[basis_idx] * cp.row(n + 2);
            }
        }

        // Fill B_reduced for p_2 to p_n
        for (int j = 2; j <= n; ++j) {
            int basis_idx = j - (span - degree_);
            if (basis_idx >= 0 && basis_idx < basis_vals.size()) {
                int col = j - 2;
                if (col >= 0 && col < sys_size) {
                    b_reduced(k - 1, col) = basis_vals[basis_idx];
                }
            }
        }
    }

    // Reduced weight matrix
    Eigen::MatrixXd w_reduced = Eigen::MatrixXd::Zero(sys_size, sys_size);
    for (int i = 0; i < sys_size; ++i) {
        w_reduced(i, i) = weights_[i + 1];
    }

    Eigen::MatrixXd a_mat = construct_a_matrix();
    Eigen::MatrixXd c_mat = construct_c_matrix();
    Eigen::MatrixXd c_reduced = c_mat.block(0, 2, c_mat.rows(), sys_size);

    // PZ vector
    Eigen::MatrixXd pz = Eigen::MatrixXd::Zero(n + 1, dim);
    pz.row(0) = c_mat(0, 0) * cp.row(0) + c_mat(0, 1) * cp.row(1);
    pz.row(n) = c_mat(n, n + 1) * cp.row(n + 1) + c_mat(n, n + 2) * cp.row(n + 2);

    Eigen::MatrixXd ctac_reduced = c_reduced.transpose() * a_mat * c_reduced;
    Eigen::MatrixXd ctat_pz = c_reduced.transpose() * a_mat * pz;

    Eigen::MatrixXd left = b_reduced.transpose() * w_reduced * b_reduced +
                           lambda_param_ * ctac_reduced;
    Eigen::MatrixXd right_side =
        b_reduced.transpose() * w_reduced * q_reduced - lambda_param_ * ctat_pz;

    Eigen::FullPivLU<Eigen::MatrixXd> lu(left);
    if (lu.isInvertible()) {
        for (int d = 0; d < dim; ++d) {
            cp.block(2, d, sys_size, 1) = lu.solve(right_side.col(d));
        }
    } else {
        for (int d = 0; d < dim; ++d) {
            cp.block(2, d, sys_size, 1) =
                left.completeOrthogonalDecomposition().solve(right_side.col(d));
        }
    }

    return cp;
}

Eigen::VectorXd SmoothingCubicBSpline::calculate_approximation_error() const {
    const int n = static_cast<int>(approximation_points_.rows());
    Eigen::VectorXd errors = Eigen::VectorXd::Zero(n);

    for (int k = 0; k < n; ++k) {
        Eigen::VectorXd pt = evaluate(u_bars_[k]);
        errors[k] = (pt - approximation_points_.row(k).transpose()).norm();
    }

    return errors;
}

double SmoothingCubicBSpline::calculate_total_error() const {
    double total = 0.0;
    const int n = static_cast<int>(approximation_points_.rows());

    for (int k = 0; k < n; ++k) {
        Eigen::VectorXd pt = evaluate(u_bars_[k]);
        Eigen::VectorXd diff = pt - approximation_points_.row(k).transpose();
        total += weights_[k] * diff.squaredNorm();
    }

    return total;
}

SmoothingCubicBSpline::SmoothingCubicBSpline(const Eigen::MatrixXd& points,
                                             const BSplineParams& params)
    : BSpline(DeferInit{})
{
    Eigen::MatrixXd pts = points;
    const int n_points = static_cast<int>(pts.rows());
    const int dim = static_cast<int>(pts.cols());
    const int n = n_points - 1;

    approximation_points_ = pts;
    mu_ = std::clamp(params.mu, 0.0, 1.0);
    lambda_param_ = (mu_ > 0.0) ? (1.0 - mu_) / (6.0 * mu_)
                                 : std::numeric_limits<double>::infinity();
    enforce_endpoints_ = params.enforce_endpoints;

    // Set weights
    if (params.weights.has_value()) {
        if (params.weights->size() != n_points) {
            throw std::invalid_argument(
                "Length of weights must match the number of points");
        }
        weights_ = params.weights.value();
    } else {
        weights_ = Eigen::VectorXd::Ones(n_points);
    }

    // Calculate parameters
    u_bars_ = calculate_parameters(pts, params.method);

    // Calculate knot vector
    Eigen::VectorXd kv = calculate_knot_vector();

    // Process endpoint derivatives
    if (params.enforce_endpoints) {
        if (params.v0.has_value()) {
            v0_ = params.v0.value();
        } else if (params.auto_derivatives && n > 0) {
            double u_diff = u_bars_[1] - u_bars_[0];
            if (std::abs(u_diff) > kEpsilon) {
                v0_ = (pts.row(1) - pts.row(0)).transpose() / u_diff;
            } else {
                v0_ = Eigen::VectorXd::Zero(dim);
            }
        } else {
            v0_ = Eigen::VectorXd::Zero(dim);
        }

        if (params.vn.has_value()) {
            vn_ = params.vn.value();
        } else if (params.auto_derivatives && n > 0) {
            double u_diff = u_bars_[n] - u_bars_[n - 1];
            if (std::abs(u_diff) > kEpsilon) {
                vn_ = (pts.row(n) - pts.row(n - 1)).transpose() / u_diff;
            } else {
                vn_ = Eigen::VectorXd::Zero(dim);
            }
        } else {
            vn_ = Eigen::VectorXd::Zero(dim);
        }
    } else {
        v0_ = Eigen::VectorXd::Zero(dim);
        vn_ = Eigen::VectorXd::Zero(dim);
    }

    // Initialize base with knot vector to enable basis_functions()
    degree_ = 3;
    knots_ = kv;
    const int n_cp = n + 3;
    control_points_ = Eigen::MatrixXd::Zero(n_cp, dim);
    u_min_ = knots_[3];
    u_max_ = knots_[static_cast<int>(knots_.size()) - 4];
    dimension_ = dim;

    // Now calculate actual control points
    control_points_ = calculate_control_points_impl();
}

}  // namespace interpolatecpp::bspline
