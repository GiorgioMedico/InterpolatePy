#include <interpolatecpp/bspline/approximation_bspline.hpp>

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>

namespace interpolatecpp::bspline {

Eigen::VectorXd ApproximationBSpline::compute_parameters(const Eigen::MatrixXd& points,
                                                         Parameterization method) {
    const int n = static_cast<int>(points.rows()) - 1;
    Eigen::VectorXd u_bar = Eigen::VectorXd::Zero(n + 1);
    u_bar[0] = 0.0;
    u_bar[n] = 1.0;

    switch (method) {
        case Parameterization::EquallySpaced: {
            for (int k = 1; k < n; ++k) {
                u_bar[k] = static_cast<double>(k) / static_cast<double>(n);
            }
            break;
        }
        case Parameterization::ChordLength: {
            double total = 0.0;
            for (int k = 1; k <= n; ++k) {
                total += (points.row(k) - points.row(k - 1)).norm();
            }
            for (int k = 1; k < n; ++k) {
                u_bar[k] =
                    u_bar[k - 1] + (points.row(k) - points.row(k - 1)).norm() / total;
            }
            break;
        }
        case Parameterization::Centripetal: {
            constexpr double mu = 0.5;
            double total = 0.0;
            for (int k = 1; k <= n; ++k) {
                total += std::pow((points.row(k) - points.row(k - 1)).norm(), mu);
            }
            for (int k = 1; k < n; ++k) {
                u_bar[k] = u_bar[k - 1] +
                           std::pow((points.row(k) - points.row(k - 1)).norm(), mu) / total;
            }
            break;
        }
    }

    return u_bar;
}

Eigen::VectorXd ApproximationBSpline::compute_knots(int degree, int num_control_points,
                                                    int num_points,
                                                    const Eigen::VectorXd& u_bar) {
    const int num_knots = num_control_points + degree + 1;
    Eigen::VectorXd knots = Eigen::VectorXd::Zero(num_knots);

    // First and last knots with multiplicity p+1
    for (int i = 0; i <= degree; ++i) {
        knots[i] = u_bar[0];
    }
    for (int i = num_knots - degree - 1; i < num_knots; ++i) {
        knots[i] = u_bar[u_bar.size() - 1];
    }

    // Internal knots (Section 8.5.1)
    const int n = num_points - 1;
    const int m = num_control_points - 1;
    const double d = static_cast<double>(n + 1) / static_cast<double>(m - degree + 1);

    for (int j = 1; j <= m - degree; ++j) {
        int i = static_cast<int>(j * d);
        double alpha = j * d - i;
        knots[j + degree] = (1.0 - alpha) * u_bar[i - 1] + alpha * u_bar[i];
    }

    return knots;
}

Eigen::MatrixXd ApproximationBSpline::approximate_control_points(
    const Eigen::MatrixXd& points, int degree, const Eigen::VectorXd& knots,
    const Eigen::VectorXd& u_bar, int num_control_points,
    const Eigen::VectorXd& weights) const {
    const int n = static_cast<int>(points.rows()) - 1;
    const int m = num_control_points - 1;
    const int dim = static_cast<int>(points.cols());

    Eigen::MatrixXd cp = Eigen::MatrixXd::Zero(num_control_points, dim);
    cp.row(0) = points.row(0);
    cp.row(m) = points.row(n);

    if (m <= 1) {
        return cp;
    }

    // Create temporary BSpline for basis function calculation
    Eigen::MatrixXd temp_ctrl = Eigen::MatrixXd::Zero(num_control_points, dim);
    BSpline temp_bs(degree, std::span<const double>(knots.data(), knots.size()), temp_ctrl);

    // Build B matrix (n-1 x m-1) and R matrix (n-1 x dim)
    Eigen::MatrixXd b_matrix = Eigen::MatrixXd::Zero(n - 1, m - 1);
    Eigen::MatrixXd r_matrix = Eigen::MatrixXd::Zero(n - 1, dim);

    for (int k = 1; k < n; ++k) {
        const double u = u_bar[k];
        const int span = temp_bs.find_knot_span(u);
        const Eigen::VectorXd basis_vals = temp_bs.basis_functions(u, span);

        // Collect all basis function values
        Eigen::VectorXd all_basis = Eigen::VectorXd::Zero(m + 1);
        for (int j = 0; j <= degree; ++j) {
            int idx = span - degree + j;
            if (idx >= 0 && idx <= m) {
                all_basis[idx] = basis_vals[j];
            }
        }

        // Fill B matrix with internal control point basis values
        for (int j = 1; j < m; ++j) {
            b_matrix(k - 1, j - 1) = all_basis[j];
        }

        // R = Q - B_0 * Q_0 - B_m * Q_n
        r_matrix.row(k - 1) =
            points.row(k) - all_basis[0] * points.row(0) - all_basis[m] * points.row(n);
    }

    // Weighted least squares: (B^T W B)^{-1} B^T W R
    Eigen::MatrixXd w_matrix = Eigen::MatrixXd::Zero(n - 1, n - 1);
    for (int i = 0; i < n - 1; ++i) {
        w_matrix(i, i) = weights[i];
    }

    Eigen::MatrixXd btw = b_matrix.transpose() * w_matrix;
    Eigen::MatrixXd btwb = btw * b_matrix;
    Eigen::MatrixXd btwr = btw * r_matrix;

    // Solve normal equations
    Eigen::MatrixXd internal_cp;
    Eigen::FullPivLU<Eigen::MatrixXd> lu(btwb);
    if (lu.isInvertible()) {
        internal_cp = lu.solve(btwr);
    } else {
        // Fallback to pseudo-inverse
        internal_cp = btwb.completeOrthogonalDecomposition().solve(btwr);
    }

    cp.block(1, 0, m - 1, dim) = internal_cp;

    return cp;
}

double ApproximationBSpline::calculate_approximation_error() const {
    double sum_sq = 0.0;
    for (int i = 0; i < static_cast<int>(original_points_.rows()); ++i) {
        Eigen::VectorXd spline_pt = evaluate(original_parameters_[i]);
        double sq_dist = (original_points_.row(i).transpose() - spline_pt).squaredNorm();
        sum_sq += sq_dist;
    }
    return sum_sq;
}

ApproximationBSpline::ApproximationBSpline(const Eigen::MatrixXd& points,
                                           int num_control_points, int degree,
                                           const std::optional<Eigen::VectorXd>& weights,
                                           Parameterization method, bool debug)
    : BSpline(DeferInit{})
      ,
      debug_(debug) {
    if (degree < 1) {
        throw std::invalid_argument("Degree must be at least 1");
    }
    if (num_control_points <= degree) {
        throw std::invalid_argument(
            "Number of control points must be greater than the degree");
    }
    if (static_cast<int>(points.rows()) <= num_control_points) {
        throw std::invalid_argument(
            "Number of points must be greater than number of control points");
    }

    const int n = static_cast<int>(points.rows()) - 1;

    Eigen::VectorXd w;
    if (weights.has_value()) {
        w = weights.value();
    } else {
        w = Eigen::VectorXd::Ones(n - 1);
    }

    Eigen::VectorXd u_bar = compute_parameters(points, method);
    Eigen::VectorXd new_knots =
        compute_knots(degree, num_control_points, static_cast<int>(points.rows()), u_bar);
    Eigen::MatrixXd new_cp =
        approximate_control_points(points, degree, new_knots, u_bar, num_control_points, w);

    // Reinitialize base
    degree_ = degree;
    knots_ = new_knots;
    control_points_ = new_cp;
    u_min_ = knots_[degree];
    u_max_ = knots_[static_cast<int>(knots_.size()) - degree - 1];
    dimension_ = static_cast<int>(points.cols());

    original_points_ = points;
    original_parameters_ = u_bar;
}

}  // namespace interpolatecpp::bspline
