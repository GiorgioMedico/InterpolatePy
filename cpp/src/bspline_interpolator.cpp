#include <interpolatecpp/bspline/bspline_interpolator.hpp>

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace interpolatecpp::bspline {

namespace {
constexpr int kCubicDegree = 3;
constexpr int kQuarticDegree = 4;
constexpr int kQuinticDegree = 5;
}  // namespace

Eigen::VectorXd BSplineInterpolator::create_knot_vector(int degree,
                                                        const Eigen::MatrixXd& points,
                                                        const Eigen::VectorXd& times) {
    const int n = static_cast<int>(points.rows()) - 1;
    const int p = degree;

    if (p % 2 == 1) {
        // Odd degree: knots at interpolation points (eq. 4.42)
        const int num_knots = n + 2 * p + 1;
        Eigen::VectorXd knots = Eigen::VectorXd::Zero(num_knots);

        for (int i = 0; i <= p; ++i) {
            knots[i] = times[0];
        }

        for (int i = 0; i < n - 1; ++i) {
            knots[p + 1 + i] = times[1 + i];
        }

        for (int i = p + n; i < num_knots; ++i) {
            knots[i] = times[n];
        }

        return knots;
    }

    // Even degree: knots at midpoints (eq. 4.43)
    const int num_knots = n + 2 * p + 2;
    Eigen::VectorXd knots = Eigen::VectorXd::Zero(num_knots);

    for (int i = 0; i <= p; ++i) {
        knots[i] = times[0];
    }

    for (int i = 0; i < n; ++i) {
        knots[p + 1 + i] = (times[i] + times[i + 1]) / 2.0;
    }

    for (int i = p + 1 + n; i < num_knots; ++i) {
        knots[i] = times[n];
    }

    return knots;
}

Eigen::MatrixXd BSplineInterpolator::compute_control_points(
    int degree, const Eigen::MatrixXd& points, const Eigen::VectorXd& times,
    const Eigen::VectorXd& knots) const {
    const int n = static_cast<int>(points.rows()) - 1;
    const int p = degree;
    const int dim = static_cast<int>(points.cols());

    const int num_cp = (p % 2 == 1) ? (n + 1 + p - 1) : (n + 1 + p);
    const int num_additional = (p % 2 == 0) ? p : (p - 1);
    const int total_rows = n + 1 + num_additional;

    // Create temporary BSpline for basis function calculation
    Eigen::MatrixXd temp_ctrl = Eigen::MatrixXd::Zero(num_cp, 1);
    BSpline temp_spline(degree, std::span<const double>(knots.data(), knots.size()),
                        temp_ctrl);

    // Build linear system A * P = b
    Eigen::MatrixXd a_matrix = Eigen::MatrixXd::Zero(total_rows, num_cp);
    Eigen::MatrixXd b = Eigen::MatrixXd::Zero(total_rows, dim);

    // Interpolation conditions
    for (int i = 0; i <= n; ++i) {
        const double t = times[i];
        const int span = temp_spline.find_knot_span(t);
        const Eigen::VectorXd basis_vals = temp_spline.basis_functions(t, span);

        for (int j = 0; j <= p; ++j) {
            int col = span - p + j;
            if (col >= 0 && col < num_cp) {
                a_matrix(i, col) = basis_vals[j];
            }
        }
        b.row(i) = points.row(i);
    }

    // Boundary conditions
    int row = n + 1;

    if (cyclic_) {
        for (int k = 1; k <= num_additional && row < total_rows; ++k) {
            const int span0 = temp_spline.find_knot_span(times[0]);
            const int spann = temp_spline.find_knot_span(times[n]);
            const Eigen::MatrixXd ders0 =
                temp_spline.basis_function_derivatives(times[0], span0, k);
            const Eigen::MatrixXd dersn =
                temp_spline.basis_function_derivatives(times[n], spann, k);

            for (int j = 0; j <= p; ++j) {
                int col0 = span0 - p + j;
                if (col0 >= 0 && col0 < num_cp) {
                    a_matrix(row, col0) = ders0(k, j);
                }
                int coln = spann - p + j;
                if (coln >= 0 && coln < num_cp) {
                    a_matrix(row, coln) -= dersn(k, j);
                }
            }
            ++row;
        }
    } else {
        // Velocity constraints
        if (initial_velocity_.has_value() && row < total_rows) {
            const int span = temp_spline.find_knot_span(times[0]);
            const Eigen::MatrixXd ders =
                temp_spline.basis_function_derivatives(times[0], span, 1);
            for (int j = 0; j <= p; ++j) {
                int col = span - p + j;
                if (col >= 0 && col < num_cp) {
                    a_matrix(row, col) = ders(1, j);
                }
            }
            b.row(row) = initial_velocity_.value().transpose();
            ++row;
        }

        if (final_velocity_.has_value() && row < total_rows) {
            const int span = temp_spline.find_knot_span(times[n]);
            const Eigen::MatrixXd ders =
                temp_spline.basis_function_derivatives(times[n], span, 1);
            for (int j = 0; j <= p; ++j) {
                int col = span - p + j;
                if (col >= 0 && col < num_cp) {
                    a_matrix(row, col) = ders(1, j);
                }
            }
            b.row(row) = final_velocity_.value().transpose();
            ++row;
        }

        // Acceleration constraints
        if (initial_acceleration_.has_value() && row < total_rows) {
            const int span = temp_spline.find_knot_span(times[0]);
            const Eigen::MatrixXd ders =
                temp_spline.basis_function_derivatives(times[0], span, 2);
            for (int j = 0; j <= p; ++j) {
                int col = span - p + j;
                if (col >= 0 && col < num_cp) {
                    a_matrix(row, col) = ders(2, j);
                }
            }
            b.row(row) = initial_acceleration_.value().transpose();
            ++row;
        }

        if (final_acceleration_.has_value() && row < total_rows) {
            const int span = temp_spline.find_knot_span(times[n]);
            const Eigen::MatrixXd ders =
                temp_spline.basis_function_derivatives(times[n], span, 2);
            for (int j = 0; j <= p; ++j) {
                int col = span - p + j;
                if (col >= 0 && col < num_cp) {
                    a_matrix(row, col) = ders(2, j);
                }
            }
            b.row(row) = final_acceleration_.value().transpose();
            ++row;
        }

        // Fill remaining rows with natural spline conditions
        while (row < total_rows) {
            int deriv_order = std::min(p - 1, 2);
            double t = (row % 2 == 0) ? times[0] : times[n];

            const int span = temp_spline.find_knot_span(t);
            const Eigen::MatrixXd ders =
                temp_spline.basis_function_derivatives(t, span, deriv_order);

            for (int j = 0; j <= p; ++j) {
                int col = span - p + j;
                if (col >= 0 && col < num_cp) {
                    a_matrix(row, col) = ders(deriv_order, j);
                }
            }
            ++row;
        }
    }

    // Solve using ColPivHouseholderQR (handles rank-deficient systems gracefully)
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> solver(a_matrix);
    solver.setThreshold(kRegularizationEps);

    Eigen::MatrixXd cp(num_cp, dim);
    for (int d = 0; d < dim; ++d) {
        cp.col(d) = solver.solve(b.col(d));
    }

    return cp;
}

BSplineInterpolator::BSplineInterpolator(
    int degree, const Eigen::MatrixXd& points,
    const std::optional<Eigen::VectorXd>& times,
    const std::optional<Eigen::VectorXd>& initial_velocity,
    const std::optional<Eigen::VectorXd>& final_velocity,
    const std::optional<Eigen::VectorXd>& initial_acceleration,
    const std::optional<Eigen::VectorXd>& final_acceleration, bool cyclic)
    : BSpline(DeferInit{})
{
    if (degree != kCubicDegree && degree != kQuarticDegree && degree != kQuinticDegree) {
        throw std::invalid_argument("Degree must be 3, 4, or 5, got " +
                                    std::to_string(degree));
    }

    Eigen::MatrixXd pts = points;
    const int num_points = static_cast<int>(pts.rows());
    const int min_points = degree + 1;

    if (num_points < min_points) {
        throw std::invalid_argument(
            "Not enough points for degree " + std::to_string(degree) +
            " B-spline interpolation. Need at least " + std::to_string(min_points) +
            " points, but got " + std::to_string(num_points) + ".");
    }

    interp_points_ = pts;
    initial_velocity_ = initial_velocity;
    final_velocity_ = final_velocity;
    initial_acceleration_ = initial_acceleration;
    final_acceleration_ = final_acceleration;
    cyclic_ = cyclic;

    if (times.has_value()) {
        times_ = times.value();
    } else {
        times_ = Eigen::VectorXd::LinSpaced(num_points, 0.0,
                                            static_cast<double>(num_points - 1));
    }

    Eigen::VectorXd new_knots = create_knot_vector(degree, pts, times_);
    Eigen::MatrixXd new_cp = compute_control_points(degree, pts, times_, new_knots);

    // Reinitialize base class
    degree_ = degree;
    knots_ = new_knots;
    control_points_ = new_cp;
    u_min_ = knots_[degree];
    u_max_ = knots_[static_cast<int>(knots_.size()) - degree - 1];
    dimension_ = static_cast<int>(pts.cols());
}

}  // namespace interpolatecpp::bspline
