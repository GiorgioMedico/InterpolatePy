#include <interpolatecpp/bspline/bspline.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace interpolatecpp::bspline {

BSpline::BSpline(int degree, std::span<const double> knots,
                 const Eigen::MatrixXd& control_points)
    : degree_(degree), control_points_(control_points) {

    if (degree < 0) {
        throw std::invalid_argument("Degree must be non-negative");
    }

    // Validate knot vector is non-decreasing
    for (size_t i = 1; i < knots.size(); ++i) {
        if (knots[i] < knots[i - 1]) {
            throw std::invalid_argument("Knot vector must be non-decreasing");
        }
    }

    const auto n_control = static_cast<int>(control_points.rows());
    const auto n_knots = static_cast<int>(knots.size());

    if (n_knots != n_control + degree + 1) {
        throw std::invalid_argument(
            "Invalid knot vector length for the given degree and number of control points. "
            "Expected " +
            std::to_string(n_control + degree + 1) + ", got " + std::to_string(n_knots) +
            ". The relationship must satisfy: n_knots = n_control_points + degree + 1");
    }

    knots_ = Eigen::Map<const Eigen::VectorXd>(knots.data(), n_knots);
    u_min_ = knots_[degree];
    u_max_ = knots_[n_knots - degree - 1];
    dimension_ = static_cast<int>(control_points.cols());
}

int BSpline::find_knot_span(double u) const {
    // Validate parameter range
    if (u < u_min_ - kEps || u > u_max_ + kEps) {
        throw std::invalid_argument("Parameter u=" + std::to_string(u) +
                                    " outside valid range [" + std::to_string(u_min_) + ", " +
                                    std::to_string(u_max_) + "]");
    }

    // Clamp to valid range
    u = std::clamp(u, u_min_, u_max_);

    // Handle endpoint case
    if (std::abs(u - u_max_) <= kEps) {
        return static_cast<int>(knots_.size()) - degree_ - 2;
    }

    // Binary search
    const int n = static_cast<int>(control_points_.rows()) - 1;
    int low = degree_;
    int high = n + 1;

    while (low < high - 1) {
        int mid = (low + high) / 2;
        if (u >= knots_[mid]) {
            low = mid;
        } else {
            high = mid;
        }
    }

    return low;
}

Eigen::VectorXd BSpline::basis_functions(double u, int span_index) const {
    const int p = degree_;
    Eigen::VectorXd n_basis = Eigen::VectorXd::Zero(p + 1);
    Eigen::VectorXd left = Eigen::VectorXd::Zero(p + 1);
    Eigen::VectorXd right = Eigen::VectorXd::Zero(p + 1);

    n_basis[0] = 1.0;

    for (int d = 1; d <= p; ++d) {
        left[d] = u - knots_[span_index + 1 - d];
        right[d] = knots_[span_index + d] - u;

        double saved = 0.0;

        for (int r = 0; r < d; ++r) {
            double denominator = right[r + 1] + left[d - r];
            double temp = (std::abs(denominator) < kEps) ? 0.0 : n_basis[r] / denominator;
            n_basis[r] = saved + right[r + 1] * temp;
            saved = left[d - r] * temp;
        }

        n_basis[d] = saved;
    }

    return n_basis;
}

Eigen::VectorXd BSpline::evaluate(double u) const {
    // Handle endpoints exactly
    if (std::abs(u - u_min_) <= kEps) {
        return control_points_.row(0).transpose();
    }
    if (std::abs(u - u_max_) <= kEps) {
        return control_points_.row(control_points_.rows() - 1).transpose();
    }

    u = std::clamp(u, u_min_, u_max_);

    const int span = find_knot_span(u);
    const Eigen::VectorXd n_basis = basis_functions(u, span);

    Eigen::VectorXd point = Eigen::VectorXd::Zero(dimension_);
    for (int i = 0; i <= degree_; ++i) {
        point += n_basis[i] * control_points_.row(span - degree_ + i).transpose();
    }

    return point;
}

Eigen::MatrixXd BSpline::basis_function_derivatives(double u, int span_index,
                                                    int order) const {
    const int p = degree_;
    order = std::min(order, p);

    Eigen::MatrixXd ders = Eigen::MatrixXd::Zero(order + 1, p + 1);
    Eigen::VectorXd left = Eigen::VectorXd::Zero(p + 1);
    Eigen::VectorXd right = Eigen::VectorXd::Zero(p + 1);
    Eigen::MatrixXd a = Eigen::MatrixXd::Zero(2, p + 1);
    Eigen::MatrixXd ndu = Eigen::MatrixXd::Zero(p + 1, p + 1);

    ndu(0, 0) = 1.0;

    for (int j = 1; j <= p; ++j) {
        left[j] = u - knots_[span_index + 1 - j];
        right[j] = knots_[span_index + j] - u;
        double saved = 0.0;

        for (int r = 0; r < j; ++r) {
            ndu(j, r) = right[r + 1] + left[j - r];
            double temp = (std::abs(ndu(j, r)) < kEps) ? 0.0 : ndu(r, j - 1) / ndu(j, r);
            ndu(r, j) = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu(j, j) = saved;
    }

    // Load basis functions
    for (int j = 0; j <= p; ++j) {
        ders(0, j) = ndu(j, p);
    }

    // Calculate derivatives
    for (int r = 0; r <= p; ++r) {
        int s1 = 0;
        int s2 = 1;
        a(0, 0) = 1.0;

        for (int k = 1; k <= order; ++k) {
            double d = 0.0;
            int rk = r - k;
            int pk = p - k;

            if (r >= k) {
                a(s2, 0) = a(s1, 0) / ndu(pk + 1, rk);
                d = a(s2, 0) * ndu(rk, pk);
            }

            int j1 = (rk >= -1) ? 1 : -rk;
            int j2 = (r - 1 <= pk) ? k - 1 : p - r;

            for (int j = j1; j <= j2; ++j) {
                a(s2, j) = (a(s1, j) - a(s1, j - 1)) / ndu(pk + 1, rk + j);
                d += a(s2, j) * ndu(rk + j, pk);
            }

            if (r <= pk) {
                a(s2, k) = -a(s1, k - 1) / ndu(pk + 1, r);
                d += a(s2, k) * ndu(r, pk);
            }

            ders(k, r) = d;
            std::swap(s1, s2);
        }
    }

    // Multiply by correct factors: p!/(p-k)!
    double r_factor = static_cast<double>(p);
    for (int k = 1; k <= order; ++k) {
        for (int j = 0; j <= p; ++j) {
            ders(k, j) *= r_factor;
        }
        r_factor *= static_cast<double>(p - k);
    }

    return ders;
}

Eigen::VectorXd BSpline::evaluate_derivative(double u, int order) const {
    if (order > degree_) {
        throw std::invalid_argument("Derivative order " + std::to_string(order) +
                                    " exceeds B-spline degree " + std::to_string(degree_));
    }

    if (order == 0) {
        return evaluate(u);
    }

    u = std::clamp(u, u_min_ + kEps, u_max_ - kEps);

    const int span = find_knot_span(u);
    const Eigen::MatrixXd ders = basis_function_derivatives(u, span, order);

    Eigen::VectorXd derivative = Eigen::VectorXd::Zero(dimension_);
    for (int j = 0; j <= degree_; ++j) {
        derivative += ders(order, j) * control_points_.row(span - degree_ + j).transpose();
    }

    return derivative;
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd>
BSpline::generate_curve_points(int num_points) const {
    Eigen::VectorXd u_values = Eigen::VectorXd::LinSpaced(num_points, u_min_, u_max_);
    Eigen::MatrixXd curve_points(num_points, dimension_);

    for (int i = 0; i < num_points; ++i) {
        curve_points.row(i) = evaluate(u_values[i]).transpose();
    }

    return {u_values, curve_points};
}

Eigen::VectorXd BSpline::create_uniform_knots(int degree, int num_control_points,
                                              double domain_min, double domain_max) {
    if (degree < 0) {
        throw std::invalid_argument("Degree must be non-negative");
    }
    if (num_control_points <= degree) {
        throw std::invalid_argument(
            "Number of control points must be greater than the degree");
    }

    const int num_knots = num_control_points + degree + 1;
    Eigen::VectorXd knots = Eigen::VectorXd::Zero(num_knots);

    const int n_internal = num_knots - 2 * (degree + 1);

    // First degree+1 knots
    for (int i = 0; i <= degree; ++i) {
        knots[i] = domain_min;
    }

    // Internal knots
    if (n_internal >= 0) {
        Eigen::VectorXd internal_vals =
            Eigen::VectorXd::LinSpaced(n_internal + 2, domain_min, domain_max);
        for (int i = 0; i < n_internal; ++i) {
            knots[degree + 1 + i] = internal_vals[i + 1];
        }
    }

    // Last degree+1 knots
    for (int i = 0; i <= degree; ++i) {
        knots[num_knots - degree - 1 + i] = domain_max;
    }

    return knots;
}

Eigen::VectorXd BSpline::create_periodic_knots(int degree, int num_control_points,
                                               double domain_min, double domain_max) {
    if (degree < 0) {
        throw std::invalid_argument("Degree must be non-negative");
    }
    if (num_control_points < degree + 1) {
        throw std::invalid_argument(
            "For a periodic B-spline, number of control points must be at least degree+1");
    }

    const int num_knots = num_control_points + degree + 1;
    const int n_regular = num_knots - 2 * degree;
    Eigen::VectorXd regular_knots = Eigen::VectorXd::LinSpaced(n_regular, domain_min, domain_max);

    Eigen::VectorXd knots = Eigen::VectorXd::Zero(num_knots);
    const double step = (domain_max - domain_min) / static_cast<double>(n_regular - 1);

    // Leading extension
    for (int i = 0; i < degree; ++i) {
        knots[i] = domain_min - static_cast<double>(degree - i) * step;
    }

    // Regular knots
    for (int i = 0; i < n_regular; ++i) {
        knots[degree + i] = regular_knots[i];
    }

    // Trailing extension
    for (int i = 0; i < degree; ++i) {
        knots[degree + n_regular + i] = domain_max + static_cast<double>(i + 1) * step;
    }

    return knots;
}

}  // namespace interpolatecpp::bspline
