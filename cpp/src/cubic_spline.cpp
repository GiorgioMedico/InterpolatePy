#include <interpolatecpp/spline/cubic_spline.hpp>
#include <interpolatecpp/tridiagonal.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace interpolatecpp::spline {

CubicSpline::CubicSpline(std::span<const double> t_points, std::span<const double> q_points,
                          double v0, double vn, bool debug)
    : v0_(v0), vn_(vn), debug_(debug) {
    if (t_points.size() != q_points.size()) {
        throw std::invalid_argument("Time points and position points must have the same length");
    }

    const auto n_points = static_cast<int>(t_points.size());
    if (n_points < 2) {
        throw std::invalid_argument("At least two points are required");
    }

    t_points_ = Eigen::Map<const Eigen::VectorXd>(t_points.data(), n_points);
    q_points_ = Eigen::Map<const Eigen::VectorXd>(q_points.data(), n_points);

    // Check strictly increasing
    for (int i = 1; i < n_points; ++i) {
        if (t_points_[i] <= t_points_[i - 1]) {
            throw std::invalid_argument("Time points must be strictly increasing");
        }
    }

    n_ = n_points - 1;
    t_intervals_ = t_points_.tail(n_).array() - t_points_.head(n_).array();

    compute_velocities();
    compute_coefficients();
}

void CubicSpline::compute_velocities() {
    if (n_ == 1) {
        velocities_.resize(2);
        velocities_(0) = v0_;
        velocities_(1) = vn_;
        return;
    }

    // Build RHS: c_i = 3/(T_i*T_{i+1}) * [T_i^2*(q_{i+2}-q_{i+1}) + T_{i+1}^2*(q_{i+1}-q_i)]
    const int m = n_ - 1;  // number of intermediate velocities
    Eigen::VectorXd rhs(m);

    for (int i = 0; i < m; ++i) {
        rhs(i) = 3.0 / (t_intervals_(i) * t_intervals_(i + 1)) *
                 (t_intervals_(i) * t_intervals_(i) * (q_points_(i + 2) - q_points_(i + 1)) +
                  t_intervals_(i + 1) * t_intervals_(i + 1) * (q_points_(i + 1) - q_points_(i)));
    }

    // Adjust for boundary velocities
    rhs(0) -= t_intervals_(1) * v0_;
    rhs(m - 1) -= t_intervals_(n_ - 2) * vn_;

    Eigen::VectorXd v_intermediate;

    if (n_ == 2) {
        // 1x1 system: simple division
        double main_diag_value = 2.0 * (t_intervals_(0) + t_intervals_(1));
        v_intermediate = rhs / main_diag_value;
    } else {
        // Build tridiagonal diagonals
        Eigen::VectorXd main_diag(m);
        Eigen::VectorXd lower_diag = Eigen::VectorXd::Zero(m);
        Eigen::VectorXd upper_diag = Eigen::VectorXd::Zero(m);

        for (int i = 0; i < m; ++i) {
            main_diag(i) = 2.0 * (t_intervals_(i) + t_intervals_(i + 1));
        }
        for (int i = 1; i < m; ++i) {
            lower_diag(i) = t_intervals_(i + 1);
        }
        for (int i = 0; i < m - 1; ++i) {
            upper_diag(i) = t_intervals_(i);
        }

        if (debug_) {
            std::cout << "\nTridiagonal Matrix components:\n";
            std::cout << "Main diagonal: " << main_diag.transpose() << "\n";
            std::cout << "Lower diagonal: " << lower_diag.transpose() << "\n";
            std::cout << "Upper diagonal: " << upper_diag.transpose() << "\n";
            std::cout << "Right-hand side vector: " << rhs.transpose() << "\n";
        }

        v_intermediate = solve_tridiagonal(lower_diag, main_diag, upper_diag, rhs);

        if (debug_) {
            std::cout << "\nIntermediate velocities: " << v_intermediate.transpose() << "\n";
        }
    }

    // Assemble full velocity vector
    velocities_.resize(n_ + 1);
    velocities_(0) = v0_;
    velocities_.segment(1, m) = v_intermediate;
    velocities_(n_) = vn_;

    if (debug_) {
        std::cout << "\nComplete velocity vector: " << velocities_.transpose() << "\n";
    }
}

void CubicSpline::compute_coefficients() {
    coefficients_.resize(n_, 4);

    for (int k = 0; k < n_; ++k) {
        double T = t_intervals_(k);
        double q_k = q_points_(k);
        double q_k1 = q_points_(k + 1);
        double v_k = velocities_(k);
        double v_k1 = velocities_(k + 1);

        coefficients_(k, 0) = q_k;                                                       // a0
        coefficients_(k, 1) = v_k;                                                       // a1
        coefficients_(k, 2) = (1.0 / T) * (3.0 * (q_k1 - q_k) / T - 2.0 * v_k - v_k1); // a2
        coefficients_(k, 3) =
            (1.0 / (T * T)) * (2.0 * (q_k - q_k1) / T + v_k + v_k1);                    // a3

        if (debug_) {
            std::cout << "\nCoefficient calculation for segment " << k << ":\n";
            std::cout << "  a0 = " << coefficients_(k, 0) << "\n";
            std::cout << "  a1 = " << coefficients_(k, 1) << "\n";
            std::cout << "  a2 = " << coefficients_(k, 2) << "\n";
            std::cout << "  a3 = " << coefficients_(k, 3) << "\n";
        }
    }
}

CubicSpline::SegmentInfo CubicSpline::find_segment(double t) const {
    if (t <= t_points_(0)) {
        return {0, 0.0};
    }
    if (t >= t_points_(n_)) {
        return {n_ - 1, t_intervals_(n_ - 1)};
    }
    // Binary search: find largest k such that t_points_[k] <= t
    auto begin = t_points_.data();
    auto end = begin + n_ + 1;
    auto it = std::upper_bound(begin, end, t);
    int k = static_cast<int>(it - begin) - 1;
    return {k, t - t_points_(k)};
}

double CubicSpline::evaluate(double t) const {
    auto [k, tau] = find_segment(t);
    double a0 = coefficients_(k, 0);
    double a1 = coefficients_(k, 1);
    double a2 = coefficients_(k, 2);
    double a3 = coefficients_(k, 3);
    return a0 + a1 * tau + a2 * tau * tau + a3 * tau * tau * tau;
}

Eigen::VectorXd CubicSpline::evaluate(const Eigen::VectorXd& t) const {
    Eigen::VectorXd result(t.size());
    for (Eigen::Index i = 0; i < t.size(); ++i) {
        result(i) = evaluate(t(i));
    }
    return result;
}

double CubicSpline::evaluate_velocity(double t) const {
    auto [k, tau] = find_segment(t);
    double a1 = coefficients_(k, 1);
    double a2 = coefficients_(k, 2);
    double a3 = coefficients_(k, 3);
    return a1 + 2.0 * a2 * tau + 3.0 * a3 * tau * tau;
}

Eigen::VectorXd CubicSpline::evaluate_velocity(const Eigen::VectorXd& t) const {
    Eigen::VectorXd result(t.size());
    for (Eigen::Index i = 0; i < t.size(); ++i) {
        result(i) = evaluate_velocity(t(i));
    }
    return result;
}

double CubicSpline::evaluate_acceleration(double t) const {
    auto [k, tau] = find_segment(t);
    double a2 = coefficients_(k, 2);
    double a3 = coefficients_(k, 3);
    return 2.0 * a2 + 6.0 * a3 * tau;
}

Eigen::VectorXd CubicSpline::evaluate_acceleration(const Eigen::VectorXd& t) const {
    Eigen::VectorXd result(t.size());
    for (Eigen::Index i = 0; i < t.size(); ++i) {
        result(i) = evaluate_acceleration(t(i));
    }
    return result;
}

}  // namespace interpolatecpp::spline
