#include <interpolatecpp/spline/cubic_smoothing_spline.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace interpolatecpp::spline {

CubicSmoothingSpline::CubicSmoothingSpline(std::span<const double> t_points,
                                            std::span<const double> q_points, double mu,
                                            std::optional<std::span<const double>> weights,
                                            double v0, double vn, bool debug)
    : mu_(mu), v0_(v0), vn_(vn), debug_(debug) {
    if (t_points.size() != q_points.size()) {
        throw std::invalid_argument("Time and position arrays must have the same length");
    }

    n_ = static_cast<int>(t_points.size());

    if (n_ < kMinPointsRequired) {
        throw std::invalid_argument("At least two points are required");
    }

    t_ = Eigen::Map<const Eigen::VectorXd>(t_points.data(), n_);
    q_ = Eigen::Map<const Eigen::VectorXd>(q_points.data(), n_);

    // Check strictly increasing
    for (int i = 1; i < n_; ++i) {
        if (t_(i) <= t_(i - 1)) {
            throw std::invalid_argument("Time points must be strictly increasing");
        }
    }

    if (mu_ <= 0.0 || mu_ > 1.0) {
        throw std::invalid_argument("Parameter mu must be in range (0, 1]");
    }

    // Compute lambda
    lambd_ = (mu_ < 1.0) ? (1.0 - mu_) / (6.0 * mu_ + 1e-10) : 0.0;

    // Compute time intervals
    time_intervals_ = t_.tail(n_ - 1) - t_.head(n_ - 1);

    // Initialize weights
    if (weights.has_value()) {
        auto w_span = weights.value();
        if (static_cast<int>(w_span.size()) != n_) {
            throw std::invalid_argument(
                "Weights array must have the same length as time and position arrays");
        }
        w_ = Eigen::Map<const Eigen::VectorXd>(w_span.data(), n_);
    } else {
        w_ = Eigen::VectorXd::Ones(n_);
    }

    // Handle infinite weights
    w_inv_ = Eigen::VectorXd::Zero(n_);
    for (int i = 0; i < n_; ++i) {
        if (std::isfinite(w_(i))) {
            w_inv_(i) = 1.0 / w_(i);
        }
    }

    if (debug_) {
        std::cout << "Smoothing parameter mu: " << mu_ << "\n";
        std::cout << "Lambda: " << lambd_ << "\n";
        std::cout << "Weights: " << w_.transpose() << "\n";
        std::cout << "Inverse weights: " << w_inv_.transpose() << "\n";
    }

    construct_matrices();
    solve_system();
    compute_positions();
    compute_coefficients();

    if (debug_) {
        std::cout << "Original points: " << q_.transpose() << "\n";
        std::cout << "Approximated points: " << s_.transpose() << "\n";
        std::cout << "Accelerations: " << omega_.transpose() << "\n";
        std::cout << "Maximum position error: " << (q_ - s_).cwiseAbs().maxCoeff() << "\n";
    }
}

void CubicSmoothingSpline::construct_matrices() {
    a_matrix_ = Eigen::MatrixXd::Zero(n_, n_);
    c_matrix_ = Eigen::MatrixXd::Zero(n_, n_);

    // Fill A matrix (equation 4.23)
    a_matrix_(0, 0) = 2.0 * time_intervals_(0);
    a_matrix_(n_ - 1, n_ - 1) = 2.0 * time_intervals_(n_ - 2);

    for (int i = 1; i < n_ - 1; ++i) {
        a_matrix_(i, i) = 2.0 * (time_intervals_(i - 1) + time_intervals_(i));
    }

    for (int i = 0; i < n_ - 1; ++i) {
        a_matrix_(i, i + 1) = time_intervals_(i);
        a_matrix_(i + 1, i) = time_intervals_(i);
    }

    // Fill C matrix (equation 4.34)
    c_matrix_(0, 0) = -6.0 / time_intervals_(0);
    c_matrix_(0, 1) = 6.0 / time_intervals_(0);
    c_matrix_(n_ - 1, n_ - 2) = 6.0 / time_intervals_(n_ - 2);
    c_matrix_(n_ - 1, n_ - 1) = -6.0 / time_intervals_(n_ - 2);

    for (int i = 1; i < n_ - 1; ++i) {
        c_matrix_(i, i - 1) = 6.0 / time_intervals_(i - 1);
        c_matrix_(i, i) = -(6.0 / time_intervals_(i - 1) + 6.0 / time_intervals_(i));
        c_matrix_(i, i + 1) = 6.0 / time_intervals_(i);
    }

    if (debug_) {
        std::cout << "Matrix A:\n" << a_matrix_ << "\n";
        std::cout << "Matrix C:\n" << c_matrix_ << "\n";
    }
}

void CubicSmoothingSpline::solve_system() {
    if (mu_ == 1.0) {
        // Pure interpolation: A*omega = c (equation 4.22/4.24)
        Eigen::VectorXd c = Eigen::VectorXd::Zero(n_);

        c(0) = 6.0 * ((q_(1) - q_(0)) / time_intervals_(0) - v0_);
        c(n_ - 1) = 6.0 * (vn_ - (q_(n_ - 1) - q_(n_ - 2)) / time_intervals_(n_ - 2));

        for (int i = 1; i < n_ - 1; ++i) {
            c(i) = 6.0 * ((q_(i + 1) - q_(i)) / time_intervals_(i) -
                           (q_(i) - q_(i - 1)) / time_intervals_(i - 1));
        }

        if (debug_) {
            std::cout << "Vector c (pure interpolation): " << c.transpose() << "\n";
        }

        omega_ = a_matrix_.colPivHouseholderQr().solve(c);
        return;
    }

    // Smoothing spline: (A + lambda*C*W_inv*C^T)*omega = C*q
    Eigen::VectorXd rhs = c_matrix_ * q_;
    Eigen::MatrixXd w_inv_diag = w_inv_.asDiagonal();
    Eigen::MatrixXd c_w_inv_ct = c_matrix_ * w_inv_diag * c_matrix_.transpose();

    // Symmetrize
    c_w_inv_ct = (c_w_inv_ct + c_w_inv_ct.transpose()) / 2.0;

    Eigen::MatrixXd system_matrix = a_matrix_ + lambd_ * c_w_inv_ct;
    system_matrix = (system_matrix + system_matrix.transpose()) / 2.0;

    if (debug_) {
        std::cout << "System matrix (smoothing):\n" << system_matrix << "\n";
        std::cout << "RHS vector: " << rhs.transpose() << "\n";
    }

    // Check condition number and regularize if needed
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(system_matrix);
    double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);

    if (cond > kHighConditionThreshold) {
        system_matrix += Eigen::MatrixXd::Identity(n_, n_) * kRegularizationFactor;
        if (debug_) {
            std::cout << "Added regularization. New condition number will be lower.\n";
        }
    }

    omega_ = system_matrix.colPivHouseholderQr().solve(rhs);
}

void CubicSmoothingSpline::compute_positions() {
    if (mu_ == 1.0) {
        s_ = q_;
        return;
    }

    Eigen::VectorXd ct_omega = c_matrix_.transpose() * omega_;
    Eigen::VectorXd adjustment = lambd_ * (w_inv_.array() * ct_omega.array()).matrix();
    s_ = q_ - adjustment;

    if (debug_) {
        std::cout << "Computed s: " << s_.transpose() << " for mu: " << mu_ << "\n";
    }
}

void CubicSmoothingSpline::compute_coefficients() {
    int n_segments = n_ - 1;
    coeffs_.resize(n_segments, 4);

    for (int k = 0; k < n_segments; ++k) {
        double T = time_intervals_(k);

        coeffs_(k, 0) = s_(k);
        coeffs_(k, 1) = (s_(k + 1) - s_(k)) / T - (T / 6.0) * (omega_(k + 1) + 2.0 * omega_(k));
        coeffs_(k, 2) = omega_(k) / 2.0;
        coeffs_(k, 3) = (omega_(k + 1) - omega_(k)) / (6.0 * T);
    }

    if (debug_) {
        std::cout << "Polynomial coefficients:\n";
        for (int k = 0; k < n_segments; ++k) {
            std::cout << "Segment " << k << ": " << coeffs_.row(k) << "\n";
        }
    }
}

CubicSmoothingSpline::SegmentInfo CubicSmoothingSpline::find_segment(double t) const {
    if (t <= t_(0)) {
        return {0, t - t_(0)};
    }
    if (t >= t_(n_ - 1)) {
        return {n_ - 2, t - t_(n_ - 2)};
    }
    auto begin = t_.data();
    auto end = begin + n_;
    auto it = std::upper_bound(begin, end, t);
    int k = static_cast<int>(it - begin) - 1;
    return {k, t - t_(k)};
}

double CubicSmoothingSpline::evaluate(double t) const {
    auto [k, tau] = find_segment(t);
    double c0 = coeffs_(k, 0);
    double c1 = coeffs_(k, 1);
    double c2 = coeffs_(k, 2);
    double c3 = coeffs_(k, 3);
    return c0 + c1 * tau + c2 * tau * tau + c3 * tau * tau * tau;
}

Eigen::VectorXd CubicSmoothingSpline::evaluate(const Eigen::VectorXd& t) const {
    Eigen::VectorXd result(t.size());
    for (Eigen::Index i = 0; i < t.size(); ++i) {
        result(i) = evaluate(t(i));
    }
    return result;
}

double CubicSmoothingSpline::evaluate_velocity(double t) const {
    auto [k, tau] = find_segment(t);
    double c1 = coeffs_(k, 1);
    double c2 = coeffs_(k, 2);
    double c3 = coeffs_(k, 3);
    return c1 + 2.0 * c2 * tau + 3.0 * c3 * tau * tau;
}

Eigen::VectorXd CubicSmoothingSpline::evaluate_velocity(const Eigen::VectorXd& t) const {
    Eigen::VectorXd result(t.size());
    for (Eigen::Index i = 0; i < t.size(); ++i) {
        result(i) = evaluate_velocity(t(i));
    }
    return result;
}

double CubicSmoothingSpline::evaluate_acceleration(double t) const {
    auto [k, tau] = find_segment(t);
    double c2 = coeffs_(k, 2);
    double c3 = coeffs_(k, 3);
    return 2.0 * c2 + 6.0 * c3 * tau;
}

Eigen::VectorXd CubicSmoothingSpline::evaluate_acceleration(const Eigen::VectorXd& t) const {
    Eigen::VectorXd result(t.size());
    for (Eigen::Index i = 0; i < t.size(); ++i) {
        result(i) = evaluate_acceleration(t(i));
    }
    return result;
}

}  // namespace interpolatecpp::spline
