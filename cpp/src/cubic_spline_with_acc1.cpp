#include <interpolatecpp/spline/cubic_spline_with_acc1.hpp>
#include <interpolatecpp/tridiagonal.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace interpolatecpp::spline {

namespace {
constexpr int kMinPoints = 2;
constexpr int kMinSegmentsForUpperDiag = 2;
constexpr int kMinSegmentsForSecondRow = 3;
constexpr int kMinSegmentsForUpperDiagSecondRow = 4;
constexpr int kMinSegmentsForSecondElement = 4;
}  // namespace

CubicSplineWithAcceleration1::CubicSplineWithAcceleration1(std::span<const double> t_points,
                                                            std::span<const double> q_points,
                                                            double v0, double vn, double a0,
                                                            double an, bool debug)
    : v0_(v0), vn_(vn), a0_(a0), an_(an), debug_(debug) {
    if (t_points.size() != q_points.size()) {
        throw std::invalid_argument("Time and position arrays must have the same length");
    }

    n_orig_ = static_cast<int>(t_points.size());
    if (n_orig_ < kMinPoints) {
        throw std::invalid_argument("At least two points are required");
    }

    t_orig_ = Eigen::Map<const Eigen::VectorXd>(t_points.data(), n_orig_);
    q_orig_ = Eigen::Map<const Eigen::VectorXd>(q_points.data(), n_orig_);

    for (int i = 1; i < n_orig_; ++i) {
        if (t_orig_(i) <= t_orig_(i - 1)) {
            throw std::invalid_argument("Time points must be strictly increasing");
        }
    }

    add_extra_points();
    n_ = static_cast<int>(t_.size());
    T_ = t_.tail(n_ - 1) - t_.head(n_ - 1);

    if (debug_) {
        std::cout << "Time interval length: " << T_.transpose() << "\n\n";
    }

    solve_accelerations();
    compute_coefficients();
    compute_original_indices();
}

void CubicSplineWithAcceleration1::add_extra_points() {
    int new_n = n_orig_ + 2;
    t_.resize(new_n);
    q_.resize(new_n);

    t_(0) = t_orig_(0);
    t_(new_n - 1) = t_orig_(n_orig_ - 1);
    q_(0) = q_orig_(0);
    q_(new_n - 1) = q_orig_(n_orig_ - 1);

    // Interior original points
    for (int i = 1; i < n_orig_ - 1; ++i) {
        t_(i + 1) = t_orig_(i);
        q_(i + 1) = q_orig_(i);
    }

    // Extra points at midpoints
    t_(1) = (t_orig_(0) + t_orig_(1)) / 2.0;
    t_(new_n - 2) = (t_orig_(n_orig_ - 2) + t_orig_(n_orig_ - 1)) / 2.0;

    // q values for extra points will be computed after solving accelerations
    q_(1) = 0.0;
    q_(new_n - 2) = 0.0;

    if (debug_) {
        std::cout << "Original times: " << t_orig_.transpose() << "\n";
        std::cout << "New times with extra points: " << t_.transpose() << "\n\n";
    }
}

void CubicSplineWithAcceleration1::solve_accelerations() {
    int n_segments = n_ - 1;
    int sys_size = n_segments - 1;

    Eigen::VectorXd main_diag = Eigen::VectorXd::Zero(sys_size);
    Eigen::VectorXd lower_diag = Eigen::VectorXd::Zero(sys_size);
    Eigen::VectorXd upper_diag = Eigen::VectorXd::Zero(sys_size);

    // First element of main diagonal
    main_diag(0) = 2.0 * T_(1) + T_(0) * (3.0 + T_(0) / T_(1));
    if (n_segments > kMinSegmentsForUpperDiag) {
        upper_diag(0) = T_(1);
    }

    // Last element of main diagonal
    main_diag(sys_size - 1) = 2.0 * T_(n_segments - 2) +
                               T_(n_segments - 1) *
                                   (3.0 + T_(n_segments - 1) / T_(n_segments - 2));
    if (n_segments > kMinSegmentsForUpperDiag) {
        lower_diag(sys_size - 1) = T_(n_segments - 2);
    }

    if (n_segments > kMinSegmentsForSecondRow) {
        // Second row
        lower_diag(1) = T_(1) - (T_(0) * T_(0) / T_(1));
        main_diag(1) = 2.0 * (T_(1) + T_(2));
        if (n_segments > kMinSegmentsForUpperDiagSecondRow) {
            upper_diag(1) = T_(2);
        }

        // Second-to-last row
        main_diag(sys_size - 2) =
            2.0 * (T_(n_segments - 3) + T_(n_segments - 2));
        lower_diag(sys_size - 2) = T_(n_segments - 3);
        upper_diag(sys_size - 2) =
            T_(n_segments - 2) -
            T_(n_segments - 1) * T_(n_segments - 1) / T_(n_segments - 2);
    }

    // Middle rows
    for (int i = 2; i < sys_size - 2; ++i) {
        lower_diag(i) = T_(i);
        main_diag(i) = 2.0 * (T_(i) + T_(i + 1));
        upper_diag(i) = T_(i + 1);
    }

    // Construct RHS vector c (equation 4.28)
    Eigen::VectorXd c = Eigen::VectorXd::Zero(sys_size);

    c(0) = 6.0 * ((q_(2) - q_(0)) / T_(1) - v0_ * (1.0 + T_(0) / T_(1)) -
                   a0_ * (0.5 + T_(0) / (3.0 * T_(1))) * T_(0));

    c(sys_size - 1) =
        6.0 * ((q_(n_ - 3) - q_(n_ - 1)) / T_(n_segments - 2) +
               vn_ * (1.0 + T_(n_segments - 1) / T_(n_segments - 2)) -
               an_ * (0.5 + T_(n_segments - 1) / (3.0 * T_(n_segments - 2))) *
                   T_(n_segments - 1));

    if (n_segments >= kMinSegmentsForSecondElement) {
        c(1) = 6.0 * ((q_(3) - q_(2)) / T_(2) - (q_(2) - q_(0)) / T_(1) +
                       v0_ * T_(0) / T_(1) + a0_ * T_(0) * T_(0) / (3.0 * T_(1)));

        c(sys_size - 2) =
            6.0 *
            ((q_(n_ - 1) - q_(n_ - 3)) / T_(n_segments - 2) -
             (q_(n_ - 3) - q_(n_ - 4)) / T_(n_segments - 3) -
             vn_ * T_(n_segments - 1) / T_(n_segments - 2) +
             an_ * T_(n_segments - 1) * T_(n_segments - 1) / (3.0 * T_(n_segments - 2)));

        // Middle elements
        for (int i = 2; i < sys_size - 2; ++i) {
            if (i == sys_size - 3) {
                continue;  // Already handled second-to-last
            }
            c(i) = 6.0 * ((q_(i + 2) - q_(i + 1)) / T_(i + 1) -
                           (q_(i + 1) - q_(i)) / T_(i));
        }
    }

    if (debug_) {
        std::cout << "Main diagonal: " << main_diag.transpose() << "\n";
        std::cout << "Lower diagonal: " << lower_diag.transpose() << "\n";
        std::cout << "Upper diagonal: " << upper_diag.transpose() << "\n";
        std::cout << "Vector c: " << c.transpose() << "\n\n";
    }

    Eigen::VectorXd interior_omega = solve_tridiagonal(lower_diag, main_diag, upper_diag, c);

    // Complete accelerations vector
    omega_.resize(n_);
    omega_(0) = a0_;
    omega_.segment(1, sys_size) = interior_omega;
    omega_(n_ - 1) = an_;

    // Adjust extra point positions (equations 4.26, 4.27)
    q_(1) = q_(0) + T_(0) * v0_ + (T_(0) * T_(0) / 3.0) * a0_ +
            (T_(0) * T_(0) / 6.0) * omega_(1);
    q_(n_ - 2) = q_(n_ - 1) - T_(n_segments - 1) * vn_ +
                  (T_(n_segments - 1) * T_(n_segments - 1) / 3.0) * an_ +
                  (T_(n_segments - 1) * T_(n_segments - 1) / 6.0) * omega_(n_ - 2);

    if (debug_) {
        std::cout << "Computed accelerations: " << omega_.transpose() << "\n";
        std::cout << "Adjusted q1: " << q_(1) << "\n";
        std::cout << "Adjusted qn-1: " << q_(n_ - 2) << "\n\n";
    }
}

void CubicSplineWithAcceleration1::compute_coefficients() {
    int n_segments = n_ - 1;
    coeffs_.resize(n_segments, 4);

    for (int k = 0; k < n_segments; ++k) {
        coeffs_(k, 0) = q_(k);
        coeffs_(k, 1) =
            (q_(k + 1) - q_(k)) / T_(k) - (T_(k) / 6.0) * (omega_(k + 1) + 2.0 * omega_(k));
        coeffs_(k, 2) = omega_(k) / 2.0;
        coeffs_(k, 3) = (omega_(k + 1) - omega_(k)) / (6.0 * T_(k));
    }

    if (debug_) {
        std::cout << "Polynomial coefficients:\n";
        for (int k = 0; k < n_segments; ++k) {
            std::cout << "Segment " << k << ": " << coeffs_.row(k) << "\n";
        }
    }
}

void CubicSplineWithAcceleration1::compute_original_indices() {
    original_indices_.clear();
    original_indices_.push_back(0);
    for (int i = 1; i < n_orig_ - 1; ++i) {
        original_indices_.push_back(i + 1);
    }
    original_indices_.push_back(n_ - 1);
}

CubicSplineWithAcceleration1::SegmentInfo
CubicSplineWithAcceleration1::find_segment(double t) const {
    if (t <= t_(0)) {
        return {0, 0.0};
    }
    if (t >= t_(n_ - 1)) {
        return {n_ - 2, T_(n_ - 2)};
    }
    auto begin = t_.data();
    auto end = begin + n_;
    auto it = std::upper_bound(begin, end, t);
    int k = static_cast<int>(it - begin) - 1;
    return {k, t - t_(k)};
}

double CubicSplineWithAcceleration1::evaluate(double t) const {
    auto [k, tau] = find_segment(t);
    auto c = coeffs_.row(k);
    return c(0) + c(1) * tau + c(2) * tau * tau + c(3) * tau * tau * tau;
}

Eigen::VectorXd CubicSplineWithAcceleration1::evaluate(const Eigen::VectorXd& t) const {
    Eigen::VectorXd result(t.size());
    for (Eigen::Index i = 0; i < t.size(); ++i) {
        result(i) = evaluate(t(i));
    }
    return result;
}

double CubicSplineWithAcceleration1::evaluate_velocity(double t) const {
    auto [k, tau] = find_segment(t);
    auto c = coeffs_.row(k);
    return c(1) + 2.0 * c(2) * tau + 3.0 * c(3) * tau * tau;
}

Eigen::VectorXd CubicSplineWithAcceleration1::evaluate_velocity(const Eigen::VectorXd& t) const {
    Eigen::VectorXd result(t.size());
    for (Eigen::Index i = 0; i < t.size(); ++i) {
        result(i) = evaluate_velocity(t(i));
    }
    return result;
}

double CubicSplineWithAcceleration1::evaluate_acceleration(double t) const {
    auto [k, tau] = find_segment(t);
    auto c = coeffs_.row(k);
    return 2.0 * c(2) + 6.0 * c(3) * tau;
}

Eigen::VectorXd
CubicSplineWithAcceleration1::evaluate_acceleration(const Eigen::VectorXd& t) const {
    Eigen::VectorXd result(t.size());
    for (Eigen::Index i = 0; i < t.size(); ++i) {
        result(i) = evaluate_acceleration(t(i));
    }
    return result;
}

}  // namespace interpolatecpp::spline
