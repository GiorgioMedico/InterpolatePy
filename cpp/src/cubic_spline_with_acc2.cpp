#include <interpolatecpp/spline/cubic_spline_with_acc2.hpp>

#include <Eigen/Dense>
#include <iostream>

namespace interpolatecpp::spline {

CubicSplineWithAcceleration2::CubicSplineWithAcceleration2(std::span<const double> t_points,
                                                            std::span<const double> q_points,
                                                            const SplineParameters& params)
    : CubicSpline(t_points, q_points, params.v0, params.vn, params.debug),
      a0_(params.a0),
      an_(params.an) {
    if (params.a0.has_value()) {
        replace_first_segment_with_quintic();
    }
    if (params.an.has_value()) {
        replace_last_segment_with_quintic();
    }
}

void CubicSplineWithAcceleration2::replace_first_segment_with_quintic() {
    double q0 = q_points_(0);
    double q1 = q_points_(1);
    double v0 = velocities_(0);
    double v1 = velocities_(1);
    double a0 = a0_.value();

    // Acceleration at end of first segment from cubic coefficients
    double a1 = 2.0 * coefficients_(0, 2) + 6.0 * coefficients_(0, 3) * t_intervals_(0);

    double T = t_intervals_(0);

    // 6x6 system: p(0)=q0, p'(0)=v0, p''(0)=a0, p(T)=q1, p'(T)=v1, p''(T)=a1
    Eigen::Matrix<double, 6, 6> A;
    A << 1, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0,
         0, 0, 2, 0, 0, 0,
         1, T, T*T, T*T*T, T*T*T*T, T*T*T*T*T,
         0, 1, 2*T, 3*T*T, 4*T*T*T, 5*T*T*T*T,
         0, 0, 2, 6*T, 12*T*T, 20*T*T*T;

    Eigen::Vector<double, 6> b;
    b << q0, v0, a0, q1, v1, a1;

    quintic_first_ = A.colPivHouseholderQr().solve(b);

    if (debug_) {
        std::cout << "\nReplaced first segment with quintic polynomial:\n";
        for (int i = 0; i < 6; ++i) {
            std::cout << "  b" << i << " = " << quintic_first_.value()(i) << "\n";
        }
    }
}

void CubicSplineWithAcceleration2::replace_last_segment_with_quintic() {
    double qn_1 = q_points_(n_);        // last point
    double qn_1_prev = q_points_(n_ - 1); // second-to-last
    double vn = velocities_(n_);
    double vn_1 = velocities_(n_ - 1);
    double an = an_.value();

    // Acceleration at start of last segment from cubic coefficients
    double an_1 = 2.0 * coefficients_(n_ - 1, 2);

    double T = t_intervals_(n_ - 1);

    // 6x6 system
    Eigen::Matrix<double, 6, 6> A;
    A << 1, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0,
         0, 0, 2, 0, 0, 0,
         1, T, T*T, T*T*T, T*T*T*T, T*T*T*T*T,
         0, 1, 2*T, 3*T*T, 4*T*T*T, 5*T*T*T*T,
         0, 0, 2, 6*T, 12*T*T, 20*T*T*T;

    Eigen::Vector<double, 6> b;
    b << qn_1_prev, vn_1, an_1, qn_1, vn, an;

    quintic_last_ = A.colPivHouseholderQr().solve(b);

    if (debug_) {
        std::cout << "\nReplaced last segment with quintic polynomial:\n";
        for (int i = 0; i < 6; ++i) {
            std::cout << "  b" << i << " = " << quintic_last_.value()(i) << "\n";
        }
    }
}

double CubicSplineWithAcceleration2::evaluate(double t) const {
    auto [k, tau] = find_segment(t);

    if (k == 0 && quintic_first_.has_value()) {
        const auto& b = quintic_first_.value();
        return b(0) + b(1) * tau + b(2) * tau * tau + b(3) * tau * tau * tau +
               b(4) * tau * tau * tau * tau + b(5) * tau * tau * tau * tau * tau;
    }
    if (k == n_ - 1 && quintic_last_.has_value()) {
        const auto& b = quintic_last_.value();
        return b(0) + b(1) * tau + b(2) * tau * tau + b(3) * tau * tau * tau +
               b(4) * tau * tau * tau * tau + b(5) * tau * tau * tau * tau * tau;
    }

    double a0 = coefficients_(k, 0);
    double a1 = coefficients_(k, 1);
    double a2 = coefficients_(k, 2);
    double a3 = coefficients_(k, 3);
    return a0 + a1 * tau + a2 * tau * tau + a3 * tau * tau * tau;
}

Eigen::VectorXd CubicSplineWithAcceleration2::evaluate(const Eigen::VectorXd& t) const {
    Eigen::VectorXd result(t.size());
    for (Eigen::Index i = 0; i < t.size(); ++i) {
        result(i) = evaluate(t(i));
    }
    return result;
}

double CubicSplineWithAcceleration2::evaluate_velocity(double t) const {
    auto [k, tau] = find_segment(t);

    if (k == 0 && quintic_first_.has_value()) {
        const auto& b = quintic_first_.value();
        return b(1) + 2.0 * b(2) * tau + 3.0 * b(3) * tau * tau +
               4.0 * b(4) * tau * tau * tau + 5.0 * b(5) * tau * tau * tau * tau;
    }
    if (k == n_ - 1 && quintic_last_.has_value()) {
        const auto& b = quintic_last_.value();
        return b(1) + 2.0 * b(2) * tau + 3.0 * b(3) * tau * tau +
               4.0 * b(4) * tau * tau * tau + 5.0 * b(5) * tau * tau * tau * tau;
    }

    double a1 = coefficients_(k, 1);
    double a2 = coefficients_(k, 2);
    double a3 = coefficients_(k, 3);
    return a1 + 2.0 * a2 * tau + 3.0 * a3 * tau * tau;
}

Eigen::VectorXd CubicSplineWithAcceleration2::evaluate_velocity(const Eigen::VectorXd& t) const {
    Eigen::VectorXd result(t.size());
    for (Eigen::Index i = 0; i < t.size(); ++i) {
        result(i) = evaluate_velocity(t(i));
    }
    return result;
}

double CubicSplineWithAcceleration2::evaluate_acceleration(double t) const {
    auto [k, tau] = find_segment(t);

    if (k == 0 && quintic_first_.has_value()) {
        const auto& b = quintic_first_.value();
        return 2.0 * b(2) + 6.0 * b(3) * tau + 12.0 * b(4) * tau * tau +
               20.0 * b(5) * tau * tau * tau;
    }
    if (k == n_ - 1 && quintic_last_.has_value()) {
        const auto& b = quintic_last_.value();
        return 2.0 * b(2) + 6.0 * b(3) * tau + 12.0 * b(4) * tau * tau +
               20.0 * b(5) * tau * tau * tau;
    }

    double a2 = coefficients_(k, 2);
    double a3 = coefficients_(k, 3);
    return 2.0 * a2 + 6.0 * a3 * tau;
}

Eigen::VectorXd
CubicSplineWithAcceleration2::evaluate_acceleration(const Eigen::VectorXd& t) const {
    Eigen::VectorXd result(t.size());
    for (Eigen::Index i = 0; i < t.size(); ++i) {
        result(i) = evaluate_acceleration(t(i));
    }
    return result;
}

}  // namespace interpolatecpp::spline
