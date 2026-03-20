#include <interpolatecpp/quat/quaternion_spline.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace interpolatecpp::quat {

QuaternionSpline::QuaternionSpline(const std::vector<double>& time_points,
                                   const std::vector<Quaternion>& quaternions,
                                   Method method)
    : times_(time_points), quaternions_(quaternions), method_(method) {
    if (times_.size() != quaternions_.size()) {
        throw std::invalid_argument("Time points and quaternions must have same length");
    }
    if (times_.size() < 2) {
        throw std::invalid_argument("Need at least 2 quaternions for interpolation");
    }

    compute_intermediates();
}

void QuaternionSpline::compute_intermediates() {
    const int n = static_cast<int>(quaternions_.size());
    intermediates_.resize(n);

    intermediates_[0] = quaternions_[0];
    intermediates_[n - 1] = quaternions_[n - 1];

    for (int i = 1; i < n - 1; ++i) {
        intermediates_[i] = Quaternion::compute_intermediate_quaternion(
            quaternions_[i - 1], quaternions_[i], quaternions_[i + 1]);
    }
}

int QuaternionSpline::find_segment(double t) const {
    const int n = static_cast<int>(times_.size());
    if (t <= times_[0]) return 0;
    if (t >= times_[n - 1]) return n - 2;

    auto it = std::upper_bound(times_.begin(), times_.end(), t);
    int idx = static_cast<int>(it - times_.begin()) - 1;
    return std::clamp(idx, 0, n - 2);
}

Quaternion QuaternionSpline::evaluate(double t) const {
    t = std::clamp(t, times_.front(), times_.back());
    int seg = find_segment(t);
    const int n = static_cast<int>(quaternions_.size());

    double u = (times_[seg + 1] - times_[seg]) > 1e-15
                   ? (t - times_[seg]) / (times_[seg + 1] - times_[seg])
                   : 0.0;

    bool use_squad = false;
    if (method_ == Method::Squad || method_ == Method::Auto) {
        use_squad = (n >= 4 && seg > 0 && seg < n - 2);
    }

    if (use_squad) {
        return Quaternion::squad(quaternions_[seg], intermediates_[seg],
                                intermediates_[seg + 1], quaternions_[seg + 1], u);
    }

    return Quaternion::slerp(quaternions_[seg], quaternions_[seg + 1], u);
}

Eigen::Vector3d QuaternionSpline::evaluate_velocity(double t) const {
    double t_lo = std::max(t - kDt, times_.front());
    double t_hi = std::min(t + kDt, times_.back());
    double dt = t_hi - t_lo;

    if (dt < 1e-15) return Eigen::Vector3d::Zero();

    Quaternion q_lo = evaluate(t_lo);
    Quaternion q_hi = evaluate(t_hi);

    // dq/dt approximation
    Quaternion dq = (q_hi - q_lo) * (1.0 / dt);

    // omega = 2 * dq * q^(-1), take vector part
    Quaternion q_now = evaluate(t);
    Quaternion omega_q = dq * q_now.inverse() * 2.0;

    return omega_q.vec();
}

Eigen::Vector3d QuaternionSpline::evaluate_acceleration(double t) const {
    double t_lo = std::max(t - kDt, times_.front());
    double t_hi = std::min(t + kDt, times_.back());
    double dt = t_hi - t_lo;

    if (dt < 1e-15) return Eigen::Vector3d::Zero();

    Eigen::Vector3d v_lo = evaluate_velocity(t_lo);
    Eigen::Vector3d v_hi = evaluate_velocity(t_hi);

    return (v_hi - v_lo) / dt;
}

}  // namespace interpolatecpp::quat
