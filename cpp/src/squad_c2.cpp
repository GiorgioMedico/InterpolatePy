#include <interpolatecpp/quat/squad_c2.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace interpolatecpp::quat {

double SquadC2::quintic_u(double t, double t0, double t1) {
    // Quintic polynomial with zero-clamped boundary conditions:
    // u(t0)=0, u'(t0)=0, u''(t0)=0, u(t1)=1, u'(t1)=0, u''(t1)=0
    double h = t1 - t0;
    if (h < 1e-15) return 0.0;

    double s = (t - t0) / h;
    s = std::clamp(s, 0.0, 1.0);

    // u(s) = 10s³ - 15s⁴ + 6s⁵
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s);
}

void SquadC2::build_extended_sequence() {
    const int n = static_cast<int>(quaternions_.size());
    if (n < 2) return;

    // Extended sequence: [q₁, q₁ᵛⁱʳᵗ, q₂, ..., qₙ₋₁, qₙ₋₁ᵛⁱʳᵗ, qₙ]
    ext_quats_.clear();
    ext_times_.clear();

    // First original point
    ext_quats_.push_back(quaternions_[0]);
    ext_times_.push_back(times_[0]);

    // First virtual = first original quaternion
    if (n == 2) {
        // Special case: dt/3 spacing for 2-waypoint case
        double dt = times_[1] - times_[0];
        ext_quats_.push_back(quaternions_[0]);
        ext_times_.push_back(times_[0] + dt / 3.0);
    } else {
        // General case: midpoint spacing
        double t_mid = (times_[0] + times_[1]) / 2.0;
        ext_quats_.push_back(quaternions_[0]);
        ext_times_.push_back(t_mid);
    }

    // Interior original points
    for (int i = 1; i < n - 1; ++i) {
        ext_quats_.push_back(quaternions_[i]);
        ext_times_.push_back(times_[i]);
    }

    // Last virtual = last original quaternion
    if (n == 2) {
        // Special case: dt/3 spacing for 2-waypoint case
        double dt = times_[1] - times_[0];
        ext_quats_.push_back(quaternions_[n - 1]);
        ext_times_.push_back(times_[1] - dt / 3.0);
    } else {
        // General case: midpoint spacing
        double t_mid2 = (times_[n - 2] + times_[n - 1]) / 2.0;
        ext_quats_.push_back(quaternions_[n - 1]);
        ext_times_.push_back(t_mid2);
    }

    // Last original point
    ext_quats_.push_back(quaternions_[n - 1]);
    ext_times_.push_back(times_[n - 1]);
}

void SquadC2::compute_intermediates() {
    const int n_ext = static_cast<int>(ext_quats_.size());
    ext_intermediates_.resize(n_ext);

    ext_intermediates_[0] = ext_quats_[0];
    ext_intermediates_[n_ext - 1] = ext_quats_[n_ext - 1];

    for (int i = 1; i < n_ext - 1; ++i) {
        double h_prev = ext_times_[i] - ext_times_[i - 1];
        double h_curr = ext_times_[i + 1] - ext_times_[i];

        Quaternion q_inv = ext_quats_[i].inverse();

        // Handle double-cover
        Quaternion next = ext_quats_[i + 1];
        if (ext_quats_[i].dot_product(next) < 0.0) next = -next;
        Quaternion prev = ext_quats_[i - 1];
        if (ext_quats_[i].dot_product(prev) < 0.0) prev = -prev;

        Quaternion log_next = Quaternion::log(q_inv * next);
        Quaternion log_prev = Quaternion::log(q_inv * prev);

        // Corrected Wittmann formula for non-uniform spacing
        double w_next = -1.0 / (2.0 * (1.0 + h_curr / std::max(h_prev, 1e-15)));
        double w_prev = -1.0 / (2.0 * (1.0 + h_prev / std::max(h_curr, 1e-15)));

        Quaternion weighted_sum = log_next * w_next + log_prev * w_prev;
        ext_intermediates_[i] = ext_quats_[i] * Quaternion::exp(weighted_sum);
    }
}

int SquadC2::find_original_segment(double t) const {
    const int n = static_cast<int>(times_.size());
    if (t <= times_[0]) return 0;
    if (t >= times_[n - 1]) return n - 2;

    auto it = std::upper_bound(times_.begin(), times_.end(), t);
    int idx = static_cast<int>(it - times_.begin()) - 1;
    return std::clamp(idx, 0, n - 2);
}

SquadC2::SquadC2(const std::vector<double>& time_points,
                 const std::vector<Quaternion>& quaternions, bool normalize_quaternions,
                 bool validate_continuity)
    : times_(time_points), validate_continuity_(validate_continuity) {
    if (time_points.size() != quaternions.size()) {
        throw std::invalid_argument("Time points and quaternions must have same length");
    }
    if (quaternions.size() < 2) {
        throw std::invalid_argument("Need at least 2 quaternions");
    }
    for (size_t i = 1; i < time_points.size(); ++i) {
        if (time_points[i] <= time_points[i - 1]) {
            throw std::invalid_argument("Time points must be strictly increasing");
        }
    }

    if (normalize_quaternions) {
        quaternions_.reserve(quaternions.size());
        for (const auto& q : quaternions) {
            quaternions_.push_back(q.unit());
        }
    } else {
        quaternions_ = quaternions;
    }

    build_extended_sequence();
    compute_intermediates();
}

SquadC2::SquadC2(const SquadC2Config& config)
    : SquadC2(config.time_points, config.quaternions, config.normalize_quaternions,
              config.validate_continuity) {}

Quaternion SquadC2::evaluate(double t) const {
    // Boundary cases — return original endpoint quaternions
    if (t <= times_.front()) return quaternions_.front();
    if (t >= times_.back()) return quaternions_.back();

    // Find segment in ORIGINAL time points
    int seg = find_original_segment(t);

    // Quintic parameterization over the original segment
    double u = quintic_u(t, times_[seg], times_[seg + 1]);

    // Map original segment index to extended sequence for SQUAD:
    // q1, q2 from original waypoints; s1, s2 from extended intermediates.
    // Extended layout: [q₁, q₁ᵛⁱʳᵗ, q₂, ..., qₙ₋₁, qₙᵛⁱʳᵗ, qₙ]
    // For original segment i: s1 = ext_intermediates_[i+1], s2 = ext_intermediates_[i+2]
    const Quaternion& q1 = quaternions_[seg];
    const Quaternion& q2 = quaternions_[seg + 1];
    const Quaternion& s1 = ext_intermediates_[static_cast<size_t>(seg) + 1];
    const Quaternion& s2 = ext_intermediates_[static_cast<size_t>(seg) + 2];

    return Quaternion::squad(q1, s1, s2, q2, u);
}

Eigen::Vector3d SquadC2::evaluate_velocity(double t) const {
    double t_lo = std::max(t - kDt, times_.front());
    double t_hi = std::min(t + kDt, times_.back());
    double dt = t_hi - t_lo;
    if (dt < 1e-15) return Eigen::Vector3d::Zero();

    Quaternion q_lo = evaluate(t_lo);
    Quaternion q_hi = evaluate(t_hi);
    Quaternion q_now = evaluate(t);

    Quaternion dq = (q_hi - q_lo) * (1.0 / dt);
    Quaternion omega_q = dq * q_now.inverse() * 2.0;
    return omega_q.vec();
}

Eigen::Vector3d SquadC2::evaluate_acceleration(double t) const {
    double t_lo = std::max(t - kDt, times_.front());
    double t_hi = std::min(t + kDt, times_.back());
    double dt = t_hi - t_lo;
    if (dt < 1e-15) return Eigen::Vector3d::Zero();

    return (evaluate_velocity(t_hi) - evaluate_velocity(t_lo)) / dt;
}

}  // namespace interpolatecpp::quat
