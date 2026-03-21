#include <interpolatecpp/motion/parabolic_blend_trajectory.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace interpolatecpp::motion {

void ParabolicBlendTrajectory::build_regions() {
    if (n_ <= 0) {
        n_regions_ = 0;
        return;
    }

    if (n_ == 1) {
        // Single waypoint: one blend region
        n_regions_ = 1;
        reg_t0_.resize(1);
        reg_t1_.resize(1);
        reg_q0_.resize(1);
        reg_v0_.resize(1);
        reg_a_.resize(1);

        reg_t0_[0] = t_[0] - dt_blend_[0] / 2.0;
        reg_t1_[0] = t_[0] + dt_blend_[0] / 2.0;
        reg_q0_[0] = q_[0];
        reg_v0_[0] = 0.0;
        reg_a_[0] = 0.0;
        return;
    }

    // Compute segment velocities
    Eigen::VectorXd v_before = Eigen::VectorXd::Zero(n_);
    Eigen::VectorXd v_after = Eigen::VectorXd::Zero(n_);

    for (int k = 0; k < n_ - 1; ++k) {
        double dt = t_[k + 1] - t_[k];
        if (std::abs(dt) > 1e-15) {
            v_before[k + 1] = (q_[k + 1] - q_[k]) / dt;
        }
    }
    if (n_ >= 2) {
        double dt0 = t_[1] - t_[0];
        if (std::abs(dt0) > 1e-15) {
            v_before[0] = 0.0;  // Zero at start
        }
    }

    // v_after[k] = v_before[k+1] for k < N-1, v_after[N-1] = 0
    for (int k = 0; k < n_ - 1; ++k) {
        v_after[k] = v_before[k + 1];
    }
    v_after[n_ - 1] = 0.0;

    // Also set v_before for segment velocities (as linear slopes)
    Eigen::VectorXd seg_v = Eigen::VectorXd::Zero(n_ - 1);
    for (int k = 0; k < n_ - 1; ++k) {
        double dt = t_[k + 1] - t_[k];
        if (std::abs(dt) > 1e-15) {
            seg_v[k] = (q_[k + 1] - q_[k]) / dt;
        }
    }

    // Compute accelerations at each waypoint
    Eigen::VectorXd acc = Eigen::VectorXd::Zero(n_);
    for (int k = 0; k < n_; ++k) {
        double vb = (k > 0) ? seg_v[k - 1] : 0.0;
        double va = (k < n_ - 1) ? seg_v[k] : 0.0;
        double db = dt_blend_[k];
        if (std::abs(db) > 1e-15) {
            acc[k] = (va - vb) / db;
        }
    }

    // Build 2N-1 regions
    n_regions_ = 2 * n_ - 1;
    reg_t0_.resize(n_regions_);
    reg_t1_.resize(n_regions_);
    reg_q0_.resize(n_regions_);
    reg_v0_.resize(n_regions_);
    reg_a_.resize(n_regions_);

    int r = 0;

    // Initial blend
    double blend_start = t_[0] - dt_blend_[0] / 2.0;
    double blend_end = t_[0] + dt_blend_[0] / 2.0;
    reg_t0_[r] = blend_start;
    reg_t1_[r] = blend_end;
    reg_q0_[r] = q_[0] - 0.0 * (dt_blend_[0] / 2.0) -
                 0.5 * acc[0] * (dt_blend_[0] / 2.0) * (dt_blend_[0] / 2.0);
    reg_v0_[r] = 0.0 - acc[0] * (dt_blend_[0] / 2.0);
    // Actually: at blend start, position from quadratic backwards
    // q at blend center = q_[0], v at blend center = (vb+va)/2
    double vb_0 = 0.0;
    reg_v0_[r] = vb_0;
    reg_q0_[r] = q_[0] - vb_0 * (dt_blend_[0] / 2.0) -
                 0.5 * acc[0] * (dt_blend_[0] / 2.0) * (dt_blend_[0] / 2.0);
    reg_a_[r] = acc[0];
    ++r;

    // For each segment k (0 to N-2)
    for (int k = 0; k < n_ - 1; ++k) {
        // Constant-velocity region
        double cv_start = t_[k] + dt_blend_[k] / 2.0;
        double cv_end = t_[k + 1] - dt_blend_[k + 1] / 2.0;

        // Position at start of CV = position at end of previous blend
        double u_prev = reg_t1_[r - 1] - reg_t0_[r - 1];
        double q_cv_start =
            reg_q0_[r - 1] + reg_v0_[r - 1] * u_prev + 0.5 * reg_a_[r - 1] * u_prev * u_prev;

        reg_t0_[r] = cv_start;
        reg_t1_[r] = cv_end;
        reg_q0_[r] = q_cv_start;
        reg_v0_[r] = seg_v[k];
        reg_a_[r] = 0.0;
        ++r;

        // Blend at waypoint k+1
        double blend_s = t_[k + 1] - dt_blend_[k + 1] / 2.0;
        double blend_e = t_[k + 1] + dt_blend_[k + 1] / 2.0;

        // Position at start of blend = position at end of CV
        double u_cv = reg_t1_[r - 1] - reg_t0_[r - 1];
        double q_blend_start = reg_q0_[r - 1] + reg_v0_[r - 1] * u_cv;

        reg_t0_[r] = blend_s;
        reg_t1_[r] = blend_e;
        reg_q0_[r] = q_blend_start;
        reg_v0_[r] = seg_v[k];
        reg_a_[r] = acc[k + 1];
        ++r;
    }
}

int ParabolicBlendTrajectory::find_region(double t_abs) const {
    // Binary search on region start times
    int lo = 0, hi = n_regions_ - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (t_abs >= reg_t0_[mid]) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

ParabolicBlendTrajectory::ParabolicBlendTrajectory(const std::vector<double>& q,
                                                   const std::vector<double>& t,
                                                   const std::vector<double>& dt_blend)
    : n_(static_cast<int>(q.size())) {
    if (q.size() != t.size() || q.size() != dt_blend.size()) {
        throw std::invalid_argument("q, t, and dt_blend must have the same length");
    }

    q_ = Eigen::Map<const Eigen::VectorXd>(q.data(), n_);
    t_ = Eigen::Map<const Eigen::VectorXd>(t.data(), n_);
    dt_blend_ = Eigen::Map<const Eigen::VectorXd>(dt_blend.data(), n_);

    build_regions();
}

TrajectoryResult ParabolicBlendTrajectory::evaluate(double t) const {
    if (n_regions_ == 0) return {0.0, 0.0, 0.0};

    // Clamp to valid range
    double t_min = reg_t0_[0];
    double t_max = reg_t1_[n_regions_ - 1];
    t = std::clamp(t, t_min, t_max);

    int r = find_region(t);
    double u = t - reg_t0_[r];

    double pos = reg_q0_[r] + reg_v0_[r] * u + 0.5 * reg_a_[r] * u * u;
    double vel = reg_v0_[r] + reg_a_[r] * u;
    double acc = reg_a_[r];

    return {pos, vel, acc};
}

double ParabolicBlendTrajectory::duration() const {
    if (n_ < 2) return 0.0;
    return t_[n_ - 1] - t_[0] + (dt_blend_[0] + dt_blend_[n_ - 1]) / 2.0;
}

}  // namespace interpolatecpp::motion
