#include <interpolatecpp/motion/trapezoidal_trajectory.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace interpolatecpp::motion {

void TrapezoidalTrajectory::plan_velocity_based(double amax, double vmax) {
    double h = std::abs(q1_ - q0_);
    // Work in positive displacement space
    double q0 = (sign_ > 0) ? q0_ : -q0_;
    double q1 = (sign_ > 0) ? q1_ : -q1_;
    double v0 = (sign_ > 0) ? v0_ : -v0_;
    double v1 = (sign_ > 0) ? v1_ : -v1_;

    h = q1 - q0;

    if (h * amax > vmax * vmax - (v0 * v0 + v1 * v1) / 2.0) {
        // vmax is reached
        vv_ = vmax;
        ta_ = (vmax - v0) / (amax + kEpsilon);
        td_ = (vmax - v1) / (amax + kEpsilon);

        double v0r = v0 / std::max(vmax, kEpsilon);
        double v1r = v1 / std::max(vmax, kEpsilon);
        v0r = std::clamp(v0r, -1.0 + kEpsilon, 1.0 - kEpsilon);
        v1r = std::clamp(v1r, -1.0 + kEpsilon, 1.0 - kEpsilon);

        duration_ = h / std::max(vmax, kEpsilon) +
                    vmax / (2.0 * amax + kEpsilon) * (1.0 - v0r) * (1.0 - v0r) +
                    vmax / (2.0 * amax + kEpsilon) * (1.0 - v1r) * (1.0 - v1r);
    } else {
        // Triangular profile
        double sqrt_term = h * amax + (v0 * v0 + v1 * v1) / 2.0;
        if (sqrt_term < 0) {
            sqrt_term = std::max(sqrt_term, 0.0);
        }
        double vlim = std::sqrt(sqrt_term);
        vv_ = vlim;
        ta_ = (vlim - v0) / (amax + kEpsilon);
        td_ = (vlim - v1) / (amax + kEpsilon);
        duration_ = ta_ + td_;
    }
}

void TrapezoidalTrajectory::plan_duration_based(double amax, double total_duration) {
    double q0 = (sign_ > 0) ? q0_ : -q0_;
    double q1 = (sign_ > 0) ? q1_ : -q1_;
    double v0 = (sign_ > 0) ? v0_ : -v0_;
    double v1 = (sign_ > 0) ? v1_ : -v1_;

    double h = q1 - q0;

    // Feasibility check
    if (amax * h < std::abs(v0 * v0 - v1 * v1) / 2.0) {
        throw std::invalid_argument(
            "Trajectory not feasible. Try increasing amax or reducing velocities.");
    }

    double T2 = total_duration * total_duration;
    double sqrt_term =
        4.0 * h * h - 4.0 * h * (v0 + v1) * total_duration +
        2.0 * (v0 * v0 + v1 * v1) * T2;

    if (sqrt_term < 0) {
        if (sqrt_term > -kEpsilon) {
            sqrt_term = 0.0;
        } else {
            throw std::invalid_argument(
                "Trajectory not feasible with given duration.");
        }
    }

    double alim =
        (2.0 * h - total_duration * (v0 + v1) + std::sqrt(sqrt_term)) /
        std::max(T2, kEpsilon);

    if (amax < alim) {
        amax = alim;
    }

    double vv_sqrt =
        amax * amax * T2 - 4.0 * amax * h +
        2.0 * amax * (v0 + v1) * total_duration - (v0 - v1) * (v0 - v1);

    if (vv_sqrt < 0) {
        vv_sqrt = std::max(vv_sqrt, 0.0);
    }

    vv_ = 0.5 * (v0 + v1 + amax * total_duration - std::sqrt(vv_sqrt));
    ta_ = (vv_ - v0) / (amax + kEpsilon);
    td_ = (vv_ - v1) / (amax + kEpsilon);
    duration_ = total_duration;
}

TrapezoidalTrajectory::TrapezoidalTrajectory(double q0, double q1, double amax,
                                             double vmax, double v0, double v1,
                                             double t0)
    : q0_(q0), q1_(q1), v0_(v0), v1_(v1), t0_(t0), duration_(0), ta_(0), td_(0),
      vv_(0) {
    amax = std::abs(amax);
    vmax = std::abs(vmax);

    double h = q1 - q0;
    sign_ = (h >= 0) ? 1 : -1;

    plan_velocity_based(amax, vmax);
}

TrapezoidalTrajectory::TrapezoidalTrajectory(DurationBased, double q0, double q1,
                                             double amax, double v0, double v1,
                                             double t0, double duration)
    : q0_(q0), q1_(q1), v0_(v0), v1_(v1), t0_(t0), duration_(0), ta_(0), td_(0),
      vv_(0), sign_(1) {
    amax = std::abs(amax);

    double h = q1 - q0;
    sign_ = (h >= 0) ? 1 : -1;

    plan_duration_based(amax, duration);
}

TrajectoryResult TrapezoidalTrajectory::evaluate(double t) const {
    double t1 = t0_ + duration_;
    t = std::clamp(t, t0_, t1);

    // Work in transformed space
    double q0 = (sign_ > 0) ? q0_ : -q0_;
    double q1 = (sign_ > 0) ? q1_ : -q1_;
    double v0 = (sign_ > 0) ? v0_ : -v0_;
    double v1 = (sign_ > 0) ? v1_ : -v1_;

    double ta_safe = std::max(ta_, kEpsilon);
    double td_safe = std::max(td_, kEpsilon);

    double pos = 0.0, vel = 0.0, acc = 0.0;

    if (t < t0_ + ta_safe) {
        double dt = t - t0_;
        pos = q0 + v0 * dt + (vv_ - v0) / (2.0 * ta_safe) * dt * dt;
        vel = v0 + (vv_ - v0) / ta_safe * dt;
        acc = (vv_ - v0) / ta_safe;
    } else if (t < t1 - td_safe) {
        pos = q0 + v0 * ta_safe / 2.0 + vv_ * (t - t0_ - ta_safe / 2.0);
        vel = vv_;
        acc = 0.0;
    } else {
        double dt = t1 - t;
        pos = q1 - v1 * dt - (vv_ - v1) / (2.0 * td_safe) * dt * dt;
        vel = v1 + (vv_ - v1) / td_safe * dt;
        acc = -(vv_ - v1) / td_safe;
    }

    if (sign_ < 0) {
        return {-pos, -vel, -acc};
    }
    return {pos, vel, acc};
}

std::vector<double> TrapezoidalTrajectory::heuristic_velocities(
    const std::vector<double>& points, const std::vector<double>& /*times*/,
    double vmax) {
    const int n = static_cast<int>(points.size());
    std::vector<double> velocities(n, 0.0);

    if (n < 2) return velocities;

    std::vector<double> h_values(n - 1);
    for (int k = 0; k < n - 1; ++k) {
        h_values[k] = points[k + 1] - points[k];
    }

    for (int k = 0; k < n - 2; ++k) {
        if (h_values[k] * h_values[k + 1] <= 0) {
            velocities[k + 1] = 0.0;
        } else {
            velocities[k + 1] = (h_values[k] > 0) ? vmax : -vmax;
        }
    }

    return velocities;
}

std::vector<TrapezoidalTrajectory> TrapezoidalTrajectory::interpolate_waypoints(
    const std::vector<double>& points, double amax, double vmax, double v0,
    double vn, const std::vector<double>& times,
    const std::vector<double>& velocities) {
    const int n = static_cast<int>(points.size());
    if (n < 2) {
        throw std::invalid_argument("Need at least 2 points for interpolation");
    }

    std::vector<double> vels;
    if (!velocities.empty()) {
        vels.resize(n, 0.0);
        vels[0] = v0;
        vels[n - 1] = vn;
        for (int i = 0; i < static_cast<int>(velocities.size()) && i < n - 2; ++i) {
            vels[i + 1] = velocities[i];
        }
    } else {
        vels = heuristic_velocities(points, times, vmax);
        vels[0] = v0;
        vels[n - 1] = vn;
    }

    std::vector<TrapezoidalTrajectory> segments;
    segments.reserve(n - 1);

    double t_current = 0.0;
    for (int i = 0; i < n - 1; ++i) {
        if (!times.empty() && times.size() >= static_cast<size_t>(n)) {
            double seg_duration = times[i + 1] - times[i];
            segments.emplace_back(DurationBased{}, points[i], points[i + 1], amax,
                                  vels[i], vels[i + 1], t_current, seg_duration);
        } else {
            segments.emplace_back(points[i], points[i + 1], amax, vmax, vels[i],
                                  vels[i + 1], t_current);
        }
        t_current += segments.back().duration();
    }

    return segments;
}

TrajectoryResult TrapezoidalTrajectory::evaluate_multipoint(
    const std::vector<TrapezoidalTrajectory>& segments, double t) {
    if (segments.empty()) return {0.0, 0.0, 0.0};

    if (t <= segments.front().t_start()) {
        return segments.front().evaluate(segments.front().t_start());
    }
    if (t >= segments.back().t_end()) {
        return segments.back().evaluate(segments.back().t_end());
    }

    // Binary search
    int lo = 0, hi = static_cast<int>(segments.size()) - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (t >= segments[mid].t_end()) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    return segments[lo].evaluate(t);
}

}  // namespace interpolatecpp::motion
