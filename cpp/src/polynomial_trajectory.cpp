#include <interpolatecpp/motion/polynomial_trajectory.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace interpolatecpp::motion {

void PolynomialTrajectory::compute_order3(const BoundaryCondition& start,
                                          const BoundaryCondition& end, double /*h*/) {
    double T = t_end_ - t_start_;
    double disp = end.position - start.position;

    coeffs_.resize(4);
    coeffs_[0] = start.position;
    coeffs_[1] = start.velocity;
    coeffs_[2] = (3.0 * disp - (2.0 * start.velocity + end.velocity) * T) / (T * T);
    coeffs_[3] = (-2.0 * disp + (start.velocity + end.velocity) * T) / (T * T * T);
}

void PolynomialTrajectory::compute_order5(const BoundaryCondition& start,
                                          const BoundaryCondition& end, double /*h*/) {
    double T = t_end_ - t_start_;
    double disp = end.position - start.position;
    double T2 = T * T;
    double T3 = T2 * T;
    double T4 = T3 * T;
    double T5 = T4 * T;

    coeffs_.resize(6);
    coeffs_[0] = start.position;
    coeffs_[1] = start.velocity;
    coeffs_[2] = start.acceleration / 2.0;
    coeffs_[3] = (1.0 / (2.0 * T3)) *
                 (20.0 * disp - (8.0 * end.velocity + 12.0 * start.velocity) * T -
                  (3.0 * start.acceleration - end.acceleration) * T2);
    coeffs_[4] = (1.0 / (2.0 * T4)) *
                 (-30.0 * disp + (14.0 * end.velocity + 16.0 * start.velocity) * T +
                  (3.0 * start.acceleration - 2.0 * end.acceleration) * T2);
    coeffs_[5] = (1.0 / (2.0 * T5)) *
                 (12.0 * disp - 6.0 * (end.velocity + start.velocity) * T +
                  (end.acceleration - start.acceleration) * T2);
}

void PolynomialTrajectory::compute_order7(const BoundaryCondition& start,
                                          const BoundaryCondition& end, double /*h*/) {
    double T = t_end_ - t_start_;
    double disp = end.position - start.position;
    double T2 = T * T;
    double T4 = T2 * T2;
    double T5 = T4 * T;
    double T6 = T5 * T;
    double T7 = T6 * T;

    coeffs_.resize(8);
    coeffs_[0] = start.position;
    coeffs_[1] = start.velocity;
    coeffs_[2] = start.acceleration / 2.0;
    coeffs_[3] = start.jerk / 6.0;
    coeffs_[4] =
        (210.0 * disp -
         T * ((30.0 * start.acceleration - 15.0 * end.acceleration) * T +
              (4.0 * start.jerk + end.jerk) * T2 + 120.0 * start.velocity +
              90.0 * end.velocity)) /
        (6.0 * T4);
    coeffs_[5] =
        (-168.0 * disp +
         T * ((20.0 * start.acceleration - 14.0 * end.acceleration) * T +
              (2.0 * start.jerk + end.jerk) * T2 + 90.0 * start.velocity +
              78.0 * end.velocity)) /
        (2.0 * T5);
    coeffs_[6] =
        (420.0 * disp -
         T * ((45.0 * start.acceleration - 39.0 * end.acceleration) * T +
              (4.0 * start.jerk + 3.0 * end.jerk) * T2 + 216.0 * start.velocity +
              204.0 * end.velocity)) /
        (6.0 * T6);
    coeffs_[7] =
        (-120.0 * disp +
         T * ((12.0 * start.acceleration - 12.0 * end.acceleration) * T +
              (start.jerk + end.jerk) * T2 + 60.0 * start.velocity +
              60.0 * end.velocity)) /
        (6.0 * T7);
}

PolynomialTrajectory::PolynomialTrajectory(const BoundaryCondition& bc_start,
                                           const BoundaryCondition& bc_end,
                                           const TimeInterval& interval, int order)
    : order_(order), t_start_(interval.start), t_end_(interval.end) {
    double h = interval.duration();
    switch (order) {
        case ORDER_3:
            compute_order3(bc_start, bc_end, h);
            break;
        case ORDER_5:
            compute_order5(bc_start, bc_end, h);
            break;
        case ORDER_7:
            compute_order7(bc_start, bc_end, h);
            break;
        default:
            throw std::invalid_argument("Order must be 3, 5, or 7");
    }
}

FullTrajectoryResult PolynomialTrajectory::evaluate(double t) const {
    t = std::clamp(t, t_start_, t_end_);
    double tau = t - t_start_;

    double q = 0.0, qd = 0.0, qdd = 0.0, qddd = 0.0;
    const int n = static_cast<int>(coeffs_.size());

    // Horner-like evaluation for each derivative
    if (n >= 4) {
        // Position
        q = coeffs_[0] + tau * (coeffs_[1] + tau * (coeffs_[2] + tau * coeffs_[3]));
        qd = coeffs_[1] + tau * (2.0 * coeffs_[2] + tau * 3.0 * coeffs_[3]);
        qdd = 2.0 * coeffs_[2] + 6.0 * coeffs_[3] * tau;
        qddd = 6.0 * coeffs_[3];
    }
    if (n >= 6) {
        double tau2 = tau * tau;
        double tau3 = tau2 * tau;
        double tau4 = tau3 * tau;
        q += coeffs_[4] * tau4 + coeffs_[5] * tau4 * tau;
        qd += 4.0 * coeffs_[4] * tau3 + 5.0 * coeffs_[5] * tau4;
        qdd += 12.0 * coeffs_[4] * tau2 + 20.0 * coeffs_[5] * tau3;
        qddd += 24.0 * coeffs_[4] * tau + 60.0 * coeffs_[5] * tau2;
    }
    if (n >= 8) {
        double tau2 = tau * tau;
        double tau3 = tau2 * tau;
        double tau4 = tau3 * tau;
        double tau5 = tau4 * tau;
        double tau6 = tau5 * tau;
        q += coeffs_[6] * tau5 * tau + coeffs_[7] * tau6 * tau;
        qd += 6.0 * coeffs_[6] * tau5 + 7.0 * coeffs_[7] * tau6;
        qdd += 30.0 * coeffs_[6] * tau4 + 42.0 * coeffs_[7] * tau5;
        qddd += 120.0 * coeffs_[6] * tau3 + 210.0 * coeffs_[7] * tau4;
    }

    return {q, qd, qdd, qddd};
}

std::vector<double> PolynomialTrajectory::heuristic_velocities(
    const std::vector<double>& points, const std::vector<double>& times) {
    const int n = static_cast<int>(points.size());
    if (n < 2) {
        throw std::invalid_argument("Need at least 2 points for velocity computation");
    }

    std::vector<double> velocities(n, 0.0);

    // Compute slopes
    std::vector<double> slopes(n - 1);
    for (int i = 0; i < n - 1; ++i) {
        double dt = times[i + 1] - times[i];
        slopes[i] = (dt > 1e-10) ? (points[i + 1] - points[i]) / dt : 0.0;
    }

    // Interior velocities: average of adjacent slopes, zero at sign changes
    for (int i = 1; i < n - 1; ++i) {
        if (slopes[i - 1] * slopes[i] > 0) {
            velocities[i] = (slopes[i - 1] + slopes[i]) / 2.0;
        } else {
            velocities[i] = 0.0;
        }
    }

    return velocities;
}

std::vector<PolynomialTrajectory> PolynomialTrajectory::multipoint_trajectory(
    const std::vector<double>& points, const std::vector<double>& times, int order,
    double v0, double vn) {
    const int n = static_cast<int>(points.size());
    if (n < 2) {
        throw std::invalid_argument("Need at least 2 points");
    }

    auto velocities = heuristic_velocities(points, times);
    velocities[0] = v0;
    velocities[n - 1] = vn;

    std::vector<PolynomialTrajectory> segments;
    segments.reserve(n - 1);

    for (int i = 0; i < n - 1; ++i) {
        BoundaryCondition bc_start{points[i], velocities[i], 0.0, 0.0};
        BoundaryCondition bc_end{points[i + 1], velocities[i + 1], 0.0, 0.0};
        TimeInterval interval{times[i], times[i + 1]};
        segments.emplace_back(bc_start, bc_end, interval, order);
    }

    return segments;
}

FullTrajectoryResult PolynomialTrajectory::evaluate_multipoint(
    const std::vector<PolynomialTrajectory>& segments, double t) {
    if (segments.empty()) {
        return {0.0, 0.0, 0.0, 0.0};
    }

    // Clamp to valid range
    if (t <= segments.front().t_start()) {
        return segments.front().evaluate(segments.front().t_start());
    }
    if (t >= segments.back().t_end()) {
        return segments.back().evaluate(segments.back().t_end());
    }

    // Binary search for segment
    int lo = 0;
    int hi = static_cast<int>(segments.size()) - 1;
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
