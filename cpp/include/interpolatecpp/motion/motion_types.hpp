#pragma once

#include <cmath>
#include <stdexcept>
#include <string>

#include <interpolatecpp/config.hpp>

namespace interpolatecpp::motion {

/// Result of trajectory evaluation: position, velocity, acceleration.
struct TrajectoryResult {
    double position;
    double velocity;
    double acceleration;
};

/// Result of full trajectory evaluation with jerk.
struct FullTrajectoryResult {
    double position;
    double velocity;
    double acceleration;
    double jerk;
};

/// Boundary condition for polynomial trajectory.
struct BoundaryCondition {
    double position = 0.0;
    double velocity = 0.0;
    double acceleration = 0.0;
    double jerk = 0.0;
};

/// Time interval for trajectory segments.
struct TimeInterval {
    double start = 0.0;
    double end = 0.0;

    [[nodiscard]] double duration() const noexcept { return end - start; }
};

/// State parameters for trajectory planning (immutable).
struct INTERPOLATECPP_API StateParams {
    double q_0;
    double q_1;
    double v_0;
    double v_1;
};

/// Trajectory bounds (positive values enforced).
struct INTERPOLATECPP_API TrajectoryBounds {
    double v_bound;
    double a_bound;
    double j_bound;

    TrajectoryBounds(double v, double a, double j)
        : v_bound(std::abs(v)), a_bound(std::abs(a)), j_bound(std::abs(j)) {
        if (v_bound <= 0.0 || a_bound <= 0.0 || j_bound <= 0.0) {
            throw std::invalid_argument("Bounds must be positive values");
        }
    }
};

}  // namespace interpolatecpp::motion
