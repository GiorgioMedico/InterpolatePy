#pragma once

#include <vector>

#include <interpolatecpp/config.hpp>
#include <interpolatecpp/motion/motion_types.hpp>

namespace interpolatecpp::motion {

/// Trapezoidal velocity profile trajectory (3-phase: accel, cruise, decel).
class INTERPOLATECPP_API TrapezoidalTrajectory {
  public:
    /// Construct with velocity/acceleration bounds (computes duration).
    TrapezoidalTrajectory(double q0, double q1, double amax, double vmax,
                          double v0 = 0.0, double v1 = 0.0, double t0 = 0.0);

    /// Tag type to disambiguate duration-based constructor.
    struct DurationBased {};

    /// Construct with fixed duration (computes required acceleration).
    TrapezoidalTrajectory(DurationBased, double q0, double q1, double amax,
                          double v0, double v1, double t0, double duration);

    /// Evaluate trajectory at time t.
    [[nodiscard]] TrajectoryResult evaluate(double t) const;

    /// Get total trajectory duration.
    [[nodiscard]] double duration() const noexcept { return duration_; }

    /// Get start time.
    [[nodiscard]] double t_start() const noexcept { return t0_; }
    [[nodiscard]] double t_end() const noexcept { return t0_ + duration_; }

    /// Compute heuristic velocities for waypoints.
    [[nodiscard]] static std::vector<double> heuristic_velocities(
        const std::vector<double>& points, const std::vector<double>& times,
        double vmax);

    /// Interpolate through waypoints using trapezoidal profiles.
    [[nodiscard]] static std::vector<TrapezoidalTrajectory> interpolate_waypoints(
        const std::vector<double>& points, double amax, double vmax,
        double v0 = 0.0, double vn = 0.0,
        const std::vector<double>& times = {},
        const std::vector<double>& velocities = {});

    /// Evaluate a multipoint trajectory at time t.
    [[nodiscard]] static TrajectoryResult evaluate_multipoint(
        const std::vector<TrapezoidalTrajectory>& segments, double t);

  private:
    static constexpr double kEpsilon = 1e-10;

    double q0_, q1_;
    double v0_, v1_;
    double t0_;
    double duration_;
    double ta_;  // Acceleration phase duration
    double td_;  // Deceleration phase duration
    double vv_;  // Cruise velocity
    int sign_;   // Direction

    void plan_velocity_based(double amax, double vmax);
    void plan_duration_based(double amax, double total_duration);
};

}  // namespace interpolatecpp::motion
