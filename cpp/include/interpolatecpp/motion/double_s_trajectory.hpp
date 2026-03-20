#pragma once

#include <map>
#include <string>

#include <interpolatecpp/config.hpp>
#include <interpolatecpp/motion/motion_types.hpp>

namespace interpolatecpp::motion {

/// Double-S (7-phase) trajectory with bounded velocity, acceleration, and jerk.
///
/// Generates smooth S-curve trajectories consisting of 7 phases:
/// jerk ramp-up, constant acceleration, jerk ramp-down, constant velocity,
/// jerk phase, constant deceleration, jerk ramp-down.
class INTERPOLATECPP_API DoubleSTrajectory {
  public:
    /// Construct a double-S trajectory.
    ///
    /// @param state  Initial and final state parameters
    /// @param bounds Velocity, acceleration, and jerk bounds
    DoubleSTrajectory(const StateParams& state, const TrajectoryBounds& bounds);

    /// Evaluate trajectory at time t (returns position, velocity, acceleration, jerk).
    [[nodiscard]] FullTrajectoryResult evaluate(double t) const;

    /// Get total trajectory duration.
    [[nodiscard]] double duration() const { return T_; }

    /// Get phase durations.
    [[nodiscard]] std::map<std::string, double> phase_durations() const;

  private:
    static constexpr double kEpsilon = 1e-6;
    static constexpr double kMinGamma = 0.01;
    static constexpr int kMaxIterations = 50;

    StateParams state_;
    TrajectoryBounds bounds_;

    // Planned trajectory parameters
    double T_;       // Total duration
    double Ta_;      // Acceleration phase duration
    double Tv_;      // Constant velocity phase duration
    double Td_;      // Deceleration phase duration
    double Tj_1_;    // Jerk time (acceleration)
    double Tj_2_;    // Jerk time (deceleration)
    double a_lim_a_; // Limiting acceleration (accel phase)
    double a_lim_d_; // Limiting acceleration (decel phase)
    double v_lim_;   // Limiting velocity
    double sigma_;   // Direction (+1 or -1)

    void plan_trajectory();
    [[nodiscard]] FullTrajectoryResult evaluate_7phase(double t) const;
};

}  // namespace interpolatecpp::motion
