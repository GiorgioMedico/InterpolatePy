#pragma once

#include <Eigen/Core>
#include <vector>

#include <interpolatecpp/config.hpp>
#include <interpolatecpp/motion/motion_types.hpp>

namespace interpolatecpp::motion {

/// Linear segments with parabolic blends trajectory.
///
/// Creates a trajectory with constant-velocity segments connected by
/// parabolic acceleration blends at each waypoint.
class INTERPOLATECPP_API ParabolicBlendTrajectory {
  public:
    /// Construct from waypoints with blend durations.
    ///
    /// @param q         Position waypoints
    /// @param t         Nominal times at each waypoint
    /// @param dt_blend  Blend duration at each waypoint
    ParabolicBlendTrajectory(const std::vector<double>& q, const std::vector<double>& t,
                             const std::vector<double>& dt_blend);

    /// Evaluate trajectory at time t.
    [[nodiscard]] TrajectoryResult evaluate(double t) const;

    /// Get total trajectory duration.
    [[nodiscard]] double duration() const;

    /// Get number of waypoints.
    [[nodiscard]] int n_waypoints() const noexcept { return n_; }

  private:
    int n_;  // Number of waypoints
    Eigen::VectorXd q_;
    Eigen::VectorXd t_;
    Eigen::VectorXd dt_blend_;

    // Region data (2*N-1 regions: blend + CV pairs)
    Eigen::VectorXd reg_t0_;
    Eigen::VectorXd reg_t1_;
    Eigen::VectorXd reg_q0_;
    Eigen::VectorXd reg_v0_;
    Eigen::VectorXd reg_a_;
    int n_regions_;

    void build_regions();
    [[nodiscard]] int find_region(double t_abs) const;
};

}  // namespace interpolatecpp::motion
