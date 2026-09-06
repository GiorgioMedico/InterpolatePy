#pragma once

#include <Eigen/Core>
#include <interpolatecpp/config.hpp>
#include <interpolatecpp/quat/quaternion.hpp>
#include <utility>
#include <vector>

namespace interpolatecpp::quat {

/// Dimensionless matching tolerance and independently verified RK4 integration.
struct ShootingConfig {
    double tolerance = 1e-8;
    int max_iterations = 30;
    int integration_steps = 16;
    int max_integration_steps = 1024;
};

/// Natural Riemannian cubic rotations, solved by sparse multiple shooting.
/// Each interval has nine unknowns (v, a, c) in local time u = (t - t_i) / h.
/// Physical angular velocity is 2v/h and angular acceleration is 2a/h^2.
/// Matching enforces orientation, velocity and acceleration continuity, with
/// zero endpoint acceleration. The solver finds a local stationary curve;
/// neither global minimality nor convergence for arbitrary data is guaranteed.
class INTERPOLATECPP_API ShootingQuaternionInterpolation {
  public:
    ShootingQuaternionInterpolation(const std::vector<double>& time_points,
                                    const std::vector<Quaternion>& quaternions,
                                    ShootingConfig config = {});
    [[nodiscard]] Quaternion evaluate(double t) const;
    [[nodiscard]] Eigen::Vector3d evaluate_velocity(double t) const;
    [[nodiscard]] Eigen::Vector3d evaluate_acceleration(double t) const;
    [[nodiscard]] std::pair<std::vector<double>, std::vector<Quaternion>>
    generate_trajectory(int num_points = 100) const;
    [[nodiscard]] double t_min() const noexcept { return time_points_.front(); }
    [[nodiscard]] double t_max() const noexcept { return time_points_.back(); }
    [[nodiscard]] int iterations_run() const noexcept { return iterations_run_; }
    [[nodiscard]] int integration_steps() const noexcept { return integration_steps_; }
    [[nodiscard]] int num_variables() const noexcept { return static_cast<int>(9 * durations_.size()); }
    [[nodiscard]] double residual_norm() const noexcept { return residual_norm_; }
    [[nodiscard]] double acceleration_energy() const noexcept { return acceleration_energy_; }
    [[nodiscard]] const std::vector<double>& time_points() const noexcept { return time_points_; }
    [[nodiscard]] const std::vector<Quaternion>& quaternions() const noexcept { return quaternions_; }

  private:
    using State = Eigen::Matrix<double, 10, 1>;
    ShootingConfig config_;
    std::vector<double> time_points_;
    std::vector<double> durations_;
    std::vector<Quaternion> quaternions_;
    Eigen::MatrixXd parameters_;
    std::vector<std::vector<State>> nodes_;
    int iterations_run_ = 0;
    int integration_steps_ = 0;
    double residual_norm_ = 0.0;
    double acceleration_energy_ = 0.0;
    [[nodiscard]] double check_time(double t) const;
    [[nodiscard]] std::pair<State, double> evaluate_state(double t) const;
};

}  // namespace interpolatecpp::quat
