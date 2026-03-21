#pragma once

#include <Eigen/Core>

#include <interpolatecpp/config.hpp>

namespace interpolatecpp::path {

/// Result of linear interpolation.
struct LinearTrajResult {
    Eigen::MatrixXd positions;
    Eigen::MatrixXd velocities;
    Eigen::MatrixXd accelerations;
};

/// Simple linear interpolation between two points or scalars.
[[nodiscard]] INTERPOLATECPP_API LinearTrajResult linear_traj(
    const Eigen::VectorXd& p0, const Eigen::VectorXd& p1, double t0, double t1,
    int num_points = 100);

}  // namespace interpolatecpp::path
