#pragma once

#include <interpolatecpp/quat/shooting_quaternion_interpolation.hpp>

namespace interpolatecpp::quat::shooting_detail {

using State = Eigen::Matrix<double, 10, 1>;

struct Solution {
    Eigen::MatrixXd parameters;
    std::vector<std::vector<State>> nodes;
    int iterations = 0;
    int steps = 0;
    double residual = 0.0;
    double energy = 0.0;
};

State evaluate_step(const State& state, const Eigen::Vector3d& constant, double step);
Solution solve(const std::vector<Quaternion>& quaternions,
               const std::vector<double>& durations, const ShootingConfig& config);

}  // namespace interpolatecpp::quat::shooting_detail
