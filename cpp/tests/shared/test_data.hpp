#pragma once

#include <Eigen/Core>

namespace interpolatecpp::test {

// Tolerance levels matching Python test conventions
inline constexpr double kRegularRtol = 1e-10;
inline constexpr double kRegularAtol = 1e-10;
inline constexpr double kNumericalRtol = 1e-6;
inline constexpr double kNumericalAtol = 1e-6;
inline constexpr double kLooseAtol = 1e-4;

// Common test data: basic waypoints
inline const Eigen::VectorXd kBasicTimes = (Eigen::VectorXd(4) << 0.0, 1.0, 2.0, 3.0).finished();
inline const Eigen::VectorXd kBasicPositions =
    (Eigen::VectorXd(4) << 0.0, 1.0, 0.0, 1.0).finished();

// Linear function test data: y = 2x
inline const Eigen::VectorXd kLinearTimes =
    (Eigen::VectorXd(4) << 0.0, 1.0, 2.0, 3.0).finished();
inline const Eigen::VectorXd kLinearPositions =
    (Eigen::VectorXd(4) << 0.0, 2.0, 4.0, 6.0).finished();

// Quadratic function test data: y = x^2
inline const Eigen::VectorXd kQuadraticTimes =
    (Eigen::VectorXd(4) << 0.0, 1.0, 2.0, 3.0).finished();
inline const Eigen::VectorXd kQuadraticPositions =
    (Eigen::VectorXd(4) << 0.0, 1.0, 4.0, 9.0).finished();

// Symmetric trajectory
inline const Eigen::VectorXd kSymmetricTimes =
    (Eigen::VectorXd(5) << 0.0, 1.0, 2.0, 3.0, 4.0).finished();
inline const Eigen::VectorXd kSymmetricPositions =
    (Eigen::VectorXd(5) << 0.0, 1.0, 2.0, 1.0, 0.0).finished();

}  // namespace interpolatecpp::test
