#pragma once

#include <Eigen/Core>
#include <functional>
#include <vector>

#include <interpolatecpp/config.hpp>

namespace interpolatecpp::path {

/// Frenet-Serret frame result.
struct FrenetFrame {
    Eigen::Vector3d tangent;
    Eigen::Vector3d normal;
    Eigen::Vector3d binormal;
    double curvature;
    double torsion;
};

/// Compute Frenet-Serret frames along a parametric curve.
///
/// @param curve Function returning (position, velocity, acceleration) at parameter s
/// @param s_values Parameter values at which to compute frames
/// @return Vector of Frenet frames
[[nodiscard]] INTERPOLATECPP_API std::vector<FrenetFrame> compute_frenet_frames(
    const std::function<std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>(
        double)>& curve,
    const Eigen::VectorXd& s_values);

/// Circular trajectory returning (position, velocity, acceleration).
///
/// @param u Parameter value
/// @param r Radius (default 2.0)
/// @return Tuple of (position, dp/du, d2p/du2)
[[nodiscard]] INTERPOLATECPP_API
std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>
circular_trajectory_with_derivatives(double u, double r = 2.0);

/// Helicoidal trajectory returning (position, velocity, acceleration).
///
/// @param u Parameter value
/// @param r Radius (default 2.0)
/// @param d Pitch parameter (default 0.5)
/// @return Tuple of (position, dp/du, d2p/du2)
[[nodiscard]] INTERPOLATECPP_API
std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>
helicoidal_trajectory_with_derivatives(double u, double r = 2.0, double d = 0.5);

}  // namespace interpolatecpp::path
