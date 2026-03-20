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

}  // namespace interpolatecpp::path
