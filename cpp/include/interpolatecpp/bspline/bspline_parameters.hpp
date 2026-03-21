#pragma once

#include <Eigen/Core>
#include <optional>
#include <string>

namespace interpolatecpp::bspline {

/// Parameterization method for computing parameter values from data points.
enum class Parameterization {
    EquallySpaced,
    ChordLength,
    Centripetal,
};

/// Configuration parameters for SmoothingCubicBSpline.
struct BSplineParams {
    double mu = 0.5;
    std::optional<Eigen::VectorXd> weights = std::nullopt;
    std::optional<Eigen::VectorXd> v0 = std::nullopt;
    std::optional<Eigen::VectorXd> vn = std::nullopt;
    Parameterization method = Parameterization::ChordLength;
    bool enforce_endpoints = false;
    bool auto_derivatives = false;
};

}  // namespace interpolatecpp::bspline
