#pragma once

#include <Eigen/Core>
#include <optional>

namespace interpolatecpp::spline {

/// Parameters for CubicSplineWithAcceleration2 construction.
/// Mirrors Python's SplineParameters dataclass.
struct SplineParameters {
    double v0 = 0.0;
    double vn = 0.0;
    std::optional<double> a0 = std::nullopt;
    std::optional<double> an = std::nullopt;
    bool debug = false;
};

/// Configuration for smoothing spline tolerance search.
/// Mirrors Python's SplineConfig dataclass.
struct SplineConfig {
    std::optional<Eigen::VectorXd> weights = std::nullopt;
    double v0 = 0.0;
    double vn = 0.0;
    int max_iterations = 50;
    bool debug = false;
};

}  // namespace interpolatecpp::spline
