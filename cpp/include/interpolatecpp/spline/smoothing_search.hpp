#pragma once

#include <Eigen/Core>
#include <span>

#include <interpolatecpp/config.hpp>
#include <interpolatecpp/spline/cubic_smoothing_spline.hpp>
#include <interpolatecpp/spline/spline_parameters.hpp>

namespace interpolatecpp::spline {

/// Result of smoothing spline tolerance search.
struct SmoothingSearchResult {
    CubicSmoothingSpline spline;
    double mu;
    double max_error;
    int iterations;
};

/// Find optimal smoothing parameter mu via binary search on tolerance.
///
/// Searches for the smallest mu (most smoothing) that keeps
/// max|q - s| below the given tolerance.
///
/// @param t_points   Time points
/// @param q_points   Position points
/// @param tolerance  Maximum allowed approximation error
/// @param config     Search configuration
/// @return SmoothingSearchResult with optimal spline, mu, error, iterations
INTERPOLATECPP_API SmoothingSearchResult
smoothing_spline_with_tolerance(std::span<const double> t_points,
                                std::span<const double> q_points, double tolerance,
                                const SplineConfig& config);

}  // namespace interpolatecpp::spline
