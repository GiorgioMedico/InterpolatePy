#include <interpolatecpp/spline/smoothing_search.hpp>

#include <cmath>
#include <iostream>
#include <optional>

namespace interpolatecpp::spline {

namespace {
constexpr double kEpsilon = 1e-6;
}

SmoothingSearchResult smoothing_spline_with_tolerance(std::span<const double> t_points,
                                                       std::span<const double> q_points,
                                                       double tolerance,
                                                       const SplineConfig& config) {
    double lower_bound = kEpsilon;
    double upper_bound = 1.0;

    if (config.debug) {
        std::cout << "Starting binary search with tolerance delta=" << tolerance << "\n";
        std::cout << "Initial lower_bound=" << lower_bound
                  << ", upper_bound=" << upper_bound << "\n";
    }

    // Weights span (may be empty)
    std::optional<std::span<const double>> weights_span;
    if (config.weights.has_value()) {
        const auto& w = config.weights.value();
        weights_span = std::span<const double>(w.data(), static_cast<size_t>(w.size()));
    }

    // Default fallback: mu=1.0
    CubicSmoothingSpline default_spline(t_points, q_points, 1.0, weights_span, config.v0,
                                         config.vn, false);
    double default_error =
        (default_spline.q_points() - default_spline.s_points()).cwiseAbs().maxCoeff();

    CubicSmoothingSpline best_spline = default_spline;
    double best_mu = 1.0;
    double best_error = default_error;

    for (int i = 0; i < config.max_iterations; ++i) {
        double mu = (lower_bound + upper_bound) / 2.0;

        if (config.debug) {
            std::cout << "\nIteration " << (i + 1) << ": mu=" << mu << "\n";
        }

        try {
            CubicSmoothingSpline spline(t_points, q_points, mu, weights_span, config.v0,
                                         config.vn, false);
            double e_max = (spline.q_points() - spline.s_points()).cwiseAbs().maxCoeff();

            if (config.debug) {
                std::cout << "  Maximum error e_max(" << i << ")=" << e_max << "\n";
            }

            if (e_max < best_error) {
                best_spline = spline;
                best_mu = mu;
                best_error = e_max;
            }

            if (e_max > tolerance) {
                lower_bound = mu;
            } else {
                upper_bound = mu;
            }

            // Convergence check
            if (std::abs(e_max - tolerance) < kEpsilon ||
                (e_max < tolerance && upper_bound - lower_bound < kEpsilon)) {
                if (config.debug) {
                    std::cout << "\nConverged with error " << e_max << " after " << (i + 1)
                              << " iterations\n";
                }
                return {std::move(spline), mu, e_max, i + 1};
            }
        } catch (const std::exception& e) {
            if (config.debug) {
                std::cout << "  Error with mu=" << mu << ": " << e.what() << "\n";
            }
            lower_bound = mu;
        }
    }

    if (config.debug) {
        std::cout << "\nReached maximum iterations (" << config.max_iterations << ")\n";
        std::cout << "Best solution found: mu=" << best_mu << ", error=" << best_error << "\n";
    }

    return {std::move(best_spline), best_mu, best_error, config.max_iterations};
}

}  // namespace interpolatecpp::spline
