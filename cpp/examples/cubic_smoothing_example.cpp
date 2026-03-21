/// Cubic smoothing spline example — C++ port of examples/c_s_smoothing_ex.py
/// AND examples/c_s_smoot_search_ex.py
///
/// Part 1: Smoothing spline with varying mu values and weighted endpoints.
/// Part 2: Tolerance-based search for optimal smoothing parameter.

#include <interpolatecpp/spline/cubic_smoothing_spline.hpp>
#include <interpolatecpp/spline/smoothing_search.hpp>
#include <interpolatecpp/spline/spline_parameters.hpp>

#include "example_utils.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace ex = interpolatecpp::examples;
using namespace interpolatecpp::spline;

// ---------------------------------------------------------------------------
// Part 1: Smoothing splines with different mu values
// ---------------------------------------------------------------------------
void smoothing_mu_example() {
    ex::print_header("Part 1: Smoothing Splines with Different mu Values");

    // Define points from the textbook
    std::vector<double> t = {0.0, 5.0, 7.0, 8.0, 10.0, 15.0, 18.0};
    std::vector<double> q = {3.0, -2.0, -5.0, 0.0, 6.0, 12.0, 8.0};

    // Weights: infinite at endpoints (fixed), 1.0 elsewhere
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> weights = {inf, 1.0, 1.0, 1.0, 1.0, 1.0, inf};

    // Create splines with different mu values
    std::vector<double> mu_values = {0.3, 0.6, 1.0};

    for (double mu : mu_values) {
        std::cout << "--- mu = " << mu << " ---\n\n";

        CubicSmoothingSpline spline(
            t, q, mu,
            std::span<const double>(weights),
            0.0, 0.0);

        // Print smoothed vs original positions
        std::cout << "  Smoothed positions (s): ";
        const auto& s = spline.s_points();
        std::cout << "[";
        for (Eigen::Index i = 0; i < s.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(4) << s(i);
        }
        std::cout << "]\n";

        // Print deviation from original
        std::cout << "  Deviation (q - s):      [";
        for (Eigen::Index i = 0; i < s.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(4) << (q[static_cast<size_t>(i)] - s(i));
        }
        std::cout << "]\n";

        ex::print_value("Lambda", spline.lambda());
        ex::print_value("Mu", spline.mu());

        std::cout << "\n";
        ex::print_trajectory_table(
            [&](double tv) { return spline.evaluate(tv); },
            [&](double tv) { return spline.evaluate_velocity(tv); },
            [&](double tv) { return spline.evaluate_acceleration(tv); },
            t.front(), t.back(), 15);
    }

    // Summary comparison: max deviation for each mu
    ex::print_separator('=');
    std::cout << "Summary: max deviation from waypoints\n\n";
    for (double mu : mu_values) {
        CubicSmoothingSpline spline(
            t, q, mu,
            std::span<const double>(weights),
            0.0, 0.0);

        const auto& s = spline.s_points();
        double max_dev = 0.0;
        for (Eigen::Index i = 0; i < s.size(); ++i) {
            max_dev = std::max(max_dev, std::abs(q[static_cast<size_t>(i)] - s(i)));
        }
        std::cout << "  mu=" << std::fixed << std::setprecision(1) << mu
                  << "  max deviation=" << std::setprecision(6) << max_dev << "\n";
    }
}

// ---------------------------------------------------------------------------
// Part 2: Smoothing spline with prescribed tolerance (binary search)
// ---------------------------------------------------------------------------
void smoothing_tolerance_example() {
    ex::print_header("Part 2: Smoothing Spline with Prescribed Tolerance");

    // Same data as Part 1
    std::vector<double> t = {0.0, 5.0, 7.0, 8.0, 10.0, 15.0, 18.0};
    std::vector<double> q = {3.0, -2.0, -5.0, 0.0, 6.0, 12.0, 8.0};

    // Weights with fixed endpoints
    const double inf = std::numeric_limits<double>::infinity();
    Eigen::VectorXd weights(7);
    weights << inf, 1.0, 1.0, 1.0, 1.0, 1.0, inf;

    // Test different tolerance values
    std::vector<double> tolerances = {0.5, 1.0, 2.0};

    SplineConfig config;
    config.weights = weights;
    config.v0 = 0.0;
    config.vn = 0.0;
    config.debug = false;

    for (double tol : tolerances) {
        std::cout << "--- Tolerance = " << std::fixed << std::setprecision(1)
                  << tol << " ---\n\n";

        SmoothingSearchResult result =
            smoothing_spline_with_tolerance(t, q, tol, config);

        ex::print_value("Found mu", result.mu);
        ex::print_value("Max error", result.max_error);
        ex::print_value("Iterations", static_cast<double>(result.iterations), 0);

        // Print smoothed positions
        const auto& s = result.spline.s_points();
        std::cout << "  Smoothed positions: [";
        for (Eigen::Index i = 0; i < s.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(4) << s(i);
        }
        std::cout << "]\n";

        std::cout << "\n";
        ex::print_trajectory_table(
            [&](double tv) { return result.spline.evaluate(tv); },
            [&](double tv) { return result.spline.evaluate_velocity(tv); },
            [&](double tv) { return result.spline.evaluate_acceleration(tv); },
            t.front(), t.back(), 15);
    }

    // Summary
    ex::print_separator('=');
    std::cout << "Tolerance search summary:\n\n";
    const int w = 14;
    std::cout << std::right
              << std::setw(w) << "Tolerance"
              << std::setw(w) << "Mu"
              << std::setw(w) << "Max Error"
              << std::setw(w) << "Iterations"
              << "\n";
    ex::print_separator('-', 4 * w);

    for (double tol : tolerances) {
        SmoothingSearchResult result =
            smoothing_spline_with_tolerance(t, q, tol, config);

        std::cout << std::fixed
                  << std::setprecision(1) << std::setw(w) << tol
                  << std::setprecision(6) << std::setw(w) << result.mu
                  << std::setprecision(6) << std::setw(w) << result.max_error
                  << std::setprecision(0) << std::setw(w) << static_cast<double>(result.iterations)
                  << "\n";
    }
    std::cout << "\n";
}

// ---------------------------------------------------------------------------
int main() {
    smoothing_mu_example();
    smoothing_tolerance_example();

    return 0;
}
