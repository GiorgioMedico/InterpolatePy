#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <interpolatecpp/spline/smoothing_search.hpp>

#include "test_data.hpp"

#include <cmath>
#include <vector>

using namespace interpolatecpp::spline;
using namespace interpolatecpp::test;
using Catch::Matchers::WithinAbs;

namespace {
// Generate noisy sine data
void make_noisy_sine(std::vector<double>& t, std::vector<double>& q, int n, double noise) {
    t.resize(static_cast<size_t>(n));
    q.resize(static_cast<size_t>(n));
    // Simple deterministic "noise" using a fixed pattern
    for (int i = 0; i < n; ++i) {
        t[static_cast<size_t>(i)] = 2.0 * M_PI * i / (n - 1);
        double pseudo_noise = noise * std::sin(17.0 * i + 0.3) * std::cos(31.0 * i + 0.7);
        q[static_cast<size_t>(i)] = std::sin(t[static_cast<size_t>(i)]) + pseudo_noise;
    }
}
}  // namespace

TEST_CASE("Smoothing spline with tolerance - basic search", "[smoothing_search]") {
    std::vector<double> t, q;
    make_noisy_sine(t, q, 30, 0.1);

    SplineConfig config;
    config.max_iterations = 30;

    SECTION("Finds spline within tolerance") {
        double tolerance = 0.2;
        auto result = smoothing_spline_with_tolerance(t, q, tolerance, config);

        REQUIRE(result.max_error <= tolerance + kNumericalAtol);
        REQUIRE(result.mu > 0.0);
        REQUIRE(result.mu <= 1.0);
        REQUIRE(result.iterations > 0);
        REQUIRE(result.iterations <= config.max_iterations);
    }

    SECTION("Tight tolerance gives mu close to 1") {
        double tolerance = 1e-10;
        auto result = smoothing_spline_with_tolerance(t, q, tolerance, config);

        REQUIRE(result.mu > 0.9);
    }

    SECTION("Loose tolerance gives lower mu") {
        double tolerance = 1.0;
        auto result = smoothing_spline_with_tolerance(t, q, tolerance, config);

        REQUIRE(result.mu < 0.5);
    }
}

TEST_CASE("Smoothing spline with tolerance - convergence", "[smoothing_search]") {
    std::vector<double> t, q;
    make_noisy_sine(t, q, 20, 0.15);

    SECTION("Converges within max iterations") {
        SplineConfig config;
        config.max_iterations = 50;
        double tolerance = 0.1;

        auto result = smoothing_spline_with_tolerance(t, q, tolerance, config);
        REQUIRE(result.iterations <= 50);
    }

    SECTION("Max iterations reached returns best solution") {
        SplineConfig config;
        config.max_iterations = 3;  // Very few iterations
        double tolerance = 0.05;

        auto result = smoothing_spline_with_tolerance(t, q, tolerance, config);
        // Should return something valid
        REQUIRE(result.mu > 0.0);
        REQUIRE(std::isfinite(result.max_error));
    }
}

TEST_CASE("Smoothing spline with tolerance - with weights", "[smoothing_search]") {
    std::vector<double> t = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> q = {0.0, 1.1, 0.9, 0.1, -0.9, -1.0};

    SplineConfig config;
    config.max_iterations = 30;
    config.weights = (Eigen::VectorXd(6) << 10.0, 1.0, 1.0, 1.0, 1.0, 10.0).finished();

    auto result = smoothing_spline_with_tolerance(t, q, 0.3, config);
    REQUIRE(result.mu > 0.0);
    REQUIRE(std::isfinite(result.max_error));
}

TEST_CASE("Smoothing spline with tolerance - with boundary conditions", "[smoothing_search]") {
    std::vector<double> t, q;
    make_noisy_sine(t, q, 20, 0.1);

    SplineConfig config;
    config.max_iterations = 20;
    config.v0 = 1.0;
    config.vn = -1.0;

    auto result = smoothing_spline_with_tolerance(t, q, 0.2, config);
    REQUIRE(result.mu > 0.0);
    REQUIRE(std::isfinite(result.max_error));
}

TEST_CASE("Smoothing spline with tolerance - result spline is usable", "[smoothing_search]") {
    std::vector<double> t, q;
    make_noisy_sine(t, q, 25, 0.1);

    SplineConfig config;
    config.max_iterations = 20;

    auto result = smoothing_spline_with_tolerance(t, q, 0.15, config);

    // The returned spline should be evaluable
    for (double ti = t.front(); ti <= t.back(); ti += 0.1) {
        REQUIRE(std::isfinite(result.spline.evaluate(ti)));
        REQUIRE(std::isfinite(result.spline.evaluate_velocity(ti)));
        REQUIRE(std::isfinite(result.spline.evaluate_acceleration(ti)));
    }
}
