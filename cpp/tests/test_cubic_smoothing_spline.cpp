#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <interpolatecpp/spline/cubic_smoothing_spline.hpp>

#include "test_data.hpp"

#include <cmath>
#include <vector>

using namespace interpolatecpp::spline;
using namespace interpolatecpp::test;
using Catch::Matchers::WithinAbs;

TEST_CASE("CubicSmoothingSpline construction", "[smoothing_spline]") {
    SECTION("Basic construction") {
        std::vector<double> t = {0.0, 1.0, 2.0, 3.0, 4.0};
        std::vector<double> q = {0.0, 1.0, 0.5, 1.5, 1.0};
        CubicSmoothingSpline spline(t, q);

        REQUIRE(spline.n_points() == 5);
        REQUIRE(spline.mu() == 0.5);
        REQUIRE(spline.coefficients().rows() == 4);
        REQUIRE(spline.coefficients().cols() == 4);
    }

    SECTION("Input validation - mismatched lengths") {
        std::vector<double> t = {0.0, 1.0, 2.0};
        std::vector<double> q = {0.0, 1.0};
        REQUIRE_THROWS_AS(CubicSmoothingSpline(t, q), std::invalid_argument);
    }

    SECTION("Input validation - insufficient points") {
        std::vector<double> t = {0.0};
        std::vector<double> q = {0.0};
        REQUIRE_THROWS_AS(CubicSmoothingSpline(t, q), std::invalid_argument);
    }

    SECTION("Input validation - non-increasing time") {
        std::vector<double> t = {0.0, 2.0, 1.0, 3.0};
        std::vector<double> q = {0.0, 1.0, 0.5, 1.5};
        REQUIRE_THROWS_AS(CubicSmoothingSpline(t, q), std::invalid_argument);
    }

    SECTION("Input validation - invalid mu (zero)") {
        std::vector<double> t = {0.0, 1.0, 2.0};
        std::vector<double> q = {0.0, 1.0, 0.5};
        REQUIRE_THROWS_AS(CubicSmoothingSpline(t, q, 0.0), std::invalid_argument);
    }

    SECTION("Input validation - invalid mu (> 1)") {
        std::vector<double> t = {0.0, 1.0, 2.0};
        std::vector<double> q = {0.0, 1.0, 0.5};
        REQUIRE_THROWS_AS(CubicSmoothingSpline(t, q, 1.1), std::invalid_argument);
    }

    SECTION("Input validation - mismatched weights") {
        std::vector<double> t = {0.0, 1.0, 2.0};
        std::vector<double> q = {0.0, 1.0, 0.5};
        std::vector<double> w = {1.0, 1.0};  // wrong length
        REQUIRE_THROWS_AS(
            CubicSmoothingSpline(t, q, 0.5, std::span<const double>(w)),
            std::invalid_argument);
    }
}

TEST_CASE("CubicSmoothingSpline exact interpolation (mu=1)", "[smoothing_spline]") {
    std::vector<double> t = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<double> q = {0.0, 1.0, 0.5, 1.5, 1.0};
    CubicSmoothingSpline spline(t, q, 1.0);

    SECTION("Passes through waypoints") {
        for (int i = 0; i < 5; ++i) {
            REQUIRE_THAT(spline.evaluate(t[static_cast<size_t>(i)]),
                         WithinAbs(q[static_cast<size_t>(i)], kNumericalAtol));
        }
    }

    SECTION("Approximated points equal original") {
        for (int i = 0; i < 5; ++i) {
            REQUIRE_THAT(spline.s_points()(i), WithinAbs(spline.q_points()(i), kRegularAtol));
        }
    }
}

TEST_CASE("CubicSmoothingSpline smoothing effect", "[smoothing_spline]") {
    // Noisy sine data
    std::vector<double> t = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0};
    std::vector<double> q = {0.0, 0.55, 0.84, 0.92, 0.91, 0.45, 0.14, -0.25, -0.76};

    SECTION("mu=1 gives exact interpolation") {
        CubicSmoothingSpline spline(t, q, 1.0);
        double max_error = (spline.q_points() - spline.s_points()).cwiseAbs().maxCoeff();
        REQUIRE_THAT(max_error, WithinAbs(0.0, kNumericalAtol));
    }

    SECTION("Small mu gives heavy smoothing") {
        CubicSmoothingSpline spline(t, q, 0.01);
        // Smoothed points should differ from original
        double max_error = (spline.q_points() - spline.s_points()).cwiseAbs().maxCoeff();
        REQUIRE(max_error > 0.01);
    }

    SECTION("Smoothing reduces acceleration magnitude") {
        CubicSmoothingSpline interp(t, q, 1.0);
        CubicSmoothingSpline smooth(t, q, 0.1);

        // Evaluate accelerations at several points
        double max_acc_interp = 0.0;
        double max_acc_smooth = 0.0;
        for (double ti = 0.0; ti <= 4.0; ti += 0.1) {
            max_acc_interp = std::max(max_acc_interp, std::abs(interp.evaluate_acceleration(ti)));
            max_acc_smooth = std::max(max_acc_smooth, std::abs(smooth.evaluate_acceleration(ti)));
        }
        REQUIRE(max_acc_smooth < max_acc_interp);
    }
}

TEST_CASE("CubicSmoothingSpline with weights", "[smoothing_spline]") {
    std::vector<double> t = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<double> q = {0.0, 1.0, 0.5, 1.5, 1.0};
    std::vector<double> high_endpoint_weights = {10.0, 1.0, 1.0, 1.0, 10.0};

    SECTION("Higher weights at endpoints enforce closer fit") {
        CubicSmoothingSpline weighted(t, q, 0.1,
                                       std::span<const double>(high_endpoint_weights));
        CubicSmoothingSpline uniform(t, q, 0.1);

        // Endpoint errors should be smaller with higher weights
        double weighted_endpoint_err =
            std::abs(weighted.s_points()(0) - q[0]) + std::abs(weighted.s_points()(4) - q[4]);
        double uniform_endpoint_err =
            std::abs(uniform.s_points()(0) - q[0]) + std::abs(uniform.s_points()(4) - q[4]);

        REQUIRE(weighted_endpoint_err <= uniform_endpoint_err + kNumericalAtol);
    }
}

TEST_CASE("CubicSmoothingSpline evaluation", "[smoothing_spline]") {
    std::vector<double> t = {0.0, 1.0, 2.0, 3.0};
    std::vector<double> q = {0.0, 1.0, 0.0, 1.0};
    CubicSmoothingSpline spline(t, q, 0.5);

    SECTION("All outputs are finite") {
        for (double ti = 0.0; ti <= 3.0; ti += 0.25) {
            REQUIRE(std::isfinite(spline.evaluate(ti)));
            REQUIRE(std::isfinite(spline.evaluate_velocity(ti)));
            REQUIRE(std::isfinite(spline.evaluate_acceleration(ti)));
        }
    }

    SECTION("Vectorized matches scalar") {
        Eigen::VectorXd tv = Eigen::VectorXd::LinSpaced(15, 0.0, 3.0);
        Eigen::VectorXd result = spline.evaluate(tv);
        for (Eigen::Index i = 0; i < tv.size(); ++i) {
            REQUIRE_THAT(result(i), WithinAbs(spline.evaluate(tv(i)), kRegularAtol));
        }
    }
}

TEST_CASE("CubicSmoothingSpline edge cases", "[smoothing_spline]") {
    SECTION("Two points") {
        std::vector<double> t = {0.0, 1.0};
        std::vector<double> q = {0.0, 1.0};
        CubicSmoothingSpline spline(t, q, 1.0);
        REQUIRE(std::isfinite(spline.evaluate(0.5)));
    }

    SECTION("Constant data") {
        std::vector<double> t = {0.0, 1.0, 2.0, 3.0};
        std::vector<double> q = {3.0, 3.0, 3.0, 3.0};
        CubicSmoothingSpline spline(t, q, 1.0);

        REQUIRE_THAT(spline.evaluate(1.5), WithinAbs(3.0, kNumericalAtol));
    }
}
