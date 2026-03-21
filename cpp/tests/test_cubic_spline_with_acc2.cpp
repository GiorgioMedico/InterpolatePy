#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <interpolatecpp/spline/cubic_spline_with_acc2.hpp>

#include "test_data.hpp"

#include <cmath>
#include <vector>

using namespace interpolatecpp::spline;
using namespace interpolatecpp::test;
using Catch::Matchers::WithinAbs;

TEST_CASE("CubicSplineWithAcceleration2 construction", "[acc2]") {
    SECTION("Basic construction (no acceleration constraints)") {
        std::vector<double> t = {0.0, 1.0, 2.0, 3.0};
        std::vector<double> q = {0.0, 1.0, 0.0, 1.0};
        CubicSplineWithAcceleration2 spline(t, q);

        REQUIRE(spline.n_segments() == 3);
        REQUIRE_FALSE(spline.has_quintic_first());
        REQUIRE_FALSE(spline.has_quintic_last());
    }

    SECTION("Inherits from CubicSpline") {
        std::vector<double> t = {0.0, 1.0, 2.0, 3.0};
        std::vector<double> q = {0.0, 1.0, 0.0, 1.0};
        CubicSplineWithAcceleration2 spline(t, q);

        // Should have all CubicSpline accessors
        REQUIRE(spline.t_points().size() == 4);
        REQUIRE(spline.q_points().size() == 4);
        REQUIRE(spline.velocities().size() == 4);
    }

    SECTION("With initial acceleration constraint") {
        std::vector<double> t = {0.0, 1.0, 2.0, 3.0};
        std::vector<double> q = {0.0, 1.0, 0.0, 1.0};
        SplineParameters params{.a0 = 2.0};
        CubicSplineWithAcceleration2 spline(t, q, params);

        REQUIRE(spline.has_quintic_first());
        REQUIRE_FALSE(spline.has_quintic_last());
    }

    SECTION("With final acceleration constraint") {
        std::vector<double> t = {0.0, 1.0, 2.0, 3.0};
        std::vector<double> q = {0.0, 1.0, 0.0, 1.0};
        SplineParameters params{.an = -1.0};
        CubicSplineWithAcceleration2 spline(t, q, params);

        REQUIRE_FALSE(spline.has_quintic_first());
        REQUIRE(spline.has_quintic_last());
    }

    SECTION("With both acceleration constraints") {
        std::vector<double> t = {0.0, 1.0, 2.0, 3.0};
        std::vector<double> q = {0.0, 1.0, 0.0, 1.0};
        SplineParameters params{.a0 = 2.0, .an = -1.0};
        CubicSplineWithAcceleration2 spline(t, q, params);

        REQUIRE(spline.has_quintic_first());
        REQUIRE(spline.has_quintic_last());
    }
}

TEST_CASE("CubicSplineWithAcceleration2 acceleration constraints", "[acc2]") {
    std::vector<double> t = {0.0, 1.0, 2.0, 3.0};
    std::vector<double> q = {0.0, 1.0, 0.0, 1.0};

    SECTION("Initial acceleration is satisfied") {
        SplineParameters params{.a0 = 2.0};
        CubicSplineWithAcceleration2 spline(t, q, params);

        REQUIRE_THAT(spline.evaluate_acceleration(0.0), WithinAbs(2.0, kNumericalAtol));
    }

    SECTION("Final acceleration is satisfied") {
        SplineParameters params{.an = -3.0};
        CubicSplineWithAcceleration2 spline(t, q, params);

        REQUIRE_THAT(spline.evaluate_acceleration(3.0), WithinAbs(-3.0, kNumericalAtol));
    }

    SECTION("Both accelerations satisfied") {
        SplineParameters params{.a0 = 2.0, .an = -3.0};
        CubicSplineWithAcceleration2 spline(t, q, params);

        REQUIRE_THAT(spline.evaluate_acceleration(0.0), WithinAbs(2.0, kNumericalAtol));
        REQUIRE_THAT(spline.evaluate_acceleration(3.0), WithinAbs(-3.0, kNumericalAtol));
    }
}

TEST_CASE("CubicSplineWithAcceleration2 evaluation", "[acc2]") {
    std::vector<double> t = {0.0, 1.0, 2.0, 3.0};
    std::vector<double> q = {0.0, 1.0, 0.0, 1.0};
    SplineParameters params{.a0 = 1.0, .an = -1.0};
    CubicSplineWithAcceleration2 spline(t, q, params);

    SECTION("Passes through waypoints") {
        for (int i = 0; i < 4; ++i) {
            REQUIRE_THAT(spline.evaluate(t[static_cast<size_t>(i)]),
                         WithinAbs(q[static_cast<size_t>(i)], kNumericalAtol));
        }
    }

    SECTION("All outputs are finite") {
        for (double ti = 0.0; ti <= 3.0; ti += 0.25) {
            REQUIRE(std::isfinite(spline.evaluate(ti)));
            REQUIRE(std::isfinite(spline.evaluate_velocity(ti)));
            REQUIRE(std::isfinite(spline.evaluate_acceleration(ti)));
        }
    }

    SECTION("Vectorized matches scalar") {
        Eigen::VectorXd tv = Eigen::VectorXd::LinSpaced(20, 0.0, 3.0);
        Eigen::VectorXd result = spline.evaluate(tv);
        for (Eigen::Index i = 0; i < tv.size(); ++i) {
            REQUIRE_THAT(result(i), WithinAbs(spline.evaluate(tv(i)), kRegularAtol));
        }
    }

    SECTION("Boundary extrapolation") {
        // Before start
        double before = spline.evaluate(-0.5);
        REQUIRE(std::isfinite(before));
        // After end
        double after = spline.evaluate(3.5);
        REQUIRE(std::isfinite(after));
    }
}

TEST_CASE("CubicSplineWithAcceleration2 derivative evaluation", "[acc2]") {
    std::vector<double> t = {0.0, 1.0, 2.0, 3.0};
    std::vector<double> q = {0.0, 1.0, 0.0, 1.0};
    SplineParameters params{.a0 = 1.0, .an = -1.0};
    CubicSplineWithAcceleration2 spline(t, q, params);

    SECTION("Finite differences match velocity") {
        // Test in quintic first segment
        double h = 1e-7;
        double t_test = 0.5;
        double numerical = (spline.evaluate(t_test + h) - spline.evaluate(t_test - h)) / (2.0 * h);
        REQUIRE_THAT(spline.evaluate_velocity(t_test), WithinAbs(numerical, kLooseAtol));

        // Test in cubic middle segment
        t_test = 1.5;
        numerical = (spline.evaluate(t_test + h) - spline.evaluate(t_test - h)) / (2.0 * h);
        REQUIRE_THAT(spline.evaluate_velocity(t_test), WithinAbs(numerical, kLooseAtol));

        // Test in quintic last segment
        t_test = 2.5;
        numerical = (spline.evaluate(t_test + h) - spline.evaluate(t_test - h)) / (2.0 * h);
        REQUIRE_THAT(spline.evaluate_velocity(t_test), WithinAbs(numerical, kLooseAtol));
    }
}

TEST_CASE("CubicSplineWithAcceleration2 error handling", "[acc2]") {
    SECTION("Insufficient points") {
        std::vector<double> t = {0.0};
        std::vector<double> q = {0.0};
        REQUIRE_THROWS_AS(CubicSplineWithAcceleration2(t, q), std::invalid_argument);
    }

    SECTION("Non-monotonic time") {
        std::vector<double> t = {0.0, 2.0, 1.0};
        std::vector<double> q = {0.0, 1.0, 0.0};
        REQUIRE_THROWS_AS(CubicSplineWithAcceleration2(t, q), std::invalid_argument);
    }
}
