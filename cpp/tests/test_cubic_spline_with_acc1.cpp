#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <interpolatecpp/spline/cubic_spline_with_acc1.hpp>

#include "test_data.hpp"

#include <cmath>
#include <vector>

using namespace interpolatecpp::spline;
using namespace interpolatecpp::test;
using Catch::Matchers::WithinAbs;

TEST_CASE("CubicSplineWithAcceleration1 construction", "[acc1]") {
    SECTION("Basic construction") {
        std::vector<double> t = {0.0, 1.0, 2.0, 3.0};
        std::vector<double> q = {0.0, 1.0, 0.0, 1.0};
        CubicSplineWithAcceleration1 spline(t, q);

        // Original 4 points + 2 extra = 6 total
        REQUIRE(spline.n_points() == 6);
        REQUIRE(spline.n_orig() == 4);
        REQUIRE(spline.original_indices().size() == 4);
    }

    SECTION("With acceleration constraints") {
        std::vector<double> t = {0.0, 1.0, 2.0, 3.0};
        std::vector<double> q = {0.0, 1.0, 0.0, 1.0};
        CubicSplineWithAcceleration1 spline(t, q, 0.0, 0.0, 2.0, 2.0);

        REQUIRE_THAT(spline.omega()(0), WithinAbs(2.0, kRegularAtol));
        REQUIRE_THAT(spline.omega()(spline.n_points() - 1), WithinAbs(2.0, kRegularAtol));
    }

    SECTION("Input validation - mismatched lengths") {
        std::vector<double> t = {0.0, 1.0, 2.0};
        std::vector<double> q = {0.0, 1.0};
        REQUIRE_THROWS_AS(CubicSplineWithAcceleration1(t, q), std::invalid_argument);
    }

    SECTION("Input validation - non-increasing time") {
        std::vector<double> t = {0.0, 2.0, 1.0, 3.0};
        std::vector<double> q = {0.0, 1.0, 0.0, 1.0};
        REQUIRE_THROWS_AS(CubicSplineWithAcceleration1(t, q), std::invalid_argument);
    }

    SECTION("Input validation - insufficient points") {
        std::vector<double> t = {0.0};
        std::vector<double> q = {0.0};
        REQUIRE_THROWS_AS(CubicSplineWithAcceleration1(t, q), std::invalid_argument);
    }
}

TEST_CASE("CubicSplineWithAcceleration1 boundary conditions", "[acc1]") {
    std::vector<double> t = {0.0, 1.0, 2.0, 3.0};
    std::vector<double> q = {0.0, 1.0, 0.0, 1.0};

    SECTION("Boundary velocity enforcement") {
        CubicSplineWithAcceleration1 spline(t, q, 1.0, -1.0, 0.0, 0.0);
        REQUIRE_THAT(spline.evaluate_velocity(0.0), WithinAbs(1.0, 1.0));
    }

    SECTION("Boundary acceleration enforcement") {
        CubicSplineWithAcceleration1 spline(t, q, 0.0, 0.0, 2.0, -2.0);
        REQUIRE_THAT(spline.evaluate_acceleration(0.0), WithinAbs(2.0, 2.0));
    }
}

TEST_CASE("CubicSplineWithAcceleration1 evaluation", "[acc1]") {
    std::vector<double> t = {0.0, 1.0, 2.0, 3.0};
    std::vector<double> q = {0.0, 1.0, 0.0, 1.0};
    CubicSplineWithAcceleration1 spline(t, q, 0.0, 0.0, 0.0, 0.0);

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

    SECTION("Passes through original endpoints") {
        REQUIRE_THAT(spline.evaluate(0.0), WithinAbs(0.0, kNumericalAtol));
        REQUIRE_THAT(spline.evaluate(3.0), WithinAbs(1.0, kNumericalAtol));
    }

    SECTION("Extra points computation") {
        // n_orig + 2 = total points
        REQUIRE(spline.t_points().size() == spline.q_points().size());
        REQUIRE(spline.t_points().size() == spline.n_orig() + 2);
    }
}

TEST_CASE("CubicSplineWithAcceleration1 derivative evaluation", "[acc1]") {
    std::vector<double> t = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<double> q = {0.0, 1.0, 4.0, 9.0, 16.0};
    CubicSplineWithAcceleration1 spline(t, q, 0.0, 8.0, 2.0, 2.0);

    SECTION("Finite differences match velocity") {
        // Use mid-segment point to avoid straddling segment boundaries
        double t_test = 1.5;
        double h = 1e-7;
        double numerical_vel = (spline.evaluate(t_test + h) - spline.evaluate(t_test - h)) / (2.0 * h);
        double vel = spline.evaluate_velocity(t_test);
        REQUIRE_THAT(vel, WithinAbs(numerical_vel, kLooseAtol));
    }

    SECTION("Finite differences match acceleration") {
        double t_test = 1.5;
        double h = 1e-5;
        double numerical_acc = (spline.evaluate_velocity(t_test + h) -
                                spline.evaluate_velocity(t_test - h)) / (2.0 * h);
        double acc = spline.evaluate_acceleration(t_test);
        REQUIRE_THAT(acc, WithinAbs(numerical_acc, kLooseAtol));
    }
}
