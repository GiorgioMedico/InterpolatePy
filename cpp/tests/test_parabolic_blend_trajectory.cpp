#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <interpolatecpp/motion/parabolic_blend_trajectory.hpp>
#include "test_data.hpp"

#include <cmath>

using namespace interpolatecpp::motion;
using namespace interpolatecpp::test;
using Catch::Matchers::WithinAbs;

TEST_CASE("ParabolicBlendTrajectory construction", "[parabolic_blend]") {
    SECTION("Basic construction") {
        std::vector<double> q = {0, 1, 2};
        std::vector<double> t = {0, 1, 2};
        std::vector<double> dt = {0.1, 0.2, 0.1};
        ParabolicBlendTrajectory traj(q, t, dt);

        REQUIRE(traj.n_waypoints() == 3);
    }

    SECTION("Single point") {
        std::vector<double> q = {1.0};
        std::vector<double> t = {0.0};
        std::vector<double> dt = {0.1};
        ParabolicBlendTrajectory traj(q, t, dt);

        REQUIRE(traj.n_waypoints() == 1);
    }

    SECTION("Mismatched lengths") {
        std::vector<double> q = {0, 1, 2};
        std::vector<double> t = {0, 1};
        std::vector<double> dt = {0.1, 0.2, 0.1};
        REQUIRE_THROWS_AS(ParabolicBlendTrajectory(q, t, dt),
                          std::invalid_argument);
    }
}

TEST_CASE("ParabolicBlendTrajectory two-point", "[parabolic_blend]") {
    std::vector<double> q = {0, 1};
    std::vector<double> t = {0, 1};
    std::vector<double> dt = {0.2, 0.2};
    ParabolicBlendTrajectory traj(q, t, dt);

    REQUIRE(traj.duration() > 0.0);

    auto r = traj.evaluate(0.5);
    REQUIRE(std::isfinite(r.position));
    REQUIRE(std::isfinite(r.velocity));
    REQUIRE(std::isfinite(r.acceleration));
}

TEST_CASE("ParabolicBlendTrajectory three-point", "[parabolic_blend]") {
    std::vector<double> q = {0, 1, 2};
    std::vector<double> t = {0, 1, 2};
    std::vector<double> dt = {0.1, 0.2, 0.1};
    ParabolicBlendTrajectory traj(q, t, dt);

    // Evaluate at several points
    for (int i = 0; i <= 10; ++i) {
        double eval_t = -0.05 + 2.05 * i / 10.0;
        eval_t = std::clamp(eval_t, -0.05, 2.05);
        auto r = traj.evaluate(eval_t);
        REQUIRE(std::isfinite(r.position));
    }
}

TEST_CASE("ParabolicBlendTrajectory position continuity", "[parabolic_blend]") {
    std::vector<double> q = {0, 2, 1, 4};
    std::vector<double> t = {0, 1, 2, 3};
    std::vector<double> dt = {0.2, 0.3, 0.3, 0.2};
    ParabolicBlendTrajectory traj(q, t, dt);

    // Check that position varies smoothly
    double prev_pos = traj.evaluate(-0.1).position;
    double max_jump = 0.0;

    for (int i = 1; i <= 100; ++i) {
        double ti = -0.1 + 3.2 * i / 100.0;
        auto r = traj.evaluate(ti);
        double jump = std::abs(r.position - prev_pos);
        max_jump = std::max(max_jump, jump);
        prev_pos = r.position;
    }

    // Should be relatively smooth (no huge jumps)
    REQUIRE(max_jump < 1.0);
}

TEST_CASE("ParabolicBlendTrajectory identical waypoints", "[parabolic_blend]") {
    std::vector<double> q = {1, 1, 1};
    std::vector<double> t = {0, 1, 2};
    std::vector<double> dt = {0.1, 0.2, 0.1};
    ParabolicBlendTrajectory traj(q, t, dt);

    auto r = traj.evaluate(1.0);
    REQUIRE(std::isfinite(r.position));
    REQUIRE(std::isfinite(r.velocity));
}

TEST_CASE("ParabolicBlendTrajectory negative positions", "[parabolic_blend]") {
    std::vector<double> q = {-2, -1, -3};
    std::vector<double> t = {0, 1, 2};
    std::vector<double> dt = {0.1, 0.2, 0.1};
    ParabolicBlendTrajectory traj(q, t, dt);

    auto r = traj.evaluate(0.5);
    REQUIRE(std::isfinite(r.position));
}

TEST_CASE("ParabolicBlendTrajectory single waypoint", "[parabolic_blend]") {
    std::vector<double> q = {5.0};
    std::vector<double> t = {0.0};
    std::vector<double> dt = {0.2};
    ParabolicBlendTrajectory traj(q, t, dt);

    auto r = traj.evaluate(0.0);
    REQUIRE(std::isfinite(r.position));
    REQUIRE_THAT(r.velocity, WithinAbs(0.0, kNumericalAtol));
}
