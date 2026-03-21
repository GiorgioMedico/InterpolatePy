#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <interpolatecpp/motion/trapezoidal_trajectory.hpp>
#include "test_data.hpp"

#include <cmath>

using namespace interpolatecpp::motion;
using namespace interpolatecpp::test;
using Catch::Matchers::WithinAbs;

TEST_CASE("TrapezoidalTrajectory velocity-based", "[trapezoidal]") {
    SECTION("Basic generation") {
        TrapezoidalTrajectory traj(0.0, 10.0, 2.0, 3.0);
        REQUIRE(traj.duration() > 0.0);

        auto r0 = traj.evaluate(traj.t_start());
        auto rT = traj.evaluate(traj.t_end());
        REQUIRE_THAT(r0.position, WithinAbs(0.0, kNumericalAtol));
        REQUIRE_THAT(rT.position, WithinAbs(10.0, kNumericalAtol));
    }

    SECTION("With initial/final velocities") {
        TrapezoidalTrajectory traj(1.0, 8.0, 2.0, 3.0, 0.5, 1.0);
        REQUIRE(traj.duration() > 0.0);

        auto r0 = traj.evaluate(traj.t_start());
        REQUIRE_THAT(r0.position, WithinAbs(1.0, kNumericalAtol));
    }

    SECTION("Velocity constraints") {
        TrapezoidalTrajectory traj(0.0, 20.0, 2.0, 3.0);
        double T = traj.duration();

        for (int i = 0; i <= 100; ++i) {
            double t = traj.t_start() + T * i / 100.0;
            auto r = traj.evaluate(t);
            REQUIRE(std::abs(r.velocity) <= 3.0 + kNumericalAtol);
        }
    }

    SECTION("Acceleration constraints") {
        TrapezoidalTrajectory traj(0.0, 15.0, 1.5, 3.0);

        for (int i = 0; i <= 100; ++i) {
            double t = traj.t_start() + traj.duration() * i / 100.0;
            auto r = traj.evaluate(t);
            REQUIRE(std::abs(r.acceleration) <= 1.5 + kNumericalAtol);
        }
    }
}

TEST_CASE("TrapezoidalTrajectory duration-based", "[trapezoidal]") {
    SECTION("Fixed duration") {
        TrapezoidalTrajectory traj(TrapezoidalTrajectory::DurationBased{},
                                   0.0, 10.0, 2.0, 0.0, 0.0, 0.0, 8.0);
        REQUIRE_THAT(traj.duration(), WithinAbs(8.0, kNumericalAtol));

        auto r0 = traj.evaluate(0.0);
        auto rT = traj.evaluate(8.0);
        REQUIRE_THAT(r0.position, WithinAbs(0.0, kNumericalAtol));
        REQUIRE_THAT(rT.position, WithinAbs(10.0, kNumericalAtol));
    }
}

TEST_CASE("TrapezoidalTrajectory edge cases", "[trapezoidal]") {
    SECTION("Zero displacement") {
        TrapezoidalTrajectory traj(5.0, 5.0, 2.0, 3.0);
        auto r = traj.evaluate(0.0);
        REQUIRE_THAT(r.position, WithinAbs(5.0, kNumericalAtol));
    }

    SECTION("Negative displacement") {
        TrapezoidalTrajectory traj(10.0, 0.0, 2.0, 3.0);
        auto rT = traj.evaluate(traj.t_end());
        REQUIRE_THAT(rT.position, WithinAbs(0.0, kNumericalAtol));
    }

    SECTION("High constraints") {
        TrapezoidalTrajectory traj(0.0, 1.0, 1000.0, 1000.0);
        REQUIRE(traj.duration() > 0.0);
    }
}

TEST_CASE("TrapezoidalTrajectory heuristic velocities", "[trapezoidal]") {
    std::vector<double> pts = {0, 3, 8, 12};
    std::vector<double> times = {0, 1, 2, 3};
    auto vels = TrapezoidalTrajectory::heuristic_velocities(pts, times, 4.0);

    REQUIRE(vels.size() == 4);
    for (double v : vels) {
        REQUIRE(std::abs(v) <= 4.0 + kNumericalAtol);
    }
}

TEST_CASE("TrapezoidalTrajectory waypoint interpolation", "[trapezoidal]") {
    SECTION("Basic interpolation") {
        std::vector<double> pts = {0, 5, 10};
        auto segments =
            TrapezoidalTrajectory::interpolate_waypoints(pts, 2.0, 3.0);

        REQUIRE(segments.size() == 2);

        auto r0 = TrapezoidalTrajectory::evaluate_multipoint(segments, 0.0);
        REQUIRE_THAT(r0.position, WithinAbs(0.0, kNumericalAtol));
    }

    SECTION("Minimum points validation") {
        std::vector<double> pts = {5.0};
        REQUIRE_THROWS_AS(
            TrapezoidalTrajectory::interpolate_waypoints(pts, 2.0, 3.0),
            std::invalid_argument);
    }
}
