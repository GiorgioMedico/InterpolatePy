#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <interpolatecpp/motion/double_s_trajectory.hpp>
#include "test_data.hpp"

#include <cmath>

using namespace interpolatecpp::motion;
using namespace interpolatecpp::test;
using Catch::Matchers::WithinAbs;

TEST_CASE("DoubleSTrajectory construction", "[double_s]") {
    SECTION("Basic construction") {
        StateParams state{0.0, 10.0, 0.0, 0.0};
        TrajectoryBounds bounds(2.0, 1.0, 0.5);
        DoubleSTrajectory traj(state, bounds);

        REQUIRE(traj.duration() > 0.0);
    }

    SECTION("Various states") {
        // Forward
        StateParams s1{0.0, 10.0, 0.0, 0.0};
        TrajectoryBounds b(5.0, 2.0, 1.0);
        DoubleSTrajectory t1(s1, b);
        REQUIRE(t1.duration() > 0.0);

        // Backward
        StateParams s2{10.0, 0.0, 0.0, 0.0};
        DoubleSTrajectory t2(s2, b);
        REQUIRE(t2.duration() > 0.0);
    }
}

TEST_CASE("DoubleSTrajectory bounds validation", "[double_s]") {
    SECTION("Zero bounds rejected") {
        REQUIRE_THROWS_AS(TrajectoryBounds(0.0, 1.0, 0.5), std::invalid_argument);
    }

    SECTION("Negative bounds convert to absolute") {
        TrajectoryBounds b(-2.0, -1.0, -0.5);
        REQUIRE(b.v_bound > 0);
        REQUIRE(b.a_bound > 0);
        REQUIRE(b.j_bound > 0);
    }
}

TEST_CASE("DoubleSTrajectory evaluation", "[double_s]") {
    StateParams state{0.0, 10.0, 0.0, 0.0};
    TrajectoryBounds bounds(3.0, 2.0, 1.0);
    DoubleSTrajectory traj(state, bounds);

    SECTION("Start and end") {
        auto r0 = traj.evaluate(0.0);
        auto rT = traj.evaluate(traj.duration());

        REQUIRE_THAT(r0.position, WithinAbs(0.0, kNumericalAtol));
        REQUIRE_THAT(rT.position, WithinAbs(10.0, kNumericalAtol));
    }

    SECTION("Midpoint finite") {
        auto rm = traj.evaluate(traj.duration() / 2.0);
        REQUIRE(std::isfinite(rm.position));
        REQUIRE(std::isfinite(rm.velocity));
        REQUIRE(std::isfinite(rm.acceleration));
        REQUIRE(std::isfinite(rm.jerk));
    }
}

TEST_CASE("DoubleSTrajectory boundary conditions", "[double_s]") {
    StateParams state{2.0, 8.0, 1.0, 0.5};
    TrajectoryBounds bounds(5.0, 3.0, 2.0);
    DoubleSTrajectory traj(state, bounds);

    auto r0 = traj.evaluate(0.0);
    auto rT = traj.evaluate(traj.duration());

    REQUIRE_THAT(r0.position, WithinAbs(2.0, kNumericalAtol));
    REQUIRE_THAT(rT.position, WithinAbs(8.0, kNumericalAtol));
}

TEST_CASE("DoubleSTrajectory velocity bounds", "[double_s]") {
    StateParams state{0.0, 20.0, 0.0, 0.0};
    TrajectoryBounds bounds(2.0, 1.0, 0.5);
    DoubleSTrajectory traj(state, bounds);

    double T = traj.duration();
    for (int i = 0; i <= 100; ++i) {
        double t = T * i / 100.0;
        auto r = traj.evaluate(t);
        REQUIRE(std::abs(r.velocity) <= bounds.v_bound + kNumericalAtol);
    }
}

TEST_CASE("DoubleSTrajectory phase durations", "[double_s]") {
    StateParams state{0.0, 10.0, 0.0, 0.0};
    TrajectoryBounds bounds(3.0, 2.0, 1.0);
    DoubleSTrajectory traj(state, bounds);

    auto phases = traj.phase_durations();
    REQUIRE(phases.count("total") == 1);
    REQUIRE(phases.count("acceleration") == 1);
    REQUIRE(phases.count("constant_velocity") == 1);
    REQUIRE(phases.count("deceleration") == 1);
    REQUIRE(phases["total"] >= 0.0);
    REQUIRE(phases["acceleration"] >= 0.0);
    REQUIRE(phases["constant_velocity"] >= 0.0);
    REQUIRE(phases["deceleration"] >= 0.0);
}

TEST_CASE("DoubleSTrajectory edge cases", "[double_s]") {
    SECTION("Zero displacement") {
        StateParams state{5.0, 5.0, 0.0, 0.0};
        TrajectoryBounds bounds(2.0, 1.0, 0.5);
        DoubleSTrajectory traj(state, bounds);

        auto r = traj.evaluate(0.0);
        REQUIRE_THAT(r.position, WithinAbs(5.0, kNumericalAtol));
    }

    SECTION("Small displacement") {
        StateParams state{0.0, 0.001, 0.0, 0.0};
        TrajectoryBounds bounds(1.0, 1.0, 1.0);
        DoubleSTrajectory traj(state, bounds);

        REQUIRE(traj.duration() >= 0.0);
    }

    SECTION("Negative displacement") {
        StateParams state{10.0, 0.0, 0.0, 0.0};
        TrajectoryBounds bounds(2.0, 1.0, 0.5);
        DoubleSTrajectory traj(state, bounds);

        auto rT = traj.evaluate(traj.duration());
        REQUIRE_THAT(rT.position, WithinAbs(0.0, kNumericalAtol));
    }

    SECTION("Non-zero initial/final velocities") {
        StateParams state{0.0, 10.0, 1.0, 2.0};
        TrajectoryBounds bounds(5.0, 3.0, 2.0);
        DoubleSTrajectory traj(state, bounds);

        auto r0 = traj.evaluate(0.0);
        REQUIRE_THAT(r0.position, WithinAbs(0.0, kNumericalAtol));
    }

    SECTION("Large displacement") {
        StateParams state{0.0, 1000.0, 0.0, 0.0};
        TrajectoryBounds bounds(10.0, 5.0, 2.0);
        DoubleSTrajectory traj(state, bounds);

        REQUIRE(traj.duration() > 0.0);
        auto rT = traj.evaluate(traj.duration());
        REQUIRE_THAT(rT.position, WithinAbs(1000.0, 0.1));
    }
}
