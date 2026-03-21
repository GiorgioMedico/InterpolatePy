#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <interpolatecpp/motion/polynomial_trajectory.hpp>
#include "test_data.hpp"

#include <cmath>

using namespace interpolatecpp::motion;
using namespace interpolatecpp::test;
using Catch::Matchers::WithinAbs;

// --- Order 3 ---

TEST_CASE("Order 3 basic trajectory", "[polynomial]") {
    BoundaryCondition start{0.0, 0.0};
    BoundaryCondition end{10.0, 0.0};
    TimeInterval interval{0.0, 2.0};
    PolynomialTrajectory traj(start, end, interval, 3);

    auto r0 = traj.evaluate(0.0);
    auto r1 = traj.evaluate(2.0);

    REQUIRE_THAT(r0.position, WithinAbs(0.0, kNumericalAtol));
    REQUIRE_THAT(r0.velocity, WithinAbs(0.0, kNumericalAtol));
    REQUIRE_THAT(r1.position, WithinAbs(10.0, kNumericalAtol));
    REQUIRE_THAT(r1.velocity, WithinAbs(0.0, kNumericalAtol));
}

TEST_CASE("Order 3 nonzero velocities", "[polynomial]") {
    BoundaryCondition start{2.0, 1.0};
    BoundaryCondition end{8.0, 3.0};
    TimeInterval interval{0.0, 3.0};
    PolynomialTrajectory traj(start, end, interval, 3);

    auto r0 = traj.evaluate(0.0);
    auto r1 = traj.evaluate(3.0);

    REQUIRE_THAT(r0.position, WithinAbs(2.0, kNumericalAtol));
    REQUIRE_THAT(r0.velocity, WithinAbs(1.0, kNumericalAtol));
    REQUIRE_THAT(r1.position, WithinAbs(8.0, kNumericalAtol));
    REQUIRE_THAT(r1.velocity, WithinAbs(3.0, kNumericalAtol));
}

TEST_CASE("Order 3 negative displacement", "[polynomial]") {
    BoundaryCondition start{10.0, 0.0};
    BoundaryCondition end{3.0, 0.0};
    TimeInterval interval{0.0, 2.0};
    PolynomialTrajectory traj(start, end, interval, 3);

    auto r0 = traj.evaluate(0.0);
    auto r1 = traj.evaluate(2.0);

    REQUIRE_THAT(r0.position, WithinAbs(10.0, kNumericalAtol));
    REQUIRE_THAT(r1.position, WithinAbs(3.0, kNumericalAtol));
}

TEST_CASE("Order 3 jerk constant", "[polynomial]") {
    BoundaryCondition start{0.0, 0.0};
    BoundaryCondition end{10.0, 0.0};
    TimeInterval interval{0.0, 2.0};
    PolynomialTrajectory traj(start, end, interval, 3);

    // Jerk should be constant for cubic
    double j0 = traj.evaluate(0.0).jerk;
    double j1 = traj.evaluate(0.5).jerk;
    double j2 = traj.evaluate(1.5).jerk;

    REQUIRE_THAT(j0, WithinAbs(j1, kNumericalAtol));
    REQUIRE_THAT(j1, WithinAbs(j2, kNumericalAtol));
}

// --- Order 5 ---

TEST_CASE("Order 5 basic trajectory", "[polynomial]") {
    BoundaryCondition start{0.0, 0.0, 0.0};
    BoundaryCondition end{8.0, 0.0, 0.0};
    TimeInterval interval{0.0, 2.0};
    PolynomialTrajectory traj(start, end, interval, 5);

    auto r0 = traj.evaluate(0.0);
    auto r1 = traj.evaluate(2.0);

    REQUIRE_THAT(r0.position, WithinAbs(0.0, kNumericalAtol));
    REQUIRE_THAT(r0.velocity, WithinAbs(0.0, kNumericalAtol));
    REQUIRE_THAT(r0.acceleration, WithinAbs(0.0, kNumericalAtol));
    REQUIRE_THAT(r1.position, WithinAbs(8.0, kNumericalAtol));
    REQUIRE_THAT(r1.velocity, WithinAbs(0.0, kNumericalAtol));
    REQUIRE_THAT(r1.acceleration, WithinAbs(0.0, kNumericalAtol));
}

TEST_CASE("Order 5 nonzero accelerations", "[polynomial]") {
    BoundaryCondition start{1.0, 2.0, 0.5};
    BoundaryCondition end{9.0, 1.0, -0.3};
    TimeInterval interval{0.0, 4.0};
    PolynomialTrajectory traj(start, end, interval, 5);

    auto r0 = traj.evaluate(0.0);
    auto r1 = traj.evaluate(4.0);

    REQUIRE_THAT(r0.position, WithinAbs(1.0, kNumericalAtol));
    REQUIRE_THAT(r0.velocity, WithinAbs(2.0, kNumericalAtol));
    REQUIRE_THAT(r0.acceleration, WithinAbs(0.5, kNumericalAtol));
    REQUIRE_THAT(r1.position, WithinAbs(9.0, kNumericalAtol));
    REQUIRE_THAT(r1.velocity, WithinAbs(1.0, kNumericalAtol));
    REQUIRE_THAT(r1.acceleration, WithinAbs(-0.3, kNumericalAtol));
}

// --- Order 7 ---

TEST_CASE("Order 7 basic trajectory", "[polynomial]") {
    BoundaryCondition start{0.0, 0.0, 0.0, 0.0};
    BoundaryCondition end{10.0, 0.0, 0.0, 0.0};
    TimeInterval interval{0.0, 3.0};
    PolynomialTrajectory traj(start, end, interval, 7);

    auto r0 = traj.evaluate(0.0);
    auto r1 = traj.evaluate(3.0);

    REQUIRE_THAT(r0.position, WithinAbs(0.0, kNumericalAtol));
    REQUIRE_THAT(r0.velocity, WithinAbs(0.0, kNumericalAtol));
    REQUIRE_THAT(r0.acceleration, WithinAbs(0.0, kNumericalAtol));
    REQUIRE_THAT(r0.jerk, WithinAbs(0.0, kNumericalAtol));
    REQUIRE_THAT(r1.position, WithinAbs(10.0, kNumericalAtol));
    REQUIRE_THAT(r1.velocity, WithinAbs(0.0, kNumericalAtol));
}

TEST_CASE("Order 7 nonzero jerks", "[polynomial]") {
    BoundaryCondition start{1.0, 1.0, 0.0, 0.5};
    BoundaryCondition end{11.0, 2.0, -1.0, -0.3};
    TimeInterval interval{0.0, 4.0};
    PolynomialTrajectory traj(start, end, interval, 7);

    auto r0 = traj.evaluate(0.0);
    auto r1 = traj.evaluate(4.0);

    REQUIRE_THAT(r0.position, WithinAbs(1.0, kNumericalAtol));
    REQUIRE_THAT(r0.velocity, WithinAbs(1.0, kNumericalAtol));
    REQUIRE_THAT(r0.jerk, WithinAbs(0.5, kNumericalAtol));
    REQUIRE_THAT(r1.position, WithinAbs(11.0, kNumericalAtol));
    REQUIRE_THAT(r1.velocity, WithinAbs(2.0, kNumericalAtol));
}

// --- Heuristic velocities ---

TEST_CASE("Heuristic velocities", "[polynomial]") {
    SECTION("Basic") {
        std::vector<double> pts = {0, 5, 10};
        std::vector<double> times = {0, 2, 4};
        auto v = PolynomialTrajectory::heuristic_velocities(pts, times);
        REQUIRE(v.size() == 3);
        REQUIRE_THAT(v[0], WithinAbs(0.0, kNumericalAtol));
        REQUIRE_THAT(v[2], WithinAbs(0.0, kNumericalAtol));
    }

    SECTION("Minimum points") {
        std::vector<double> pts = {0};
        std::vector<double> times = {0};
        REQUIRE_THROWS_AS(PolynomialTrajectory::heuristic_velocities(pts, times),
                          std::invalid_argument);
    }

    SECTION("Direction change") {
        std::vector<double> pts = {0, 5, 3, 8};
        std::vector<double> times = {0, 1, 2, 3};
        auto v = PolynomialTrajectory::heuristic_velocities(pts, times);
        // At direction change (5→3), velocity should be 0
        REQUIRE_THAT(v[2], WithinAbs(0.0, kNumericalAtol));
    }
}

// --- Multipoint ---

TEST_CASE("Multipoint trajectory", "[polynomial]") {
    std::vector<double> pts = {0, 5, 10};
    std::vector<double> times = {0, 2, 4};

    SECTION("Order 3") {
        auto segments = PolynomialTrajectory::multipoint_trajectory(pts, times, 3);
        REQUIRE(segments.size() == 2);

        auto r = PolynomialTrajectory::evaluate_multipoint(segments, 1.0);
        REQUIRE(std::isfinite(r.position));
    }

    SECTION("Order 5") {
        auto segments = PolynomialTrajectory::multipoint_trajectory(pts, times, 5);
        auto r = PolynomialTrajectory::evaluate_multipoint(segments, 1.0);
        REQUIRE(std::isfinite(r.position));
    }

    SECTION("Boundary evaluation") {
        auto segments = PolynomialTrajectory::multipoint_trajectory(pts, times, 3);
        auto r0 = PolynomialTrajectory::evaluate_multipoint(segments, 0.0);
        auto r1 = PolynomialTrajectory::evaluate_multipoint(segments, 4.0);
        REQUIRE_THAT(r0.position, WithinAbs(0.0, kNumericalAtol));
        REQUIRE_THAT(r1.position, WithinAbs(10.0, kNumericalAtol));
    }
}

// --- Edge cases ---

TEST_CASE("Polynomial edge cases", "[polynomial]") {
    SECTION("Invalid order") {
        BoundaryCondition start{0.0, 0.0};
        BoundaryCondition end{10.0, 0.0};
        TimeInterval interval{0.0, 2.0};
        REQUIRE_THROWS_AS(PolynomialTrajectory(start, end, interval, 4),
                          std::invalid_argument);
    }

    SECTION("Zero displacement") {
        BoundaryCondition start{5.0, 2.0};
        BoundaryCondition end{5.0, -2.0};
        TimeInterval interval{0.0, 1.0};
        PolynomialTrajectory traj(start, end, interval, 3);

        auto r0 = traj.evaluate(0.0);
        auto r1 = traj.evaluate(1.0);
        REQUIRE_THAT(r0.position, WithinAbs(5.0, kNumericalAtol));
        REQUIRE_THAT(r1.position, WithinAbs(5.0, kNumericalAtol));
    }
}
