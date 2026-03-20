#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <interpolatecpp/path/circular_path.hpp>
#include <interpolatecpp/path/frenet_frame.hpp>
#include <interpolatecpp/path/linear_path.hpp>
#include <interpolatecpp/path/linear_traj.hpp>
#include "test_data.hpp"

#include <cmath>

using namespace interpolatecpp::path;
using namespace interpolatecpp::test;
using Catch::Matchers::WithinAbs;

// ===== LinearPath =====

TEST_CASE("LinearPath construction", "[linear_path]") {
    Eigen::Vector3d pi(0, 0, 0);
    Eigen::Vector3d pf(1, 0, 0);
    LinearPath path(pi, pf);

    REQUIRE_THAT(path.length(), WithinAbs(1.0, kRegularAtol));
}

TEST_CASE("LinearPath position", "[linear_path]") {
    Eigen::Vector3d pi(0, 0, 0);
    Eigen::Vector3d pf(2, 0, 0);
    LinearPath path(pi, pf);

    auto p0 = path.position(0.0);
    auto pm = path.position(1.0);
    auto p1 = path.position(2.0);

    REQUIRE_THAT(p0(0), WithinAbs(0.0, kRegularAtol));
    REQUIRE_THAT(pm(0), WithinAbs(1.0, kRegularAtol));
    REQUIRE_THAT(p1(0), WithinAbs(2.0, kRegularAtol));
}

TEST_CASE("LinearPath velocity and acceleration", "[linear_path]") {
    Eigen::Vector3d pi(0, 0, 0);
    Eigen::Vector3d pf(1, 1, 1);
    LinearPath path(pi, pf);

    auto v = path.velocity(0.5);
    auto a = path.acceleration(0.5);

    double expected_v = 1.0 / std::sqrt(3.0);
    REQUIRE_THAT(v(0), WithinAbs(expected_v, kNumericalAtol));
    REQUIRE_THAT(a.norm(), WithinAbs(0.0, kRegularAtol));
}

TEST_CASE("LinearPath vector evaluation", "[linear_path]") {
    Eigen::Vector3d pi(0, 0, 0);
    Eigen::Vector3d pf(3, 0, 0);
    LinearPath path(pi, pf);

    Eigen::VectorXd s(3);
    s << 0, 1.5, 3;
    auto pts = path.position(s);

    REQUIRE(pts.rows() == 3);
    REQUIRE(pts.cols() == 3);
    REQUIRE_THAT(pts(1, 0), WithinAbs(1.5, kRegularAtol));
}

TEST_CASE("LinearPath zero length", "[linear_path]") {
    Eigen::Vector3d p(1, 2, 3);
    LinearPath path(p, p);

    REQUIRE_THAT(path.length(), WithinAbs(0.0, kRegularAtol));
    auto pos = path.position(0.0);
    REQUIRE_THAT(pos(0), WithinAbs(1.0, kRegularAtol));
}

// ===== CircularPath =====

TEST_CASE("CircularPath XY plane", "[circular_path]") {
    Eigen::Vector3d axis(0, 0, 1);
    Eigen::Vector3d axis_point(0, 0, 0);
    Eigen::Vector3d circle_point(1, 0, 0);
    CircularPath path(axis, axis_point, circle_point);

    REQUIRE_THAT(path.radius(), WithinAbs(1.0, kRegularAtol));

    SECTION("Start point") {
        auto p = path.position(0.0);
        REQUIRE_THAT(p(0), WithinAbs(1.0, kNumericalAtol));
        REQUIRE_THAT(p(1), WithinAbs(0.0, kNumericalAtol));
    }

    SECTION("Quarter circle") {
        auto p = path.position(M_PI / 2.0);
        REQUIRE_THAT(p(0), WithinAbs(0.0, kNumericalAtol));
        REQUIRE_THAT(p(1), WithinAbs(1.0, kNumericalAtol));
    }

    SECTION("Half circle") {
        auto p = path.position(M_PI);
        REQUIRE_THAT(p(0), WithinAbs(-1.0, kNumericalAtol));
        REQUIRE_THAT(p(1), WithinAbs(0.0, kNumericalAtol));
    }
}

TEST_CASE("CircularPath velocity", "[circular_path]") {
    Eigen::Vector3d axis(0, 0, 1);
    Eigen::Vector3d axis_point(0, 0, 0);
    Eigen::Vector3d circle_point(1, 0, 0);
    CircularPath path(axis, axis_point, circle_point);

    auto v = path.velocity(0.0);
    // At s=0, velocity should be in +y direction
    REQUIRE_THAT(v(0), WithinAbs(0.0, kNumericalAtol));
    REQUIRE_THAT(v(1), WithinAbs(1.0, kNumericalAtol));
    REQUIRE_THAT(v.norm(), WithinAbs(1.0, kNumericalAtol));
}

TEST_CASE("CircularPath acceleration", "[circular_path]") {
    Eigen::Vector3d axis(0, 0, 1);
    Eigen::Vector3d axis_point(0, 0, 0);
    Eigen::Vector3d circle_point(2, 0, 0);
    CircularPath path(axis, axis_point, circle_point);

    auto a = path.acceleration(0.0);
    // Centripetal: points toward center = -x direction, magnitude = 1/R
    REQUIRE_THAT(a(0), WithinAbs(-0.5, kNumericalAtol));  // -1/R = -1/2
    REQUIRE_THAT(a(1), WithinAbs(0.0, kNumericalAtol));
}

TEST_CASE("CircularPath point on axis", "[circular_path]") {
    Eigen::Vector3d axis(0, 0, 1);
    Eigen::Vector3d axis_point(0, 0, 0);
    Eigen::Vector3d on_axis(0, 0, 5);

    REQUIRE_THROWS_AS(CircularPath(axis, axis_point, on_axis), std::invalid_argument);
}

TEST_CASE("CircularPath vector evaluation", "[circular_path]") {
    Eigen::Vector3d axis(0, 0, 1);
    Eigen::Vector3d axis_point(0, 0, 0);
    Eigen::Vector3d circle_point(1, 0, 0);
    CircularPath path(axis, axis_point, circle_point);

    Eigen::VectorXd s(4);
    s << 0, M_PI / 2, M_PI, 3 * M_PI / 2;
    auto pts = path.position(s);

    REQUIRE(pts.rows() == 4);
    REQUIRE(pts.cols() == 3);
}

TEST_CASE("CircularPath 3D", "[circular_path]") {
    Eigen::Vector3d axis(1, 1, 1);
    Eigen::Vector3d axis_point(0, 0, 0);
    Eigen::Vector3d circle_point(1, 0, 0);
    CircularPath path(axis, axis_point, circle_point);

    REQUIRE(path.radius() > 0.0);
    auto p = path.position(0.5);
    REQUIRE(p.allFinite());
}

// ===== Frenet Frame =====

TEST_CASE("Frenet frame linear path", "[frenet_frame]") {
    auto curve = [](double s) -> std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d> {
        return {Eigen::Vector3d(s, 0, 0), Eigen::Vector3d(1, 0, 0),
                Eigen::Vector3d(0, 0, 0)};
    };

    Eigen::VectorXd s(3);
    s << 0, 1, 2;
    auto frames = compute_frenet_frames(curve, s);

    REQUIRE(frames.size() == 3);
    REQUIRE_THAT(frames[0].curvature, WithinAbs(0.0, kNumericalAtol));
    REQUIRE_THAT(frames[0].tangent(0), WithinAbs(1.0, kNumericalAtol));
}

TEST_CASE("Frenet frame circular path", "[frenet_frame]") {
    auto curve = [](double s) -> std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d> {
        return {Eigen::Vector3d(std::cos(s), std::sin(s), 0),
                Eigen::Vector3d(-std::sin(s), std::cos(s), 0),
                Eigen::Vector3d(-std::cos(s), -std::sin(s), 0)};
    };

    Eigen::VectorXd s(1);
    s << 0;
    auto frames = compute_frenet_frames(curve, s);

    REQUIRE_THAT(frames[0].curvature, WithinAbs(1.0, kNumericalAtol));
}

// ===== Frenet Helper Functions =====

TEST_CASE("circular_trajectory_with_derivatives", "[frenet_helpers]") {
    SECTION("At u=0") {
        auto [p, dp, d2p] = circular_trajectory_with_derivatives(0.0, 2.0);

        REQUIRE_THAT(p(0), WithinAbs(2.0, kRegularAtol));
        REQUIRE_THAT(p(1), WithinAbs(0.0, kRegularAtol));
        REQUIRE_THAT(p(2), WithinAbs(0.0, kRegularAtol));

        REQUIRE_THAT(dp(0), WithinAbs(0.0, kRegularAtol));
        REQUIRE_THAT(dp(1), WithinAbs(2.0, kRegularAtol));
        REQUIRE_THAT(dp(2), WithinAbs(0.0, kRegularAtol));

        REQUIRE_THAT(d2p(0), WithinAbs(-2.0, kRegularAtol));
        REQUIRE_THAT(d2p(1), WithinAbs(0.0, kRegularAtol));
    }

    SECTION("At u=PI/2") {
        auto [p, dp, d2p] = circular_trajectory_with_derivatives(M_PI / 2.0, 1.0);

        REQUIRE_THAT(p(0), WithinAbs(0.0, kNumericalAtol));
        REQUIRE_THAT(p(1), WithinAbs(1.0, kNumericalAtol));
    }
}

TEST_CASE("helicoidal_trajectory_with_derivatives", "[frenet_helpers]") {
    SECTION("At u=0") {
        auto [p, dp, d2p] = helicoidal_trajectory_with_derivatives(0.0, 2.0, 0.5);

        REQUIRE_THAT(p(0), WithinAbs(2.0, kRegularAtol));
        REQUIRE_THAT(p(1), WithinAbs(0.0, kRegularAtol));
        REQUIRE_THAT(p(2), WithinAbs(0.0, kRegularAtol));

        REQUIRE_THAT(dp(2), WithinAbs(0.5, kRegularAtol));  // z-velocity = d
    }

    SECTION("Z component grows linearly") {
        double u = 3.0;
        double d = 0.5;
        auto [p, dp, d2p] = helicoidal_trajectory_with_derivatives(u, 2.0, d);

        REQUIRE_THAT(p(2), WithinAbs(d * u, kRegularAtol));
        REQUIRE_THAT(d2p(2), WithinAbs(0.0, kRegularAtol));  // No z-acceleration
    }
}

TEST_CASE("Frenet frames with helicoidal helper", "[frenet_helpers]") {
    Eigen::VectorXd s(5);
    for (int i = 0; i < 5; ++i) s[i] = M_PI * i / 4.0;

    auto curve = [](double u) {
        return helicoidal_trajectory_with_derivatives(u, 2.0, 0.5);
    };

    auto frames = compute_frenet_frames(curve, s);
    REQUIRE(frames.size() == 5);

    for (const auto& f : frames) {
        REQUIRE(f.tangent.allFinite());
        REQUIRE(f.curvature >= 0.0);
    }
}

// ===== Linear Trajectory =====

TEST_CASE("Linear trajectory 1D", "[linear_traj]") {
    Eigen::VectorXd p0(1);
    p0 << 0;
    Eigen::VectorXd p1(1);
    p1 << 10;

    auto result = linear_traj(p0, p1, 0.0, 5.0, 6);

    REQUIRE(result.positions.rows() == 6);
    REQUIRE_THAT(result.positions(0, 0), WithinAbs(0.0, kRegularAtol));
    REQUIRE_THAT(result.positions(5, 0), WithinAbs(10.0, kRegularAtol));
    REQUIRE_THAT(result.velocities(0, 0), WithinAbs(2.0, kRegularAtol));
}

TEST_CASE("Linear trajectory 3D", "[linear_traj]") {
    Eigen::VectorXd p0(3);
    p0 << 0, 0, 0;
    Eigen::VectorXd p1(3);
    p1 << 1, 2, 3;

    auto result = linear_traj(p0, p1, 0.0, 1.0, 11);

    REQUIRE(result.positions.rows() == 11);
    REQUIRE(result.positions.cols() == 3);

    // Midpoint
    REQUIRE_THAT(result.positions(5, 0), WithinAbs(0.5, kRegularAtol));
    REQUIRE_THAT(result.positions(5, 1), WithinAbs(1.0, kRegularAtol));
    REQUIRE_THAT(result.positions(5, 2), WithinAbs(1.5, kRegularAtol));

    // Zero acceleration
    REQUIRE_THAT(result.accelerations.norm(), WithinAbs(0.0, kRegularAtol));
}
