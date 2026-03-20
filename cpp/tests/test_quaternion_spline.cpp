#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <interpolatecpp/quat/log_quaternion_interpolation.hpp>
#include <interpolatecpp/quat/quaternion_spline.hpp>
#include <interpolatecpp/quat/squad_c2.hpp>
#include "test_data.hpp"

#include <cmath>

using namespace interpolatecpp::quat;
using namespace interpolatecpp::test;
using Catch::Matchers::WithinAbs;

// Helper: create test quaternion sequence
static std::vector<Quaternion> make_test_quats(int n = 5) {
    std::vector<Quaternion> quats;
    for (int i = 0; i < n; ++i) {
        double a = 0.1 * (i + 1);
        quats.push_back(Quaternion::from_euler_angles(a, a * 2, a * 3));
    }
    return quats;
}

static std::vector<double> make_test_times(int n = 5) {
    std::vector<double> times(n);
    for (int i = 0; i < n; ++i) times[i] = static_cast<double>(i);
    return times;
}

// ===== QuaternionSpline =====

TEST_CASE("QuaternionSpline SLERP", "[quat_spline]") {
    auto times = make_test_times();
    auto quats = make_test_quats();
    QuaternionSpline spline(times, quats, QuaternionSpline::Method::Slerp);

    SECTION("Endpoints") {
        auto r0 = spline.evaluate(0.0);
        REQUIRE_THAT(std::abs(r0.dot_product(quats[0])), WithinAbs(1.0, kNumericalAtol));

        auto rn = spline.evaluate(4.0);
        REQUIRE_THAT(std::abs(rn.dot_product(quats[4])), WithinAbs(1.0, kNumericalAtol));
    }

    SECTION("Unit norm throughout") {
        for (int i = 0; i <= 20; ++i) {
            double t = 4.0 * i / 20.0;
            auto r = spline.evaluate(t);
            REQUIRE_THAT(r.norm(), WithinAbs(1.0, kNumericalAtol));
        }
    }

    SECTION("Velocity finite") {
        auto v = spline.evaluate_velocity(2.0);
        REQUIRE(v.allFinite());
    }

    SECTION("Acceleration finite") {
        auto a = spline.evaluate_acceleration(2.0);
        REQUIRE(a.allFinite());
    }
}

TEST_CASE("QuaternionSpline SQUAD", "[quat_spline]") {
    auto times = make_test_times();
    auto quats = make_test_quats();
    QuaternionSpline spline(times, quats, QuaternionSpline::Method::Squad);

    for (int i = 0; i <= 20; ++i) {
        double t = 4.0 * i / 20.0;
        auto r = spline.evaluate(t);
        REQUIRE_THAT(r.norm(), WithinAbs(1.0, kNumericalAtol));
    }
}

TEST_CASE("QuaternionSpline Auto", "[quat_spline]") {
    auto times = make_test_times();
    auto quats = make_test_quats();
    QuaternionSpline spline(times, quats, QuaternionSpline::Method::Auto);

    auto r = spline.evaluate(2.0);
    REQUIRE_THAT(r.norm(), WithinAbs(1.0, kNumericalAtol));
}

TEST_CASE("QuaternionSpline validation", "[quat_spline]") {
    SECTION("Mismatched sizes") {
        std::vector<double> t = {0, 1};
        std::vector<Quaternion> q = {Quaternion::identity()};
        REQUIRE_THROWS_AS(QuaternionSpline(t, q), std::invalid_argument);
    }

    SECTION("Too few points") {
        std::vector<double> t = {0};
        std::vector<Quaternion> q = {Quaternion::identity()};
        REQUIRE_THROWS_AS(QuaternionSpline(t, q), std::invalid_argument);
    }
}

// ===== SquadC2 =====

TEST_CASE("SquadC2 construction", "[squad_c2]") {
    auto times = make_test_times(6);
    auto quats = make_test_quats(6);
    SquadC2 spline(times, quats);

    SECTION("Unit norm throughout") {
        for (int i = 0; i <= 20; ++i) {
            double t = 5.0 * i / 20.0;
            auto r = spline.evaluate(t);
            REQUIRE_THAT(r.norm(), WithinAbs(1.0, kNumericalAtol));
        }
    }
}

TEST_CASE("SquadC2 endpoints", "[squad_c2]") {
    auto times = make_test_times(6);
    auto quats = make_test_quats(6);
    SquadC2 spline(times, quats);

    auto r0 = spline.evaluate(0.0);
    auto rn = spline.evaluate(5.0);

    REQUIRE_THAT(std::abs(r0.dot_product(quats[0])), WithinAbs(1.0, kNumericalAtol));
    REQUIRE_THAT(std::abs(rn.dot_product(quats[5])), WithinAbs(1.0, kNumericalAtol));
}

TEST_CASE("SquadC2 zero-clamped boundaries", "[squad_c2]") {
    auto times = make_test_times(6);
    auto quats = make_test_quats(6);
    SquadC2 spline(times, quats);

    // Angular velocity should be near-zero at boundaries
    auto v_start = spline.evaluate_velocity(0.0);
    auto v_end = spline.evaluate_velocity(5.0);

    REQUIRE(v_start.norm() < 0.1);
    REQUIRE(v_end.norm() < 0.1);
}

TEST_CASE("SquadC2 velocity and acceleration", "[squad_c2]") {
    auto times = make_test_times(6);
    auto quats = make_test_quats(6);
    SquadC2 spline(times, quats);

    auto v = spline.evaluate_velocity(2.5);
    auto a = spline.evaluate_acceleration(2.5);

    REQUIRE(v.allFinite());
    REQUIRE(a.allFinite());
}

// ===== LogQuaternionInterpolation =====

TEST_CASE("LogQuaternionInterpolation construction", "[log_quat]") {
    auto times = make_test_times(6);
    auto quats = make_test_quats(6);
    LogQuaternionInterpolation lqi(times, quats, 3);

    SECTION("Unit norm throughout") {
        for (int i = 0; i <= 20; ++i) {
            double t = 5.0 * i / 20.0;
            auto r = lqi.evaluate(t);
            REQUIRE_THAT(r.norm(), WithinAbs(1.0, 1e-4));
        }
    }
}

TEST_CASE("LogQuaternionInterpolation endpoints", "[log_quat]") {
    auto times = make_test_times(6);
    auto quats = make_test_quats(6);
    LogQuaternionInterpolation lqi(times, quats, 3);

    auto r0 = lqi.evaluate(0.0);
    auto rn = lqi.evaluate(5.0);

    REQUIRE_THAT(std::abs(r0.dot_product(quats[0])), WithinAbs(1.0, 1e-4));
    REQUIRE_THAT(std::abs(rn.dot_product(quats[5])), WithinAbs(1.0, 1e-4));
}

TEST_CASE("LogQuaternionInterpolation velocity", "[log_quat]") {
    auto times = make_test_times(6);
    auto quats = make_test_quats(6);
    LogQuaternionInterpolation lqi(times, quats, 3);

    auto v = lqi.evaluate_velocity(2.5);
    auto a = lqi.evaluate_acceleration(2.5);

    REQUIRE(v.allFinite());
    REQUIRE(a.allFinite());
}

TEST_CASE("LogQuaternionInterpolation validation", "[log_quat]") {
    SECTION("Too few points") {
        std::vector<double> t = {0};
        std::vector<Quaternion> q = {Quaternion::identity()};
        REQUIRE_THROWS_AS(LogQuaternionInterpolation(t, q), std::invalid_argument);
    }
}
