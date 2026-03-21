#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <interpolatecpp/quat/quaternion.hpp>
#include "test_data.hpp"

#include <cmath>

using namespace interpolatecpp::quat;
using namespace interpolatecpp::test;
using Catch::Matchers::WithinAbs;

TEST_CASE("Quaternion construction", "[quaternion]") {
    SECTION("Default is identity") {
        Quaternion q;
        REQUIRE_THAT(q.w(), WithinAbs(1.0, kRegularAtol));
        REQUIRE_THAT(q.x(), WithinAbs(0.0, kRegularAtol));
    }

    SECTION("Identity factory") {
        auto q = Quaternion::identity();
        REQUIRE_THAT(q.norm(), WithinAbs(1.0, kRegularAtol));
    }

    SECTION("From angle-axis") {
        auto q = Quaternion::from_angle_axis(M_PI / 2, Eigen::Vector3d::UnitZ());
        REQUIRE_THAT(q.norm(), WithinAbs(1.0, kRegularAtol));
        REQUIRE_THAT(q.w(), WithinAbs(std::cos(M_PI / 4), kNumericalAtol));
    }

    SECTION("From euler angles") {
        auto q = Quaternion::from_euler_angles(0.1, 0.2, 0.3);
        REQUIRE_THAT(q.norm(), WithinAbs(1.0, kNumericalAtol));
    }
}

TEST_CASE("Quaternion arithmetic", "[quaternion]") {
    auto q1 = Quaternion::from_angle_axis(M_PI / 4, Eigen::Vector3d::UnitZ());
    auto q2 = Quaternion::from_angle_axis(M_PI / 4, Eigen::Vector3d::UnitX());

    SECTION("Multiplication") {
        auto q3 = q1 * q2;
        REQUIRE_THAT(q3.norm(), WithinAbs(1.0, kNumericalAtol));
    }

    SECTION("Conjugate") {
        auto qc = q1.conjugate();
        REQUIRE_THAT(qc.w(), WithinAbs(q1.w(), kRegularAtol));
        REQUIRE_THAT(qc.x(), WithinAbs(-q1.x(), kRegularAtol));
    }

    SECTION("Inverse") {
        auto qi = q1.inverse();
        auto prod = q1 * qi;
        REQUIRE_THAT(prod.w(), WithinAbs(1.0, kNumericalAtol));
        REQUIRE_THAT(prod.x(), WithinAbs(0.0, kNumericalAtol));
    }

    SECTION("Dot product") {
        double d = q1.dot_product(q1);
        REQUIRE_THAT(d, WithinAbs(1.0, kNumericalAtol));
    }
}

TEST_CASE("Quaternion exp/log", "[quaternion]") {
    SECTION("Exp of zero") {
        Quaternion zero(0, 0, 0, 0);
        auto result = Quaternion::exp(zero);
        REQUIRE_THAT(result.w(), WithinAbs(1.0, kNumericalAtol));
    }

    SECTION("Log of identity") {
        auto result = Quaternion::log(Quaternion::identity());
        REQUIRE_THAT(result.w(), WithinAbs(0.0, kNumericalAtol));
        REQUIRE(result.vec().norm() < kNumericalAtol);
    }

    SECTION("Exp(log(q)) = q") {
        auto q = Quaternion::from_angle_axis(1.0, Eigen::Vector3d::UnitZ());
        auto result = Quaternion::exp(Quaternion::log(q));
        REQUIRE_THAT(result.w(), WithinAbs(q.w(), kNumericalAtol));
        REQUIRE_THAT(result.x(), WithinAbs(q.x(), kNumericalAtol));
    }

    SECTION("Power") {
        auto q = Quaternion::from_angle_axis(M_PI / 2, Eigen::Vector3d::UnitZ());
        auto q_half = Quaternion::power(q, 0.5);
        REQUIRE_THAT(q_half.norm(), WithinAbs(1.0, kNumericalAtol));
    }
}

TEST_CASE("Quaternion SLERP", "[quaternion]") {
    auto q0 = Quaternion::identity();
    auto q1 = Quaternion::from_angle_axis(M_PI / 2, Eigen::Vector3d::UnitZ());

    SECTION("Endpoints") {
        auto r0 = Quaternion::slerp(q0, q1, 0.0);
        auto r1 = Quaternion::slerp(q0, q1, 1.0);
        REQUIRE_THAT(std::abs(r0.dot_product(q0)), WithinAbs(1.0, kNumericalAtol));
        REQUIRE_THAT(std::abs(r1.dot_product(q1)), WithinAbs(1.0, kNumericalAtol));
    }

    SECTION("Unit norm throughout") {
        for (int i = 0; i <= 10; ++i) {
            double t = i / 10.0;
            auto r = Quaternion::slerp(q0, q1, t);
            REQUIRE_THAT(r.norm(), WithinAbs(1.0, kNumericalAtol));
        }
    }

    SECTION("Double-cover handling") {
        auto q_neg = -q1;
        auto r = Quaternion::slerp(q0, q_neg, 0.5);
        REQUIRE_THAT(r.norm(), WithinAbs(1.0, kNumericalAtol));
    }
}

TEST_CASE("Quaternion SQUAD", "[quaternion]") {
    auto p = Quaternion::identity();
    auto q = Quaternion::from_angle_axis(M_PI / 2, Eigen::Vector3d::UnitZ());
    auto a = Quaternion::slerp(p, q, 0.25);
    auto b = Quaternion::slerp(p, q, 0.75);

    SECTION("Endpoints") {
        auto r0 = Quaternion::squad(p, a, b, q, 0.0);
        auto r1 = Quaternion::squad(p, a, b, q, 1.0);
        REQUIRE_THAT(std::abs(r0.dot_product(p)), WithinAbs(1.0, kNumericalAtol));
        REQUIRE_THAT(std::abs(r1.dot_product(q)), WithinAbs(1.0, kNumericalAtol));
    }

    SECTION("Unit norm") {
        for (int i = 0; i <= 10; ++i) {
            auto r = Quaternion::squad(p, a, b, q, i / 10.0);
            REQUIRE_THAT(r.norm(), WithinAbs(1.0, kNumericalAtol));
        }
    }
}

TEST_CASE("Quaternion intermediate", "[quaternion]") {
    auto q0 = Quaternion::identity();
    auto q1 = Quaternion::from_euler_angles(0.1, 0.2, 0.3);
    auto q2 = Quaternion::from_euler_angles(0.2, 0.4, 0.6);

    auto s = Quaternion::compute_intermediate_quaternion(q0, q1, q2);
    REQUIRE_THAT(s.norm(), WithinAbs(1.0, kNumericalAtol));
}

TEST_CASE("Quaternion conversions", "[quaternion]") {
    auto q = Quaternion::from_angle_axis(M_PI / 3, Eigen::Vector3d(1, 1, 0).normalized());

    SECTION("Rotation matrix") {
        auto R = q.to_rotation_matrix();
        REQUIRE_THAT(R.determinant(), WithinAbs(1.0, kNumericalAtol));
    }

    SECTION("Axis-angle roundtrip") {
        auto [axis, angle] = q.to_axis_angle();
        auto q2 = Quaternion::from_angle_axis(angle, axis);
        REQUIRE_THAT(std::abs(q.dot_product(q2)), WithinAbs(1.0, kNumericalAtol));
    }
}
