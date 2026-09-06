#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <interpolatecpp/quat/shooting_quaternion_interpolation.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace interpolatecpp::quat;

namespace {

std::vector<Quaternion> curved_keyframes() {
    return {Quaternion::identity(), Quaternion::from_euler_angles(0.9, 0.1, 0.2),
            Quaternion::from_euler_angles(0.2, 1.1, 0.5), Quaternion::from_euler_angles(-0.4, 0.3, 1.4)};
}

}  // namespace

TEST_CASE("Shooting preserves keyframes and matches natural C2 boundaries", "[shooting]") {
    const auto keyframes = curved_keyframes();
    const ShootingQuaternionInterpolation curve({0.0, 1.0, 2.0, 3.0}, keyframes);
    REQUIRE(curve.num_variables() == 27);
    REQUIRE(curve.residual_norm() <= 1e-8);
    REQUIRE(curve.iterations_run() <= 30);
    REQUIRE(curve.integration_steps() > 16);
    REQUIRE(curve.acceleration_energy() > 0.0);
    for (int index = 0; index < 4; ++index) {
        REQUIRE_THAT(std::abs(curve.evaluate(index).dot_product(keyframes[index])), WithinAbs(1.0, 1e-12));
    }
    for (double time : {1.0, 2.0}) {
        REQUIRE((curve.evaluate_velocity(time - 1e-8) - curve.evaluate_velocity(time + 1e-8)).norm() < 2e-7);
        REQUIRE((curve.evaluate_acceleration(time - 1e-8) - curve.evaluate_acceleration(time + 1e-8)).norm() < 2e-7);
    }
    REQUIRE(curve.evaluate_acceleration(0.0).norm() < 1e-7);
    REQUIRE(curve.evaluate_acceleration(3.0).norm() < 1e-7);
}

TEST_CASE("Shooting preserves a rotated constant speed geodesic", "[shooting]") {
    const Quaternion start = Quaternion::from_euler_angles(0.8, -0.2, 0.4);
    const Eigen::Vector3d axis = Eigen::Vector3d(1.0, 2.0, 3.0).normalized();
    const Quaternion end = start * Quaternion::from_angle_axis(1.6, axis);
    const ShootingQuaternionInterpolation curve({0.0, 2.0}, {start, end});
    for (int index = 0; index <= 20; ++index) {
        const double time = static_cast<double>(index) / 10.0;
        const Quaternion expected = Quaternion::slerp(start, end, time / 2.0);
        REQUIRE_THAT(std::abs(curve.evaluate(time).dot_product(expected)), WithinAbs(1.0, 1e-12));
        REQUIRE((curve.evaluate_velocity(time) - 0.8 * axis).norm() < 1e-8);
        REQUIRE(curve.evaluate_acceleration(time).norm() < 1e-8);
    }
    REQUIRE(curve.acceleration_energy() < 1e-12);
}

TEST_CASE("Shooting agrees with a known scalar natural cubic", "[shooting]") {
    const ShootingQuaternionInterpolation curve({0.0, 1.0, 2.0},
        {Quaternion::identity(), Quaternion::from_angle_axis(1.0, Eigen::Vector3d::UnitZ()),
         Quaternion::identity()});
    for (int index = 0; index <= 20; ++index) {
        const double time = static_cast<double>(index) / 10.0;
        const double u = time <= 1.0 ? time : 2.0 - time;
        const double angle = 1.5 * u - 0.5 * u * u * u;
        const double velocity = (time <= 1.0 ? 1.0 : -1.0) * (1.5 - 1.5 * u * u);
        const Quaternion value = curve.evaluate(time);
        REQUIRE_THAT(2.0 * std::atan2(value.z(), value.w()), WithinAbs(angle, 1e-8));
        REQUIRE_THAT(curve.evaluate_velocity(time).z(), WithinAbs(velocity, 1e-8));
        REQUIRE_THAT(curve.evaluate_acceleration(time).z(), WithinAbs(-3.0 * u, 1e-8));
    }
    REQUIRE_THAT(curve.acceleration_energy(), WithinAbs(6.0, 1e-7));
}

TEST_CASE("Shooting is invariant to quaternion scale signs and time units", "[shooting]") {
    const auto keyframes = curved_keyframes();
    auto scaled_keyframes = keyframes;
    for (std::size_t index = 0; index < keyframes.size(); ++index) {
        scaled_keyframes[index] = keyframes[index] * (index % 2 == 0 ? 1e-8 : -1e-8);
    }
    const ShootingQuaternionInterpolation base({0.0, 1.0, 2.0, 3.0}, keyframes);
    const ShootingQuaternionInterpolation scaled({10.0, 12.0, 14.0, 16.0}, scaled_keyframes);
    for (double time : {0.4, 1.1, 2.8}) {
        REQUIRE_THAT(std::abs(base.evaluate(time).dot_product(scaled.evaluate(10.0 + 2.0 * time))), WithinAbs(1.0, 1e-12));
        REQUIRE((0.5 * base.evaluate_velocity(time) - scaled.evaluate_velocity(10.0 + 2.0 * time)).norm() < 1e-8);
        REQUIRE((0.25 * base.evaluate_acceleration(time) - scaled.evaluate_acceleration(10.0 + 2.0 * time)).norm() < 1e-8);
    }
    REQUIRE_THAT(scaled.acceleration_energy(), WithinAbs(base.acceleration_energy() / 8.0, 1e-8));
}

TEST_CASE("Shooting output resolution does not alter the solve", "[shooting]") {
    const ShootingQuaternionInterpolation curve({-100.0, 0.1},
        {Quaternion::identity(), Quaternion::from_euler_angles(0.0, 0.0, 1.2)});
    const int iterations = curve.iterations_run();
    for (int count : {17, 1001}) {
        const auto [times, quaternions] = curve.generate_trajectory(count);
        REQUIRE(times.size() == static_cast<std::size_t>(count));
        REQUIRE(times.front() == -100.0);
        REQUIRE(times.back() == 0.1);
        for (const auto& value : quaternions) REQUIRE_THAT(value.norm(), WithinAbs(1.0, 1e-12));
    }
    REQUIRE(curve.iterations_run() == iterations);
    REQUIRE(curve.num_variables() == 9);
    REQUIRE_THAT(curve.evaluate(std::nextafter(0.1, -100.0)).norm(), WithinAbs(1.0, 1e-12));
}

TEST_CASE("Shooting handles extreme finite time units without false convergence or NaN energy", "[shooting]") {
    const std::vector<Quaternion> geodesic{
        Quaternion::identity(), Quaternion::from_euler_angles(0.4, 0.0, 0.0),
        Quaternion::from_euler_angles(0.8, 0.0, 0.0)};
    for (const double scale : {1e-160, 1e-110, 1e110, 1e160}) {
        const ShootingQuaternionInterpolation curve({0.0, scale, 2.0 * scale}, geodesic);
        REQUIRE(curve.acceleration_energy() == 0.0);
        REQUIRE(std::isfinite(curve.residual_norm()));
        for (const double fraction : {0.5, 1.5}) {
            const auto expected = Quaternion::from_euler_angles(0.4 * fraction, 0.0, 0.0);
            REQUIRE_THAT(std::abs(curve.evaluate(fraction * scale).dot_product(expected)), WithinAbs(1.0, 1e-12));
            REQUIRE((curve.evaluate_velocity(fraction * scale) * scale - Eigen::Vector3d(0.4, 0.0, 0.0)).norm() < 1e-8);
            REQUIRE(curve.evaluate_acceleration(fraction * scale) == Eigen::Vector3d::Zero());
        }
    }
    const auto keyframes = curved_keyframes();
    const ShootingQuaternionInterpolation base({0.0, 0.2, 0.9, 2.7}, keyframes);
    const ShootingQuaternionInterpolation scaled({0.0, 0.2e160, 0.9e160, 2.7e160}, keyframes);
    REQUIRE(scaled.residual_norm() <= 1e-8);
    for (const double time : {0.1, 0.5, 1.8}) {
        REQUIRE_THAT(std::abs(base.evaluate(time).dot_product(scaled.evaluate(time * 1e160))), WithinAbs(1.0, 1e-12));
        REQUIRE((scaled.evaluate_acceleration(time * 1e160) * 1e160 * 1e160 - base.evaluate_acceleration(time)).norm() < 1e-3);
    }
}

TEST_CASE("Shooting rejects invalid data and failed solves", "[shooting]") {
    const auto identity = Quaternion::identity();
    REQUIRE_THROWS_AS(ShootingQuaternionInterpolation({0.0}, {identity}), std::invalid_argument);
    REQUIRE_THROWS_AS(ShootingQuaternionInterpolation({0.0, 0.0}, {identity, identity}), std::invalid_argument);
    REQUIRE_THROWS_AS(ShootingQuaternionInterpolation({0.0, 1.0}, {identity}), std::invalid_argument);
    REQUIRE_THROWS_AS(ShootingQuaternionInterpolation({0.0, 1.0}, {identity, Quaternion(0, 0, 0, 0)}), std::invalid_argument);
    ShootingConfig config;
    config.max_iterations = 1;
    REQUIRE_THROWS_AS(ShootingQuaternionInterpolation({0.0, 1.0, 2.0, 3.0}, curved_keyframes(), config), std::runtime_error);
    config = ShootingConfig{};
    config.integration_steps = 1;
    config.max_integration_steps = 2;
    REQUIRE_THROWS_AS(ShootingQuaternionInterpolation({0.0, 1.0, 2.0, 3.0}, curved_keyframes(), config), std::runtime_error);
    config.tolerance = 0.0;
    REQUIRE_THROWS_AS(ShootingQuaternionInterpolation({0.0, 1.0}, {identity, identity}, config), std::invalid_argument);
    const ShootingQuaternionInterpolation curve({0.0, 1.0}, {identity, identity});
    REQUIRE(curve.iterations_run() == 0);
    for (double time : {-0.1, 1.1, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        REQUIRE_THROWS_AS(curve.evaluate(time), std::invalid_argument);
        REQUIRE_THROWS_AS(curve.evaluate_velocity(time), std::invalid_argument);
        REQUIRE_THROWS_AS(curve.evaluate_acceleration(time), std::invalid_argument);
    }
    REQUIRE_THROWS_AS(curve.generate_trajectory(1), std::invalid_argument);
}
