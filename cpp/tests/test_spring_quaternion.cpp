#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <interpolatecpp/quat/spring_quaternion_interpolation.hpp>

#include <Eigen/Core>

#include <cmath>
#include <stdexcept>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace interpolatecpp::quat;

namespace {

std::vector<double> curved_times() { return {0.0, 1.0, 2.0, 3.0}; }

std::vector<Quaternion> curved_keyframes() {
    return {
        Quaternion::identity(),
        Quaternion::from_euler_angles(0.9, 0.1, 0.2),
        Quaternion::from_euler_angles(0.2, 1.1, 0.5),
        Quaternion::from_euler_angles(-0.4, 0.3, 1.4),
    };
}

bool same_orientation(const Quaternion& lhs, const Quaternion& rhs,
                      double tolerance = 1e-9) {
    return std::abs(std::abs(lhs.dot_product(rhs)) - 1.0) <= tolerance;
}

SpringConfig test_config(int samples = 41, int iterations = 150) {
    SpringConfig config;
    config.num_samples = samples;
    config.iterations = iterations;
    return config;
}

}  // namespace

TEST_CASE("SPRING preserves keyframes and the unit sphere", "[spring]") {
    const auto times = curved_times();
    const auto keyframes = curved_keyframes();
    const SpringQuaternionInterpolation spring(times, keyframes, test_config());

    REQUIRE(spring.samples().size() == 41);
    REQUIRE(spring.sample_times().size() == 41);
    REQUIRE(spring.keyframe_indices().size() == keyframes.size());
    for (std::size_t index = 0; index < keyframes.size(); ++index) {
        const std::size_t sample_index = spring.keyframe_indices()[index];
        REQUIRE_THAT(spring.sample_times()[sample_index], WithinAbs(times[index], 1e-12));
        REQUIRE(same_orientation(spring.evaluate(times[index]), keyframes[index]));
    }
    for (const Quaternion& sample : spring.samples()) {
        REQUIRE_THAT(sample.norm(), WithinAbs(1.0, 1e-12));
    }
}

TEST_CASE("SPRING lowers curvature energy on nested grids", "[spring]") {
    SpringConfig config = test_config(101, 300);
    const SpringQuaternionInterpolation spring(curved_times(), curved_keyframes(), config);

    REQUIRE(spring.final_energy() < spring.initial_energy());
    REQUIRE(spring.iterations_run() > 0);
    REQUIRE(spring.iterations_run() <= config.iterations);
    REQUIRE(spring.refinement_sample_counts() == std::vector<int>{7, 20, 101});
    REQUIRE(spring.stage_energy_history().size() == 3);
    for (const auto& history : spring.stage_energy_history()) {
        REQUIRE_FALSE(history.empty());
        for (std::size_t index = 1; index < history.size(); ++index) {
            REQUIRE(history[index] < history[index - 1]);
        }
    }
}

TEST_CASE("SPRING leaves one geodesic unchanged", "[spring]") {
    const Eigen::Vector3d axis = Eigen::Vector3d::UnitZ();
    const std::vector<Quaternion> keyframes = {
        Quaternion::from_angle_axis(0.0, axis),
        Quaternion::from_angle_axis(0.8, axis),
        Quaternion::from_angle_axis(1.6, axis),
    };
    const SpringQuaternionInterpolation spring(
        {0.0, 1.0, 2.0}, keyframes, test_config(21, 100));

    for (int index = 0; index <= 16; ++index) {
        const double time = 2.0 * static_cast<double>(index) / 16.0;
        const Quaternion expected = Quaternion::slerp(
            keyframes.front(), keyframes.back(), time / 2.0);
        REQUIRE(same_orientation(spring.evaluate(time), expected, 1e-8));
    }
}

TEST_CASE("SPRING is invariant to quaternion signs", "[spring]") {
    const auto keyframes = curved_keyframes();
    const std::vector<Quaternion> flipped = {
        keyframes[0], -keyframes[1], keyframes[2], -keyframes[3]};
    const SpringConfig config = test_config(31, 100);
    const SpringQuaternionInterpolation original(curved_times(), keyframes, config);
    const SpringQuaternionInterpolation equivalent(curved_times(), flipped, config);

    for (int index = 0; index <= 12; ++index) {
        const double time = 3.0 * static_cast<double>(index) / 12.0;
        REQUIRE(same_orientation(original.evaluate(time),
                                 equivalent.evaluate(time)));
    }
}

TEST_CASE("SPRING refinement preserves constant angular speed", "[spring]") {
    const Eigen::Vector3d axis = Eigen::Vector3d::UnitZ();
    const std::vector<Quaternion> keyframes = {
        Quaternion::identity(), Quaternion::from_angle_axis(1.6, axis)};
    for (const int sample_count : {21, 31, 101, 201}) {
        CAPTURE(sample_count);
        SpringConfig config = test_config(sample_count, 300);
        // Even with no gradient tolerance, refinement must not distort a
        // stationary curve through coarse-grid truncation error.
        config.tolerance = 0.0;
        const SpringQuaternionInterpolation spring({0.0, 2.0}, keyframes, config);
        for (int index = 0; index <= 40; ++index) {
            const double time = 2.0 * static_cast<double>(index) / 40.0;
            const Quaternion value = spring.evaluate(time);
            REQUIRE_THAT(2.0 * std::atan2(value.z(), value.w()),
                         WithinAbs(0.8 * time, 1e-10));
            REQUIRE((spring.evaluate_velocity(time) - 0.8 * axis).norm() < 1e-9);
            REQUIRE(spring.evaluate_acceleration(time).norm() < 1e-8);
        }
        REQUIRE(spring.final_energy() < 1e-20);
    }
}

TEST_CASE("SPRING refinement reduces curvature near a geodesic", "[spring]") {
    const std::vector<Quaternion> keyframes = {
        Quaternion::identity(), Quaternion::from_euler_angles(1e-4, 0.0, 0.8),
        Quaternion::from_euler_angles(0.0, 0.0, 1.6)};
    const SpringQuaternionInterpolation spring({0.0, 1.0, 2.0}, keyframes);
    REQUIRE(spring.final_energy() < spring.initial_energy());
    REQUIRE(spring.iterations_run() <= spring.config().iterations);
    for (std::size_t index = 0; index < keyframes.size(); ++index) {
        REQUIRE(same_orientation(spring.evaluate(static_cast<double>(index)), keyframes[index], 1e-12));
    }
}

TEST_CASE("SPRING preserves exact endpoint times", "[spring]") {
    const std::vector<double> times = {-100.0, 0.1};
    const Quaternion end = Quaternion::from_euler_angles(0.0, 0.0, 1.2);
    for (const int sample_count : {2, 101}) {
        CAPTURE(sample_count);
        const SpringQuaternionInterpolation spring(
            times, {Quaternion::identity(), end}, test_config(sample_count, 0));
        REQUIRE(spring.sample_times().front() == times.front());
        REQUIRE(spring.sample_times().back() == times.back());
        for (const double time : {std::nextafter(times.front(), times.back()),
                                  std::nextafter(times.back(), times.front()), times.back()}) {
            REQUIRE_THAT(spring.evaluate(time).norm(), WithinAbs(1.0, 1e-12));
            REQUIRE(spring.evaluate_velocity(time).allFinite());
            REQUIRE(spring.evaluate_acceleration(time).allFinite());
        }
        const auto [trajectory_times, samples] = spring.generate_trajectory(7);
        REQUIRE(trajectory_times.front() == times.front());
        REQUIRE(trajectory_times.back() == times.back());
        REQUIRE(same_orientation(samples.back(), end));
    }
}

TEST_CASE("SPRING preserves exact internal keyframe times", "[spring]") {
    const std::vector<double> times = {-100.0, 0.1, 1.0};
    const std::vector<Quaternion> keyframes = {
        Quaternion::identity(), Quaternion::from_euler_angles(0.0, 0.0, 0.6),
        Quaternion::from_euler_angles(0.0, 0.0, 1.2)};
    const SpringQuaternionInterpolation spring(times, keyframes, test_config(21, 0));
    for (std::size_t index = 0; index < times.size(); ++index) {
        REQUIRE(spring.sample_times()[spring.keyframe_indices()[index]] == times[index]);
    }
}

TEST_CASE("SPRING normalizes small nonzero keyframes", "[spring]") {
    const Quaternion end = Quaternion::from_euler_angles(0.0, 0.0, 1.2);
    for (const double scale : {1e-8, -1e-8, 2e-12}) {
        const SpringQuaternionInterpolation spring(
            {0.0, 1.0}, {Quaternion::identity() * scale, end * scale}, test_config(2, 0));
        for (const double time : {0.0, 0.5, 1.0}) {
            REQUIRE(same_orientation(spring.evaluate(time),
                                     Quaternion::slerp(Quaternion::identity(), end, time), 1e-12));
        }
    }
}

TEST_CASE("SPRING returns finite kinematics and trajectories", "[spring]") {
    const SpringQuaternionInterpolation spring(
        curved_times(), curved_keyframes(), test_config(21, 20));

    REQUIRE(spring.evaluate_velocity(1.5).allFinite());
    REQUIRE(spring.evaluate_acceleration(1.5).allFinite());
    const auto [times, samples] = spring.generate_trajectory(17);
    REQUIRE(times.size() == 17);
    REQUIRE(samples.size() == 17);
    REQUIRE_THROWS_AS(spring.generate_trajectory(1), std::invalid_argument);
    REQUIRE_THROWS_AS(spring.evaluate(-0.1), std::invalid_argument);
}

TEST_CASE("SPRING validates its input", "[spring]") {
    const auto keyframes = curved_keyframes();
    REQUIRE_THROWS_AS(
        SpringQuaternionInterpolation({0.0}, {Quaternion::identity()},
                                      test_config(8, 5)),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        SpringQuaternionInterpolation({0.0, 0.0},
                                      {keyframes[0], keyframes[1]},
                                      test_config(8, 5)),
        std::invalid_argument);

    SpringConfig config = test_config(2, 5);
    REQUIRE_THROWS_AS(
        SpringQuaternionInterpolation(curved_times(), keyframes, config),
        std::invalid_argument);
    config = test_config(8, 5);
    config.step_size = 0.0;
    REQUIRE_THROWS_AS(
        SpringQuaternionInterpolation({0.0, 1.0},
                                      {keyframes[0], keyframes[1]}, config),
        std::invalid_argument);
}
