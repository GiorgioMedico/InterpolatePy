/// Native SPRING quaternion interpolation and timing example.

#include <interpolatecpp/quat/quaternion.hpp>
#include <interpolatecpp/quat/spring_quaternion_interpolation.hpp>

#include "example_utils.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

namespace ex = interpolatecpp::examples;
using namespace interpolatecpp::quat;

int main() {
    ex::print_header("Native SPRING Quaternion Interpolation");

    const std::vector<double> times = {0.0, 0.8, 1.8, 3.0, 4.2, 5.0};
    const std::vector<Quaternion> keyframes = {
        Quaternion::identity(),
        Quaternion::from_euler_angles(0.8, 0.1, 0.2),
        Quaternion::from_euler_angles(1.2, 0.9, 0.1),
        Quaternion::from_euler_angles(0.1, 1.3, 1.0),
        Quaternion::from_euler_angles(-0.7, 0.3, 1.6),
        Quaternion::from_euler_angles(-0.2, -0.5, 2.0),
    };
    SpringConfig config;
    config.num_samples = 151;
    config.iterations = 450;
    config.refinement_levels = 3;

    const auto construction_start = std::chrono::steady_clock::now();
    const SpringQuaternionInterpolation spring(times, keyframes, config);
    const auto construction_end = std::chrono::steady_clock::now();

    constexpr int evaluation_count = 1000;
    double checksum = 0.0;
    const auto evaluation_start = std::chrono::steady_clock::now();
    for (int index = 0; index < evaluation_count; ++index) {
        const double fraction = static_cast<double>(index) /
                                static_cast<double>(evaluation_count - 1);
        checksum += spring.evaluate(times.front() +
                                    fraction * (times.back() - times.front()))
                        .w();
    }
    const auto evaluation_end = std::chrono::steady_clock::now();

    const auto construction_us =
        std::chrono::duration<double, std::micro>(construction_end -
                                                  construction_start)
            .count();
    const auto evaluation_us =
        std::chrono::duration<double, std::micro>(evaluation_end -
                                                  evaluation_start)
            .count();
    const double reduction =
        1.0 - spring.final_energy() / spring.initial_energy();

    std::cout << std::fixed << std::setprecision(3)
              << "  Construction: " << construction_us / 1000.0 << " ms\n"
              << "  " << evaluation_count << " evaluations: "
              << evaluation_us / 1000.0 << " ms ("
              << evaluation_us / static_cast<double>(evaluation_count)
              << " us/sample)\n"
              << "  Curvature-energy reduction: " << reduction * 100.0 << "%\n"
              << "  Iterations accepted: " << spring.iterations_run() << "\n"
              << "  Refinement samples:";
    for (const int count : spring.refinement_sample_counts()) {
        std::cout << ' ' << count;
    }
    std::cout << "\n  Evaluation checksum: " << checksum << '\n';

    for (std::size_t index = 0; index < keyframes.size(); ++index) {
        const double agreement = std::abs(
            spring.evaluate(times[index]).dot_product(keyframes[index]));
        if (agreement < 1.0 - 1e-9) return 1;
    }
    return 0;
}
