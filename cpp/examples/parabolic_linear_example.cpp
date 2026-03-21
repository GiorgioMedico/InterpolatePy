/// Parabolic blend and linear trajectory example -- C++ port of
/// examples/lin_poly_parabolic_ex.py and examples/linear_ex.py
///
/// Part 1: Linear trajectory interpolation (scalar and 2D)
/// Part 2: Parabolic blend trajectory with waypoints

#include <interpolatecpp/path/linear_traj.hpp>
#include <interpolatecpp/motion/parabolic_blend_trajectory.hpp>
#include <interpolatecpp/motion/motion_types.hpp>

#include "example_utils.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace ex = interpolatecpp::examples;
using namespace interpolatecpp::motion;
using namespace interpolatecpp::path;

// =====================================================================
// Part 1: Linear Trajectory
// =====================================================================

static void example_linear_scalar() {
    ex::print_header("Part 1a: Linear Trajectory -- Scalar Case");

    std::cout << "  p0=0, p1=10, t=[0, 2], 100 points\n\n";

    Eigen::VectorXd p0(1);
    p0 << 0.0;
    Eigen::VectorXd p1(1);
    p1 << 10.0;

    auto result = linear_traj(p0, p1, 0.0, 2.0, 100);

    // result.positions is (100 x 1)
    ex::print_value("Start position", result.positions(0, 0), 4);
    ex::print_value("End position", result.positions(result.positions.rows() - 1, 0), 4);
    ex::print_value("Constant velocity", result.velocities(0, 0), 4);
    ex::print_value("Acceleration", result.accelerations(0, 0), 4);

    // Print a few sample rows
    std::cout << "\n  Sample points (every 20th):\n";
    std::cout << std::right << std::setw(10) << "Index"
              << std::setw(14) << "Position"
              << std::setw(14) << "Velocity"
              << std::setw(14) << "Acceleration" << "\n";
    ex::print_separator('-', 52);

    for (int i = 0; i < result.positions.rows(); i += 20) {
        std::cout << std::setw(10) << i
                  << std::fixed << std::setprecision(4)
                  << std::setw(14) << result.positions(i, 0)
                  << std::setw(14) << result.velocities(i, 0)
                  << std::setw(14) << result.accelerations(i, 0) << "\n";
    }
    // Always print last row
    {
        int last = static_cast<int>(result.positions.rows()) - 1;
        std::cout << std::setw(10) << last
                  << std::fixed << std::setprecision(4)
                  << std::setw(14) << result.positions(last, 0)
                  << std::setw(14) << result.velocities(last, 0)
                  << std::setw(14) << result.accelerations(last, 0) << "\n";
    }
    std::cout << "\n";
}

static void example_linear_vector() {
    ex::print_header("Part 1b: Linear Trajectory -- 2D Vector Case");

    std::cout << "  p0=(0,0), p1=(10,5), t=[0, 3], 100 points\n\n";

    Eigen::VectorXd p0(2);
    p0 << 0.0, 0.0;
    Eigen::VectorXd p1(2);
    p1 << 10.0, 5.0;

    auto result = linear_traj(p0, p1, 0.0, 3.0, 100);

    // result.positions is (100 x 2)
    std::cout << "  Start position: (" << result.positions(0, 0) << ", "
              << result.positions(0, 1) << ")\n";
    int last = static_cast<int>(result.positions.rows()) - 1;
    std::cout << "  End position:   (" << result.positions(last, 0) << ", "
              << result.positions(last, 1) << ")\n";
    std::cout << "  Velocity:       (" << result.velocities(0, 0) << ", "
              << result.velocities(0, 1) << ")\n";
    std::cout << "  Acceleration:   (" << result.accelerations(0, 0) << ", "
              << result.accelerations(0, 1) << ")\n";

    // Print sample points
    std::cout << "\n  Sample points (every 20th):\n";
    std::cout << std::right << std::setw(10) << "Index"
              << std::setw(12) << "X pos"
              << std::setw(12) << "Y pos"
              << std::setw(12) << "X vel"
              << std::setw(12) << "Y vel" << "\n";
    ex::print_separator('-', 58);

    for (int i = 0; i < result.positions.rows(); i += 20) {
        std::cout << std::setw(10) << i
                  << std::fixed << std::setprecision(4)
                  << std::setw(12) << result.positions(i, 0)
                  << std::setw(12) << result.positions(i, 1)
                  << std::setw(12) << result.velocities(i, 0)
                  << std::setw(12) << result.velocities(i, 1) << "\n";
    }
    std::cout << std::setw(10) << last
              << std::fixed << std::setprecision(4)
              << std::setw(12) << result.positions(last, 0)
              << std::setw(12) << result.positions(last, 1)
              << std::setw(12) << result.velocities(last, 0)
              << std::setw(12) << result.velocities(last, 1) << "\n";
    std::cout << "\n";
}

// =====================================================================
// Part 2: Parabolic Blend Trajectory
// =====================================================================

static void example_parabolic_blend() {
    ex::print_header("Part 2: Parabolic Blend Trajectory");

    const double pi = std::acos(-1.0);

    const std::vector<double> q = {0.0, 2.0 * pi, pi / 2.0, pi};
    const std::vector<double> t = {0.0, 2.0, 3.0, 5.0};
    const std::vector<double> dt_blend = {0.6, 0.6, 0.6, 0.6};

    std::cout << "  Waypoints: [";
    for (size_t i = 0; i < q.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(4) << q[i];
    }
    std::cout << "]\n";
    std::cout << "  Times:     [";
    for (size_t i = 0; i < t.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << t[i];
    }
    std::cout << "]\n";
    std::cout << "  Blend durations: [";
    for (size_t i = 0; i < dt_blend.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << dt_blend[i];
    }
    std::cout << "]\n\n";

    ParabolicBlendTrajectory traj(q, t, dt_blend);

    const double dur = traj.duration();
    ex::print_value("Total duration", dur, 2);
    ex::print_value("Number of waypoints", static_cast<double>(traj.n_waypoints()), 0);

    // Evaluate at specific times (matching the Python example)
    const std::vector<double> eval_times = {0.5, 2.1, 3.5, 4.8};

    std::cout << "\n  Evaluating at specific times:\n";
    std::cout << std::right
              << std::setw(12) << "Time"
              << std::setw(14) << "Position"
              << std::setw(14) << "Velocity"
              << std::setw(14) << "Acceleration" << "\n";
    ex::print_separator('-', 54);

    for (double time_pt : eval_times) {
        auto r = traj.evaluate(time_pt);
        std::cout << std::fixed << std::setprecision(2) << std::setw(12) << time_pt
                  << std::setprecision(4)
                  << std::setw(14) << r.position
                  << std::setw(14) << r.velocity
                  << std::setw(14) << r.acceleration << "\n";
    }

    // Full trajectory table
    std::cout << "\n  Full trajectory profile:\n\n";
    ex::print_trajectory_table(
        [&](double tv) { return traj.evaluate(tv).position; },
        [&](double tv) { return traj.evaluate(tv).velocity; },
        [&](double tv) { return traj.evaluate(tv).acceleration; },
        0.0, dur, 20);

    // Out-of-bounds test
    std::cout << "  Out-of-bounds evaluation (t = duration + 1.0):\n";
    auto r_oob = traj.evaluate(dur + 1.0);
    ex::print_value("Position", r_oob.position, 4);
    ex::print_value("Velocity", r_oob.velocity, 4);
    ex::print_value("Acceleration", r_oob.acceleration, 4);
}

int main() {
    std::cout << "Parabolic Blend & Linear Trajectory -- C++ Usage Examples\n";
    ex::print_separator('=');

    // Part 1: Linear trajectory
    example_linear_scalar();
    example_linear_vector();

    // Part 2: Parabolic blend
    example_parabolic_blend();

    std::cout << "\nAll parabolic blend and linear trajectory examples completed.\n";
    return 0;
}
