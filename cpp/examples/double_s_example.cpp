/// Double-S trajectory example -- C++ port of examples/double_s_ex.py
///
/// Demonstrates double-S (7-phase) trajectory with bounded jerk:
/// 1. Standard trajectory (q0=0, q1=30, v0=0, v1=0)
/// 2. Velocity matching (same position, different velocity)
/// 3. Negative displacement
/// 4. Asymmetric velocities
/// 5. Phase duration inspection

#include <interpolatecpp/motion/double_s_trajectory.hpp>
#include <interpolatecpp/motion/motion_types.hpp>

#include "example_utils.hpp"

#include <iostream>
#include <map>
#include <string>
#include <tuple>

namespace ex = interpolatecpp::examples;
using namespace interpolatecpp::motion;

static void print_phase_durations(const std::map<std::string, double>& phases) {
    std::cout << "  Phase durations:\n";
    for (const auto& [name, dur] : phases) {
        std::cout << "    " << std::left << std::setw(24) << name
                  << ": " << std::fixed << std::setprecision(6) << dur << " s\n";
    }
    std::cout << std::right;
}

static void example_standard() {
    ex::print_header("Example 1: Standard Double-S Trajectory");

    std::cout << "  q0=0, q1=30, v0=0, v1=0\n";
    std::cout << "  Bounds: v=20, a=60, j=120\n\n";

    StateParams state{0.0, 30.0, 0.0, 0.0};
    TrajectoryBounds bounds(20.0, 60.0, 120.0);

    DoubleSTrajectory traj(state, bounds);

    const double dur = traj.duration();
    ex::print_value("Duration", dur, 3);

    // Evaluate at start and end
    auto r_start = traj.evaluate(0.0);
    auto r_end = traj.evaluate(dur);

    ex::print_value("Start position", r_start.position, 3);
    ex::print_value("Start velocity", r_start.velocity, 3);
    ex::print_value("End position", r_end.position, 3);
    ex::print_value("End velocity", r_end.velocity, 3);

    // Phase durations
    std::cout << "\n";
    auto phases = traj.phase_durations();
    print_phase_durations(phases);

    std::cout << "\n";
    ex::print_full_trajectory_table(
        [&](double t) -> std::tuple<double, double, double, double> {
            auto r = traj.evaluate(t);
            return {r.position, r.velocity, r.acceleration, r.jerk};
        },
        0.0, dur, 20);
}

static void example_velocity_matching() {
    ex::print_header("Example 2: Velocity Matching (Same Position)");

    std::cout << "  q0=10, q1=10, v0=0, v1=5\n";
    std::cout << "  Bounds: v=20, a=60, j=120\n\n";

    StateParams state{10.0, 10.0, 0.0, 5.0};
    TrajectoryBounds bounds(20.0, 60.0, 120.0);

    DoubleSTrajectory traj(state, bounds);

    const double dur = traj.duration();
    ex::print_value("Duration", dur, 3);

    auto r_start = traj.evaluate(0.0);
    auto r_end = traj.evaluate(dur);

    ex::print_value("Start position", r_start.position, 3);
    ex::print_value("Start velocity", r_start.velocity, 3);
    ex::print_value("End position", r_end.position, 3);
    ex::print_value("End velocity", r_end.velocity, 3);

    std::cout << "\n";
    ex::print_full_trajectory_table(
        [&](double t) -> std::tuple<double, double, double, double> {
            auto r = traj.evaluate(t);
            return {r.position, r.velocity, r.acceleration, r.jerk};
        },
        0.0, dur, 15);
}

static void example_negative_displacement() {
    ex::print_header("Example 3: Negative Displacement");

    std::cout << "  q0=0, q1=-30, v0=0, v1=0\n";
    std::cout << "  Bounds: v=20, a=60, j=120\n\n";

    StateParams state{0.0, -30.0, 0.0, 0.0};
    TrajectoryBounds bounds(20.0, 60.0, 120.0);

    DoubleSTrajectory traj(state, bounds);

    const double dur = traj.duration();
    ex::print_value("Duration", dur, 3);

    auto r_start = traj.evaluate(0.0);
    auto r_end = traj.evaluate(dur);

    ex::print_value("Start position", r_start.position, 3);
    ex::print_value("End position", r_end.position, 3);

    // Phase durations
    std::cout << "\n";
    auto phases = traj.phase_durations();
    print_phase_durations(phases);

    std::cout << "\n";
    ex::print_full_trajectory_table(
        [&](double t) -> std::tuple<double, double, double, double> {
            auto r = traj.evaluate(t);
            return {r.position, r.velocity, r.acceleration, r.jerk};
        },
        0.0, dur, 15);
}

static void example_asymmetric_velocities() {
    ex::print_header("Example 4: Asymmetric Velocities");

    std::cout << "  q0=0, q1=50, v0=10, v1=-5\n";
    std::cout << "  Bounds: v=20, a=60, j=120\n\n";

    StateParams state{0.0, 50.0, 10.0, -5.0};
    TrajectoryBounds bounds(20.0, 60.0, 120.0);

    DoubleSTrajectory traj(state, bounds);

    const double dur = traj.duration();
    ex::print_value("Duration", dur, 3);

    auto r_start = traj.evaluate(0.0);
    auto r_end = traj.evaluate(dur);

    ex::print_value("Start position", r_start.position, 3);
    ex::print_value("Start velocity", r_start.velocity, 3);
    ex::print_value("End position", r_end.position, 3);
    ex::print_value("End velocity", r_end.velocity, 3);

    // Phase durations
    std::cout << "\n";
    auto phases = traj.phase_durations();
    print_phase_durations(phases);

    std::cout << "\n";
    ex::print_full_trajectory_table(
        [&](double t) -> std::tuple<double, double, double, double> {
            auto r = traj.evaluate(t);
            return {r.position, r.velocity, r.acceleration, r.jerk};
        },
        0.0, dur, 20);
}

static void example_phase_durations() {
    ex::print_header("Example 5: Phase Duration Comparison");

    std::cout << "  Comparing phase structure across different scenarios.\n\n";

    struct Scenario {
        const char* name;
        StateParams state;
        TrajectoryBounds bounds;
    };

    const Scenario scenarios[] = {
        {"Short displacement",
         StateParams{0.0, 5.0, 0.0, 0.0},
         TrajectoryBounds(20.0, 60.0, 120.0)},
        {"Long displacement",
         StateParams{0.0, 100.0, 0.0, 0.0},
         TrajectoryBounds(20.0, 60.0, 120.0)},
        {"High jerk limit",
         StateParams{0.0, 30.0, 0.0, 0.0},
         TrajectoryBounds(20.0, 60.0, 500.0)},
        {"Low velocity bound",
         StateParams{0.0, 30.0, 0.0, 0.0},
         TrajectoryBounds(5.0, 60.0, 120.0)},
    };

    for (const auto& [name, state, bounds] : scenarios) {
        DoubleSTrajectory traj(state, bounds);
        std::cout << "  " << name << ":\n";
        ex::print_value("    Total duration", traj.duration(), 4);

        auto phases = traj.phase_durations();
        for (const auto& [phase_name, phase_dur] : phases) {
            std::cout << "      " << std::left << std::setw(24) << phase_name
                      << ": " << std::fixed << std::setprecision(4) << phase_dur << " s\n";
        }
        std::cout << std::right << "\n";
    }
}

int main() {
    std::cout << "Double-S Trajectory -- C++ Usage Examples\n";
    ex::print_separator('=');

    example_standard();
    example_velocity_matching();
    example_negative_displacement();
    example_asymmetric_velocities();
    example_phase_durations();

    std::cout << "\nAll double-S trajectory examples completed.\n";
    return 0;
}
