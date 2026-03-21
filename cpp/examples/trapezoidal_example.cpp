/// Trapezoidal trajectory example -- C++ port of examples/trapezoidal_ex.py
///
/// Demonstrates trapezoidal velocity profile trajectory generation:
/// 1. Basic trajectory (q0=0, q1=10, amax=2, vmax=3)
/// 2. Non-zero boundary velocities
/// 3. Duration-based construction
/// 4. Multi-point trajectory with heuristic velocities
/// 5. Time-constrained waypoints

#include <interpolatecpp/motion/trapezoidal_trajectory.hpp>

#include "example_utils.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace ex = interpolatecpp::examples;
using namespace interpolatecpp::motion;

static void example_basic_trajectory() {
    ex::print_header("Example 1: Basic Trapezoidal Trajectory");

    std::cout << "  q0=0, q1=10, amax=3, vmax=4, v0=0, v1=0\n\n";

    TrapezoidalTrajectory traj(0.0, 10.0, 3.0, 4.0);

    const double dur = traj.duration();
    ex::print_value("Duration", dur, 2);

    auto r_start = traj.evaluate(0.0);
    auto r_mid = traj.evaluate(dur / 2.0);
    auto r_end = traj.evaluate(dur);

    ex::print_value("Position at t=0", r_start.position, 2);
    ex::print_value("Position at t=mid", r_mid.position, 2);
    ex::print_value("Position at t=end", r_end.position, 2);

    std::cout << "\n";
    ex::print_trajectory_table(
        [&](double t) { return traj.evaluate(t).position; },
        [&](double t) { return traj.evaluate(t).velocity; },
        [&](double t) { return traj.evaluate(t).acceleration; },
        traj.t_start(), traj.t_end(), 15);
}

static void example_nonzero_velocities() {
    ex::print_header("Example 2: Non-Zero Boundary Velocities");

    std::cout << "  q0=0, q1=10, amax=5, vmax=4, v0=1.5, v1=2.0\n\n";

    TrapezoidalTrajectory traj(0.0, 10.0, 5.0, 4.0, 1.5, 2.0);

    const double dur = traj.duration();
    ex::print_value("Duration", dur, 2);

    auto r_start = traj.evaluate(0.0);
    auto r_end = traj.evaluate(dur);

    ex::print_value("Initial velocity", r_start.velocity, 2);
    ex::print_value("Final velocity", r_end.velocity, 2);

    std::cout << "\n";
    ex::print_trajectory_table(
        [&](double t) { return traj.evaluate(t).position; },
        [&](double t) { return traj.evaluate(t).velocity; },
        [&](double t) { return traj.evaluate(t).acceleration; },
        traj.t_start(), traj.t_end(), 15);
}

static void example_duration_based() {
    ex::print_header("Example 3: Duration-Based Construction");

    std::cout << "  q0=0, q1=10, amax=5, v0=0, v1=0, duration=4.0\n\n";

    TrapezoidalTrajectory traj(
        TrapezoidalTrajectory::DurationBased{},
        0.0, 10.0, 5.0, 0.0, 0.0, 0.0, 4.0);

    const double dur = traj.duration();
    ex::print_value("Specified duration", 4.0, 2);
    ex::print_value("Actual duration", dur, 2);

    auto r_start = traj.evaluate(0.0);
    auto r_end = traj.evaluate(dur);

    ex::print_value("Start position", r_start.position, 4);
    ex::print_value("End position", r_end.position, 4);

    std::cout << "\n";
    ex::print_trajectory_table(
        [&](double t) { return traj.evaluate(t).position; },
        [&](double t) { return traj.evaluate(t).velocity; },
        [&](double t) { return traj.evaluate(t).acceleration; },
        traj.t_start(), traj.t_end(), 15);
}

static void example_multipoint_heuristic() {
    ex::print_header("Example 4: Multi-Point Trajectory with Heuristic Velocities");

    const std::vector<double> points = {0.0, 5.0, -3.0, 8.0, -2.0, 4.0, 0.0};
    const double amax = 8.0;
    const double vmax = 6.0;

    std::cout << "  Waypoints: [";
    for (size_t i = 0; i < points.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << points[i];
    }
    std::cout << "]\n";
    std::cout << "  amax=" << amax << ", vmax=" << vmax << "\n\n";

    auto segments = TrapezoidalTrajectory::interpolate_waypoints(
        points, amax, vmax);

    // Compute total duration from segments
    double total_dur = 0.0;
    if (!segments.empty()) {
        total_dur = segments.back().t_end() - segments.front().t_start();
    }
    ex::print_value("Total duration", total_dur, 2);
    ex::print_value("Number of segments", static_cast<double>(segments.size()), 0);

    // Verify waypoint passage
    std::cout << "\n  Verifying waypoint positions:\n";
    for (size_t i = 0; i < segments.size(); ++i) {
        auto r = TrapezoidalTrajectory::evaluate_multipoint(
            segments, segments[i].t_start());
        std::cout << "    Segment " << i << " start: position = "
                  << std::fixed << std::setprecision(2) << r.position << "\n";
    }
    {
        auto r_final = TrapezoidalTrajectory::evaluate_multipoint(
            segments, segments.back().t_end());
        std::cout << "    Final: position = "
                  << std::fixed << std::setprecision(2) << r_final.position << "\n";
    }

    std::cout << "\n";
    ex::print_trajectory_table(
        [&](double t) {
            return TrapezoidalTrajectory::evaluate_multipoint(segments, t).position;
        },
        [&](double t) {
            return TrapezoidalTrajectory::evaluate_multipoint(segments, t).velocity;
        },
        [&](double t) {
            return TrapezoidalTrajectory::evaluate_multipoint(segments, t).acceleration;
        },
        segments.front().t_start(), segments.back().t_end(), 20);
}

static void example_time_constrained() {
    ex::print_header("Example 5: Time-Constrained Multi-Point Trajectory");

    const std::vector<double> points = {0.0, 5.0, 10.0, 7.0, 15.0};
    const std::vector<double> times = {0.0, 2.0, 4.0, 7.0, 10.0};
    const double amax = 15.0;
    const double vmax = 100.0;  // Large vmax to allow time-based planning

    std::cout << "  Waypoints: [";
    for (size_t i = 0; i < points.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << points[i];
    }
    std::cout << "]\n";
    std::cout << "  Times:     [";
    for (size_t i = 0; i < times.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << times[i];
    }
    std::cout << "]\n\n";

    auto segments = TrapezoidalTrajectory::interpolate_waypoints(
        points, amax, vmax, 0.0, 0.0, times);

    double total_dur = 0.0;
    if (!segments.empty()) {
        total_dur = segments.back().t_end() - segments.front().t_start();
    }
    ex::print_value("Total duration", total_dur, 2);

    // Verify position at specified times
    std::cout << "\n  Verifying position at specified times:\n";
    for (size_t i = 0; i < times.size(); ++i) {
        auto r = TrapezoidalTrajectory::evaluate_multipoint(segments, times[i]);
        std::cout << "    At t=" << std::fixed << std::setprecision(1) << times[i]
                  << "s: Expected=" << std::setprecision(1) << points[i]
                  << ", Actual=" << std::setprecision(1) << r.position << "\n";
    }

    std::cout << "\n";
    ex::print_trajectory_table(
        [&](double t) {
            return TrapezoidalTrajectory::evaluate_multipoint(segments, t).position;
        },
        [&](double t) {
            return TrapezoidalTrajectory::evaluate_multipoint(segments, t).velocity;
        },
        [&](double t) {
            return TrapezoidalTrajectory::evaluate_multipoint(segments, t).acceleration;
        },
        times.front(), times.back(), 20);
}

int main() {
    std::cout << "TrapezoidalTrajectory -- C++ Usage Examples\n";
    ex::print_separator('=');

    example_basic_trajectory();
    example_nonzero_velocities();
    example_duration_based();
    example_multipoint_heuristic();
    example_time_constrained();

    std::cout << "\nAll trapezoidal trajectory examples completed.\n";
    return 0;
}
