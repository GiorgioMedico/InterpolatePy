/// Polynomial trajectory example -- C++ port of examples/polynomials_ex.py
///
/// Demonstrates polynomial trajectory generation with orders 3, 5, and 7:
/// 1. Order 3 two-point trajectory
/// 2. Order 5 two-point trajectory
/// 3. Order 7 two-point trajectory (comparison of all three)
/// 4. Multi-point trajectory with heuristic velocities

#include <interpolatecpp/motion/polynomial_trajectory.hpp>
#include <interpolatecpp/motion/motion_types.hpp>

#include "example_utils.hpp"

#include <iostream>
#include <tuple>
#include <vector>

namespace ex = interpolatecpp::examples;
using namespace interpolatecpp::motion;

static void example_order3_two_point() {
    ex::print_header("Example 1: Order 3 Two-Point Trajectory");

    std::cout << "  Boundary: pos 0->1, vel 0->0, t=[0, 2]\n\n";

    BoundaryCondition bc_start{0.0, 0.0, 0.0, 0.0};
    BoundaryCondition bc_end{1.0, 0.0, 0.0, 0.0};
    TimeInterval interval{0.0, 2.0};

    PolynomialTrajectory traj(bc_start, bc_end, interval, PolynomialTrajectory::ORDER_3);

    ex::print_value("Order", static_cast<double>(traj.order()), 0);
    ex::print_value("Duration", traj.duration(), 2);

    // Show boundary values
    auto r_start = traj.evaluate(0.0);
    auto r_end = traj.evaluate(2.0);
    auto r_mid = traj.evaluate(1.0);

    std::cout << "\n  Boundary check:\n";
    ex::print_value("  Start position", r_start.position, 4);
    ex::print_value("  Start velocity", r_start.velocity, 4);
    ex::print_value("  End position", r_end.position, 4);
    ex::print_value("  End velocity", r_end.velocity, 4);
    ex::print_value("  Mid position", r_mid.position, 4);

    std::cout << "\n";
    ex::print_full_trajectory_table(
        [&](double t) -> std::tuple<double, double, double, double> {
            auto r = traj.evaluate(t);
            return {r.position, r.velocity, r.acceleration, r.jerk};
        },
        interval.start, interval.end, 12);
}

static void example_order5_two_point() {
    ex::print_header("Example 2: Order 5 Two-Point Trajectory");

    std::cout << "  Boundary: pos 0->1, vel 0->0, acc 0->0, t=[0, 2]\n\n";

    BoundaryCondition bc_start{0.0, 0.0, 0.0, 0.0};
    BoundaryCondition bc_end{1.0, 0.0, 0.0, 0.0};
    TimeInterval interval{0.0, 2.0};

    PolynomialTrajectory traj(bc_start, bc_end, interval, PolynomialTrajectory::ORDER_5);

    ex::print_value("Order", static_cast<double>(traj.order()), 0);
    ex::print_value("Duration", traj.duration(), 2);

    auto r_start = traj.evaluate(0.0);
    auto r_end = traj.evaluate(2.0);

    std::cout << "\n  Boundary check:\n";
    ex::print_value("  Start position", r_start.position, 4);
    ex::print_value("  Start velocity", r_start.velocity, 4);
    ex::print_value("  Start acceleration", r_start.acceleration, 4);
    ex::print_value("  End position", r_end.position, 4);
    ex::print_value("  End velocity", r_end.velocity, 4);
    ex::print_value("  End acceleration", r_end.acceleration, 4);

    std::cout << "\n";
    ex::print_full_trajectory_table(
        [&](double t) -> std::tuple<double, double, double, double> {
            auto r = traj.evaluate(t);
            return {r.position, r.velocity, r.acceleration, r.jerk};
        },
        interval.start, interval.end, 12);
}

static void example_order7_comparison() {
    ex::print_header("Example 3: Order 7 Two-Point (Comparison of Orders 3, 5, 7)");

    std::cout << "  Boundary: pos 0->1, vel 0->0, acc 0->0, jerk 0->0, t=[0, 2]\n\n";

    BoundaryCondition bc_start{0.0, 0.0, 0.0, 0.0};
    BoundaryCondition bc_end{1.0, 0.0, 0.0, 0.0};
    TimeInterval interval{0.0, 2.0};

    PolynomialTrajectory traj3(bc_start, bc_end, interval, PolynomialTrajectory::ORDER_3);
    PolynomialTrajectory traj5(bc_start, bc_end, interval, PolynomialTrajectory::ORDER_5);
    PolynomialTrajectory traj7(bc_start, bc_end, interval, PolynomialTrajectory::ORDER_7);

    // Compare at midpoint t=1.0
    auto r3 = traj3.evaluate(1.0);
    auto r5 = traj5.evaluate(1.0);
    auto r7 = traj7.evaluate(1.0);

    std::cout << "  Values at midpoint (t=1.0):\n";
    std::cout << "                     Order 3     Order 5     Order 7\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "    Position:     " << std::setw(11) << r3.position
              << " " << std::setw(11) << r5.position
              << " " << std::setw(11) << r7.position << "\n";
    std::cout << "    Velocity:     " << std::setw(11) << r3.velocity
              << " " << std::setw(11) << r5.velocity
              << " " << std::setw(11) << r7.velocity << "\n";
    std::cout << "    Acceleration: " << std::setw(11) << r3.acceleration
              << " " << std::setw(11) << r5.acceleration
              << " " << std::setw(11) << r7.acceleration << "\n";
    std::cout << "    Jerk:         " << std::setw(11) << r3.jerk
              << " " << std::setw(11) << r5.jerk
              << " " << std::setw(11) << r7.jerk << "\n";

    // Print coefficients
    std::cout << "\n";
    ex::print_vector("Order 3 coefficients", traj3.coefficients());
    ex::print_vector("Order 5 coefficients", traj5.coefficients());
    ex::print_vector("Order 7 coefficients", traj7.coefficients());

    // Full table for order 7
    std::cout << "\n  Order 7 trajectory table:\n\n";
    ex::print_full_trajectory_table(
        [&](double t) -> std::tuple<double, double, double, double> {
            auto r = traj7.evaluate(t);
            return {r.position, r.velocity, r.acceleration, r.jerk};
        },
        interval.start, interval.end, 15);
}

static void example_multipoint_heuristic() {
    ex::print_header("Example 4: Multi-Point with Heuristic Velocities");

    const std::vector<double> points = {0.0, 0.5, 0.8, 1.0, 0.7, 0.3, 0.0};
    const std::vector<double> times = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    std::cout << "  Points: [";
    for (size_t i = 0; i < points.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << points[i];
    }
    std::cout << "]\n";
    std::cout << "  Times:  [";
    for (size_t i = 0; i < times.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << times[i];
    }
    std::cout << "]\n\n";

    // Compute heuristic velocities
    auto heur_vels = PolynomialTrajectory::heuristic_velocities(points, times);
    std::cout << "  Heuristic velocities: [";
    for (size_t i = 0; i < heur_vels.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(4) << heur_vels[i];
    }
    std::cout << "]\n\n";

    // Create multipoint trajectories at each order
    const int orders[] = {PolynomialTrajectory::ORDER_3,
                          PolynomialTrajectory::ORDER_5,
                          PolynomialTrajectory::ORDER_7};
    const char* order_names[] = {"Order 3", "Order 5", "Order 7"};

    for (int idx = 0; idx < 3; ++idx) {
        auto segments = PolynomialTrajectory::multipoint_trajectory(
            points, times, orders[idx]);

        std::cout << "  --- " << order_names[idx] << " multi-point trajectory ---\n";
        ex::print_value("  Number of segments",
                        static_cast<double>(segments.size()), 0);

        // Verify waypoint passage
        std::cout << "  Waypoint verification:\n";
        for (size_t i = 0; i < times.size(); ++i) {
            auto r = PolynomialTrajectory::evaluate_multipoint(segments, times[i]);
            std::cout << "    t=" << std::fixed << std::setprecision(1) << times[i]
                      << ": pos=" << std::setprecision(4) << r.position
                      << " (expected " << points[i] << ")\n";
        }

        std::cout << "\n";
        ex::print_full_trajectory_table(
            [&](double t) -> std::tuple<double, double, double, double> {
                auto r = PolynomialTrajectory::evaluate_multipoint(segments, t);
                return {r.position, r.velocity, r.acceleration, r.jerk};
            },
            times.front(), times.back(), 18);
    }
}

int main() {
    std::cout << "Polynomial Trajectory -- C++ Usage Examples\n";
    ex::print_separator('=');

    example_order3_two_point();
    example_order5_two_point();
    example_order7_comparison();
    example_multipoint_heuristic();

    std::cout << "\nAll polynomial trajectory examples completed.\n";
    return 0;
}
