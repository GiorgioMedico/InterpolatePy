/// Cubic spline example — C++ port of examples/cubic_spline_ex.py
///
/// Demonstrates basic CubicSpline construction with boundary velocities,
/// trajectory evaluation, and accessor inspection.

#include <interpolatecpp/spline/cubic_spline.hpp>

#include "example_utils.hpp"

#include <iostream>
#include <vector>

namespace ex = interpolatecpp::examples;
using namespace interpolatecpp::spline;

int main() {
    ex::print_header("Cubic Spline Example");

    // Define waypoints (same as the Python example)
    std::vector<double> t_points = {0.0, 5.0, 7.0, 8.0, 10.0, 15.0, 18.0};
    std::vector<double> q_points = {3.0, -2.0, -5.0, 0.0, 6.0, 12.0, 8.0};

    // Create cubic spline with initial and final velocities
    const double v0 = 2.0;
    const double vn = -3.0;
    CubicSpline spline(t_points, q_points, v0, vn);

    // Print trajectory table at 15 samples
    std::cout << "Trajectory evaluation (15 samples):\n\n";
    ex::print_trajectory_table(
        [&](double t) { return spline.evaluate(t); },
        [&](double t) { return spline.evaluate_velocity(t); },
        [&](double t) { return spline.evaluate_acceleration(t); },
        t_points.front(), t_points.back(), 15);

    // Print accessors
    ex::print_separator('=');
    std::cout << "Spline properties:\n\n";
    ex::print_value("Number of segments", static_cast<double>(spline.n_segments()), 0);
    ex::print_value("Initial velocity (v0)", v0);
    ex::print_value("Final velocity (vn)", vn);

    std::cout << "\n";
    ex::print_vector("Time points", spline.t_points());
    ex::print_vector("Position points", spline.q_points());
    ex::print_vector("Computed velocities", spline.velocities());

    std::cout << "\n";
    ex::print_matrix("Coefficients (a0, a1, a2, a3 per segment)", spline.coefficients());

    // Verify boundary velocities
    ex::print_separator('=');
    std::cout << "Boundary verification:\n\n";
    ex::print_value("Velocity at t_start", spline.evaluate_velocity(t_points.front()));
    ex::print_value("Expected v0", v0);
    ex::print_value("Velocity at t_end", spline.evaluate_velocity(t_points.back()));
    ex::print_value("Expected vn", vn);

    // Evaluate at a specific point
    std::cout << "\n";
    const double t_sample = 6.0;
    ex::print_value("Position at t=6.0", spline.evaluate(t_sample));
    ex::print_value("Velocity at t=6.0", spline.evaluate_velocity(t_sample));
    ex::print_value("Acceleration at t=6.0", spline.evaluate_acceleration(t_sample));

    return 0;
}
