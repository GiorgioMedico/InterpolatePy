/// Cubic spline with acceleration constraints (method 2, quintic segments) —
/// C++ port of examples/c_s_with_acc2_ex.py
///
/// Demonstrates CubicSplineWithAcceleration2 with SplineParameters,
/// trajectory evaluation, and boundary condition verification.

#include <interpolatecpp/spline/cubic_spline_with_acc2.hpp>
#include <interpolatecpp/spline/spline_parameters.hpp>

#include "example_utils.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace ex = interpolatecpp::examples;
using namespace interpolatecpp::spline;

int main() {
    ex::print_header("Cubic Spline with Acceleration (Method 2) Example");

    // Define waypoints
    std::vector<double> t_points = {0.0, 5.0, 7.0, 8.0, 10.0, 15.0, 18.0};
    std::vector<double> q_points = {3.0, -2.0, -5.0, 0.0, 6.0, 12.0, 8.0};

    // Create parameters object
    SplineParameters params;
    params.v0 = 2.0;      // Initial velocity
    params.vn = -3.0;     // Final velocity
    params.a0 = 0.0;      // Initial acceleration
    params.an = 0.0;      // Final acceleration
    params.debug = false;

    // Create cubic spline with quintic boundary segments
    CubicSplineWithAcceleration2 spline(t_points, q_points, params);

    // Print trajectory table
    std::cout << "Trajectory evaluation (15 samples):\n\n";
    ex::print_trajectory_table(
        [&](double t) { return spline.evaluate(t); },
        [&](double t) { return spline.evaluate_velocity(t); },
        [&](double t) { return spline.evaluate_acceleration(t); },
        t_points.front(), t_points.back(), 15);

    // Verify boundary conditions
    ex::print_separator('=');
    std::cout << "Verifying boundary conditions:\n\n";

    const double initial_vel = spline.evaluate_velocity(t_points.front());
    ex::print_value("Initial velocity", initial_vel);
    ex::print_value("Expected v0", params.v0);

    const double final_vel = spline.evaluate_velocity(t_points.back());
    ex::print_value("Final velocity", final_vel);
    ex::print_value("Expected vn", params.vn);

    const double initial_acc = spline.evaluate_acceleration(t_points.front());
    const double expected_a0 = params.a0.value_or(0.0);
    ex::print_value("Initial acceleration", initial_acc);
    ex::print_value("Expected a0", expected_a0);

    const double final_acc = spline.evaluate_acceleration(t_points.back());
    const double expected_an = params.an.value_or(0.0);
    ex::print_value("Final acceleration", final_acc);
    ex::print_value("Expected an", expected_an);

    // Report quintic segment presence
    ex::print_separator('=');
    std::cout << "Quintic segment info:\n\n";
    std::cout << "  Has quintic first segment: " << (spline.has_quintic_first() ? "yes" : "no") << "\n";
    std::cout << "  Has quintic last segment:  " << (spline.has_quintic_last() ? "yes" : "no") << "\n";

    // Verify errors are small
    std::cout << "\n";
    const double vel_err_start = std::abs(initial_vel - params.v0);
    const double vel_err_end = std::abs(final_vel - params.vn);
    const double acc_err_start = std::abs(initial_acc - expected_a0);
    const double acc_err_end = std::abs(final_acc - expected_an);

    ex::print_value("Velocity error at start", vel_err_start);
    ex::print_value("Velocity error at end", vel_err_end);
    ex::print_value("Acceleration error at start", acc_err_start);
    ex::print_value("Acceleration error at end", acc_err_end);

    // Demonstrate with non-zero acceleration constraints
    ex::print_header("Variant: Non-zero acceleration constraints");

    SplineParameters params2;
    params2.v0 = 1.0;
    params2.vn = -1.0;
    params2.a0 = 2.0;
    params2.an = -2.0;

    CubicSplineWithAcceleration2 spline2(t_points, q_points, params2);

    ex::print_trajectory_table(
        [&](double t) { return spline2.evaluate(t); },
        [&](double t) { return spline2.evaluate_velocity(t); },
        [&](double t) { return spline2.evaluate_acceleration(t); },
        t_points.front(), t_points.back(), 15);

    std::cout << "Boundary verification (non-zero accel):\n\n";
    ex::print_value("Initial velocity", spline2.evaluate_velocity(t_points.front()));
    ex::print_value("Expected v0", params2.v0);
    ex::print_value("Final velocity", spline2.evaluate_velocity(t_points.back()));
    ex::print_value("Expected vn", params2.vn);
    ex::print_value("Initial acceleration", spline2.evaluate_acceleration(t_points.front()));
    ex::print_value("Expected a0", params2.a0.value_or(0.0));
    ex::print_value("Final acceleration", spline2.evaluate_acceleration(t_points.back()));
    ex::print_value("Expected an", params2.an.value_or(0.0));

    return 0;
}
