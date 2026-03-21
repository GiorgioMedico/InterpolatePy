/// Cubic spline with acceleration constraints (method 1) — C++ port of
/// examples/c_s_with_acc1_ex.py
///
/// Demonstrates all 5 sub-examples: simple/textbook, robot joint, camera pan,
/// drone height with time scaling, and boundary condition comparison.

#include <interpolatecpp/spline/cubic_spline.hpp>
#include <interpolatecpp/spline/cubic_spline_with_acc1.hpp>

#include "example_utils.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace ex = interpolatecpp::examples;
using namespace interpolatecpp::spline;

// ---------------------------------------------------------------------------
// Example 0: Simple textbook example
// ---------------------------------------------------------------------------
void simple_example() {
    ex::print_header("Example 0: Textbook Example");

    std::vector<double> t = {0.0, 5.0, 7.0, 8.0, 10.0, 15.0, 18.0};
    std::vector<double> q = {3.0, -2.0, -5.0, 0.0, 6.0, 12.0, 8.0};

    CubicSplineWithAcceleration1 spline(t, q, 2.0, -3.0, 0.0, 0.0);

    ex::print_trajectory_table(
        [&](double tv) { return spline.evaluate(tv); },
        [&](double tv) { return spline.evaluate_velocity(tv); },
        [&](double tv) { return spline.evaluate_acceleration(tv); },
        t.front(), t.back(), 15);
}

// ---------------------------------------------------------------------------
// Example 1: Robot joint motion planning
// ---------------------------------------------------------------------------
void robot_joint_example() {
    ex::print_header("Example 1: Robot Joint Motion Planning");

    // Joint angle waypoints (radians) and times
    std::vector<double> t = {0.0, 2.0, 4.0, 7.0, 10.0};
    std::vector<double> q = {0.0, 1.57, 0.5, -0.5, 0.0};

    // Zero initial/final velocity and acceleration for smooth start/stop
    CubicSplineWithAcceleration1 spline(t, q, 0.0, 0.0, 0.0, 0.0);

    const double t_sample = 3.0;
    ex::print_value("Position at t=3.0 (rad)", spline.evaluate(t_sample));
    ex::print_value("Velocity at t=3.0 (rad/s)", spline.evaluate_velocity(t_sample));
    ex::print_value("Acceleration at t=3.0 (rad/s^2)", spline.evaluate_acceleration(t_sample));

    std::cout << "\n";
    ex::print_trajectory_table(
        [&](double tv) { return spline.evaluate(tv); },
        [&](double tv) { return spline.evaluate_velocity(tv); },
        [&](double tv) { return spline.evaluate_acceleration(tv); },
        t.front(), t.back(), 15);
}

// ---------------------------------------------------------------------------
// Example 2: Camera pan trajectory
// ---------------------------------------------------------------------------
void camera_pan_example() {
    ex::print_header("Example 2: Camera Pan Trajectory");

    // Camera angle waypoints (degrees) with times
    std::vector<double> t = {0.0, 3.0, 7.0, 12.0, 15.0};
    std::vector<double> q = {0.0, 45.0, 90.0, 120.0, 180.0};

    // Non-zero initial/final velocities for natural camera motion
    CubicSplineWithAcceleration1 spline(t, q, 10.0, 5.0, 0.0, 0.0);

    // Print sampled trajectory table
    const int n_samples = 15;
    const int w = 16;
    const int p = 2;

    std::cout << std::right
              << std::setw(w) << "Time (s)"
              << std::setw(w) << "Position (deg)"
              << std::setw(w) << "Velocity (deg/s)"
              << "\n";
    ex::print_separator('-', 3 * w);

    for (int i = 0; i <= n_samples; ++i) {
        double tv = t.front() + (t.back() - t.front()) * static_cast<double>(i) / n_samples;
        std::cout << std::fixed << std::setprecision(p)
                  << std::setw(w) << tv
                  << std::setw(w) << spline.evaluate(tv)
                  << std::setw(w) << spline.evaluate_velocity(tv)
                  << "\n";
    }
    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// Example 3: Drone height trajectory with time scaling
// ---------------------------------------------------------------------------
void drone_height_example() {
    ex::print_header("Example 3: Drone Height Trajectory with Time Scaling");

    std::vector<double> t = {0.0, 5.0, 10.0, 20.0, 25.0};
    std::vector<double> q = {0.0, 10.0, 15.0, 5.0, 2.0};

    // Limited initial/final accelerations
    CubicSplineWithAcceleration1 spline(t, q, 0.0, 0.0, 0.5, -0.5);

    // Time-scaled version (1.5x slower)
    const double time_scale = 1.5;
    std::vector<double> t_scaled(t.size());
    for (size_t i = 0; i < t.size(); ++i) {
        t_scaled[i] = t[i] * time_scale;
    }
    CubicSplineWithAcceleration1 spline_scaled(t_scaled, q, 0.0, 0.0, 0.5, -0.5);

    // Print original trajectory
    std::cout << "--- Original trajectory ---\n";
    ex::print_trajectory_table(
        [&](double tv) { return spline.evaluate(tv); },
        [&](double tv) { return spline.evaluate_velocity(tv); },
        [&](double tv) { return spline.evaluate_acceleration(tv); },
        t.front(), t.back(), 15);

    // Print time-scaled trajectory
    std::cout << "--- Time-scaled trajectory (1.5x) ---\n";
    ex::print_trajectory_table(
        [&](double tv) { return spline_scaled.evaluate(tv); },
        [&](double tv) { return spline_scaled.evaluate_velocity(tv); },
        [&](double tv) { return spline_scaled.evaluate_acceleration(tv); },
        t_scaled.front(), t_scaled.back(), 15);

    // Summary
    ex::print_value("Original duration (s)", t.back(), 1);
    ex::print_value("Scaled duration (s)", t_scaled.back(), 1);

    // Compute max absolute acceleration for both
    const int n_eval = 100;
    double max_acc_orig = 0.0;
    double max_acc_scaled = 0.0;
    for (int i = 0; i <= n_eval; ++i) {
        double frac = static_cast<double>(i) / n_eval;
        double tv_orig = t.front() + (t.back() - t.front()) * frac;
        double tv_sc = t_scaled.front() + (t_scaled.back() - t_scaled.front()) * frac;
        max_acc_orig = std::max(max_acc_orig, std::abs(spline.evaluate_acceleration(tv_orig)));
        max_acc_scaled = std::max(max_acc_scaled, std::abs(spline_scaled.evaluate_acceleration(tv_sc)));
    }
    ex::print_value("Max acceleration - original (m/s^2)", max_acc_orig, 2);
    ex::print_value("Max acceleration - scaled (m/s^2)", max_acc_scaled, 2);
}

// ---------------------------------------------------------------------------
// Example 4: Multi-dimensional trajectory (3D path)
// ---------------------------------------------------------------------------
void multi_dimensional_example() {
    ex::print_header("Example 4: Multi-dimensional Trajectory (3D Path)");

    std::vector<double> t = {0.0, 2.0, 5.0, 8.0, 10.0, 15.0};
    std::vector<double> x_pts = {0.0, 2.0, 5.0, 8.0, 6.0, 10.0};
    std::vector<double> y_pts = {0.0, 3.0, 8.0, 4.0, 2.0, 5.0};
    std::vector<double> z_pts = {0.0, 1.0, 2.0, 4.0, 3.0, 1.0};

    CubicSplineWithAcceleration1 sx(t, x_pts, 0.0, 0.0, 0.0, 0.0);
    CubicSplineWithAcceleration1 sy(t, y_pts, 0.0, 0.0, 0.0, 0.0);
    CubicSplineWithAcceleration1 sz(t, z_pts, 0.0, 0.0, 0.0, 0.0);

    // Print 3D trajectory table
    const int n_samples = 15;
    const int w = 12;
    const int p = 4;

    std::cout << std::right
              << std::setw(w) << "Time"
              << std::setw(w) << "X"
              << std::setw(w) << "Y"
              << std::setw(w) << "Z"
              << std::setw(w) << "Speed"
              << "\n";
    ex::print_separator('-', 5 * w);

    double total_path = 0.0;
    double max_speed = 0.0;
    double sum_speed = 0.0;
    double prev_x = sx.evaluate(t.front());
    double prev_y = sy.evaluate(t.front());
    double prev_z = sz.evaluate(t.front());

    for (int i = 0; i <= n_samples; ++i) {
        double tv = t.front() + (t.back() - t.front()) * static_cast<double>(i) / n_samples;
        double xv = sx.evaluate(tv);
        double yv = sy.evaluate(tv);
        double zv = sz.evaluate(tv);
        double vx = sx.evaluate_velocity(tv);
        double vy = sy.evaluate_velocity(tv);
        double vz = sz.evaluate_velocity(tv);
        double speed = std::sqrt(vx * vx + vy * vy + vz * vz);

        std::cout << std::fixed << std::setprecision(p)
                  << std::setw(w) << tv
                  << std::setw(w) << xv
                  << std::setw(w) << yv
                  << std::setw(w) << zv
                  << std::setw(w) << speed
                  << "\n";

        if (i > 0) {
            double dx = xv - prev_x;
            double dy = yv - prev_y;
            double dz = zv - prev_z;
            total_path += std::sqrt(dx * dx + dy * dy + dz * dz);
        }
        max_speed = std::max(max_speed, speed);
        sum_speed += speed;
        prev_x = xv;
        prev_y = yv;
        prev_z = zv;
    }

    std::cout << "\n";
    ex::print_value("Total path length (approx)", total_path, 2);
    ex::print_value("Maximum speed", max_speed, 2);
    ex::print_value("Average speed", sum_speed / (n_samples + 1), 2);
}

// ---------------------------------------------------------------------------
// Example 5: Comparing different boundary conditions
// ---------------------------------------------------------------------------
void compare_boundary_conditions() {
    ex::print_header("Example 5: Comparing Different Boundary Conditions");

    std::vector<double> t = {0.0, 4.0, 8.0, 12.0, 16.0};
    std::vector<double> q = {0.0, 10.0, 5.0, 15.0, 10.0};

    // Also create a basic CubicSpline for reference
    CubicSpline basic_spline(t, q, 0.0, 0.0);
    std::cout << "Reference: CubicSpline (no accel constraints)\n";
    ex::print_trajectory_table(
        [&](double tv) { return basic_spline.evaluate(tv); },
        [&](double tv) { return basic_spline.evaluate_velocity(tv); },
        [&](double tv) { return basic_spline.evaluate_acceleration(tv); },
        t.front(), t.back(), 15);

    // Spline 1: v0=vn=0, a0=an=0
    CubicSplineWithAcceleration1 s1(t, q, 0.0, 0.0, 0.0, 0.0);
    std::cout << "Spline 1: v0=vn=0, a0=an=0\n";
    ex::print_trajectory_table(
        [&](double tv) { return s1.evaluate(tv); },
        [&](double tv) { return s1.evaluate_velocity(tv); },
        [&](double tv) { return s1.evaluate_acceleration(tv); },
        t.front(), t.back(), 15);

    // Spline 2: v0=2, vn=-2, a0=an=0
    CubicSplineWithAcceleration1 s2(t, q, 2.0, -2.0, 0.0, 0.0);
    std::cout << "Spline 2: v0=2, vn=-2, a0=an=0\n";
    ex::print_trajectory_table(
        [&](double tv) { return s2.evaluate(tv); },
        [&](double tv) { return s2.evaluate_velocity(tv); },
        [&](double tv) { return s2.evaluate_acceleration(tv); },
        t.front(), t.back(), 15);

    // Spline 3: v0=vn=0, a0=1, an=-1
    CubicSplineWithAcceleration1 s3(t, q, 0.0, 0.0, 1.0, -1.0);
    std::cout << "Spline 3: v0=vn=0, a0=1, an=-1\n";
    ex::print_trajectory_table(
        [&](double tv) { return s3.evaluate(tv); },
        [&](double tv) { return s3.evaluate_velocity(tv); },
        [&](double tv) { return s3.evaluate_acceleration(tv); },
        t.front(), t.back(), 15);

    // Spline 4: v0=2, vn=-2, a0=1, an=-1
    CubicSplineWithAcceleration1 s4(t, q, 2.0, -2.0, 1.0, -1.0);
    std::cout << "Spline 4: v0=2, vn=-2, a0=1, an=-1\n";
    ex::print_trajectory_table(
        [&](double tv) { return s4.evaluate(tv); },
        [&](double tv) { return s4.evaluate_velocity(tv); },
        [&](double tv) { return s4.evaluate_acceleration(tv); },
        t.front(), t.back(), 15);

    // Smoothness comparison via approximate jerk integral
    ex::print_separator('=');
    std::cout << "Smoothness comparison (sum of squared jerk, smaller is better):\n\n";

    auto compute_jerk_metric = [&](auto& spline) {
        const int n = 1000;
        const double dt = (t.back() - t.front()) / n;
        double metric = 0.0;
        double prev_acc = spline.evaluate_acceleration(t.front());
        for (int i = 1; i <= n; ++i) {
            double tv = t.front() + dt * i;
            double acc = spline.evaluate_acceleration(tv);
            double jerk = (acc - prev_acc) / dt;
            metric += jerk * jerk;
            prev_acc = acc;
        }
        return metric;
    };

    ex::print_value("Spline 1 (v0=vn=0, a0=an=0)", compute_jerk_metric(s1), 2);
    ex::print_value("Spline 2 (v0=2, vn=-2, a0=an=0)", compute_jerk_metric(s2), 2);
    ex::print_value("Spline 3 (v0=vn=0, a0=1, an=-1)", compute_jerk_metric(s3), 2);
    ex::print_value("Spline 4 (v0=2, vn=-2, a0=1, an=-1)", compute_jerk_metric(s4), 2);
}

// ---------------------------------------------------------------------------
int main() {
    simple_example();
    robot_joint_example();
    camera_pan_example();
    drone_height_example();
    multi_dimensional_example();
    compare_boundary_conditions();

    return 0;
}
