/// B-spline interpolator example -- C++ port of examples/b_spline_interpolate_ex.py
///
/// Demonstrates BSplineInterpolator with multiple configurations:
/// cubic with velocity constraints, degree 4 (jerk-continuous),
/// degree 5 with acceleration constraints, cyclic, and 3D interpolation.

#include <interpolatecpp/bspline/bspline_interpolator.hpp>

#include "example_utils.hpp"

#include <iostream>
#include <stdexcept>

namespace ex = interpolatecpp::examples;
using namespace interpolatecpp::bspline;

/// Helper: print a scalar trajectory table for BSplineInterpolator.
static void print_scalar_trajectory(const BSplineInterpolator& interp,
                                    double t_start, double t_end,
                                    int num_samples = 15) {
    ex::print_trajectory_table(
        [&](double t) { return interp.evaluate(t)(0); },
        [&](double t) { return interp.evaluate_derivative(t, 1)(0); },
        [&](double t) { return interp.evaluate_derivative(t, 2)(0); },
        t_start, t_end, num_samples);
}

/// Example 1: Cubic B-spline with velocity constraints (Example 4.16).
static void example_cubic_bspline() {
    ex::print_header("Example 1 -- Cubic B-spline with Velocity Constraints");

    Eigen::VectorXd times(7);
    times << 0, 5, 7, 8, 10, 15, 18;

    Eigen::MatrixXd points(7, 1);
    points << 3, -2, -5, 0, 6, 12, 8;

    Eigen::VectorXd v0(1);
    v0 << 2.0;
    Eigen::VectorXd vn(1);
    vn << -3.0;

    const BSplineInterpolator interp(
        3,          // degree
        points,
        times,
        v0,         // initial_velocity
        vn          // final_velocity
    );

    std::cout << "Properties:\n";
    ex::print_value("Degree", static_cast<double>(interp.degree()), 0);
    ex::print_value("Num control points", static_cast<double>(interp.n_control_points()), 0);
    ex::print_value("u_min", interp.u_min());
    ex::print_value("u_max", interp.u_max());

    std::cout << "\n";
    ex::print_vector("Times", interp.times());
    ex::print_vector("Knots", interp.knots());

    std::cout << "\nTrajectory evaluation:\n\n";
    print_scalar_trajectory(interp, times(0), times(times.size() - 1));

    // Verify boundary velocities
    ex::print_separator('=');
    std::cout << "Boundary verification:\n";
    ex::print_value("Velocity at t_start", interp.evaluate_derivative(times(0), 1)(0));
    ex::print_value("Expected v0", 2.0);
    ex::print_value("Velocity at t_end", interp.evaluate_derivative(times(times.size() - 1), 1)(0));
    ex::print_value("Expected vn", -3.0);
}

/// Example 2: Degree 4 B-spline (jerk-continuous).
static void example_jerk_continuous() {
    ex::print_header("Example 2 -- Degree 4 B-spline (Jerk Continuous)");

    Eigen::VectorXd times(7);
    times << 0, 5, 7, 8, 10, 15, 18;

    Eigen::MatrixXd points(7, 1);
    points << 3, -2, -5, 0, 6, 12, 8;

    Eigen::VectorXd v0(1), vn(1), a0(1), an(1);
    v0 << 2.0;
    vn << -3.0;
    a0 << 0.0;
    an << 0.0;

    const BSplineInterpolator interp(
        4,          // degree
        points,
        times,
        v0,         // initial_velocity
        vn,         // final_velocity
        a0,         // initial_acceleration
        an          // final_acceleration
    );

    std::cout << "Properties:\n";
    ex::print_value("Degree", static_cast<double>(interp.degree()), 0);
    ex::print_value("Num control points", static_cast<double>(interp.n_control_points()), 0);

    std::cout << "\n";
    ex::print_vector("Knots", interp.knots());

    // Print trajectory with jerk
    std::cout << "\nTrajectory evaluation (with jerk):\n\n";
    const double t_start = times(0);
    const double t_end = times(times.size() - 1);
    const int num_samples = 15;

    ex::print_full_trajectory_table(
        [&](double t) -> std::tuple<double, double, double, double> {
            return {interp.evaluate(t)(0),
                    interp.evaluate_derivative(t, 1)(0),
                    interp.evaluate_derivative(t, 2)(0),
                    interp.evaluate_derivative(t, 3)(0)};
        },
        t_start, t_end, num_samples);

    // Verify boundary conditions
    ex::print_separator('=');
    std::cout << "Boundary verification:\n";
    ex::print_value("Velocity at t_start", interp.evaluate_derivative(t_start, 1)(0));
    ex::print_value("Acceleration at t_start", interp.evaluate_derivative(t_start, 2)(0));
    ex::print_value("Velocity at t_end", interp.evaluate_derivative(t_end, 1)(0));
    ex::print_value("Acceleration at t_end", interp.evaluate_derivative(t_end, 2)(0));
}

/// Example 3: Degree 5 B-spline with acceleration constraints.
static void example_degree5() {
    ex::print_header("Example 3 -- Degree 5 B-spline with Acceleration Constraints");

    Eigen::VectorXd times(7);
    times << 0, 1, 2, 3, 4, 5, 6;

    Eigen::MatrixXd points(7, 1);
    points << 0, 2, 1, 3, 2, 4, 3;

    Eigen::VectorXd v0(1), vn(1), a0(1), an(1);
    v0 << 1.0;
    vn << -1.0;
    a0 << 0.0;
    an << 0.0;

    const BSplineInterpolator interp(
        5,          // degree
        points,
        times,
        v0, vn,
        a0, an
    );

    std::cout << "Properties:\n";
    ex::print_value("Degree", static_cast<double>(interp.degree()), 0);
    ex::print_value("Num control points", static_cast<double>(interp.n_control_points()), 0);

    std::cout << "\n";
    ex::print_vector("Knots", interp.knots());

    std::cout << "\nTrajectory evaluation:\n\n";
    print_scalar_trajectory(interp, times(0), times(times.size() - 1));

    // Verify boundary conditions
    ex::print_separator('=');
    std::cout << "Boundary verification:\n";
    const double t_start = times(0);
    const double t_end = times(times.size() - 1);
    ex::print_value("Velocity at t_start", interp.evaluate_derivative(t_start, 1)(0));
    ex::print_value("Acceleration at t_start", interp.evaluate_derivative(t_start, 2)(0));
    ex::print_value("Velocity at t_end", interp.evaluate_derivative(t_end, 1)(0));
    ex::print_value("Acceleration at t_end", interp.evaluate_derivative(t_end, 2)(0));
}

/// Example 4: Cyclic B-spline (Example 4.17).
static void example_cyclic() {
    ex::print_header("Example 4 -- Cyclic B-spline (Degree 4)");

    Eigen::VectorXd times(7);
    times << 0, 5, 7, 8, 10, 15, 18;

    // Last point equals first point for cyclic
    Eigen::MatrixXd points(7, 1);
    points << 3, -2, -5, 0, 6, 12, 3;

    const BSplineInterpolator interp(
        4,              // degree
        points,
        times,
        std::nullopt,   // initial_velocity
        std::nullopt,   // final_velocity
        std::nullopt,   // initial_acceleration
        std::nullopt,   // final_acceleration
        true            // cyclic
    );

    std::cout << "Properties:\n";
    ex::print_value("Degree", static_cast<double>(interp.degree()), 0);
    ex::print_value("Num control points", static_cast<double>(interp.n_control_points()), 0);
    ex::print_value("Cyclic", 1.0, 0);

    std::cout << "\n";
    ex::print_vector("Knots", interp.knots());

    // Trajectory with jerk and snap
    std::cout << "\nTrajectory evaluation:\n\n";
    const double t_start = times(0);
    const double t_end = times(times.size() - 1);

    const int w = 14;
    const int p = 6;
    std::cout << std::right
              << std::setw(w) << "Time"
              << std::setw(w) << "Position"
              << std::setw(w) << "Velocity"
              << std::setw(w) << "Acceleration"
              << std::setw(w) << "Jerk"
              << std::setw(w) << "Snap" << "\n";
    ex::print_separator('-', 6 * w);

    const int num_samples = 15;
    for (int i = 0; i <= num_samples; ++i) {
        const double t = t_start + (t_end - t_start) * static_cast<double>(i) / num_samples;
        std::cout << std::fixed << std::setprecision(p)
                  << std::setw(w) << t
                  << std::setw(w) << interp.evaluate(t)(0)
                  << std::setw(w) << interp.evaluate_derivative(t, 1)(0)
                  << std::setw(w) << interp.evaluate_derivative(t, 2)(0)
                  << std::setw(w) << interp.evaluate_derivative(t, 3)(0)
                  << std::setw(w) << interp.evaluate_derivative(t, 4)(0)
                  << "\n";
    }
    std::cout << "\n";

    // Verify cyclic continuity: values at start and end should match
    ex::print_separator('=');
    std::cout << "Cyclic continuity verification:\n";
    ex::print_value("Position at t_start", interp.evaluate(t_start)(0));
    ex::print_value("Position at t_end", interp.evaluate(t_end)(0));
    ex::print_value("Velocity at t_start", interp.evaluate_derivative(t_start, 1)(0));
    ex::print_value("Velocity at t_end", interp.evaluate_derivative(t_end, 1)(0));
    ex::print_value("Acceleration at t_start", interp.evaluate_derivative(t_start, 2)(0));
    ex::print_value("Acceleration at t_end", interp.evaluate_derivative(t_end, 2)(0));
}

/// Example 5: 3D B-spline interpolation.
static void example_3d() {
    ex::print_header("Example 5 -- 3D B-spline Interpolation");

    Eigen::VectorXd times(5);
    times << 0, 1, 2, 3, 4;

    Eigen::MatrixXd points(5, 3);
    points << 0,  0, 0,
              1,  1, 2,
              2,  0, 3,
              3, -1, 2,
              4,  0, 0;

    const int degree = 3;
    const BSplineInterpolator interp(degree, points, times);

    std::cout << "Properties:\n";
    ex::print_value("Degree", static_cast<double>(interp.degree()), 0);
    ex::print_value("Num control points", static_cast<double>(interp.n_control_points()), 0);
    ex::print_value("Dimension", static_cast<double>(interp.dimension()), 0);
    std::cout << "Continuity: C^" << (degree - 1) << " (continuous acceleration)\n";

    std::cout << "\n";
    ex::print_vector("Knots", interp.knots());
    ex::print_matrix("Control points (3D)", interp.control_points());

    // Print original and interpolated points
    std::cout << "\nOriginal points to interpolate:\n";
    for (Eigen::Index i = 0; i < points.rows(); ++i) {
        std::cout << "  Point " << i << " (t=" << times(i) << "): ("
                  << std::fixed << std::setprecision(1)
                  << points(i, 0) << ", " << points(i, 1) << ", " << points(i, 2) << ")\n";
    }

    // Evaluate at intermediate times
    std::cout << "\nInterpolated points at intermediate times:\n";
    const std::vector<double> t_samples = {0.5, 1.5, 2.5, 3.5};
    for (const double t : t_samples) {
        const Eigen::VectorXd pt = interp.evaluate(t);
        std::cout << "  t = " << std::fixed << std::setprecision(1) << t << ": ("
                  << std::setprecision(3) << pt(0) << ", " << pt(1) << ", " << pt(2) << ")\n";
    }

    // Print full 3D trajectory
    std::cout << "\n3D trajectory table:\n\n";
    const int w = 14;
    std::cout << std::right
              << std::setw(w) << "Time"
              << std::setw(w) << "X" << std::setw(w) << "Y" << std::setw(w) << "Z" << "\n";
    ex::print_separator('-', 4 * w);

    const int num_samples = 12;
    for (int i = 0; i <= num_samples; ++i) {
        const double t = times(0) + (times(times.size() - 1) - times(0))
                         * static_cast<double>(i) / num_samples;
        const Eigen::VectorXd pt = interp.evaluate(t);
        std::cout << std::fixed << std::setprecision(4)
                  << std::setw(w) << t
                  << std::setw(w) << pt(0)
                  << std::setw(w) << pt(1)
                  << std::setw(w) << pt(2) << "\n";
    }

    // Try higher degrees with more points
    ex::print_separator('=');
    std::cout << "Degree sensitivity with 5 points:\n\n";

    // Degree 3 works (already demonstrated). Try degree 4 and 5.
    for (const int deg : {4, 5}) {
        std::cout << "  Trying degree " << deg << " with " << points.rows() << " points... ";
        try {
            const BSplineInterpolator test_interp(deg, points, times);
            std::cout << "Succeeded! (n_cp=" << test_interp.n_control_points() << ")\n";
        } catch (const std::exception& e) {
            std::cout << "Failed: " << e.what() << "\n";
        }
    }

    // Extended points for higher degrees
    ex::print_separator();
    std::cout << "Extended points (7 points) for higher degrees:\n\n";

    Eigen::VectorXd times_ext(7);
    times_ext << 0, 1, 2, 3, 4, 5, 6;

    Eigen::MatrixXd points_ext(7, 3);
    points_ext << 0,  0,  0,
                  1,  1,  2,
                  2,  0,  3,
                  3, -1,  2,
                  4,  0,  0,
                  5,  1, -1,
                  6,  0, -2;

    for (const int deg : {3, 4, 5}) {
        std::cout << "  Degree " << deg << " with 7 points: ";
        try {
            const BSplineInterpolator ext_interp(deg, points_ext, times_ext);
            std::cout << "Succeeded! (n_cp=" << ext_interp.n_control_points() << ")\n";
        } catch (const std::exception& e) {
            std::cout << "Failed: " << e.what() << "\n";
        }
    }
}

int main() {
    example_cubic_bspline();
    example_jerk_continuous();
    example_degree5();
    example_cyclic();
    example_3d();

    return 0;
}
