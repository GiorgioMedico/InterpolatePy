/// Cubic B-spline interpolation example -- C++ port of examples/b_spline_cubic_ex.py
///
/// Demonstrates CubicBSplineInterpolation through 3D points (Example 8.8)
/// with chord-length parameterization and auto-derivative computation.

#include <interpolatecpp/bspline/cubic_bspline_interpolation.hpp>
#include <interpolatecpp/bspline/bspline_parameters.hpp>

#include "example_utils.hpp"

#include <iostream>

namespace ex = interpolatecpp::examples;
using namespace interpolatecpp::bspline;

/// Example 8.8: 3D cubic B-spline interpolation.
static void example_8_8() {
    ex::print_header("Example 8.8 -- 3D Cubic B-spline Interpolation");

    // Define the interpolation points from the example
    Eigen::MatrixXd points(10, 3);
    points <<  83, -54, 119,
              -64,  10, 124,
               42,  79, 226,
              -98,  23, 222,
              -13, 125, 102,
              140,  81,  92,
               43,  32,  92,
              -65, -17, 134,
              -45, -89, 182,
               71,  90, 192;

    // Create cubic B-spline interpolation with chord-length parameterization
    const CubicBSplineInterpolation interpolation(
        points,
        std::nullopt,  // v0 (auto)
        std::nullopt,  // vn (auto)
        Parameterization::ChordLength,
        true           // auto_derivatives
    );

    // Print basic info
    std::cout << "Cubic B-spline Interpolation Information:\n";
    ex::print_value("Number of interpolation points",
                    static_cast<double>(points.rows()), 0);
    ex::print_value("Degree", static_cast<double>(interpolation.degree()), 0);
    ex::print_value("Dimension", static_cast<double>(interpolation.dimension()), 0);
    ex::print_value("u_min", interpolation.u_min());
    ex::print_value("u_max", interpolation.u_max());

    // Print parameter values (u-bars)
    std::cout << "\n";
    ex::print_vector("Parameter values (u_bars)", interpolation.u_bars());

    // Print the knot vector
    ex::print_vector("Knot vector", interpolation.knots());

    // Print start/end derivatives
    std::cout << "\n";
    ex::print_vector("Start derivative (v0)", interpolation.start_derivative());
    ex::print_vector("End derivative (vn)", interpolation.end_derivative());

    // Print control points
    std::cout << "\n";
    ex::print_matrix("Control points", interpolation.control_points());

    // Print interpolation points
    std::cout << "\n";
    ex::print_matrix("Interpolation points", interpolation.interpolation_points());

    // Verify interpolation: evaluate at each parameter value
    ex::print_separator('=');
    std::cout << "Interpolation verification (evaluate at each u_bar):\n\n";

    const int w = 12;
    std::cout << std::right
              << std::setw(8) << "u_bar"
              << std::setw(w) << "X_eval"
              << std::setw(w) << "Y_eval"
              << std::setw(w) << "Z_eval"
              << std::setw(w) << "X_orig"
              << std::setw(w) << "Y_orig"
              << std::setw(w) << "Z_orig" << "\n";
    ex::print_separator('-', 8 + 6 * w);

    for (Eigen::Index i = 0; i < points.rows(); ++i) {
        const double u = interpolation.u_bars()(i);
        const Eigen::VectorXd evaluated = interpolation.evaluate(u);

        std::cout << std::fixed << std::setprecision(4)
                  << std::setw(8) << u
                  << std::setw(w) << evaluated(0)
                  << std::setw(w) << evaluated(1)
                  << std::setw(w) << evaluated(2)
                  << std::setw(w) << points(i, 0)
                  << std::setw(w) << points(i, 1)
                  << std::setw(w) << points(i, 2) << "\n";
    }

    // Evaluate derivatives at a few points along the curve
    ex::print_separator('=');
    std::cout << "Derivative evaluation at sample parameter values:\n\n";

    const int n_samples = 5;
    const double u_start = interpolation.u_min();
    const double u_end = interpolation.u_max();

    for (int i = 0; i <= n_samples; ++i) {
        const double u = u_start + (u_end - u_start) * static_cast<double>(i) / n_samples;
        const Eigen::VectorXd pos = interpolation.evaluate(u);
        const Eigen::VectorXd vel = interpolation.evaluate_derivative(u, 1);
        const Eigen::VectorXd acc = interpolation.evaluate_derivative(u, 2);

        std::cout << "u = " << std::fixed << std::setprecision(4) << u << ":\n";
        ex::print_vector("  Position", pos);
        ex::print_vector("  Velocity", vel);
        ex::print_vector("  Acceleration", acc);
        std::cout << "\n";
    }
}

/// Demonstrate with explicit endpoint derivatives.
static void example_with_derivatives() {
    ex::print_header("Cubic B-spline with Explicit Derivatives");

    // Simple 2D points
    Eigen::MatrixXd points(5, 2);
    points << 0, 0,
              1, 2,
              3, 3,
              5, 1,
              6, 0;

    // Specify endpoint derivatives
    Eigen::VectorXd v0(2);
    v0 << 1.0, 3.0;

    Eigen::VectorXd vn(2);
    vn << 1.0, -2.0;

    const CubicBSplineInterpolation interpolation(
        points, v0, vn,
        Parameterization::ChordLength,
        false  // not auto_derivatives
    );

    std::cout << "Interpolation with explicit derivatives:\n";
    ex::print_value("Number of points", static_cast<double>(points.rows()), 0);
    ex::print_value("Degree", static_cast<double>(interpolation.degree()), 0);
    ex::print_vector("Start derivative (v0)", interpolation.start_derivative());
    ex::print_vector("End derivative (vn)", interpolation.end_derivative());

    std::cout << "\n";
    ex::print_vector("u_bars", interpolation.u_bars());
    ex::print_vector("Knots", interpolation.knots());

    std::cout << "\n";
    ex::print_matrix("Control points", interpolation.control_points());

    // Generate curve points and print a summary
    const auto [params, curve_pts] = interpolation.generate_curve_points(15);
    std::cout << "\nCurve samples:\n";

    const int w = 14;
    std::cout << std::right << std::setw(w) << "u"
              << std::setw(w) << "X" << std::setw(w) << "Y" << "\n";
    ex::print_separator('-', 3 * w);

    for (Eigen::Index i = 0; i < curve_pts.rows(); ++i) {
        std::cout << std::fixed << std::setprecision(4)
                  << std::setw(w) << params(i)
                  << std::setw(w) << curve_pts(i, 0)
                  << std::setw(w) << curve_pts(i, 1) << "\n";
    }
}

int main() {
    example_8_8();
    example_with_derivatives();

    return 0;
}
