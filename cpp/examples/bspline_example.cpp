/// B-spline example -- C++ port of examples/b_spline_ex.py
///
/// Demonstrates BSpline construction, basis function evaluation,
/// derivative computation, curve point generation, and 3D curves.

#include <interpolatecpp/bspline/bspline.hpp>

#include "example_utils.hpp"

#include <iostream>
#include <span>
#include <vector>

namespace ex = interpolatecpp::examples;
using namespace interpolatecpp::bspline;

/// Create the 2D B-spline from the document example.
static BSpline create_example_bspline() {
    const int degree = 3;

    Eigen::MatrixXd control_points(7, 2);
    control_points << 1, 2,
                      2, 3,
                      3, -3,
                      4, 4,
                      5, 5,
                      6, -5,
                      7, -6;

    const std::vector<double> knots = {0, 0, 0, 0, 1, 2, 4, 7, 7, 7, 7};

    return BSpline(degree, std::span<const double>(knots), control_points);
}

/// Demonstrate basic B-spline evaluation and basis functions.
static void demonstrate_basic_bspline() {
    ex::print_header("Basic B-spline Evaluation");

    const auto bspline = create_example_bspline();

    std::cout << "B-spline properties:\n";
    ex::print_value("Degree", static_cast<double>(bspline.degree()), 0);
    ex::print_value("Number of control points", static_cast<double>(bspline.n_control_points()), 0);
    ex::print_value("Dimension", static_cast<double>(bspline.dimension()), 0);
    ex::print_value("u_min", bspline.u_min());
    ex::print_value("u_max", bspline.u_max());

    // Evaluate at u = 1.5
    const double u_value = 1.5;
    const Eigen::VectorXd point = bspline.evaluate(u_value);

    std::cout << "\nPoint at u = " << u_value << ":\n";
    ex::print_vector("Position", point);

    // Basis functions at u = 1.5
    const int span = bspline.find_knot_span(u_value);
    const Eigen::VectorXd basis = bspline.basis_functions(u_value, span);

    std::cout << "\nKnot span index at u = " << u_value << ": " << span << "\n";
    std::cout << "Non-zero basis functions:\n";
    for (Eigen::Index i = 0; i < basis.size(); ++i) {
        std::cout << "  B^" << bspline.degree() << "_"
                  << (span - bspline.degree() + static_cast<int>(i))
                  << " = " << std::fixed << std::setprecision(4) << basis(i) << "\n";
    }

    // Print knot vector
    std::cout << "\n";
    ex::print_vector("Knot vector", bspline.knots());
    ex::print_matrix("Control points", bspline.control_points());
}

/// Example B.6: Basis function derivatives at u = 4.5.
static void example_b6() {
    ex::print_header("Example B.6 -- Basis Function Derivatives");

    const int degree = 3;
    const std::vector<double> knots = {0, 0, 0, 0, 1, 2, 4, 7, 7, 7, 7};

    // Dummy control points (basis functions don't depend on them)
    Eigen::MatrixXd control_points = Eigen::MatrixXd::Zero(7, 2);

    const BSpline bspline(degree, std::span<const double>(knots), control_points);

    const double u_value = 4.5;
    const int span = bspline.find_knot_span(u_value);
    std::cout << "For u = " << u_value << ", the knot span index is: " << span << "\n";

    // Calculate derivatives up to order 3
    const Eigen::MatrixXd derivatives = bspline.basis_function_derivatives(u_value, span, 3);

    std::cout << "\nBasis function values and derivatives at u = 4.5:\n";
    ex::print_separator('-', 80);

    for (int k = 0; k < 4; ++k) {
        std::cout << "Ders[" << k << "]: ";
        for (int j = 0; j < 4; ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(4) << derivatives(k, j);
        }
        std::cout << "\n";
    }

    std::cout << "\nWhich correspond to:\n";
    ex::print_separator('-', 80);

    const std::vector<std::string> labels = {"B_i^3    ", "B_i^3(1) ", "B_i^3(2) ", "B_i^3(3) "};
    for (int k = 0; k < 4; ++k) {
        std::cout << labels[static_cast<size_t>(k)] << ": ";
        for (int j = 0; j < 4; ++j) {
            const int idx = span - degree + j;
            if (j > 0) std::cout << ", ";
            std::cout << "B_" << idx << " = " << std::fixed << std::setprecision(4)
                      << derivatives(k, j);
        }
        std::cout << "\n";
    }

    std::cout << "\nAll the other terms B_j^3(k) are null.\n";
}

/// Demonstrate curve point generation.
static void demonstrate_curve_generation() {
    ex::print_header("Curve Point Generation");

    const auto bspline = create_example_bspline();

    // Generate curve points
    const int num_points = 20;
    const auto [params, curve_points] = bspline.generate_curve_points(num_points);

    std::cout << "Generated " << curve_points.rows() << " curve points:\n\n";

    const int w = 14;
    std::cout << std::right << std::setw(w) << "u"
              << std::setw(w) << "X" << std::setw(w) << "Y" << "\n";
    ex::print_separator('-', 3 * w);

    for (Eigen::Index i = 0; i < curve_points.rows(); ++i) {
        std::cout << std::fixed << std::setprecision(4)
                  << std::setw(w) << params(i)
                  << std::setw(w) << curve_points(i, 0)
                  << std::setw(w) << curve_points(i, 1) << "\n";
    }
}

/// Demonstrate 3D B-spline with uniform knots.
static void demonstrate_3d_bspline() {
    ex::print_header("3D B-spline Curve");

    const int degree = 3;

    Eigen::MatrixXd control_points(6, 3);
    control_points << 0, 0, 0,
                      1, 1, 2,
                      2, -1, 1,
                      3, 0, 3,
                      4, 2, 0,
                      5, 0, 1;

    const Eigen::VectorXd knots = BSpline::create_uniform_knots(degree, 6);

    const BSpline bspline(degree, std::span<const double>(knots.data(), static_cast<size_t>(knots.size())),
                          control_points);

    std::cout << "B-spline properties:\n";
    ex::print_value("Degree", static_cast<double>(bspline.degree()), 0);
    ex::print_value("Number of control points", static_cast<double>(bspline.n_control_points()), 0);
    ex::print_value("Dimension", static_cast<double>(bspline.dimension()), 0);
    ex::print_value("u_min", bspline.u_min());
    ex::print_value("u_max", bspline.u_max());

    std::cout << "\n";
    ex::print_vector("Uniform knot vector", bspline.knots());
    ex::print_matrix("Control points (3D)", bspline.control_points());

    // Evaluate at midpoint
    const double u_mid = (bspline.u_min() + bspline.u_max()) / 2.0;
    const Eigen::VectorXd mid_point = bspline.evaluate(u_mid);
    std::cout << "\nPoint at u_mid = " << std::fixed << std::setprecision(4) << u_mid << ":\n";
    ex::print_vector("Position", mid_point);

    // First derivative at midpoint
    const Eigen::VectorXd deriv1 = bspline.evaluate_derivative(u_mid, 1);
    ex::print_vector("1st derivative", deriv1);

    // Second derivative at midpoint
    const Eigen::VectorXd deriv2 = bspline.evaluate_derivative(u_mid, 2);
    ex::print_vector("2nd derivative", deriv2);

    // Generate 3D curve points
    const auto [params, curve_pts] = bspline.generate_curve_points(10);
    std::cout << "\nSample 3D curve points:\n";
    const int w = 14;
    std::cout << std::right << std::setw(w) << "u"
              << std::setw(w) << "X" << std::setw(w) << "Y" << std::setw(w) << "Z" << "\n";
    ex::print_separator('-', 4 * w);
    for (Eigen::Index i = 0; i < curve_pts.rows(); ++i) {
        std::cout << std::fixed << std::setprecision(4)
                  << std::setw(w) << params(i)
                  << std::setw(w) << curve_pts(i, 0)
                  << std::setw(w) << curve_pts(i, 1)
                  << std::setw(w) << curve_pts(i, 2) << "\n";
    }
}

/// Demonstrate periodic knot generation.
static void demonstrate_periodic_knots() {
    ex::print_header("Periodic Knots");

    const int degree = 3;
    const int n_cp = 6;

    const Eigen::VectorXd uniform_knots = BSpline::create_uniform_knots(degree, n_cp, 0.0, 1.0);
    const Eigen::VectorXd periodic_knots = BSpline::create_periodic_knots(degree, n_cp, 0.0, 1.0);

    ex::print_vector("Uniform knots (p=3, n_cp=6)", uniform_knots);
    ex::print_vector("Periodic knots (p=3, n_cp=6)", periodic_knots);
}

int main() {
    demonstrate_basic_bspline();
    example_b6();
    demonstrate_curve_generation();
    demonstrate_3d_bspline();
    demonstrate_periodic_knots();

    return 0;
}
