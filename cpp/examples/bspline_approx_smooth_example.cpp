/// B-spline approximation and smoothing example -- C++ port of
/// examples/b_spline_approx_ex.py AND examples/b_spline_smooth_ex.py
///
/// Part 1: ApproximationBSpline -- least-squares fitting with varying control
///         point counts and degrees.
/// Part 2: SmoothingCubicBSpline -- smoothing with different lambda/mu values
///         (Example 8.12).

#include <interpolatecpp/bspline/approximation_bspline.hpp>
#include <interpolatecpp/bspline/bspline.hpp>
#include <interpolatecpp/bspline/bspline_parameters.hpp>
#include <interpolatecpp/bspline/smoothing_cubic_bspline.hpp>

#include "example_utils.hpp"

#include <cmath>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace ex = interpolatecpp::examples;
using namespace interpolatecpp::bspline;

// ============================================================================
// Part 1: Approximation B-spline
// ============================================================================

/// Generate sample points from a B-spline defined by the Example 8.10 control points.
static Eigen::MatrixXd generate_sample_points(int num_samples) {
    // Control points from Example 8.10
    Eigen::MatrixXd control_points(10, 2);
    control_points << 137, 229,
                      101, 201,
                      177, 121,
                       93,  44,
                       62, 203,
                       49, 272,
                      104, 402,
                      141, 277,
                      147, 258,
                      138, 231;

    const int degree = 3;
    const Eigen::VectorXd knots = BSpline::create_uniform_knots(degree, 10);

    const BSpline spline(degree,
                         std::span<const double>(knots.data(), static_cast<size_t>(knots.size())),
                         control_points);

    const auto [params, curve_pts] = spline.generate_curve_points(num_samples);
    return curve_pts;
}

/// Example from Section 8.5: Approximation with different configurations.
static void example_approximation() {
    ex::print_header("Part 1 -- B-spline Approximation (Section 8.5)");

    // Print the source control points
    std::cout << "Control points from Example 8.10:\n";
    const std::vector<std::pair<int, int>> cp_coords = {
        {137, 229}, {101, 201}, {177, 121}, {93, 44}, {62, 203},
        {49, 272}, {104, 402}, {141, 277}, {147, 258}, {138, 231}
    };
    for (size_t i = 0; i < cp_coords.size(); ++i) {
        std::cout << "  P" << i << ": (" << cp_coords[i].first
                  << ", " << cp_coords[i].second << ")\n";
    }

    // Generate sample points
    const Eigen::MatrixXd sample_points = generate_sample_points(84);
    std::cout << "\nGenerated " << sample_points.rows() << " sample points.\n";

    // Test cases: {num_cps, degree, title}
    struct TestCase {
        int num_cps;
        int degree;
        std::string title;
    };
    const std::vector<TestCase> test_cases = {
        {10, 3, "Cubic (p=3) with 10 control points"},
        {10, 4, "Quartic (p=4) with 10 control points"},
        {20, 3, "Cubic (p=3) with 20 control points"},
    };

    for (size_t i = 0; i < test_cases.size(); ++i) {
        const auto& tc = test_cases[i];
        ex::print_separator('=');
        std::cout << "Test case " << (i + 1) << ": " << tc.title << "\n\n";

        const ApproximationBSpline approx(
            sample_points, tc.num_cps, tc.degree);

        const double error = approx.calculate_approximation_error();
        ex::print_value("Approximation error", error, 2);
        ex::print_value("Num control points", static_cast<double>(approx.n_control_points()), 0);
        ex::print_value("Degree", static_cast<double>(approx.degree()), 0);

        std::cout << "\n";
        ex::print_vector("Knot vector", approx.knots());
        ex::print_matrix("Control points", approx.control_points());

        // Generate and print a few curve points
        const auto [params, curve_pts] = approx.generate_curve_points(10);
        std::cout << "\nSample curve points:\n";
        const int w = 14;
        std::cout << std::right << std::setw(w) << "u"
                  << std::setw(w) << "X" << std::setw(w) << "Y" << "\n";
        ex::print_separator('-', 3 * w);
        for (Eigen::Index j = 0; j < curve_pts.rows(); ++j) {
            std::cout << std::fixed << std::setprecision(4)
                      << std::setw(w) << params(j)
                      << std::setw(w) << curve_pts(j, 0)
                      << std::setw(w) << curve_pts(j, 1) << "\n";
        }
    }
}

/// Demonstrate degree comparison on a heart-shaped curve.
static void example_degree_comparison() {
    ex::print_header("Approximation -- Degree Comparison");

    // Generate heart-shaped points
    const int n = 100;
    Eigen::MatrixXd heart_points(n, 2);
    for (int i = 0; i < n; ++i) {
        const double t = 2.0 * M_PI * static_cast<double>(i) / n;
        heart_points(i, 0) = 16.0 * std::pow(std::sin(t), 3) * 10.0 + 150.0;
        heart_points(i, 1) = (13.0 * std::cos(t) - 5.0 * std::cos(2.0 * t)
                               - 2.0 * std::cos(3.0 * t) - std::cos(4.0 * t)) * 10.0 + 150.0;
    }

    std::cout << "Heart shape: " << heart_points.rows() << " sample points\n\n";

    const int num_cps = 12;
    const std::vector<int> degrees = {2, 3, 4};

    for (const int degree : degrees) {
        const ApproximationBSpline approx(heart_points, num_cps, degree);
        const double error = approx.calculate_approximation_error();

        std::cout << "Degree " << degree
                  << " (n_cp=" << num_cps << "): error = "
                  << std::fixed << std::setprecision(2) << error << "\n";
    }
}

/// Demonstrate control-point-count comparison on a spiral.
static void example_cp_count_comparison() {
    ex::print_header("Approximation -- Control Point Count Comparison");

    // Generate spiral points
    const int n = 100;
    Eigen::MatrixXd spiral_points(n, 2);
    for (int i = 0; i < n; ++i) {
        const double t = 6.0 * M_PI * static_cast<double>(i) / (n - 1);
        const double r = 5.0 + 15.0 * t;
        spiral_points(i, 0) = r * std::cos(t) + 150.0;
        spiral_points(i, 1) = r * std::sin(t) + 150.0;
    }

    std::cout << "Spiral: " << spiral_points.rows() << " sample points\n\n";

    const int degree = 3;
    const std::vector<int> cp_counts = {8, 15, 25};

    for (const int num_cps : cp_counts) {
        const ApproximationBSpline approx(spiral_points, num_cps, degree);
        const double error = approx.calculate_approximation_error();

        std::cout << "CPs = " << std::setw(3) << num_cps
                  << " (degree=" << degree << "): error = "
                  << std::fixed << std::setprecision(2) << error << "\n";
    }
}

/// Demonstrate weighted approximation.
static void example_weighted_approximation() {
    ex::print_header("Approximation -- Weighted Fit");

    // Circle with noise
    const int n = 60;
    Eigen::MatrixXd circle_points(n, 2);
    for (int i = 0; i < n; ++i) {
        const double t = 2.0 * M_PI * static_cast<double>(i) / n;
        circle_points(i, 0) = 100.0 * std::cos(t) + 150.0;
        circle_points(i, 1) = 100.0 * std::sin(t) + 150.0;
    }

    const int num_cps = 10;
    const int degree = 3;

    // Uniform weights
    const ApproximationBSpline approx_uniform(circle_points, num_cps, degree);
    const double error_uniform = approx_uniform.calculate_approximation_error();

    // Custom weights: emphasize first half of the points
    Eigen::VectorXd weights = Eigen::VectorXd::Ones(n);
    for (int i = 0; i < n / 2; ++i) {
        weights(i) = 5.0;
    }

    const ApproximationBSpline approx_weighted(
        circle_points, num_cps, degree, weights);
    const double error_weighted = approx_weighted.calculate_approximation_error();

    std::cout << "Circle approximation (" << n << " points, " << num_cps << " CPs):\n";
    ex::print_value("Uniform weights error", error_uniform, 2);
    ex::print_value("Weighted (first half emphasized) error", error_weighted, 2);
}

// ============================================================================
// Part 2: Smoothing Cubic B-spline
// ============================================================================

/// Example 8.12: Smoothing B-spline with different lambda values.
static void example_8_12() {
    ex::print_header("Part 2 -- Smoothing Cubic B-spline (Example 8.12)");

    // Points from the example
    Eigen::MatrixXd points(6, 3);
    points << 0, 0, 0,
              1, 2, 1,
              2, 3, 0,
              4, 3, 0,
              5, 2, 2,
              6, 0, 2;

    std::cout << "Approximation points:\n";
    ex::print_matrix("Points", points);

    // Test different lambda values
    const std::vector<double> lambda_values = {1e-4, 1e-5, 1e-6};

    for (const double lambda_val : lambda_values) {
        ex::print_separator('=');

        // Convert lambda to mu: lambda = (1 - mu) / (6 * mu) => mu = 1 / (6*lambda + 1)
        const double mu = 1.0 / (6.0 * lambda_val + 1.0);

        std::cout << "Lambda = " << std::scientific << std::setprecision(6) << lambda_val
                  << ", Mu = " << std::fixed << std::setprecision(6) << mu << "\n\n";

        BSplineParams params;
        params.mu = mu;
        params.method = Parameterization::ChordLength;
        params.enforce_endpoints = true;
        params.auto_derivatives = true;

        const SmoothingCubicBSpline spline(points, params);

        ex::print_value("Mu (stored)", spline.mu());
        ex::print_value("Lambda (stored)", spline.lambda_param());
        ex::print_value("Degree", static_cast<double>(spline.degree()), 0);
        ex::print_value("Num control points", static_cast<double>(spline.n_control_points()), 0);

        std::cout << "\n";
        ex::print_matrix("Control points", spline.control_points());

        // Approximation errors
        const Eigen::VectorXd errors = spline.calculate_approximation_error();
        std::cout << "\n";
        ex::print_vector("Per-point errors", errors);

        const double total_error = spline.calculate_total_error();
        ex::print_value("Total error", total_error);

        // Evaluate along the curve
        std::cout << "\nCurve samples:\n";
        const int w = 14;
        std::cout << std::right
                  << std::setw(w) << "u"
                  << std::setw(w) << "X"
                  << std::setw(w) << "Y"
                  << std::setw(w) << "Z" << "\n";
        ex::print_separator('-', 4 * w);

        const int n_samples = 10;
        const double u_start = spline.u_min();
        const double u_end = spline.u_max();
        for (int i = 0; i <= n_samples; ++i) {
            const double u = u_start + (u_end - u_start) * static_cast<double>(i) / n_samples;
            const Eigen::VectorXd pt = spline.evaluate(u);
            std::cout << std::fixed << std::setprecision(4)
                      << std::setw(w) << u
                      << std::setw(w) << pt(0)
                      << std::setw(w) << pt(1)
                      << std::setw(w) << pt(2) << "\n";
        }
    }
}

/// Demonstrate smoothing with different mu values on 2D data.
static void example_smoothing_mu_comparison() {
    ex::print_header("Smoothing -- Mu Comparison (2D)");

    // Simple 2D points
    Eigen::MatrixXd points(8, 2);
    points << 0, 0,
              1, 3,
              2, 1,
              3, 4,
              4, 2,
              5, 5,
              6, 1,
              7, 0;

    const std::vector<double> mu_values = {0.1, 0.5, 0.9};

    for (const double mu : mu_values) {
        BSplineParams params;
        params.mu = mu;
        params.method = Parameterization::ChordLength;
        params.enforce_endpoints = true;
        params.auto_derivatives = true;

        const SmoothingCubicBSpline spline(points, params);

        const double total_error = spline.calculate_total_error();
        std::cout << "Mu = " << std::fixed << std::setprecision(2) << mu
                  << ": total_error = " << std::setprecision(6) << total_error
                  << ", n_cp = " << spline.n_control_points() << "\n";
    }
}

/// Demonstrate smoothing with explicit endpoint derivatives.
static void example_smoothing_with_derivatives() {
    ex::print_header("Smoothing -- Explicit Endpoint Derivatives");

    Eigen::MatrixXd points(6, 3);
    points << 0, 0, 0,
              1, 2, 1,
              2, 3, 0,
              4, 3, 0,
              5, 2, 2,
              6, 0, 2;

    Eigen::VectorXd v0(3), vn(3);
    v0 << 4.43, 8.87, 4.43;
    vn << 4.85, -9.71, 0.0;

    BSplineParams params;
    params.mu = 0.999;
    params.v0 = v0;
    params.vn = vn;
    params.method = Parameterization::ChordLength;
    params.enforce_endpoints = true;
    params.auto_derivatives = false;

    const SmoothingCubicBSpline spline(points, params);

    std::cout << "With explicit endpoint derivatives:\n";
    ex::print_vector("v0", v0);
    ex::print_vector("vn", vn);
    ex::print_value("Mu", spline.mu());
    ex::print_value("Total error", spline.calculate_total_error());
    std::cout << "\n";
    ex::print_matrix("Control points", spline.control_points());

    // Second derivative magnitudes at a few points
    ex::print_separator();
    std::cout << "Second derivative magnitude along curve:\n\n";
    const int n_samples = 8;
    for (int i = 0; i <= n_samples; ++i) {
        const double u = spline.u_min()
            + (spline.u_max() - spline.u_min()) * static_cast<double>(i) / n_samples;
        const Eigen::VectorXd d2 = spline.evaluate_derivative(u, 2);
        const double mag = d2.norm();
        std::cout << "  u = " << std::fixed << std::setprecision(4) << u
                  << ": ||s''(u)|| = " << std::setprecision(4) << mag << "\n";
    }
}

int main() {
    // Part 1: Approximation
    example_approximation();
    example_degree_comparison();
    example_cp_count_comparison();
    example_weighted_approximation();

    // Part 2: Smoothing
    example_8_12();
    example_smoothing_mu_comparison();
    example_smoothing_with_derivatives();

    return 0;
}
