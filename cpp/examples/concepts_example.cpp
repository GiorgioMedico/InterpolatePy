/// C++20 concepts example -- C++ port of examples/protocols_ex.py
///
/// Demonstrates how C++20 concepts serve as the compile-time equivalent of
/// Python protocols (PEP 544).  Template functions constrained by each concept
/// accept any concrete type that satisfies the required interface, with no
/// inheritance needed.

#include <interpolatecpp/concepts.hpp>

// Concrete types for ScalarTrajectory
#include <interpolatecpp/spline/cubic_spline.hpp>
#include <interpolatecpp/spline/cubic_smoothing_spline.hpp>

// Concrete types for CurveEvaluator
#include <interpolatecpp/bspline/bspline.hpp>
#include <interpolatecpp/bspline/bspline_interpolator.hpp>

// Concrete types for GeometricPath
#include <interpolatecpp/path/linear_path.hpp>
#include <interpolatecpp/path/circular_path.hpp>

// Concrete types for QuaternionTrajectory
#include <interpolatecpp/quat/quaternion_spline.hpp>
#include <interpolatecpp/quat/squad_c2.hpp>
#include <interpolatecpp/quat/log_quaternion_interpolation.hpp>
#include <interpolatecpp/quat/quaternion.hpp>

#include "example_utils.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

namespace ex = interpolatecpp::examples;

// ---------------------------------------------------------------------------
// 1. ScalarTrajectory -- generic sampling of position, velocity, acceleration
// ---------------------------------------------------------------------------

template <interpolatecpp::ScalarTrajectory T>
void sample_scalar(const T& traj, double t_start, double t_end, int n,
                   const std::string& label) {
    std::cout << "  " << label << ":\n";

    const int w = 14;
    const int p = 6;

    std::cout << std::right
              << std::setw(w) << "t"
              << std::setw(w) << "Position"
              << std::setw(w) << "Velocity"
              << std::setw(w) << "Acceleration"
              << "\n";
    ex::print_separator('-', 4 * w);

    for (int i = 0; i <= n; ++i) {
        const double t = t_start + (t_end - t_start) * static_cast<double>(i) / n;
        std::cout << std::fixed << std::setprecision(p)
                  << std::setw(w) << t
                  << std::setw(w) << traj.evaluate(t)
                  << std::setw(w) << traj.evaluate_velocity(t)
                  << std::setw(w) << traj.evaluate_acceleration(t)
                  << "\n";
    }
    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// 2. CurveEvaluator -- generic sampling of parametric curve and derivatives
// ---------------------------------------------------------------------------

template <interpolatecpp::CurveEvaluator T>
void sample_curve(const T& curve, double u_start, double u_end, int n,
                  const std::string& label) {
    std::cout << "  " << label << ":\n";

    const int w = 14;
    const int p = 6;

    // Determine dimension from the first evaluation
    const Eigen::VectorXd first_pt = curve.evaluate(u_start);
    const int dim = static_cast<int>(first_pt.size());

    // Print header
    std::cout << std::right << std::setw(w) << "u";
    for (int d = 0; d < dim; ++d) {
        std::cout << std::setw(w) << ("P[" + std::to_string(d) + "]");
    }
    for (int d = 0; d < dim; ++d) {
        std::cout << std::setw(w) << ("D[" + std::to_string(d) + "]");
    }
    std::cout << "\n";
    ex::print_separator('-', (1 + 2 * dim) * w);

    for (int i = 0; i <= n; ++i) {
        const double u = u_start + (u_end - u_start) * static_cast<double>(i) / n;
        const Eigen::VectorXd pt = curve.evaluate(u);
        const Eigen::VectorXd deriv = curve.evaluate_derivative(u, 1);

        std::cout << std::fixed << std::setprecision(p) << std::setw(w) << u;
        for (int d = 0; d < dim; ++d) {
            std::cout << std::setw(w) << pt(d);
        }
        for (int d = 0; d < dim; ++d) {
            std::cout << std::setw(w) << deriv(d);
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// 3. GeometricPath -- generic 3D path sampling
// ---------------------------------------------------------------------------

template <interpolatecpp::GeometricPath T>
void sample_path(const T& path, double s_start, double s_end, int n,
                 const std::string& label) {
    std::cout << "  " << label << ":\n";

    const int w = 12;
    const int p = 4;

    std::cout << std::right
              << std::setw(w) << "s"
              << std::setw(w) << "Pos X"
              << std::setw(w) << "Pos Y"
              << std::setw(w) << "Pos Z"
              << std::setw(w) << "Vel X"
              << std::setw(w) << "Vel Y"
              << std::setw(w) << "Vel Z"
              << std::setw(w) << "Acc X"
              << std::setw(w) << "Acc Y"
              << std::setw(w) << "Acc Z"
              << "\n";
    ex::print_separator('-', 10 * w);

    for (int i = 0; i <= n; ++i) {
        const double s = s_start + (s_end - s_start) * static_cast<double>(i) / n;
        const Eigen::Vector3d pos = path.position(s);
        const Eigen::Vector3d vel = path.velocity(s);
        const Eigen::Vector3d acc = path.acceleration(s);

        std::cout << std::fixed << std::setprecision(p)
                  << std::setw(w) << s
                  << std::setw(w) << pos.x()
                  << std::setw(w) << pos.y()
                  << std::setw(w) << pos.z()
                  << std::setw(w) << vel.x()
                  << std::setw(w) << vel.y()
                  << std::setw(w) << vel.z()
                  << std::setw(w) << acc.x()
                  << std::setw(w) << acc.y()
                  << std::setw(w) << acc.z()
                  << "\n";
    }
    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// 4. QuaternionTrajectory -- generic quaternion sampling with angular velocity
// ---------------------------------------------------------------------------

template <interpolatecpp::QuaternionTrajectory T>
void sample_quaternion(const T& traj, double t_start, double t_end, int n,
                       const std::string& label) {
    std::cout << "  " << label << ":\n";

    const int w = 12;
    const int p = 4;

    std::cout << std::right
              << std::setw(w) << "t"
              << std::setw(w) << "q.w"
              << std::setw(w) << "q.x"
              << std::setw(w) << "q.y"
              << std::setw(w) << "q.z"
              << std::setw(w) << "omega.x"
              << std::setw(w) << "omega.y"
              << std::setw(w) << "omega.z"
              << std::setw(w) << "|omega|"
              << "\n";
    ex::print_separator('-', 9 * w);

    for (int i = 0; i <= n; ++i) {
        const double t = t_start + (t_end - t_start) * static_cast<double>(i) / n;
        const Eigen::Quaterniond q = traj.evaluate(t);
        const Eigen::Vector3d omega = traj.evaluate_velocity(t);

        std::cout << std::fixed << std::setprecision(p)
                  << std::setw(w) << t
                  << std::setw(w) << q.w()
                  << std::setw(w) << q.x()
                  << std::setw(w) << q.y()
                  << std::setw(w) << q.z()
                  << std::setw(w) << omega.x()
                  << std::setw(w) << omega.y()
                  << std::setw(w) << omega.z()
                  << std::setw(w) << omega.norm()
                  << "\n";
    }
    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// Compile-time concept verification helpers
// ---------------------------------------------------------------------------

static void print_concept_conformance() {
    ex::print_header("Concept Conformance Summary");

    // ScalarTrajectory
    std::cout << "ScalarTrajectory concept:\n";
    std::cout << "  CubicSpline          : "
              << (interpolatecpp::ScalarTrajectory<interpolatecpp::spline::CubicSpline>
                      ? "YES" : "NO") << "\n";
    std::cout << "  CubicSmoothingSpline : "
              << (interpolatecpp::ScalarTrajectory<interpolatecpp::spline::CubicSmoothingSpline>
                      ? "YES" : "NO") << "\n";
    std::cout << "\n";

    // CurveEvaluator
    std::cout << "CurveEvaluator concept:\n";
    std::cout << "  BSpline              : "
              << (interpolatecpp::CurveEvaluator<interpolatecpp::bspline::BSpline>
                      ? "YES" : "NO") << "\n";
    std::cout << "  BSplineInterpolator  : "
              << (interpolatecpp::CurveEvaluator<interpolatecpp::bspline::BSplineInterpolator>
                      ? "YES" : "NO") << "\n";
    std::cout << "\n";

    // GeometricPath
    std::cout << "GeometricPath concept:\n";
    std::cout << "  LinearPath           : "
              << (interpolatecpp::GeometricPath<interpolatecpp::path::LinearPath>
                      ? "YES" : "NO") << "\n";
    std::cout << "  CircularPath         : "
              << (interpolatecpp::GeometricPath<interpolatecpp::path::CircularPath>
                      ? "YES" : "NO") << "\n";
    std::cout << "\n";

    // QuaternionTrajectory
    std::cout << "QuaternionTrajectory concept:\n";
    std::cout << "  QuaternionSpline     : "
              << (interpolatecpp::QuaternionTrajectory<interpolatecpp::quat::QuaternionSpline>
                      ? "YES" : "NO") << "\n";
    std::cout << "  SquadC2              : "
              << (interpolatecpp::QuaternionTrajectory<interpolatecpp::quat::SquadC2>
                      ? "YES" : "NO") << "\n";
    std::cout << "  LogQuaternionInterp  : "
              << (interpolatecpp::QuaternionTrajectory<interpolatecpp::quat::LogQuaternionInterpolation>
                      ? "YES" : "NO") << "\n";
}

// ---------------------------------------------------------------------------
// Example: ScalarTrajectory with CubicSpline and CubicSmoothingSpline
// ---------------------------------------------------------------------------

static void example_scalar_trajectory() {
    ex::print_header("1. ScalarTrajectory Concept");

    std::cout << "The same template function 'sample_scalar' works with any type\n"
              << "that satisfies the ScalarTrajectory concept.\n\n";

    // CubicSpline
    const std::vector<double> t_points = {0.0, 1.0, 2.0, 3.0, 4.0};
    const std::vector<double> q_points = {0.0, 5.0, 3.0, 8.0, 10.0};
    const interpolatecpp::spline::CubicSpline spline(t_points, q_points);
    sample_scalar(spline, 0.0, 4.0, 10, "CubicSpline");

    // CubicSmoothingSpline
    const interpolatecpp::spline::CubicSmoothingSpline smooth(t_points, q_points, 0.8);
    sample_scalar(smooth, 0.0, 4.0, 10, "CubicSmoothingSpline (mu=0.8)");
}

// ---------------------------------------------------------------------------
// Example: CurveEvaluator with BSpline and BSplineInterpolator
// ---------------------------------------------------------------------------

static void example_curve_evaluator() {
    ex::print_header("2. CurveEvaluator Concept");

    std::cout << "The same template function 'sample_curve' works with any type\n"
              << "that satisfies the CurveEvaluator concept.\n\n";

    // BSplineInterpolator (interpolates through points)
    Eigen::VectorXd times(5);
    times << 0.0, 1.0, 2.0, 3.0, 4.0;

    Eigen::MatrixXd points(5, 2);
    points << 0.0, 0.0,
              1.0, 2.0,
              3.0, 3.0,
              5.0, 1.0,
              6.0, 4.0;

    const interpolatecpp::bspline::BSplineInterpolator interpolator(3, points, times);
    sample_curve(interpolator, interpolator.u_min(), interpolator.u_max(), 10,
                 "BSplineInterpolator (degree 3, 2D)");

    // BSpline (approximation with explicit control polygon)
    Eigen::MatrixXd ctrl_pts(6, 2);
    ctrl_pts << 0.0, 0.0,
                1.0, 3.0,
                2.5, 4.0,
                4.0, 2.0,
                5.5, 3.5,
                6.0, 1.0;

    const int degree = 3;
    const Eigen::VectorXd knots = interpolatecpp::bspline::BSpline::create_uniform_knots(
        degree, static_cast<int>(ctrl_pts.rows()));

    const interpolatecpp::bspline::BSpline bspline(degree, std::vector<double>(
        knots.data(), knots.data() + knots.size()), ctrl_pts);
    sample_curve(bspline, bspline.u_min(), bspline.u_max(), 10,
                 "BSpline (degree 3, 6 control points, 2D)");
}

// ---------------------------------------------------------------------------
// Example: GeometricPath with LinearPath and CircularPath
// ---------------------------------------------------------------------------

static void example_geometric_path() {
    ex::print_header("3. GeometricPath Concept");

    std::cout << "The same template function 'sample_path' works with any type\n"
              << "that satisfies the GeometricPath concept.\n\n";

    // LinearPath
    const Eigen::Vector3d start(0.0, 0.0, 0.0);
    const Eigen::Vector3d end(5.0, 3.0, 2.0);
    const interpolatecpp::path::LinearPath linear(start, end);
    sample_path(linear, 0.0, linear.length(), 8, "LinearPath (0,0,0) -> (5,3,2)");

    // CircularPath
    const Eigen::Vector3d axis(0.0, 0.0, 1.0);
    const Eigen::Vector3d center(0.0, 0.0, 0.0);
    const Eigen::Vector3d circle_pt(3.0, 0.0, 0.0);
    const interpolatecpp::path::CircularPath circular(axis, center, circle_pt);
    const double half_arc = M_PI * circular.radius();
    sample_path(circular, 0.0, half_arc, 8, "CircularPath (half circle, r=3)");
}

// ---------------------------------------------------------------------------
// Example: QuaternionTrajectory with SquadC2, QuaternionSpline, and LogQuat
// ---------------------------------------------------------------------------

static void example_quaternion_trajectory() {
    ex::print_header("4. QuaternionTrajectory Concept");

    std::cout << "The same template function 'sample_quaternion' works with any type\n"
              << "that satisfies the QuaternionTrajectory concept.\n\n";

    using interpolatecpp::quat::Quaternion;

    // Shared waypoints: identity -> 90deg Z -> 90deg X+Z -> identity
    const Quaternion q1 = Quaternion::identity();
    const Quaternion q2 = Quaternion::from_euler_angles(0.0, 0.0, M_PI / 2.0);
    const Quaternion q3 = Quaternion::from_euler_angles(M_PI / 2.0, 0.0, M_PI / 2.0);
    const Quaternion q4 = Quaternion::identity();
    const std::vector<double> times = {0.0, 1.0, 2.0, 3.0};
    const std::vector<Quaternion> quats = {q1, q2, q3, q4};

    // SquadC2
    const interpolatecpp::quat::SquadC2 squad(times, quats);
    sample_quaternion(squad, 0.0, 3.0, 10, "SquadC2");

    // QuaternionSpline (Squad method)
    const interpolatecpp::quat::QuaternionSpline qspline(
        times, quats, interpolatecpp::quat::QuaternionSpline::Method::Squad);
    sample_quaternion(qspline, 0.0, 3.0, 10, "QuaternionSpline (SQUAD method)");

    // LogQuaternionInterpolation
    const interpolatecpp::quat::LogQuaternionInterpolation logquat(times, quats);
    sample_quaternion(logquat, 0.0, 3.0, 10, "LogQuaternionInterpolation");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    std::cout << "C++20 Concepts -- Compile-Time Protocol Equivalents\n";
    ex::print_separator('=');

    print_concept_conformance();
    example_scalar_trajectory();
    example_curve_evaluator();
    example_geometric_path();
    example_quaternion_trajectory();

    std::cout << "All concept examples completed.\n";
    return 0;
}
