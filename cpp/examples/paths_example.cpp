/// Paths and Frenet frames example -- C++ port of examples/simple_paths_ex.py
/// and examples/frenet_frame_ex.py
///
/// Demonstrates LinearPath, CircularPath, and Frenet-Serret frame computation
/// on helicoidal and circular trajectories.

#include <interpolatecpp/path/linear_path.hpp>
#include <interpolatecpp/path/circular_path.hpp>
#include <interpolatecpp/path/frenet_frame.hpp>
#include <interpolatecpp/path/linear_traj.hpp>

#include "example_utils.hpp"

#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>

namespace ex = interpolatecpp::examples;
using namespace interpolatecpp::path;

// ---------------------------------------------------------------------------
// 1. LinearPath -- evaluate position, velocity, acceleration at arc-length samples
// ---------------------------------------------------------------------------

static void linear_path_example() {
    ex::print_header("Linear Path Example");

    const Eigen::Vector3d pi(0.0, 0.0, 0.0);
    const Eigen::Vector3d pf(10.0, 5.0, 3.0);
    const LinearPath path(pi, pf);

    const double path_length = path.length();
    ex::print_value("Path length", path_length, 4);
    ex::print_vector3("Start point", pi);
    ex::print_vector3("End point", pf);

    // Evaluate at 10 arc-length samples
    const int n_samples = 10;
    const int w = 14;
    const int p = 6;

    std::cout << "\nTrajectory along linear path (" << n_samples << " samples):\n\n";

    std::cout << std::right
              << std::setw(w) << "s"
              << std::setw(w) << "Pos X"
              << std::setw(w) << "Pos Y"
              << std::setw(w) << "Pos Z"
              << std::setw(w) << "Vel X"
              << std::setw(w) << "Vel Y"
              << std::setw(w) << "Vel Z"
              << "\n";
    ex::print_separator('-', 7 * w);

    for (int i = 0; i <= n_samples; ++i) {
        const double s = path_length * static_cast<double>(i) / n_samples;
        const Eigen::Vector3d pos = path.position(s);
        const Eigen::Vector3d vel = path.velocity(s);

        std::cout << std::fixed << std::setprecision(p)
                  << std::setw(w) << s
                  << std::setw(w) << pos.x()
                  << std::setw(w) << pos.y()
                  << std::setw(w) << pos.z()
                  << std::setw(w) << vel.x()
                  << std::setw(w) << vel.y()
                  << std::setw(w) << vel.z()
                  << "\n";
    }

    // Acceleration is zero for a linear path
    ex::print_separator('=');
    std::cout << "Linear path properties:\n";
    ex::print_vector3("Acceleration (constant)", path.acceleration(0.0));
    std::cout << "  Note: acceleration is zero for a straight-line path.\n";

    // Vectorized evaluation
    Eigen::VectorXd s_vec(5);
    s_vec << 0.0, path_length * 0.25, path_length * 0.5, path_length * 0.75, path_length;
    const Eigen::MatrixXd positions = path.position(s_vec);

    std::cout << "\nVectorized evaluation at 5 points:\n";
    ex::print_matrix("Positions", positions);
}

// ---------------------------------------------------------------------------
// 2. CircularPath -- arc-length parameterized circle in 3D
// ---------------------------------------------------------------------------

static void circular_path_example() {
    ex::print_header("Circular Path Example");

    const Eigen::Vector3d axis(0.0, 0.0, 1.0);
    const Eigen::Vector3d axis_point(0.0, 0.0, 0.0);
    const Eigen::Vector3d circle_point(2.0, 0.0, 0.0);
    const CircularPath path(axis, axis_point, circle_point);

    const double radius = path.radius();
    const double half_arc = M_PI * radius;

    ex::print_value("Radius", radius, 4);
    ex::print_vector3("Center", path.center());
    ex::print_value("Half-circle arc length", half_arc, 4);

    // Evaluate at angular samples over a full circle
    const int n_samples = 12;
    const double full_arc = 2.0 * M_PI * radius;
    const int w = 14;
    const int p = 6;

    std::cout << "\nTrajectory along circular path (" << n_samples << " angular samples):\n\n";

    std::cout << std::right
              << std::setw(w) << "s"
              << std::setw(w) << "Pos X"
              << std::setw(w) << "Pos Y"
              << std::setw(w) << "Pos Z"
              << std::setw(w) << "Vel X"
              << std::setw(w) << "Vel Y"
              << std::setw(w) << "Vel Z"
              << "\n";
    ex::print_separator('-', 7 * w);

    for (int i = 0; i <= n_samples; ++i) {
        const double s = full_arc * static_cast<double>(i) / n_samples;
        const Eigen::Vector3d pos = path.position(s);
        const Eigen::Vector3d vel = path.velocity(s);

        std::cout << std::fixed << std::setprecision(p)
                  << std::setw(w) << s
                  << std::setw(w) << pos.x()
                  << std::setw(w) << pos.y()
                  << std::setw(w) << pos.z()
                  << std::setw(w) << vel.x()
                  << std::setw(w) << vel.y()
                  << std::setw(w) << vel.z()
                  << "\n";
    }

    // Print acceleration at a few points
    ex::print_separator('=');
    std::cout << "Circular path acceleration (centripetal, always points inward):\n\n";
    const std::vector<double> check_points = {0.0, half_arc * 0.5, half_arc, half_arc * 1.5};
    for (const double s : check_points) {
        const Eigen::Vector3d acc = path.acceleration(s);
        std::cout << "  s = " << std::fixed << std::setprecision(2) << s
                  << " -> acc = (" << std::setprecision(4)
                  << acc.x() << ", " << acc.y() << ", " << acc.z() << ")"
                  << "  |acc| = " << acc.norm() << "\n";
    }
}

// ---------------------------------------------------------------------------
// 3. Frenet frames on helicoidal trajectory
// ---------------------------------------------------------------------------

static void frenet_frame_helicoidal() {
    ex::print_header("Frenet Frames -- Helicoidal Trajectory");

    const double r = 2.0;
    const double d = 0.5;
    const int n_samples = 8;

    std::cout << "Parameters: r = " << r << ", d = " << d << "\n";
    std::cout << "Helicoidal curve: p(u) = (r*cos(u), r*sin(u), d*u)\n\n";

    // Build parameter values
    Eigen::VectorXd u_values(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        u_values(i) = 4.0 * M_PI * static_cast<double>(i) / (n_samples - 1);
    }

    // Curve function returning (position, velocity, acceleration)
    auto helix_fn = [r, d](double u)
        -> std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d> {
        return helicoidal_trajectory_with_derivatives(u, r, d);
    };

    const std::vector<FrenetFrame> frames = compute_frenet_frames(helix_fn, u_values);

    // Print table of Frenet frame data
    const int w = 12;
    const int p = 4;

    std::cout << std::right
              << std::setw(8)  << "u"
              << std::setw(w)  << "T.x"
              << std::setw(w)  << "T.y"
              << std::setw(w)  << "T.z"
              << std::setw(w)  << "N.x"
              << std::setw(w)  << "N.y"
              << std::setw(w)  << "N.z"
              << std::setw(w)  << "B.x"
              << std::setw(w)  << "B.y"
              << std::setw(w)  << "B.z"
              << std::setw(w)  << "curv"
              << std::setw(w)  << "torsion"
              << "\n";
    ex::print_separator('-', 8 + 11 * w);

    for (int i = 0; i < n_samples; ++i) {
        const auto& f = frames[static_cast<size_t>(i)];
        std::cout << std::fixed << std::setprecision(p)
                  << std::setw(8) << u_values(i)
                  << std::setw(w) << f.tangent.x()
                  << std::setw(w) << f.tangent.y()
                  << std::setw(w) << f.tangent.z()
                  << std::setw(w) << f.normal.x()
                  << std::setw(w) << f.normal.y()
                  << std::setw(w) << f.normal.z()
                  << std::setw(w) << f.binormal.x()
                  << std::setw(w) << f.binormal.y()
                  << std::setw(w) << f.binormal.z()
                  << std::setw(w) << f.curvature
                  << std::setw(w) << f.torsion
                  << "\n";
    }

    // Print summary for first and last frames
    ex::print_separator('=');
    std::cout << "Frame details:\n\n";

    const auto& first = frames.front();
    std::cout << "  First frame (u = 0):\n";
    ex::print_vector3("    Tangent", first.tangent);
    ex::print_vector3("    Normal", first.normal);
    ex::print_vector3("    Binormal", first.binormal);
    ex::print_value("    Curvature", first.curvature);
    ex::print_value("    Torsion", first.torsion);

    const auto& last = frames.back();
    std::cout << "\n  Last frame (u = " << std::fixed << std::setprecision(2)
              << u_values(n_samples - 1) << "):\n";
    ex::print_vector3("    Tangent", last.tangent);
    ex::print_vector3("    Normal", last.normal);
    ex::print_vector3("    Binormal", last.binormal);
    ex::print_value("    Curvature", last.curvature);
    ex::print_value("    Torsion", last.torsion);
}

// ---------------------------------------------------------------------------
// 4. Frenet frames on circular trajectory
// ---------------------------------------------------------------------------

static void frenet_frame_circular() {
    ex::print_header("Frenet Frames -- Circular Trajectory");

    const double r = 2.0;
    const int n_samples = 8;

    std::cout << "Parameters: r = " << r << "\n";
    std::cout << "Circular curve: p(u) = (r*cos(u), r*sin(u), 0)\n\n";

    // Build parameter values over one full circle
    Eigen::VectorXd u_values(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        u_values(i) = 2.0 * M_PI * static_cast<double>(i) / (n_samples - 1);
    }

    // Curve function
    auto circle_fn = [r](double u)
        -> std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d> {
        return circular_trajectory_with_derivatives(u, r);
    };

    const std::vector<FrenetFrame> frames = compute_frenet_frames(circle_fn, u_values);

    // Print table
    const int w = 12;
    const int p = 4;

    std::cout << std::right
              << std::setw(8)  << "u"
              << std::setw(w)  << "T.x"
              << std::setw(w)  << "T.y"
              << std::setw(w)  << "T.z"
              << std::setw(w)  << "N.x"
              << std::setw(w)  << "N.y"
              << std::setw(w)  << "N.z"
              << std::setw(w)  << "curv"
              << std::setw(w)  << "torsion"
              << "\n";
    ex::print_separator('-', 8 + 8 * w);

    for (int i = 0; i < n_samples; ++i) {
        const auto& f = frames[static_cast<size_t>(i)];
        std::cout << std::fixed << std::setprecision(p)
                  << std::setw(8) << u_values(i)
                  << std::setw(w) << f.tangent.x()
                  << std::setw(w) << f.tangent.y()
                  << std::setw(w) << f.tangent.z()
                  << std::setw(w) << f.normal.x()
                  << std::setw(w) << f.normal.y()
                  << std::setw(w) << f.normal.z()
                  << std::setw(w) << f.curvature
                  << std::setw(w) << f.torsion
                  << "\n";
    }

    // Verify expected properties of a circle
    ex::print_separator('=');
    std::cout << "Circular Frenet frame properties:\n";
    ex::print_value("  Expected curvature (1/r)", 1.0 / r);
    ex::print_value("  Actual curvature (frame 0)", frames.front().curvature);
    ex::print_value("  Expected torsion", 0.0);
    ex::print_value("  Actual torsion (frame 0)", frames.front().torsion);
    ex::print_vector3("  Binormal (should be +Z or -Z)", frames.front().binormal);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    std::cout << "Path and Frenet Frame Examples\n";
    ex::print_separator('=');

    linear_path_example();
    circular_path_example();
    frenet_frame_helicoidal();
    frenet_frame_circular();

    std::cout << "\nAll path examples completed.\n";
    return 0;
}
