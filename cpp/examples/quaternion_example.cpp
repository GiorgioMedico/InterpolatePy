/// Quaternion interpolation example — C++ port of
///   examples/squad_c2_ex.py, examples/log_quat_new_ex.py,
///   examples/quat_visualization_ex.py
///
/// Demonstrates quaternion construction, SLERP vs SQUAD comparison,
/// C2-continuous SQUAD, logarithmic quaternion interpolation (LQI),
/// modified LQI (mLQI), and a side-by-side method comparison.

#include <interpolatecpp/quat/quaternion.hpp>
#include <interpolatecpp/quat/quaternion_spline.hpp>
#include <interpolatecpp/quat/squad_c2.hpp>
#include <interpolatecpp/quat/log_quaternion_interpolation.hpp>
#include <interpolatecpp/quat/modified_log_quaternion_interpolation.hpp>

#include "example_utils.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

namespace ex = interpolatecpp::examples;
using namespace interpolatecpp::quat;

// ---------------------------------------------------------------------------
// Shared test data
// ---------------------------------------------------------------------------

static constexpr double kRadToDeg = 180.0 / M_PI;

/// Five waypoints from euler angles (degrees) at uniform time spacing.
static std::pair<std::vector<double>, std::vector<Quaternion>> make_waypoints() {
    // Euler angles in degrees: (roll, pitch, yaw)
    //   (0, 0, 0)  (30, 45, 0)  (60, -30, 90)  (0, 45, 180)  (-30, 0, 270)
    const auto deg = [](double d) { return d * M_PI / 180.0; };

    std::vector<double> times = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<Quaternion> quats = {
        Quaternion::from_euler_angles(deg(0), deg(0), deg(0)),
        Quaternion::from_euler_angles(deg(30), deg(45), deg(0)),
        Quaternion::from_euler_angles(deg(60), deg(-30), deg(90)),
        Quaternion::from_euler_angles(deg(0), deg(45), deg(180)),
        Quaternion::from_euler_angles(deg(-30), deg(0), deg(270)),
    };
    return {times, quats};
}

// ---------------------------------------------------------------------------
// Helpers to format quaternion output
// ---------------------------------------------------------------------------

/// Print a quaternion as "(w, x, y, z)".
static void print_quat(const std::string& label, const Quaternion& q, int precision = 4) {
    std::cout << "  " << label << ": ("
              << std::fixed << std::setprecision(precision)
              << q.w() << ", " << q.x() << ", " << q.y() << ", " << q.z()
              << ")\n";
}

/// Print a quaternion's euler angles as "(roll, pitch, yaw) deg".
static void print_euler_deg(const std::string& label, const Quaternion& q, int precision = 2) {
    auto [roll, pitch, yaw] = q.to_euler_angles();
    std::cout << "  " << label << ": ("
              << std::fixed << std::setprecision(precision)
              << roll * kRadToDeg << ", "
              << pitch * kRadToDeg << ", "
              << yaw * kRadToDeg << ") deg\n";
}

/// Print both forms on one logical block.
static void print_quat_full(const std::string& label, const Quaternion& q) {
    print_quat(label, q);
    print_euler_deg(label + " euler", q);
}

// ---------------------------------------------------------------------------
// 1. Quaternion basics
// ---------------------------------------------------------------------------

static void quaternion_basics() {
    ex::print_header("1. Quaternion Basics");

    // Construction methods
    const Quaternion q_id = Quaternion::identity();
    print_quat_full("identity", q_id);

    const auto deg = [](double d) { return d * M_PI / 180.0; };
    const Quaternion q_euler = Quaternion::from_euler_angles(deg(30), deg(45), deg(0));
    print_quat_full("from_euler(30, 45, 0)", q_euler);

    const Quaternion q_aa = Quaternion::from_angle_axis(
        M_PI / 4.0, Eigen::Vector3d::UnitZ());
    print_quat_full("from_angle_axis(pi/4, Z)", q_aa);

    const Quaternion q_raw(0.707, 0.0, 0.707, 0.0);
    print_quat_full("Quaternion(0.707, 0, 0.707, 0)", q_raw);

    // Component access
    ex::print_separator();
    std::cout << "  Component access on q_euler:\n";
    ex::print_value("w", q_euler.w());
    ex::print_value("x", q_euler.x());
    ex::print_value("y", q_euler.y());
    ex::print_value("z", q_euler.z());
    ex::print_vector3("vec", q_euler.vec());
    ex::print_value("norm", q_euler.norm());

    // Arithmetic (immutable)
    ex::print_separator();
    std::cout << "  Quaternion arithmetic:\n";
    const Quaternion q_product = q_euler * q_aa;
    print_quat("q_euler * q_aa", q_product);

    const Quaternion q_conj = q_euler.conjugate();
    print_quat("q_euler.conjugate()", q_conj);

    const Quaternion q_inv = q_euler.inverse();
    print_quat("q_euler.inverse()", q_inv);

    const Quaternion q_unit = q_raw.unit();
    print_quat("q_raw.unit()", q_unit);
    ex::print_value("q_raw.unit().norm()", q_unit.norm());

    ex::print_value("q_euler.dot(q_aa)", q_euler.dot_product(q_aa));

    // Exp/log/power
    ex::print_separator();
    std::cout << "  Exp / log / power:\n";
    const Quaternion q_log = Quaternion::log(q_euler);
    print_quat("log(q_euler)", q_log);
    const Quaternion q_exp = Quaternion::exp(q_log);
    print_quat("exp(log(q_euler))", q_exp);
    const Quaternion q_pow = Quaternion::power(q_euler, 0.5);
    print_quat("power(q_euler, 0.5)", q_pow);

    // Conversions
    ex::print_separator();
    std::cout << "  Conversions:\n";
    const Eigen::Matrix3d rot = q_euler.to_rotation_matrix();
    ex::print_matrix("to_rotation_matrix()", rot);

    auto [axis, angle] = q_euler.to_axis_angle();
    ex::print_vector3("axis", axis);
    ex::print_value("angle (deg)", angle * kRadToDeg, 2);
}

// ---------------------------------------------------------------------------
// 2. SLERP vs SQUAD comparison
// ---------------------------------------------------------------------------

static void slerp_squad_comparison() {
    ex::print_header("2. SLERP vs SQUAD Comparison");

    auto [times, quats] = make_waypoints();

    const QuaternionSpline slerp_spline(times, quats, QuaternionSpline::Method::Slerp);
    const QuaternionSpline squad_spline(times, quats, QuaternionSpline::Method::Squad);

    std::cout << "  Waypoints (5 quaternions from euler angles):\n";
    for (size_t i = 0; i < quats.size(); ++i) {
        print_euler_deg("  q[" + std::to_string(i) + "]", quats[i]);
    }

    // Print euler angles at sample times
    const int n_samples = 12;
    const double t0 = times.front();
    const double t1 = times.back();

    std::cout << "\n  Interpolated euler angles (deg):\n\n";

    const int tw = 10;
    const int cw = 36;
    std::cout << std::right
              << std::setw(tw) << "Time"
              << std::setw(cw) << "SLERP (roll, pitch, yaw)"
              << std::setw(cw) << "SQUAD (roll, pitch, yaw)" << "\n";
    ex::print_separator('-', tw + 2 * cw);

    for (int i = 0; i <= n_samples; ++i) {
        const double t = t0 + (t1 - t0) * static_cast<double>(i) / n_samples;

        const Quaternion q_slerp = slerp_spline.evaluate(t);
        const Quaternion q_squad = squad_spline.evaluate(t);

        auto [sr, sp, sy] = q_slerp.to_euler_angles();
        auto [qr, qp, qy] = q_squad.to_euler_angles();

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(tw) << t
                  << "   (" << std::setw(8) << sr * kRadToDeg
                  << ", "   << std::setw(8) << sp * kRadToDeg
                  << ", "   << std::setw(8) << sy * kRadToDeg << ")"
                  << "   (" << std::setw(8) << qr * kRadToDeg
                  << ", "   << std::setw(8) << qp * kRadToDeg
                  << ", "   << std::setw(8) << qy * kRadToDeg << ")"
                  << "\n";
    }

    std::cout << "\n  Key: SLERP is C0 (linear per segment); SQUAD is C1 (smoother).\n";
}

// ---------------------------------------------------------------------------
// 3. SquadC2 — C2-continuous
// ---------------------------------------------------------------------------

static void squad_c2_example() {
    ex::print_header("3. SQUAD C2 — C2-Continuous Interpolation");

    auto [times, quats] = make_waypoints();
    const SquadC2 spline(times, quats);

    std::cout << "  SquadC2 constructed with " << quats.size() << " waypoints\n";
    std::cout << "  Time range: [" << spline.t_min() << ", " << spline.t_max() << "]\n\n";

    // Print velocity and acceleration norms to show C2 continuity
    const int n_samples = 16;
    const double t0 = spline.t_min();
    const double t1 = spline.t_max();

    const int tw = 10;
    const int cw = 18;
    std::cout << std::right
              << std::setw(tw) << "Time"
              << std::setw(cw) << "||vel||"
              << std::setw(cw) << "||accel||"
              << "\n";
    ex::print_separator('-', tw + 2 * cw);

    for (int i = 0; i <= n_samples; ++i) {
        const double t = t0 + (t1 - t0) * static_cast<double>(i) / n_samples;
        const Eigen::Vector3d vel = spline.evaluate_velocity(t);
        const Eigen::Vector3d acc = spline.evaluate_acceleration(t);

        std::cout << std::fixed << std::setprecision(4)
                  << std::setw(tw) << t
                  << std::setw(cw) << vel.norm()
                  << std::setw(cw) << acc.norm()
                  << "\n";
    }

    std::cout << "\n  Note: zero-clamped boundaries => vel norms are zero at t=0 and t=4.\n"
              << "  Smooth (continuous) velocity and acceleration confirm C2 continuity.\n";
}

// ---------------------------------------------------------------------------
// 4. Log-quaternion interpolation (LQI)
// ---------------------------------------------------------------------------

static void log_quaternion_example() {
    ex::print_header("4. Logarithmic Quaternion Interpolation (LQI)");

    auto [times, quats] = make_waypoints();
    const LogQuaternionInterpolation lqi(times, quats, /*degree=*/3);

    std::cout << "  LQI constructed with degree 3, "
              << quats.size() << " waypoints\n\n";

    const int n_samples = 12;
    const double t0 = lqi.t_min();
    const double t1 = lqi.t_max();

    const int tw = 10;
    const int ew = 36;
    const int vw = 14;
    std::cout << std::right
              << std::setw(tw) << "Time"
              << std::setw(ew) << "Euler (roll, pitch, yaw) deg"
              << std::setw(vw) << "||vel||"
              << "\n";
    ex::print_separator('-', tw + ew + vw);

    for (int i = 0; i <= n_samples; ++i) {
        const double t = t0 + (t1 - t0) * static_cast<double>(i) / n_samples;
        const Quaternion q = lqi.evaluate(t);
        const Eigen::Vector3d vel = lqi.evaluate_velocity(t);
        auto [r, p, y] = q.to_euler_angles();

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(tw) << t
                  << "   (" << std::setw(8) << r * kRadToDeg
                  << ", "   << std::setw(8) << p * kRadToDeg
                  << ", "   << std::setw(8) << y * kRadToDeg << ")"
                  << std::setprecision(4)
                  << std::setw(vw) << vel.norm()
                  << "\n";
    }

    std::cout << "\n  LQI uses axis-angle representation with B-spline interpolation.\n"
              << "  Provides C2 continuous trajectories in the tangent space.\n";
}

// ---------------------------------------------------------------------------
// 5. Modified log-quaternion interpolation (mLQI)
// ---------------------------------------------------------------------------

static void modified_log_quaternion_example() {
    ex::print_header("5. Modified Log-Quaternion Interpolation (mLQI)");

    auto [times, quats] = make_waypoints();
    const ModifiedLogQuaternionInterpolation mlqi(
        times, quats, /*degree=*/3, /*normalize_axis=*/true);

    std::cout << "  mLQI constructed with degree 3, normalize_axis=true, "
              << quats.size() << " waypoints\n\n";

    const int n_samples = 12;
    const double t0 = mlqi.t_min();
    const double t1 = mlqi.t_max();

    const int tw = 10;
    const int ew = 36;
    const int vw = 14;
    std::cout << std::right
              << std::setw(tw) << "Time"
              << std::setw(ew) << "Euler (roll, pitch, yaw) deg"
              << std::setw(vw) << "||vel||"
              << "\n";
    ex::print_separator('-', tw + ew + vw);

    for (int i = 0; i <= n_samples; ++i) {
        const double t = t0 + (t1 - t0) * static_cast<double>(i) / n_samples;
        const Quaternion q = mlqi.evaluate(t);
        const Eigen::Vector4d vel = mlqi.evaluate_velocity(t);
        auto [r, p, y] = q.to_euler_angles();

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(tw) << t
                  << "   (" << std::setw(8) << r * kRadToDeg
                  << ", "   << std::setw(8) << p * kRadToDeg
                  << ", "   << std::setw(8) << y * kRadToDeg << ")"
                  << std::setprecision(4)
                  << std::setw(vw) << vel.norm()
                  << "\n";
    }

    std::cout << "\n  mLQI decouples angle (theta) from unit axis (X, Y, Z).\n"
              << "  Provides improved numerical stability for complex trajectories.\n";
}

// ---------------------------------------------------------------------------
// 6. Side-by-side method comparison
// ---------------------------------------------------------------------------

static void method_comparison() {
    ex::print_header("6. Method Comparison — All 4 Methods");

    auto [times, quats] = make_waypoints();

    const QuaternionSpline slerp_sp(times, quats, QuaternionSpline::Method::Slerp);
    const QuaternionSpline squad_sp(times, quats, QuaternionSpline::Method::Squad);
    const SquadC2 sc2(times, quats);
    const LogQuaternionInterpolation lqi(times, quats, 3);

    const int n_samples = 10;
    const double t0 = times.front();
    const double t1 = times.back();

    // Table header
    const int tw = 8;
    const int mw = 30;
    std::cout << std::right
              << std::setw(tw) << "Time"
              << std::setw(mw) << "SLERP euler (deg)"
              << std::setw(mw) << "SQUAD euler (deg)"
              << std::setw(mw) << "SQUAD-C2 euler (deg)"
              << std::setw(mw) << "LQI euler (deg)"
              << "\n";
    ex::print_separator('-', tw + 4 * mw);

    for (int i = 0; i <= n_samples; ++i) {
        const double t = t0 + (t1 - t0) * static_cast<double>(i) / n_samples;

        const auto fmt = [&](const Quaternion& q) -> std::string {
            auto [r, p, y] = q.to_euler_angles();
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(1)
                << "(" << std::setw(6) << r * kRadToDeg
                << ", " << std::setw(6) << p * kRadToDeg
                << ", " << std::setw(6) << y * kRadToDeg << ")";
            return oss.str();
        };

        const Quaternion q_sl = slerp_sp.evaluate(t);
        const Quaternion q_sq = squad_sp.evaluate(t);
        const Quaternion q_c2 = sc2.evaluate(t);
        const Quaternion q_lq = lqi.evaluate(t);

        std::cout << std::fixed << std::setprecision(2)
                  << std::setw(tw) << t
                  << std::setw(mw) << fmt(q_sl)
                  << std::setw(mw) << fmt(q_sq)
                  << std::setw(mw) << fmt(q_c2)
                  << std::setw(mw) << fmt(q_lq)
                  << "\n";
    }

    std::cout << "\n  Summary of continuity guarantees:\n"
              << "    SLERP    — C0 (piecewise linear on S3)\n"
              << "    SQUAD    — C1 (spherical cubic, smooth velocity)\n"
              << "    SQUAD-C2 — C2 (quintic, smooth velocity + acceleration)\n"
              << "    LQI      — C2 (B-spline in log space)\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    std::cout << "Quaternion Interpolation Examples\n";
    ex::print_separator('=', 60);
    std::cout << "Demonstrates quaternion basics, SLERP/SQUAD comparison,\n"
              << "C2-continuous SQUAD, logarithmic interpolation (LQI),\n"
              << "modified LQI (mLQI), and side-by-side method comparison.\n";

    quaternion_basics();
    slerp_squad_comparison();
    squad_c2_example();
    log_quaternion_example();
    modified_log_quaternion_example();
    method_comparison();

    std::cout << "\n";
    ex::print_separator('=', 60);
    std::cout << "All quaternion interpolation examples completed.\n";

    return 0;
}
