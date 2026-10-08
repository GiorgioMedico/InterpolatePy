#include <interpolatecpp/quat/modified_log_quaternion_interpolation.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace interpolatecpp::quat {

void ModifiedLogQuaternionInterpolation::ensure_quaternion_continuity() {
    for (size_t i = 1; i < quaternions_.size(); ++i) {
        double dot_pos = quaternions_[i - 1].dot_product(quaternions_[i]);
        double dot_neg = quaternions_[i - 1].dot_product(-quaternions_[i]);
        if (dot_neg > dot_pos) {
            quaternions_[i] = -quaternions_[i];
        }
    }
}

namespace {

// Unlike to_axis_angle(), keeps the sign of w so the angle spans [0, 2*pi].
std::pair<Eigen::Vector3d, double> axis_angle_raw(const Quaternion& q, double eps) {
    const double s = q.w();
    const double sin_half = std::sqrt(std::max(0.0, 1.0 - s * s));
    const Eigen::Vector3d axis =
        sin_half < eps ? Eigen::Vector3d::UnitX() : Eigen::Vector3d(q.vec() / sin_half);
    return {axis, 2.0 * std::acos(std::clamp(s, -1.0, 1.0))};
}

}  // namespace

std::pair<Eigen::VectorXd, Eigen::MatrixXd>
ModifiedLogQuaternionInterpolation::transform_to_theta_xyz() const {
    const int n = static_cast<int>(quaternions_.size());
    Eigen::VectorXd theta_values(n);
    Eigen::MatrixXd xyz_values(n, 3);

    for (int i = 0; i < n; ++i) {
        auto [axis, angle] = axis_angle_raw(quaternions_[i], kEpsilon);
        // Keep the axis on the previous sample's side so theta stays on one branch.
        if (i > 0 && axis.dot(xyz_values.row(i - 1).transpose()) < 0.0) {
            axis = -axis;
            angle = 2.0 * std::numbers::pi - angle;
        }
        theta_values[i] = angle;
        xyz_values.row(i) = axis.transpose();
    }

    // Same algorithm as np.unwrap, so jumps of exactly pi resolve identically.
    constexpr double pi = std::numbers::pi;
    double correction = 0.0;
    double prev_raw = theta_values[0];
    for (int i = 1; i < n; ++i) {
        const double raw = theta_values[i];
        const double dd = raw - prev_raw;
        prev_raw = raw;
        if (std::abs(dd) >= pi) {
            double dd_mod = dd + pi - 2.0 * pi * std::floor((dd + pi) / (2.0 * pi)) - pi;
            if (dd_mod == -pi && dd > 0.0) {
                dd_mod = pi;
            }
            correction += dd_mod - dd;
        }
        theta_values[i] = raw + correction;
    }

    return {theta_values, xyz_values};
}

ModifiedLogQuaternionInterpolation::ModifiedLogQuaternionInterpolation(
    const std::vector<double>& time_points, const std::vector<Quaternion>& quaternions,
    int degree, bool normalize_axis,
    const std::optional<Eigen::VectorXd>& initial_velocity,
    const std::optional<Eigen::VectorXd>& final_velocity)
    : times_(time_points), quaternions_(quaternions), normalize_axis_(normalize_axis) {
    if (time_points.size() != quaternions.size()) {
        throw std::invalid_argument("Time points and quaternions must have same length");
    }
    if (quaternions.size() < 2) {
        throw std::invalid_argument("Need at least 2 quaternions");
    }
    if (degree < 3 || degree > 5) {
        throw std::invalid_argument("Degree must be 3, 4, or 5");
    }
    for (size_t i = 1; i < time_points.size(); ++i) {
        if (time_points[i] <= time_points[i - 1]) {
            throw std::invalid_argument("Time points must be strictly increasing");
        }
    }

    // Normalize quaternions
    for (auto& q : quaternions_) {
        q = q.unit();
    }

    // Handle double-cover for continuity
    ensure_quaternion_continuity();

    // Transform to (theta, X, Y, Z) representation
    auto [theta_values, xyz_values] = transform_to_theta_xyz();

    // Create time vector
    Eigen::VectorXd times_eigen(times_.size());
    for (size_t i = 0; i < times_.size(); ++i) {
        times_eigen[static_cast<Eigen::Index>(i)] = times_[i];
    }

    // Split velocity constraints if provided
    std::optional<Eigen::VectorXd> theta_init_vel = std::nullopt;
    std::optional<Eigen::VectorXd> xyz_init_vel = std::nullopt;
    std::optional<Eigen::VectorXd> theta_final_vel = std::nullopt;
    std::optional<Eigen::VectorXd> xyz_final_vel = std::nullopt;

    if (initial_velocity.has_value()) {
        const auto& iv = initial_velocity.value();
        if (iv.size() != 4) {
            throw std::invalid_argument("initial_velocity must have exactly 4 elements");
        }
        theta_init_vel = Eigen::VectorXd::Constant(1, iv[0]);
        xyz_init_vel = iv.segment(1, 3);
    }

    if (final_velocity.has_value()) {
        const auto& fv = final_velocity.value();
        if (fv.size() != 4) {
            throw std::invalid_argument("final_velocity must have exactly 4 elements");
        }
        theta_final_vel = Eigen::VectorXd::Constant(1, fv[0]);
        xyz_final_vel = fv.segment(1, 3);
    }

    // Create separate B-spline interpolators for theta (1D) and XYZ (3D)
    Eigen::MatrixXd theta_matrix = theta_values.reshaped(theta_values.size(), 1);

    theta_spline_ = std::make_unique<bspline::BSplineInterpolator>(
        degree, theta_matrix, times_eigen, theta_init_vel, theta_final_vel);

    xyz_spline_ = std::make_unique<bspline::BSplineInterpolator>(
        degree, xyz_values, times_eigen, xyz_init_vel, xyz_final_vel);
}

Quaternion ModifiedLogQuaternionInterpolation::evaluate(double t) const {
    t = std::clamp(t, times_.front(), times_.back());

    // Boundary cases
    if (std::abs(t - times_.front()) <= kEpsilon) {
        return quaternions_.front();
    }
    if (std::abs(t - times_.back()) <= kEpsilon) {
        return quaternions_.back();
    }

    // Evaluate both interpolators
    double theta = theta_spline_->evaluate(t)[0];
    Eigen::VectorXd xyz_vec = xyz_spline_->evaluate(t);
    Eigen::Vector3d xyz(xyz_vec[0], xyz_vec[1], xyz_vec[2]);

    // Optionally normalize the axis
    if (normalize_axis_) {
        double norm_xyz = xyz.norm();
        if (norm_xyz > kEpsilon) {
            xyz /= norm_xyz;
        } else {
            xyz = Eigen::Vector3d::UnitX();
        }
    }

    // Convert back to quaternion: q = [cos(theta/2), sin(theta/2) * axis]
    if (std::abs(theta) < kEpsilon) {
        return Quaternion::identity();
    }

    double cos_half = std::cos(theta / 2.0);
    double sin_half = std::sin(theta / 2.0);

    return Quaternion(cos_half, sin_half * xyz[0], sin_half * xyz[1], sin_half * xyz[2]);
}

Eigen::Vector4d ModifiedLogQuaternionInterpolation::evaluate_velocity(double t) const {
    t = std::clamp(t, times_.front(), times_.back());

    double theta_dot = theta_spline_->evaluate_derivative(t, 1)[0];
    Eigen::VectorXd xyz_dot = xyz_spline_->evaluate_derivative(t, 1);

    return Eigen::Vector4d(theta_dot, xyz_dot[0], xyz_dot[1], xyz_dot[2]);
}

Eigen::Vector4d ModifiedLogQuaternionInterpolation::evaluate_acceleration(double t) const {
    t = std::clamp(t, times_.front(), times_.back());

    double theta_ddot = theta_spline_->evaluate_derivative(t, 2)[0];
    Eigen::VectorXd xyz_ddot = xyz_spline_->evaluate_derivative(t, 2);

    return Eigen::Vector4d(theta_ddot, xyz_ddot[0], xyz_ddot[1], xyz_ddot[2]);
}

Eigen::Vector3d ModifiedLogQuaternionInterpolation::angular_velocity(double t) const {
    t = std::clamp(t, times_.front(), times_.back());

    const double theta = theta_spline_->evaluate(t)[0];
    const double theta_dot = theta_spline_->evaluate_derivative(t, 1)[0];
    const Eigen::Vector3d u_raw = xyz_spline_->evaluate(t);
    const Eigen::Vector3d u_dot_raw = xyz_spline_->evaluate_derivative(t, 1);

    Eigen::Vector3d u = u_raw;
    Eigen::Vector3d u_dot = u_dot_raw;
    if (normalize_axis_) {
        const double r = u_raw.norm();
        if (r < kEpsilon) {
            u = Eigen::Vector3d::UnitX();
            u_dot.setZero();
        } else {
            // Project out the radial component so u_dot is tangent to the unit sphere.
            u = u_raw / r;
            u_dot = (u_dot_raw - u.dot(u_dot_raw) * u) / r;
        }
    }

    return u * theta_dot + u_dot * std::sin(theta) +
           u.cross(u_dot) * (1.0 - std::cos(theta));
}

}  // namespace interpolatecpp::quat
