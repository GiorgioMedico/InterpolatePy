#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <interpolatecpp/config.hpp>

namespace interpolatecpp::quat {

/// Quaternion class with exp/log/power/slerp/squad operations.
///
/// Stores as [w, x, y, z] (scalar-first convention).
/// Wraps Eigen::Quaterniond but adds mathematical operations needed
/// for trajectory interpolation that Eigen doesn't provide.
class INTERPOLATECPP_API Quaternion {
  public:
    /// Construct from components (scalar-first).
    explicit Quaternion(double w = 1.0, double x = 0.0, double y = 0.0, double z = 0.0);

    /// Construct from Eigen quaternion.
    explicit Quaternion(const Eigen::Quaterniond& eq);

    // Factory methods
    [[nodiscard]] static Quaternion identity();
    [[nodiscard]] static Quaternion from_angle_axis(double angle,
                                                    const Eigen::Vector3d& axis);
    [[nodiscard]] static Quaternion from_euler_angles(double roll, double pitch,
                                                      double yaw);

    // Component access
    [[nodiscard]] double w() const noexcept { return q_.w(); }
    [[nodiscard]] double x() const noexcept { return q_.x(); }
    [[nodiscard]] double y() const noexcept { return q_.y(); }
    [[nodiscard]] double z() const noexcept { return q_.z(); }
    [[nodiscard]] Eigen::Vector3d vec() const noexcept { return q_.vec(); }
    [[nodiscard]] const Eigen::Quaterniond& eigen() const noexcept { return q_; }

    // Arithmetic (immutable - returns new Quaternion)
    [[nodiscard]] Quaternion operator*(const Quaternion& other) const;
    [[nodiscard]] Quaternion operator*(double scalar) const;
    [[nodiscard]] Quaternion operator+(const Quaternion& other) const;
    [[nodiscard]] Quaternion operator-(const Quaternion& other) const;
    [[nodiscard]] Quaternion operator-() const;

    // Core operations
    [[nodiscard]] Quaternion conjugate() const;
    [[nodiscard]] Quaternion inverse() const;
    [[nodiscard]] Quaternion unit() const;
    [[nodiscard]] double norm() const noexcept;
    [[nodiscard]] double norm_squared() const noexcept;
    [[nodiscard]] double dot_product(const Quaternion& other) const;

    // Exponential/logarithmic maps
    [[nodiscard]] static Quaternion exp(const Quaternion& q);
    [[nodiscard]] static Quaternion log(const Quaternion& q);
    [[nodiscard]] static Quaternion power(const Quaternion& q, double t);

    // Interpolation
    [[nodiscard]] static Quaternion slerp(const Quaternion& q0, const Quaternion& q1,
                                          double t);
    [[nodiscard]] static Quaternion slerp_prime(const Quaternion& q0,
                                                const Quaternion& q1, double t);
    [[nodiscard]] static Quaternion squad(const Quaternion& p, const Quaternion& a,
                                          const Quaternion& b, const Quaternion& q,
                                          double t);
    [[nodiscard]] static Quaternion compute_intermediate_quaternion(
        const Quaternion& q_prev, const Quaternion& q_curr, const Quaternion& q_next);

    // Conversions
    [[nodiscard]] Eigen::Matrix3d to_rotation_matrix() const;
    [[nodiscard]] Eigen::Matrix4d to_transformation_matrix() const;
    [[nodiscard]] std::pair<Eigen::Vector3d, double> to_axis_angle() const;
    [[nodiscard]] std::tuple<double, double, double> to_euler_angles() const;

    // Factory: construct from rotation matrix (3x3)
    [[nodiscard]] static Quaternion from_rotation_matrix(
        const Eigen::Matrix3d& rotation_matrix);

    // Dynamics
    /// E-matrix for quaternion kinematics.  sign=0 → base frame, sign=1 → body frame.
    [[nodiscard]] Eigen::Matrix3d E(int sign) const;

    /// Quaternion time derivative  qdot = 0.5 * E(sign) * omega.
    [[nodiscard]] Quaternion dot(const Eigen::Vector3d& omega, int sign) const;

    /// Extract angular velocity from q and qdot.
    [[nodiscard]] static Eigen::Vector3d Omega(const Quaternion& q,
                                                const Quaternion& q_dot);

    // Conversion to Eigen::Quaterniond for concept conformance
    operator Eigen::Quaterniond() const noexcept { return q_; }

  private:
    Eigen::Quaterniond q_;

    static constexpr double kEpsilon = 1e-7;
};

}  // namespace interpolatecpp::quat
