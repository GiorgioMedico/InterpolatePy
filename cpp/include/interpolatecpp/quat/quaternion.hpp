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
    [[nodiscard]] double w() const { return q_.w(); }
    [[nodiscard]] double x() const { return q_.x(); }
    [[nodiscard]] double y() const { return q_.y(); }
    [[nodiscard]] double z() const { return q_.z(); }
    [[nodiscard]] Eigen::Vector3d vec() const { return q_.vec(); }
    [[nodiscard]] const Eigen::Quaterniond& eigen() const { return q_; }

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
    [[nodiscard]] double norm() const;
    [[nodiscard]] double norm_squared() const;
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
    [[nodiscard]] std::pair<Eigen::Vector3d, double> to_axis_angle() const;

    // Conversion to Eigen::Quaterniond for concept conformance
    operator Eigen::Quaterniond() const { return q_; }

  private:
    Eigen::Quaterniond q_;

    static constexpr double kEpsilon = 1e-7;
};

}  // namespace interpolatecpp::quat
