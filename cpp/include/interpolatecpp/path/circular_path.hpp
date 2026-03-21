#pragma once

#include <Eigen/Core>

#include <interpolatecpp/config.hpp>

namespace interpolatecpp::path {

/// Arc-length parameterized circular path in 3D space.
class INTERPOLATECPP_API CircularPath {
  public:
    /// Construct from axis vector, point on axis, and point on circle.
    CircularPath(const Eigen::Vector3d& axis, const Eigen::Vector3d& axis_point,
                 const Eigen::Vector3d& circle_point);

    [[nodiscard]] Eigen::Vector3d position(double s) const;
    [[nodiscard]] Eigen::MatrixXd position(const Eigen::VectorXd& s) const;
    [[nodiscard]] Eigen::Vector3d velocity(double s) const;
    [[nodiscard]] Eigen::Vector3d acceleration(double s) const;
    [[nodiscard]] double radius() const noexcept { return radius_; }
    [[nodiscard]] const Eigen::Vector3d& center() const noexcept { return center_; }

  private:
    Eigen::Vector3d axis_, center_;
    Eigen::Matrix3d R_;  // Local-to-global rotation
    double radius_;
};

}  // namespace interpolatecpp::path
