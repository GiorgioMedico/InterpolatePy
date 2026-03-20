#pragma once

#include <Eigen/Core>

#include <interpolatecpp/config.hpp>

namespace interpolatecpp::path {

/// Arc-length parameterized linear path in 3D space.
class INTERPOLATECPP_API LinearPath {
  public:
    LinearPath(const Eigen::Vector3d& pi, const Eigen::Vector3d& pf);

    [[nodiscard]] Eigen::Vector3d position(double s) const;
    [[nodiscard]] Eigen::MatrixXd position(const Eigen::VectorXd& s) const;
    [[nodiscard]] Eigen::Vector3d velocity(double s) const;
    [[nodiscard]] Eigen::Vector3d acceleration(double s) const;
    [[nodiscard]] double length() const { return length_; }

  private:
    Eigen::Vector3d pi_, pf_, tangent_;
    double length_;
};

}  // namespace interpolatecpp::path
