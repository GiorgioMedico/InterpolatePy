#include <interpolatecpp/path/circular_path.hpp>

#include <Eigen/Geometry>
#include <cmath>
#include <stdexcept>

namespace interpolatecpp::path {

CircularPath::CircularPath(const Eigen::Vector3d& axis,
                           const Eigen::Vector3d& axis_point,
                           const Eigen::Vector3d& circle_point) {
    axis_ = axis.normalized();

    Eigen::Vector3d delta = circle_point - axis_point;

    if (std::abs(delta.dot(axis_)) >= delta.norm()) {
        throw std::invalid_argument("The point must not be on the circle axis");
    }

    // Center = projection of circle_point onto axis line
    center_ = axis_point + delta.dot(axis_) * axis_;
    radius_ = (circle_point - center_).norm();

    // Rotation matrix: local (x',y',z') -> global
    Eigen::Vector3d x_prime = (circle_point - center_) / radius_;
    Eigen::Vector3d z_prime = axis_;
    Eigen::Vector3d y_prime = z_prime.cross(x_prime);

    R_.col(0) = x_prime;
    R_.col(1) = y_prime;
    R_.col(2) = z_prime;
}

Eigen::Vector3d CircularPath::position(double s) const {
    double angle = s / radius_;
    Eigen::Vector3d p_local(radius_ * std::cos(angle), radius_ * std::sin(angle), 0.0);
    return center_ + R_ * p_local;
}

Eigen::MatrixXd CircularPath::position(const Eigen::VectorXd& s) const {
    Eigen::MatrixXd result(s.size(), 3);
    for (Eigen::Index i = 0; i < s.size(); ++i) {
        result.row(i) = position(s[i]).transpose();
    }
    return result;
}

Eigen::Vector3d CircularPath::velocity(double s) const {
    double angle = s / radius_;
    Eigen::Vector3d dp_local(-std::sin(angle), std::cos(angle), 0.0);
    return R_ * dp_local;
}

Eigen::Vector3d CircularPath::acceleration(double s) const {
    double angle = s / radius_;
    Eigen::Vector3d d2p_local(-std::cos(angle) / radius_, -std::sin(angle) / radius_,
                              0.0);
    return R_ * d2p_local;
}

}  // namespace interpolatecpp::path
