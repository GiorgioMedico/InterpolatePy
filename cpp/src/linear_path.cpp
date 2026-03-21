#include <interpolatecpp/path/linear_path.hpp>

#include <algorithm>

namespace interpolatecpp::path {

LinearPath::LinearPath(const Eigen::Vector3d& pi, const Eigen::Vector3d& pf)
    : pi_(pi), pf_(pf) {
    length_ = (pf_ - pi_).norm();
    if (length_ > 0) {
        tangent_ = (pf_ - pi_) / length_;
    } else {
        tangent_ = Eigen::Vector3d::Zero();
    }
}

Eigen::Vector3d LinearPath::position(double s) const {
    s = std::clamp(s, 0.0, length_);
    return (length_ > 0) ? pi_ + (s / length_) * (pf_ - pi_) : pi_;
}

Eigen::MatrixXd LinearPath::position(const Eigen::VectorXd& s) const {
    Eigen::MatrixXd result(s.size(), 3);
    for (Eigen::Index i = 0; i < s.size(); ++i) {
        result.row(i) = position(s[i]).transpose();
    }
    return result;
}

Eigen::Vector3d LinearPath::velocity(double /*s*/) const { return tangent_; }

Eigen::Vector3d LinearPath::acceleration(double /*s*/) const {
    return Eigen::Vector3d::Zero();
}

}  // namespace interpolatecpp::path
