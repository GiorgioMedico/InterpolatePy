#include <interpolatecpp/quat/log_quaternion_interpolation.hpp>

#include <cmath>
#include <stdexcept>

namespace interpolatecpp::quat {

Eigen::MatrixXd LogQuaternionInterpolation::recover_continuous_axis_angle() const {
    const int n = static_cast<int>(quaternions_.size());
    Eigen::MatrixXd aa_vectors(n, 3);

    // Working copy for double-cover handling
    std::vector<Quaternion> quats = quaternions_;

    // Ensure double-cover continuity
    for (int i = 1; i < n; ++i) {
        double dot_pos = quats[i - 1].dot_product(quats[i]);
        double dot_neg = quats[i - 1].dot_product(-quats[i]);
        if (dot_neg > dot_pos) {
            quats[i] = -quats[i];
        }
    }

    // Extract axis-angle and handle continuity
    std::vector<double> angles(n);
    std::vector<Eigen::Vector3d> axes(n);

    for (int i = 0; i < n; ++i) {
        auto [axis, angle] = quats[i].to_axis_angle();
        angles[i] = angle;
        axes[i] = axis;
    }

    // Axis continuity: flip axis if closer to -prev_axis
    for (int i = 1; i < n; ++i) {
        if ((axes[i - 1] - axes[i]).norm() > (axes[i - 1] + axes[i]).norm()) {
            angles[i] = -angles[i];
            axes[i] = -axes[i];
        }
    }

    // Phase unwrap angles
    for (int i = 1; i < n; ++i) {
        double diff = angles[i] - angles[i - 1];
        while (diff > M_PI) {
            angles[i] -= 2.0 * M_PI;
            diff -= 2.0 * M_PI;
        }
        while (diff < -M_PI) {
            angles[i] += 2.0 * M_PI;
            diff += 2.0 * M_PI;
        }
    }

    // Convert to axis-angle vectors: r = theta * axis
    for (int i = 0; i < n; ++i) {
        aa_vectors.row(i) = angles[i] * axes[i].transpose();
    }

    return aa_vectors;
}

LogQuaternionInterpolation::LogQuaternionInterpolation(
    const std::vector<double>& time_points, const std::vector<Quaternion>& quaternions,
    int degree, const std::optional<Eigen::VectorXd>& initial_velocity,
    const std::optional<Eigen::VectorXd>& final_velocity)
    : times_(time_points), quaternions_(quaternions) {
    if (time_points.size() != quaternions.size()) {
        throw std::invalid_argument("Time points and quaternions must have same length");
    }
    if (quaternions.size() < 2) {
        throw std::invalid_argument("Need at least 2 quaternions");
    }
    if (static_cast<int>(quaternions.size()) < degree + 1) {
        throw std::invalid_argument("Need at least degree+1 quaternions");
    }

    // Normalize quaternions
    for (auto& q : quaternions_) {
        q = q.unit();
    }

    // Recover continuous axis-angle representation
    Eigen::MatrixXd aa_vectors = recover_continuous_axis_angle();

    // Create time vector for B-spline
    Eigen::VectorXd times_eigen(times_.size());
    for (size_t i = 0; i < times_.size(); ++i) {
        times_eigen[i] = times_[i];
    }

    // Create B-spline interpolator in axis-angle space
    spline_ = std::make_unique<bspline::BSplineInterpolator>(
        degree, aa_vectors, times_eigen, initial_velocity, final_velocity);
}

Quaternion LogQuaternionInterpolation::evaluate(double t) const {
    t = std::clamp(t, times_.front(), times_.back());

    Eigen::VectorXd r = spline_->evaluate(t);

    // Convert axis-angle vector back to quaternion
    double theta = r.norm();
    if (theta < kEpsilon) {
        return Quaternion::identity();
    }

    Eigen::Vector3d axis = r / theta;
    return Quaternion::from_angle_axis(theta, axis);
}

Eigen::Vector3d LogQuaternionInterpolation::evaluate_velocity(double t) const {
    t = std::clamp(t, times_.front(), times_.back());
    Eigen::VectorXd dr = spline_->evaluate_derivative(t, 1);
    return Eigen::Vector3d(dr[0], dr[1], dr[2]);
}

Eigen::Vector3d LogQuaternionInterpolation::evaluate_acceleration(double t) const {
    t = std::clamp(t, times_.front(), times_.back());
    Eigen::VectorXd ddr = spline_->evaluate_derivative(t, 2);
    return Eigen::Vector3d(ddr[0], ddr[1], ddr[2]);
}

}  // namespace interpolatecpp::quat
