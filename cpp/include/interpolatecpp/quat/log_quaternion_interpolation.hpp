#pragma once

#include <Eigen/Core>
#include <memory>
#include <vector>

#include <interpolatecpp/bspline/bspline_interpolator.hpp>
#include <interpolatecpp/config.hpp>
#include <interpolatecpp/quat/quaternion.hpp>

namespace interpolatecpp::quat {

/// Logarithmic quaternion interpolation (Parker Algorithm 1).
///
/// Transforms quaternions to axis-angle space, interpolates with B-splines,
/// and maps back. Handles double-cover and axis continuity.
class INTERPOLATECPP_API LogQuaternionInterpolation {
  public:
    LogQuaternionInterpolation(
        const std::vector<double>& time_points,
        const std::vector<Quaternion>& quaternions, int degree = 3,
        const std::optional<Eigen::VectorXd>& initial_velocity = std::nullopt,
        const std::optional<Eigen::VectorXd>& final_velocity = std::nullopt);

    [[nodiscard]] Quaternion evaluate(double t) const;
    [[nodiscard]] Eigen::Vector3d evaluate_velocity(double t) const;
    [[nodiscard]] Eigen::Vector3d evaluate_acceleration(double t) const;

    [[nodiscard]] double t_min() const { return times_.front(); }
    [[nodiscard]] double t_max() const { return times_.back(); }

  private:
    std::vector<double> times_;
    std::vector<Quaternion> quaternions_;
    std::unique_ptr<bspline::BSplineInterpolator> spline_;

    static constexpr double kEpsilon = 1e-10;

    [[nodiscard]] Eigen::MatrixXd recover_continuous_axis_angle() const;
};

}  // namespace interpolatecpp::quat
