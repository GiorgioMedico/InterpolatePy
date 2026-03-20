#pragma once

#include <Eigen/Core>
#include <memory>
#include <vector>

#include <interpolatecpp/bspline/bspline_interpolator.hpp>
#include <interpolatecpp/config.hpp>
#include <interpolatecpp/quat/quaternion.hpp>

namespace interpolatecpp::quat {

/// Modified Logarithmic Quaternion Interpolation (mLQI).
///
/// Interpolates quaternions as (theta, X, Y, Z) where X^2+Y^2+Z^2=1,
/// using separate B-spline interpolators for angle and axis components.
/// Based on Parker et al. (2023).
class INTERPOLATECPP_API ModifiedLogQuaternionInterpolation {
  public:
    /// Construct a modified log-quaternion interpolator.
    ///
    /// @param time_points       Strictly increasing time values
    /// @param quaternions       Unit quaternions at each time point
    /// @param degree            B-spline degree (3, 4, or 5)
    /// @param normalize_axis    Whether to normalize (X,Y,Z) after interpolation
    /// @param initial_velocity  Initial velocity constraint (4D: [theta_dot, X_dot, Y_dot, Z_dot])
    /// @param final_velocity    Final velocity constraint (4D)
    ModifiedLogQuaternionInterpolation(
        const std::vector<double>& time_points,
        const std::vector<Quaternion>& quaternions, int degree = 3,
        bool normalize_axis = true,
        const std::optional<Eigen::VectorXd>& initial_velocity = std::nullopt,
        const std::optional<Eigen::VectorXd>& final_velocity = std::nullopt);

    [[nodiscard]] Quaternion evaluate(double t) const;
    [[nodiscard]] Eigen::Vector4d evaluate_velocity(double t) const;
    [[nodiscard]] Eigen::Vector4d evaluate_acceleration(double t) const;

    [[nodiscard]] double t_min() const { return times_.front(); }
    [[nodiscard]] double t_max() const { return times_.back(); }
    [[nodiscard]] bool normalize_axis() const { return normalize_axis_; }

  private:
    std::vector<double> times_;
    std::vector<Quaternion> quaternions_;
    bool normalize_axis_;

    std::unique_ptr<bspline::BSplineInterpolator> theta_spline_;
    std::unique_ptr<bspline::BSplineInterpolator> xyz_spline_;

    static constexpr double kEpsilon = 1e-10;

    void ensure_quaternion_continuity();
    [[nodiscard]] std::pair<Eigen::VectorXd, Eigen::MatrixXd>
    transform_to_theta_xyz() const;
};

}  // namespace interpolatecpp::quat
