#pragma once

#include <Eigen/Core>
#include <optional>

#include <interpolatecpp/bspline/bspline.hpp>
#include <interpolatecpp/config.hpp>

namespace interpolatecpp::bspline {

/// B-spline interpolator supporting degrees 3, 4, and 5.
///
/// Computes knot vector and control points to interpolate given data points
/// with appropriate continuity (C2 for degree 3, C3 for degree 4, C4 for degree 5).
class INTERPOLATECPP_API BSplineInterpolator : public BSpline {
  public:
    /// Construct a B-spline interpolator.
    ///
    /// @param degree                Degree (3, 4, or 5)
    /// @param points                Points to interpolate (n x d matrix)
    /// @param times                 Time instants (nullopt for uniform spacing)
    /// @param initial_velocity      Initial velocity constraint
    /// @param final_velocity        Final velocity constraint
    /// @param initial_acceleration  Initial acceleration constraint
    /// @param final_acceleration    Final acceleration constraint
    /// @param cyclic                Use cyclic (periodic) conditions
    BSplineInterpolator(
        int degree, const Eigen::MatrixXd& points,
        const std::optional<Eigen::VectorXd>& times = std::nullopt,
        const std::optional<Eigen::VectorXd>& initial_velocity = std::nullopt,
        const std::optional<Eigen::VectorXd>& final_velocity = std::nullopt,
        const std::optional<Eigen::VectorXd>& initial_acceleration = std::nullopt,
        const std::optional<Eigen::VectorXd>& final_acceleration = std::nullopt,
        bool cyclic = false);

    // Accessors
    [[nodiscard]] const Eigen::MatrixXd& interp_points() const { return interp_points_; }
    [[nodiscard]] const Eigen::VectorXd& times() const { return times_; }

  private:
    Eigen::MatrixXd interp_points_;
    Eigen::VectorXd times_;
    std::optional<Eigen::VectorXd> initial_velocity_;
    std::optional<Eigen::VectorXd> final_velocity_;
    std::optional<Eigen::VectorXd> initial_acceleration_;
    std::optional<Eigen::VectorXd> final_acceleration_;
    bool cyclic_;

    static constexpr double kConditionThreshold = 1e12;
    static constexpr double kRegularizationEps = 1e-10;

    [[nodiscard]] static Eigen::VectorXd create_knot_vector(int degree,
                                                            const Eigen::MatrixXd& points,
                                                            const Eigen::VectorXd& times);
    [[nodiscard]] Eigen::MatrixXd compute_control_points(int degree,
                                                         const Eigen::MatrixXd& points,
                                                         const Eigen::VectorXd& times,
                                                         const Eigen::VectorXd& knots) const;
};

}  // namespace interpolatecpp::bspline
