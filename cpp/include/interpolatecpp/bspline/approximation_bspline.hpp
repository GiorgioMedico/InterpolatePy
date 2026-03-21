#pragma once

#include <Eigen/Core>
#include <optional>

#include <interpolatecpp/bspline/bspline.hpp>
#include <interpolatecpp/bspline/bspline_parameters.hpp>
#include <interpolatecpp/config.hpp>

namespace interpolatecpp::bspline {

/// Least-squares B-spline approximation of data points.
///
/// Fits a B-spline with fewer control points than data points,
/// exactly interpolating the endpoints and approximating internal points.
/// Follows "The NURBS Book" Section 8.5.
class INTERPOLATECPP_API ApproximationBSpline : public BSpline {
  public:
    /// Construct an approximation B-spline.
    ///
    /// @param points             Points to approximate (n x d matrix)
    /// @param num_control_points Number of control points to use
    /// @param degree             Degree (default 3 for cubic)
    /// @param weights            Weights for internal points (nullopt for uniform)
    /// @param method             Parameterization method
    /// @param debug              Print debug information
    ApproximationBSpline(const Eigen::MatrixXd& points, int num_control_points,
                         int degree = 3,
                         const std::optional<Eigen::VectorXd>& weights = std::nullopt,
                         Parameterization method = Parameterization::ChordLength,
                         bool debug = false);

    /// Calculate sum of squared approximation errors.
    [[nodiscard]] double calculate_approximation_error() const;

    // Accessors
    [[nodiscard]] const Eigen::MatrixXd& original_points() const noexcept { return original_points_; }
    [[nodiscard]] const Eigen::VectorXd& original_parameters() const noexcept {
        return original_parameters_;
    }

  private:
    Eigen::MatrixXd original_points_;
    Eigen::VectorXd original_parameters_;
    bool debug_;

    [[nodiscard]] static Eigen::VectorXd compute_parameters(
        const Eigen::MatrixXd& points, Parameterization method);
    [[nodiscard]] static Eigen::VectorXd compute_knots(int degree, int num_control_points,
                                                       int num_points,
                                                       const Eigen::VectorXd& u_bar);
    [[nodiscard]] Eigen::MatrixXd approximate_control_points(
        const Eigen::MatrixXd& points, int degree, const Eigen::VectorXd& knots,
        const Eigen::VectorXd& u_bar, int num_control_points,
        const Eigen::VectorXd& weights) const;
};

}  // namespace interpolatecpp::bspline
