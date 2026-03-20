#pragma once

#include <Eigen/Core>

#include <interpolatecpp/bspline/bspline.hpp>
#include <interpolatecpp/bspline/bspline_parameters.hpp>
#include <interpolatecpp/config.hpp>

namespace interpolatecpp::bspline {

/// Smoothing cubic B-spline approximation.
///
/// Creates a cubic B-spline that balances fitting accuracy with smoothness,
/// controlled by the smoothing parameter mu in [0, 1].
/// Implements the Tikhonov regularization: (B^T W B + lambda C^T A C) P = B^T W Q
/// as described in "The NURBS Book" Section 8.7.
class INTERPOLATECPP_API SmoothingCubicBSpline : public BSpline {
  public:
    /// Construct a smoothing cubic B-spline.
    ///
    /// @param points Points to approximate (n x d matrix)
    /// @param params Configuration parameters (nullopt for defaults)
    SmoothingCubicBSpline(const Eigen::MatrixXd& points,
                          const BSplineParams& params = BSplineParams{});

    /// Calculate per-point approximation errors.
    [[nodiscard]] Eigen::VectorXd calculate_approximation_error() const;

    /// Calculate total weighted approximation error.
    [[nodiscard]] double calculate_total_error() const;

    // Accessors
    [[nodiscard]] const Eigen::MatrixXd& approximation_points() const {
        return approximation_points_;
    }
    [[nodiscard]] const Eigen::VectorXd& u_bars() const { return u_bars_; }
    [[nodiscard]] double mu() const { return mu_; }
    [[nodiscard]] double lambda_param() const { return lambda_param_; }

  private:
    Eigen::MatrixXd approximation_points_;
    Eigen::VectorXd u_bars_;
    Eigen::VectorXd weights_;
    double mu_;
    double lambda_param_;
    bool enforce_endpoints_;
    Eigen::VectorXd v0_;
    Eigen::VectorXd vn_;

    static constexpr double kEpsilon = 1e-10;

    [[nodiscard]] static Eigen::VectorXd calculate_parameters(
        const Eigen::MatrixXd& points, Parameterization method);
    [[nodiscard]] Eigen::VectorXd calculate_knot_vector() const;
    [[nodiscard]] Eigen::MatrixXd construct_b_matrix() const;
    [[nodiscard]] Eigen::MatrixXd construct_a_matrix() const;
    [[nodiscard]] Eigen::MatrixXd construct_c_matrix() const;
    [[nodiscard]] Eigen::MatrixXd calculate_control_points_impl() const;
    [[nodiscard]] Eigen::MatrixXd calculate_control_points_with_endpoints() const;
};

}  // namespace interpolatecpp::bspline
