#pragma once

#include <Eigen/Core>
#include <optional>
#include <span>

#include <interpolatecpp/bspline/bspline.hpp>
#include <interpolatecpp/bspline/bspline_parameters.hpp>
#include <interpolatecpp/config.hpp>

namespace interpolatecpp::bspline {

/// Cubic B-spline interpolation through a set of points.
///
/// Creates a degree-3 B-spline that passes through all specified points
/// with C2 continuity. Follows "The NURBS Book" Section 8.4.2.
class INTERPOLATECPP_API CubicBSplineInterpolation : public BSpline {
  public:
    /// Construct a cubic B-spline interpolation.
    ///
    /// @param points           Points to interpolate (n x d matrix)
    /// @param v0               Initial endpoint derivative (nullopt for zero or auto)
    /// @param vn               Final endpoint derivative (nullopt for zero or auto)
    /// @param method           Parameterization method
    /// @param auto_derivatives Whether to auto-calculate derivatives
    CubicBSplineInterpolation(
        const Eigen::MatrixXd& points,
        const std::optional<Eigen::VectorXd>& v0 = std::nullopt,
        const std::optional<Eigen::VectorXd>& vn = std::nullopt,
        Parameterization method = Parameterization::ChordLength,
        bool auto_derivatives = false);

    // Accessors
    [[nodiscard]] const Eigen::MatrixXd& interpolation_points() const {
        return interpolation_points_;
    }
    [[nodiscard]] const Eigen::VectorXd& u_bars() const { return u_bars_; }
    [[nodiscard]] const Eigen::VectorXd& start_derivative() const { return v0_; }
    [[nodiscard]] const Eigen::VectorXd& end_derivative() const { return vn_; }

  private:
    Eigen::MatrixXd interpolation_points_;
    Eigen::VectorXd u_bars_;
    Eigen::VectorXd v0_;
    Eigen::VectorXd vn_;

    static constexpr double kParamDiffThreshold = 1e-10;
    static constexpr int kMinPointsForTridiagonal = 2;

    [[nodiscard]] static Eigen::VectorXd calculate_parameters(
        const Eigen::MatrixXd& points, Parameterization method);
    [[nodiscard]] static Eigen::VectorXd calculate_knot_vector(const Eigen::VectorXd& u_bars,
                                                               int n_points);
    [[nodiscard]] Eigen::MatrixXd calculate_control_points(const Eigen::VectorXd& knots) const;
};

}  // namespace interpolatecpp::bspline
