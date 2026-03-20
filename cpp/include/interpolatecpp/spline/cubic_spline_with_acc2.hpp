#pragma once

#include <Eigen/Core>
#include <optional>
#include <span>

#include <interpolatecpp/config.hpp>
#include <interpolatecpp/spline/cubic_spline.hpp>
#include <interpolatecpp/spline/spline_parameters.hpp>

namespace interpolatecpp::spline {

/// Cubic spline with quintic polynomial segments for acceleration constraints.
///
/// Extends CubicSpline: replaces first/last segments with 5th-degree polynomials
/// when acceleration constraints are provided (section 4.4.4 of the reference).
class INTERPOLATECPP_API CubicSplineWithAcceleration2 : public CubicSpline {
  public:
    using QuinticCoeffs = Eigen::Vector<double, 6>;

    CubicSplineWithAcceleration2(std::span<const double> t_points,
                                 std::span<const double> q_points,
                                 const SplineParameters& params = SplineParameters{});

    // Override evaluation to dispatch to quintic segments
    [[nodiscard]] double evaluate(double t) const;
    [[nodiscard]] Eigen::VectorXd evaluate(const Eigen::VectorXd& t) const;
    [[nodiscard]] double evaluate_velocity(double t) const;
    [[nodiscard]] Eigen::VectorXd evaluate_velocity(const Eigen::VectorXd& t) const;
    [[nodiscard]] double evaluate_acceleration(double t) const;
    [[nodiscard]] Eigen::VectorXd evaluate_acceleration(const Eigen::VectorXd& t) const;

    // Accessors
    [[nodiscard]] std::optional<double> a0() const { return a0_; }
    [[nodiscard]] std::optional<double> an() const { return an_; }
    [[nodiscard]] bool has_quintic_first() const { return quintic_first_.has_value(); }
    [[nodiscard]] bool has_quintic_last() const { return quintic_last_.has_value(); }

  private:
    std::optional<double> a0_;
    std::optional<double> an_;
    std::optional<QuinticCoeffs> quintic_first_;
    std::optional<QuinticCoeffs> quintic_last_;

    void replace_first_segment_with_quintic();
    void replace_last_segment_with_quintic();
};

}  // namespace interpolatecpp::spline
