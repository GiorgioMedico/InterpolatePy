#pragma once

#include <Eigen/Core>
#include <span>

#include <interpolatecpp/config.hpp>

namespace interpolatecpp::spline {

/// C2-continuous cubic spline trajectory planning.
///
/// Generates a smooth trajectory passing through specified waypoints
/// with continuous position, velocity, and acceleration profiles.
/// Polynomial per segment: q(tau) = a0 + a1*tau + a2*tau^2 + a3*tau^3
class INTERPOLATECPP_API CubicSpline {
  public:
    /// Construct a cubic spline through the given waypoints.
    ///
    /// @param t_points  Time points (must be strictly increasing)
    /// @param q_points  Position points (same length as t_points)
    /// @param v0        Initial velocity (default 0)
    /// @param vn        Final velocity (default 0)
    /// @param debug     Print debug info (default false)
    /// @throws std::invalid_argument if lengths mismatch or times not increasing
    CubicSpline(std::span<const double> t_points, std::span<const double> q_points,
                double v0 = 0.0, double vn = 0.0, bool debug = false);

    virtual ~CubicSpline() = default;

    // Evaluation (satisfies ScalarTrajectory concept)
    [[nodiscard]] double evaluate(double t) const;
    [[nodiscard]] Eigen::VectorXd evaluate(const Eigen::VectorXd& t) const;

    [[nodiscard]] double evaluate_velocity(double t) const;
    [[nodiscard]] Eigen::VectorXd evaluate_velocity(const Eigen::VectorXd& t) const;

    [[nodiscard]] double evaluate_acceleration(double t) const;
    [[nodiscard]] Eigen::VectorXd evaluate_acceleration(const Eigen::VectorXd& t) const;

    // Accessors
    [[nodiscard]] const Eigen::VectorXd& t_points() const noexcept { return t_points_; }
    [[nodiscard]] const Eigen::VectorXd& q_points() const noexcept { return q_points_; }
    [[nodiscard]] const Eigen::VectorXd& t_intervals() const noexcept { return t_intervals_; }
    [[nodiscard]] const Eigen::VectorXd& velocities() const noexcept { return velocities_; }
    [[nodiscard]] const Eigen::MatrixXd& coefficients() const noexcept { return coefficients_; }
    [[nodiscard]] int n_segments() const noexcept { return n_; }

  protected:
    Eigen::VectorXd t_points_;
    Eigen::VectorXd q_points_;
    double v0_;
    double vn_;
    bool debug_;
    int n_;  // number of segments
    Eigen::VectorXd t_intervals_;
    Eigen::VectorXd velocities_;
    Eigen::MatrixXd coefficients_;  // (n_ x 4)

    virtual void compute_velocities();
    virtual void compute_coefficients();

    struct SegmentInfo {
        int index;
        double tau;
    };

    /// Find segment containing time t and local parameter tau.
    [[nodiscard]] SegmentInfo find_segment(double t) const;
};

}  // namespace interpolatecpp::spline
