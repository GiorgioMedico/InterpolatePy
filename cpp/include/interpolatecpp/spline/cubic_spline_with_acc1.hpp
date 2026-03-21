#pragma once

#include <Eigen/Core>
#include <span>
#include <vector>

#include <interpolatecpp/config.hpp>

namespace interpolatecpp::spline {

/// Cubic spline with velocity AND acceleration constraints at endpoints.
///
/// Adds two extra points at midpoints of first/last segments to satisfy
/// acceleration boundary conditions (section 4.4.4 of the reference).
class INTERPOLATECPP_API CubicSplineWithAcceleration1 {
  public:
    /// @param t_points  Original time points (strictly increasing)
    /// @param q_points  Original position points
    /// @param v0        Initial velocity (default 0)
    /// @param vn        Final velocity (default 0)
    /// @param a0        Initial acceleration (default 0)
    /// @param an        Final acceleration (default 0)
    /// @param debug     Print debug info (default false)
    CubicSplineWithAcceleration1(std::span<const double> t_points,
                                 std::span<const double> q_points, double v0 = 0.0,
                                 double vn = 0.0, double a0 = 0.0, double an = 0.0,
                                 bool debug = false);

    // Evaluation (satisfies ScalarTrajectory concept)
    [[nodiscard]] double evaluate(double t) const;
    [[nodiscard]] Eigen::VectorXd evaluate(const Eigen::VectorXd& t) const;
    [[nodiscard]] double evaluate_velocity(double t) const;
    [[nodiscard]] Eigen::VectorXd evaluate_velocity(const Eigen::VectorXd& t) const;
    [[nodiscard]] double evaluate_acceleration(double t) const;
    [[nodiscard]] Eigen::VectorXd evaluate_acceleration(const Eigen::VectorXd& t) const;

    // Accessors
    [[nodiscard]] const Eigen::VectorXd& t_points() const noexcept { return t_; }
    [[nodiscard]] const Eigen::VectorXd& q_points() const noexcept { return q_; }
    [[nodiscard]] const Eigen::VectorXd& t_orig() const noexcept { return t_orig_; }
    [[nodiscard]] const Eigen::VectorXd& q_orig() const noexcept { return q_orig_; }
    [[nodiscard]] const Eigen::VectorXd& omega() const noexcept { return omega_; }
    [[nodiscard]] const Eigen::MatrixXd& coefficients() const noexcept { return coeffs_; }
    [[nodiscard]] const std::vector<int>& original_indices() const noexcept { return original_indices_; }
    [[nodiscard]] int n_points() const noexcept { return n_; }
    [[nodiscard]] int n_orig() const noexcept { return n_orig_; }

  private:
    Eigen::VectorXd t_orig_;
    Eigen::VectorXd q_orig_;
    Eigen::VectorXd t_;      // expanded with extra points
    Eigen::VectorXd q_;      // expanded with extra points
    Eigen::VectorXd T_;      // time intervals
    Eigen::VectorXd omega_;  // accelerations
    Eigen::MatrixXd coeffs_; // (n-1) x 4

    double v0_, vn_, a0_, an_;
    bool debug_;
    int n_orig_;
    int n_;

    std::vector<int> original_indices_;

    void add_extra_points();
    void solve_accelerations();
    void compute_coefficients();
    void compute_original_indices();

    struct SegmentInfo {
        int index;
        double tau;
    };
    [[nodiscard]] SegmentInfo find_segment(double t) const;
};

}  // namespace interpolatecpp::spline
