#pragma once

#include <Eigen/Core>
#include <optional>
#include <span>

#include <interpolatecpp/config.hpp>

namespace interpolatecpp::spline {

/// Cubic smoothing spline with accuracy-smoothness trade-off.
///
/// Minimizes a weighted sum of waypoint error and acceleration magnitude.
/// mu=1 gives exact interpolation; mu->0 gives maximum smoothing.
class INTERPOLATECPP_API CubicSmoothingSpline {
  public:
    static constexpr int kMinPointsRequired = 2;
    static constexpr double kHighConditionThreshold = 1e12;
    static constexpr double kRegularizationFactor = 1e-8;

    /// @param t_points  Time points (strictly increasing)
    /// @param q_points  Position points
    /// @param mu        Smoothing parameter in (0, 1]. Default 0.5
    /// @param weights   Per-point weights (nullopt = uniform 1.0)
    /// @param v0        Initial velocity (default 0)
    /// @param vn        Final velocity (default 0)
    /// @param debug     Print debug info (default false)
    CubicSmoothingSpline(std::span<const double> t_points, std::span<const double> q_points,
                         double mu = 0.5,
                         std::optional<std::span<const double>> weights = std::nullopt,
                         double v0 = 0.0, double vn = 0.0, bool debug = false);

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
    [[nodiscard]] const Eigen::VectorXd& s_points() const noexcept { return s_; }
    [[nodiscard]] const Eigen::VectorXd& omega() const noexcept { return omega_; }
    [[nodiscard]] const Eigen::MatrixXd& coefficients() const noexcept { return coeffs_; }
    [[nodiscard]] double mu() const noexcept { return mu_; }
    [[nodiscard]] double lambda() const noexcept { return lambd_; }
    [[nodiscard]] int n_points() const noexcept { return n_; }

  private:
    Eigen::VectorXd t_;
    Eigen::VectorXd q_;
    Eigen::VectorXd s_;        // approximated positions
    Eigen::VectorXd omega_;    // accelerations
    Eigen::VectorXd w_;        // weights
    Eigen::VectorXd w_inv_;    // inverse weights
    Eigen::MatrixXd coeffs_;   // (n-1) x 4
    Eigen::MatrixXd a_matrix_; // tridiagonal A
    Eigen::MatrixXd c_matrix_; // C matrix

    double mu_;
    double lambd_;
    double v0_;
    double vn_;
    bool debug_;
    int n_; // number of points
    Eigen::VectorXd time_intervals_;

    void construct_matrices();
    void solve_system();
    void compute_positions();
    void compute_coefficients();

    struct SegmentInfo {
        int index;
        double tau;
    };
    [[nodiscard]] SegmentInfo find_segment(double t) const;
};

}  // namespace interpolatecpp::spline
