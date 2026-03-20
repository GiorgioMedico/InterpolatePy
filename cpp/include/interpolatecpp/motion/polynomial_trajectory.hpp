#pragma once

#include <Eigen/Core>
#include <vector>

#include <interpolatecpp/config.hpp>
#include <interpolatecpp/motion/motion_types.hpp>

namespace interpolatecpp::motion {

/// Polynomial trajectory generation supporting orders 3, 5, and 7.
///
/// Generates smooth trajectories using polynomial interpolation between
/// boundary conditions, with analytical derivative computation.
class INTERPOLATECPP_API PolynomialTrajectory {
  public:
    static constexpr int ORDER_3 = 3;
    static constexpr int ORDER_5 = 5;
    static constexpr int ORDER_7 = 7;

    /// Generate an order-3 (cubic) polynomial trajectory.
    /// Satisfies position and velocity boundary conditions.
    PolynomialTrajectory(const BoundaryCondition& bc_start, const BoundaryCondition& bc_end,
                         const TimeInterval& interval, int order);

    /// Evaluate trajectory at time t.
    [[nodiscard]] FullTrajectoryResult evaluate(double t) const;

    // Accessors
    [[nodiscard]] int order() const { return order_; }
    [[nodiscard]] double t_start() const { return t_start_; }
    [[nodiscard]] double t_end() const { return t_end_; }
    [[nodiscard]] double duration() const { return t_end_ - t_start_; }
    [[nodiscard]] const Eigen::VectorXd& coefficients() const { return coeffs_; }

    /// Compute heuristic velocities for intermediate waypoints.
    [[nodiscard]] static std::vector<double> heuristic_velocities(
        const std::vector<double>& points, const std::vector<double>& times);

    /// Create multipoint trajectory from waypoints.
    [[nodiscard]] static std::vector<PolynomialTrajectory> multipoint_trajectory(
        const std::vector<double>& points, const std::vector<double>& times,
        int order = ORDER_3, double v0 = 0.0, double vn = 0.0);

    /// Evaluate a multipoint trajectory at time t.
    [[nodiscard]] static FullTrajectoryResult evaluate_multipoint(
        const std::vector<PolynomialTrajectory>& segments, double t);

  private:
    int order_;
    double t_start_;
    double t_end_;
    Eigen::VectorXd coeffs_;

    void compute_order3(const BoundaryCondition& start, const BoundaryCondition& end,
                        double h);
    void compute_order5(const BoundaryCondition& start, const BoundaryCondition& end,
                        double h);
    void compute_order7(const BoundaryCondition& start, const BoundaryCondition& end,
                        double h);
};

}  // namespace interpolatecpp::motion
