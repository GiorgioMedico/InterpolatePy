#pragma once

#include <Eigen/Core>
#include <vector>

#include <interpolatecpp/config.hpp>
#include <interpolatecpp/quat/quaternion.hpp>

namespace interpolatecpp::quat {

/// C2-continuous SQUAD quaternion interpolation (Wittmann et al.).
///
/// Uses quintic polynomial parameterization for zero-clamped boundary
/// conditions, ensuring continuous angular velocity and acceleration.
class INTERPOLATECPP_API SquadC2 {
  public:
    SquadC2(const std::vector<double>& time_points,
            const std::vector<Quaternion>& quaternions,
            bool normalize_quaternions = true);

    [[nodiscard]] Quaternion evaluate(double t) const;
    [[nodiscard]] Eigen::Vector3d evaluate_velocity(double t) const;
    [[nodiscard]] Eigen::Vector3d evaluate_acceleration(double t) const;

    [[nodiscard]] double t_min() const { return times_.front(); }
    [[nodiscard]] double t_max() const { return times_.back(); }

  private:
    std::vector<double> times_;
    std::vector<Quaternion> quaternions_;

    // Extended sequence (with virtual waypoints)
    std::vector<double> ext_times_;
    std::vector<Quaternion> ext_quats_;
    std::vector<Quaternion> ext_intermediates_;

    static constexpr double kDt = 1e-6;

    void build_extended_sequence();
    void compute_intermediates();
    [[nodiscard]] int find_segment(double t) const;
    [[nodiscard]] static double quintic_u(double t, double t0, double t1);
};

}  // namespace interpolatecpp::quat
