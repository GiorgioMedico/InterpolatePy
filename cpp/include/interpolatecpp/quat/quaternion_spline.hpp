#pragma once

#include <Eigen/Core>
#include <string>
#include <vector>

#include <interpolatecpp/config.hpp>
#include <interpolatecpp/quat/quaternion.hpp>

namespace interpolatecpp::quat {

/// Quaternion spline interpolation with SLERP/SQUAD/Auto methods.
class INTERPOLATECPP_API QuaternionSpline {
  public:
    enum class Method { Slerp, Squad, Auto };

    QuaternionSpline(const std::vector<double>& time_points,
                     const std::vector<Quaternion>& quaternions,
                     Method method = Method::Auto);

    [[nodiscard]] Quaternion evaluate(double t) const;
    [[nodiscard]] Eigen::Vector3d evaluate_velocity(double t) const;
    [[nodiscard]] Eigen::Vector3d evaluate_acceleration(double t) const;

    [[nodiscard]] double t_min() const noexcept { return times_.front(); }
    [[nodiscard]] double t_max() const noexcept { return times_.back(); }

  private:
    std::vector<double> times_;
    std::vector<Quaternion> quaternions_;
    std::vector<Quaternion> intermediates_;
    Method method_;

    static constexpr double kDt = 1e-6;

    void compute_intermediates();
    [[nodiscard]] int find_segment(double t) const;
};

}  // namespace interpolatecpp::quat
