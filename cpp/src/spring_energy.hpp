#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace interpolatecpp::quat::detail {

/// Fixed stencil for one grid; rejected trials do not accumulate gradients.
class SpringEnergy {
  public:
    using Frame = Eigen::Vector4d;
    using Frames = std::vector<Frame>;

    SpringEnergy(std::size_t count, const std::vector<double>& weights,
                 double norm_penalty, const std::vector<std::size_t>& indices = {})
        : weights_(weights), norm_penalty_(norm_penalty), coefficients_(count - 2) {
        const double mean = indices.empty() ? 1.0
            : static_cast<double>(indices.back() - indices.front()) /
              static_cast<double>(indices.size() - 1);
        for (std::size_t index = 1; index + 1 < count; ++index) {
            const double h_left = indices.empty() ? 1.0
                : static_cast<double>(indices[index] - indices[index - 1]) / mean;
            const double h_right = indices.empty() ? 1.0
                : static_cast<double>(indices[index + 1] - indices[index]) / mean;
            const double left = 2.0 / (h_left * (h_left + h_right));
            const double right = 2.0 / (h_right * (h_left + h_right));
            coefficients_[index - 1] = Eigen::Vector3d(left, -(left + right), right);
        }
    }

    [[nodiscard]] const std::vector<Eigen::Vector3d>& coefficients() const { return coefficients_; }

    [[nodiscard]] double energy(const Frames& frames) const {
        Frames unused;
        return evaluate<false>(frames, unused);
    }

    double energy_gradient(const Frames& frames, Frames& gradient) const {
        gradient.resize(frames.size());
        std::fill(gradient.begin(), gradient.end(), Frame::Zero());
        return evaluate<true>(frames, gradient);
    }

  private:
    const std::vector<double>& weights_;
    double norm_penalty_;
    std::vector<Eigen::Vector3d> coefficients_;

    template<bool Gradient>
    double evaluate(const Frames& frames, Frames& gradient) const {
        double curve_energy = 0.0;
        for (std::size_t index = 1; index + 1 < frames.size(); ++index) {
            const auto& coefficient = coefficients_[index - 1];
            const Frame difference = coefficient[0] * frames[index - 1] +
                coefficient[1] * frames[index] + coefficient[2] * frames[index + 1];
            const double radial = difference.dot(frames[index]) / frames[index].squaredNorm();
            const Frame curvature = difference - radial * frames[index];
            if constexpr (Gradient) {
                const Frame weighted = weights_[index - 1] * curvature;
                gradient[index - 1] += 2.0 * coefficient[0] * weighted;
                gradient[index] += 2.0 * coefficient[1] * weighted;
                gradient[index] += -2.0 * radial * weighted;
                gradient[index + 1] += 2.0 * coefficient[2] * weighted;
            }
            curve_energy += weights_[index - 1] * curvature.squaredNorm();
        }
        double penalty_energy = 0.0;
        for (std::size_t index = 0; index < frames.size(); ++index) {
            const double residual = frames[index].squaredNorm() - 1.0;
            if constexpr (Gradient) gradient[index] += 4.0 * norm_penalty_ * residual * frames[index];
            penalty_energy += norm_penalty_ * residual * residual;
        }
        return curve_energy + penalty_energy;
    }
};

}  // namespace interpolatecpp::quat::detail
