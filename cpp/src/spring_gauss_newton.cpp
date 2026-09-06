#include <interpolatecpp/quat/spring_quaternion_interpolation.hpp>

#include "spring_energy.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace interpolatecpp::quat {
namespace {

using Frame = Eigen::Vector4d;
using Frames = std::vector<Frame>;
using Band = std::vector<std::array<double, 12>>;
constexpr std::size_t kBandwidth = 11;

// Lower storage: band[column][row - column]. Three-frame residuals give
// block-pentadiagonal normal equations, regardless of the sample count.
Band normal_matrix(const Frames& frames, const std::vector<double>& weights,
                   double penalty, const std::vector<bool>& fixed,
                   const std::vector<Eigen::Vector3d>& coefficients) {
    Band band(4 * frames.size(), std::array<double, 12>{});
    const auto add_block = [&](std::size_t row_node, std::size_t column_node, const Eigen::Matrix4d& block) {
        if (fixed[row_node] || fixed[column_node]) return;
        for (std::size_t row = 0; row < 4; ++row) {
            for (std::size_t column = 0; column < 4; ++column) {
                const std::size_t j = 4 * column_node + column;
                const int distance = 4 * static_cast<int>(row_node - column_node) +
                                     static_cast<int>(row) - static_cast<int>(column);
                if (distance >= 0 && distance <= static_cast<int>(kBandwidth)) {
                    band[j][static_cast<std::size_t>(distance)] +=
                        block(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column));
                }
            }
        }
    };
    for (std::size_t index = 1; index + 1 < frames.size(); ++index) {
        const double left = coefficients[index - 1][0];
        const double center = coefficients[index - 1][1];
        const double right = coefficients[index - 1][2];
        const Frame& q = frames[index];
        const Frame difference = left * frames[index - 1] + center * q + right * frames[index + 1];
        const double squared_norm = q.squaredNorm();
        const double radial = q.dot(difference) / squared_norm;
        const Frame curvature = difference - radial * q;
        const Eigen::Matrix4d projection = Eigen::Matrix4d::Identity() - q * q.transpose() / squared_norm;
        const std::array<Eigen::Matrix4d, 3> jacobian{
            left * projection,
            (center - radial) * projection - q * curvature.transpose() / squared_norm,
            right * projection};
        for (std::size_t row = 0; row < 3; ++row) {
            for (std::size_t column = 0; column <= row; ++column) {
                add_block(index - 1 + row, index - 1 + column,
                          2.0 * weights[index - 1] * jacobian[row].transpose() * jacobian[column]);
            }
        }
    }
    for (std::size_t index = 0; index < frames.size(); ++index) {
        add_block(index, index, 8.0 * penalty * frames[index] * frames[index].transpose());
        if (fixed[index]) {
            for (std::size_t component = 0; component < 4; ++component) band[4 * index + component][0] = 1.0;
        }
    }
    return band;
}

bool cholesky_solve(Band band, std::vector<double>& rhs, double damping) {
    const std::size_t count = band.size();
    for (std::size_t column = 0; column < count; ++column) {
        band[column][0] += damping;
        for (std::size_t row = column; row <= std::min(column + kBandwidth, count - 1); ++row) {
            double value = band[column][row - column];
            const std::size_t start = row > kBandwidth ? row - kBandwidth : 0;
            for (std::size_t k = start; k < column; ++k) value -= band[k][row - k] * band[k][column - k];
            if (row == column) {
                if (!std::isfinite(value) || value <= 0.0) return false;
                band[column][0] = std::sqrt(value);
            } else {
                band[column][row - column] = value / band[column][0];
            }
        }
    }
    for (std::size_t row = 0; row < count; ++row) {
        const std::size_t start = row > kBandwidth ? row - kBandwidth : 0;
        for (std::size_t column = start; column < row; ++column) rhs[row] -= band[column][row - column] * rhs[column];
        rhs[row] /= band[row][0];
    }
    for (std::size_t column = count; column-- > 0;) {
        for (std::size_t row = column + 1; row <= std::min(column + kBandwidth, count - 1); ++row) {
            rhs[column] -= band[column][row - column] * rhs[row];
        }
        rhs[column] /= band[column][0];
    }
    return true;
}

Frames descent_direction(const Band& band, const Frames& gradient) {
    double scale = 1.0;
    for (const auto& column : band) scale = std::max(scale, column[0]);
    double damping = 0.0;
    for (int attempt = 0; attempt < 8; ++attempt) {
        std::vector<double> rhs(band.size());
        for (std::size_t index = 0; index < rhs.size(); ++index) rhs[index] = -gradient[index / 4][index % 4];
        if (cholesky_solve(band, rhs, damping)) {
            Frames direction(gradient.size());
            double slope = 0.0;
            bool finite = true;
            for (std::size_t index = 0; index < gradient.size(); ++index) {
                direction[index] = Eigen::Map<const Frame>(rhs.data() + 4 * index);
                slope += direction[index].dot(gradient[index]);
                finite = finite && direction[index].allFinite();
            }
            if (finite && slope < 0.0) return direction;
        }
        damping = damping == 0.0 ? scale * 1e-12 : damping * 10.0;
    }
    double squared_norm = 0.0;
    for (const auto& value : gradient) squared_norm += value.squaredNorm();
    Frames direction = gradient;
    for (auto& value : direction) value /= -std::sqrt(squared_norm);
    return direction;
}

}  // namespace

std::pair<SpringQuaternionInterpolation::Frames, std::vector<double>>
SpringQuaternionInterpolation::minimize_gauss_newton(
    const Frames& initial, const std::vector<bool>& fixed, const std::vector<double>& weights,
    int iterations, const Indices& indices) const {
    Frames frames = initial;
    const detail::SpringEnergy model(frames.size(), weights, config_.norm_penalty, indices);
    EnergyGradient current{};
    current.energy = model.energy_gradient(frames, current.gradient);
    EnergyGradient trial{};
    Frames candidate = frames;
    Frames tangents(frames.size());
    std::vector<double> lengths(frames.size()), radial(frames.size());
    std::vector<double> history{current.energy};
    for (int iteration = 0; iteration < iterations; ++iteration) {
        double squared_gradient = 0.0;
        for (std::size_t index = 0; index < frames.size(); ++index) {
            if (fixed[index]) current.gradient[index].setZero();
            squared_gradient += current.gradient[index].squaredNorm();
        }
        if (std::sqrt(squared_gradient) <= config_.tolerance) break;
        const auto direction = descent_direction(normal_matrix(frames, weights, config_.norm_penalty, fixed,
                                                               model.coefficients()),
                                                 current.gradient);
        for (std::size_t index = 0; index < frames.size(); ++index) {
            if (fixed[index]) continue;
            lengths[index] = frames[index].norm();
            radial[index] = frames[index].dot(direction[index]) / lengths[index];
            tangents[index] = direction[index] - (radial[index] / lengths[index]) * frames[index];
        }
        double step = 1.0;
        bool accepted = false;
        for (int backtrack = 0; backtrack < kMaxBacktracks; ++backtrack) {
            candidate = frames;
            bool valid = true;
            for (std::size_t index = 0; index < frames.size(); ++index) {
                if (fixed[index]) continue;
                const double target_length = lengths[index] + step * radial[index];
                if (target_length <= 0.0) { valid = false; break; }
                candidate[index] = (frames[index] + step * tangents[index]).normalized() * target_length;
            }
            if (valid) {
                trial.energy = model.energy(candidate);
                const bool near_roundoff = std::abs(trial.energy - current.energy) <=
                    64.0 * std::numeric_limits<double>::epsilon() * std::max(std::abs(current.energy), 1e-30);
                if (trial.energy < current.energy || near_roundoff) {
                    trial.energy = model.energy_gradient(candidate, trial.gradient);
                    double trial_squared_gradient = 0.0;
                    for (std::size_t index = 0; index < frames.size(); ++index) {
                        if (!fixed[index]) trial_squared_gradient += trial.gradient[index].squaredNorm();
                    }
                    if (trial.energy < current.energy || (near_roundoff && trial_squared_gradient < 0.25 * squared_gradient)) {
                        frames.swap(candidate);
                        std::swap(current, trial);
                        history.push_back(current.energy);
                        accepted = true;
                        break;
                    }
                }
            }
            step *= 0.5;
            if (step < kMinBacktrackStep) break;
        }
        if (!accepted) break;
    }
    return {std::move(frames), std::move(history)};
}

}  // namespace interpolatecpp::quat
