#include <interpolatecpp/quat/shooting_quaternion_interpolation.hpp>

#include "shooting_solver.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace interpolatecpp::quat {

ShootingQuaternionInterpolation::ShootingQuaternionInterpolation(
    const std::vector<double>& time_points, const std::vector<Quaternion>& quaternions,
    ShootingConfig config) : config_(config), time_points_(time_points) {
    if (!std::isfinite(config.tolerance) || config.tolerance <= 0.0) {
        throw std::invalid_argument("tolerance must be positive and finite");
    }
    if (config.max_iterations < 1 || config.integration_steps < 1) {
        throw std::invalid_argument("max_iterations and integration_steps must be positive integers");
    }
    if (config.max_integration_steps <= config.integration_steps) {
        throw std::invalid_argument("max_integration_steps must exceed integration_steps for independent verification");
    }
    if (time_points.size() != quaternions.size()) {
        throw std::invalid_argument("Number of time points must match number of quaternions");
    }
    if (quaternions.size() < 2) throw std::invalid_argument("At least 2 quaternions are required");
    for (std::size_t index = 0; index < time_points.size(); ++index) {
        if (!std::isfinite(time_points[index])) throw std::invalid_argument("time_points must contain only finite values");
        if (index > 0) {
            const double duration = time_points[index] - time_points[index - 1];
            if (!std::isfinite(duration) || duration <= 0.0) {
                throw std::invalid_argument("Time points must have finite, strictly positive spacing");
            }
            durations_.push_back(duration);
        }
        Eigen::Vector4d value(quaternions[index].w(), quaternions[index].x(),
                              quaternions[index].y(), quaternions[index].z());
        const double norm = value.norm();
        if (!std::isfinite(norm) || norm <= 1e-12) {
            throw std::invalid_argument("Quaternion must be finite and non-zero");
        }
        value /= norm;
        Quaternion current(value[0], value[1], value[2], value[3]);
        if (!quaternions_.empty() && quaternions_.back().dot_product(current) < 0.0) current = -current;
        quaternions_.push_back(current);
    }
    auto solution = shooting_detail::solve(quaternions_, durations_, config_);
    parameters_ = std::move(solution.parameters);
    nodes_ = std::move(solution.nodes);
    iterations_run_ = solution.iterations;
    integration_steps_ = solution.steps;
    residual_norm_ = solution.residual;
    acceleration_energy_ = solution.energy;
}

double ShootingQuaternionInterpolation::check_time(double t) const {
    if (!std::isfinite(t)) throw std::invalid_argument("Time must be finite");
    if (t < t_min() - 1e-12 || t > t_max() + 1e-12) {
        throw std::invalid_argument("Time outside valid range");
    }
    return std::clamp(t, t_min(), t_max());
}

std::pair<ShootingQuaternionInterpolation::State, double>
ShootingQuaternionInterpolation::evaluate_state(double t) const {
    t = check_time(t);
    const auto upper = std::upper_bound(time_points_.begin(), time_points_.end(), t);
    const auto segment = std::min(static_cast<std::size_t>(upper - time_points_.begin() - 1), durations_.size() - 1);
    const double duration = durations_[segment];
    const double fraction = std::clamp((t - time_points_[segment]) / duration, 0.0, 1.0);
    const int node = std::min(static_cast<int>(fraction * integration_steps_), integration_steps_);
    State state = nodes_[segment][static_cast<std::size_t>(node)];
    const double remainder = fraction - static_cast<double>(node) / integration_steps_;
    if (remainder > 0.0) {
        const Eigen::Vector3d constant = parameters_.block<1, 3>(static_cast<Eigen::Index>(segment), 6).transpose();
        state = shooting_detail::evaluate_step(state, constant, remainder);
    }
    return {state, duration};
}

Quaternion ShootingQuaternionInterpolation::evaluate(double t) const {
    t = check_time(t);
    const auto keyframe = std::lower_bound(time_points_.begin(), time_points_.end(), t);
    if (keyframe != time_points_.end() && *keyframe == t) {
        return quaternions_[static_cast<std::size_t>(keyframe - time_points_.begin())];
    }
    const auto [state, duration] = evaluate_state(t);
    return Quaternion(state[0], state[1], state[2], state[3]);
}

Eigen::Vector3d ShootingQuaternionInterpolation::evaluate_velocity(double t) const {
    const auto [state, duration] = evaluate_state(t);
    return 2.0 * state.segment<3>(4) / duration;
}

Eigen::Vector3d ShootingQuaternionInterpolation::evaluate_acceleration(double t) const {
    const auto [state, duration] = evaluate_state(t);
    return 2.0 * state.tail<3>() / duration / duration;
}

std::pair<std::vector<double>, std::vector<Quaternion>>
ShootingQuaternionInterpolation::generate_trajectory(int num_points) const {
    if (num_points < 2) throw std::invalid_argument("num_points must be at least 2");
    std::vector<double> times;
    std::vector<Quaternion> quaternions;
    times.reserve(static_cast<std::size_t>(num_points));
    quaternions.reserve(static_cast<std::size_t>(num_points));
    for (int index = 0; index < num_points; ++index) {
        const double t = index == num_points - 1 ? t_max() :
            t_min() + (static_cast<double>(index) / (num_points - 1)) * (t_max() - t_min());
        times.push_back(t);
        quaternions.push_back(evaluate(t));
    }
    return {std::move(times), std::move(quaternions)};
}

}  // namespace interpolatecpp::quat
