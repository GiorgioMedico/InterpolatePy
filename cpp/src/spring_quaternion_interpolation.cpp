#include <interpolatecpp/quat/spring_quaternion_interpolation.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace interpolatecpp::quat {

SpringQuaternionInterpolation::Frame
SpringQuaternionInterpolation::quaternion_to_frame(const Quaternion& quaternion) {
    return {quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z()};
}

Quaternion SpringQuaternionInterpolation::frame_to_quaternion(const Frame& frame) {
    return Quaternion(frame[0], frame[1], frame[2], frame[3]);
}

void SpringQuaternionInterpolation::validate_config(const SpringConfig& config) {
    if (config.final_iterations < -1) {
        throw std::invalid_argument("final_iterations must be non-negative or -1 for the shared budget");
    }
    if (config.solver != "gradient_descent" && config.solver != "gauss_newton") {
        throw std::invalid_argument("solver must be 'gradient_descent' or 'gauss_newton'");
    }
    if (config.num_samples < 2) {
        throw std::invalid_argument("num_samples must be at least 2");
    }
    if (config.iterations < 0) {
        throw std::invalid_argument("iterations must be non-negative");
    }
    if (config.refinement_levels < 1) {
        throw std::invalid_argument("refinement_levels must be at least 1");
    }
    if (!std::isfinite(config.step_size) || config.step_size <= 0.0) {
        throw std::invalid_argument("step_size must be positive and finite");
    }
    if (!std::isfinite(config.norm_penalty) || config.norm_penalty <= 0.0) {
        throw std::invalid_argument("norm_penalty must be positive and finite");
    }
    if (!std::isfinite(config.keyframe_curvature_weight) ||
        config.keyframe_curvature_weight <= 0.0) {
        throw std::invalid_argument(
            "keyframe_curvature_weight must be positive and finite");
    }
    if (!std::isfinite(config.tolerance) || config.tolerance < 0.0) {
        throw std::invalid_argument("tolerance must be non-negative and finite");
    }
}

SpringQuaternionInterpolation::SpringQuaternionInterpolation(
    const std::vector<double>& time_points,
    const std::vector<Quaternion>& quaternions,
    SpringConfig config)
    : config_(config), time_points_(time_points) {
    validate_config(config_);
    validate_and_prepare_keyframes(quaternions);
    if (config_.num_samples < static_cast<int>(quaternions_.size())) {
        throw std::invalid_argument(
            "num_samples must be at least the number of keyframes");
    }

    const auto intervals = allocate_intervals();
    Frames initial_frames = create_initial_curve(intervals);
    refinement_sample_counts_ = create_refinement_sample_counts();
    const auto level_indices = create_nested_level_indices();
    const auto budgets = create_iteration_budgets(config_.iterations,
                                                   level_indices.size());
    Frames optimized_frames = optimize_levels(initial_frames, level_indices, budgets);
    const auto final_curvature_weights = curvature_weights(level_indices.back());

    for (Frame& frame : optimized_frames) frame.normalize();
    for (std::size_t index = 0; index < keyframe_indices_.size(); ++index) {
        optimized_frames[keyframe_indices_[index]] =
            quaternion_to_frame(quaternions_[index]);
    }

    samples_.reserve(optimized_frames.size());
    for (const Frame& frame : optimized_frames) {
        samples_.push_back(frame_to_quaternion(frame));
    }
    initial_energy_ = curvature_energy(initial_frames, final_curvature_weights);
    final_energy_ = curvature_energy(optimized_frames, final_curvature_weights);
    for (const auto& history : stage_energy_history_) {
        iterations_run_ += static_cast<int>(history.size()) - 1;
    }
    t_min_ = time_points_.front();
    t_max_ = time_points_.back();
    derivative_step_ = std::numeric_limits<double>::infinity();
    for (std::size_t index = 1; index < sample_times_.size(); ++index) {
        derivative_step_ = std::min(
            derivative_step_, (sample_times_[index] - sample_times_[index - 1]) * 0.5);
    }
}

void SpringQuaternionInterpolation::validate_and_prepare_keyframes(
    const std::vector<Quaternion>& quaternions) {
    if (time_points_.size() != quaternions.size()) {
        throw std::invalid_argument(
            "Number of time points must match number of quaternions");
    }
    if (quaternions.size() < 2) {
        throw std::invalid_argument(
            "At least 2 quaternions are required for interpolation");
    }
    for (std::size_t index = 0; index < time_points_.size(); ++index) {
        if (!std::isfinite(time_points_[index])) {
            throw std::invalid_argument(
                "time_points must contain only finite values");
        }
        if (index > 0 && time_points_[index] <= time_points_[index - 1]) {
            throw std::invalid_argument("Time points must be strictly increasing");
        }
    }

    quaternions_.reserve(quaternions.size());
    for (std::size_t index = 0; index < quaternions.size(); ++index) {
        const Frame values = quaternion_to_frame(quaternions[index]);
        const double norm = values.norm();
        if (!std::isfinite(norm) || norm <= kEpsilon) {
            throw std::invalid_argument("Quaternion " + std::to_string(index) +
                                        " must be finite and non-zero");
        }
        Quaternion current = frame_to_quaternion(values / norm);
        if (!quaternions_.empty() &&
            quaternions_.back().dot_product(current) < 0.0) {
            current = -current;
        }
        quaternions_.push_back(current);
    }
}

std::vector<int> SpringQuaternionInterpolation::allocate_intervals() const {
    const std::size_t segment_count = quaternions_.size() - 1;
    std::vector<int> intervals(segment_count, 1);
    const int remaining = config_.num_samples - 1 -
                          static_cast<int>(segment_count);
    if (remaining == 0) return intervals;

    std::vector<double> chord_lengths(segment_count);
    for (std::size_t index = 0; index < segment_count; ++index) {
        chord_lengths[index] =
            (quaternion_to_frame(quaternions_[index + 1]) -
             quaternion_to_frame(quaternions_[index]))
                .norm();
    }
    double total = std::accumulate(chord_lengths.begin(), chord_lengths.end(), 0.0);
    if (total <= kEpsilon) {
        for (std::size_t index = 0; index < segment_count; ++index) {
            chord_lengths[index] = time_points_[index + 1] - time_points_[index];
        }
        total = std::accumulate(chord_lengths.begin(), chord_lengths.end(), 0.0);
    }

    std::vector<double> fractions(segment_count);
    int assigned = 0;
    for (std::size_t index = 0; index < segment_count; ++index) {
        const double exact = static_cast<double>(remaining) *
                             chord_lengths[index] / total;
        const int extra = static_cast<int>(std::floor(exact));
        intervals[index] += extra;
        fractions[index] = exact - static_cast<double>(extra);
        assigned += extra;
    }

    std::vector<std::size_t> order(segment_count);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&fractions](std::size_t lhs, std::size_t rhs) {
                         return fractions[lhs] > fractions[rhs];
                     });
    const int leftover = remaining - assigned;
    for (int index = 0; index < leftover; ++index) {
        intervals[order[static_cast<std::size_t>(index)]] += 1;
    }
    return intervals;
}

SpringQuaternionInterpolation::Frames
SpringQuaternionInterpolation::create_initial_curve(
    const std::vector<int>& intervals) {
    sample_times_.reserve(static_cast<std::size_t>(config_.num_samples));
    Frames frames;
    frames.reserve(static_cast<std::size_t>(config_.num_samples));
    keyframe_indices_.reserve(quaternions_.size());

    sample_times_.push_back(time_points_.front());
    frames.push_back(quaternion_to_frame(quaternions_.front()));
    keyframe_indices_.push_back(0);

    for (std::size_t index = 0; index < intervals.size(); ++index) {
        const int count = intervals[index];
        const double start_time = time_points_[index];
        const double end_time = time_points_[index + 1];
        for (int offset = 1; offset <= count; ++offset) {
            const double fraction = static_cast<double>(offset) /
                                    static_cast<double>(count);
            sample_times_.push_back(offset == count ? end_time :
                start_time + fraction * (end_time - start_time));
            frames.push_back(quaternion_to_frame(Quaternion::slerp(
                quaternions_[index], quaternions_[index + 1], fraction)));
        }
        keyframe_indices_.push_back(frames.size() - 1);
    }
    return frames;
}

double SpringQuaternionInterpolation::check_time(double t) const {
    if (!std::isfinite(t)) throw std::invalid_argument("Time must be finite");
    if (t < t_min_ - kEpsilon || t > t_max_ + kEpsilon) {
        throw std::invalid_argument("Time " + std::to_string(t) +
                                    " outside valid range [" +
                                    std::to_string(t_min_) + ", " +
                                    std::to_string(t_max_) + "]");
    }
    return std::clamp(t, t_min_, t_max_);
}

Quaternion SpringQuaternionInterpolation::evaluate(double t) const {
    t = check_time(t);
    if (t <= t_min_) return samples_.front();
    if (t >= t_max_) return samples_.back();

    const auto upper = std::upper_bound(sample_times_.begin(),
                                        sample_times_.end(), t);
    if (upper == sample_times_.end()) return samples_.back();
    const std::size_t upper_index =
        static_cast<std::size_t>(upper - sample_times_.begin());
    const std::size_t lower_index = upper_index - 1;
    const double fraction =
        (t - sample_times_[lower_index]) /
        (sample_times_[upper_index] - sample_times_[lower_index]);
    return Quaternion::slerp(samples_[lower_index], samples_[upper_index],
                             fraction)
        .unit();
}

Eigen::Vector3d SpringQuaternionInterpolation::evaluate_velocity(double t) const {
    t = check_time(t);
    const double left = std::max(t_min_, t - derivative_step_);
    const double right = std::min(t_max_, t + derivative_step_);
    if (right - left <= kEpsilon) return Eigen::Vector3d::Zero();

    const Quaternion q_left = evaluate(left);
    const Quaternion q_right = evaluate(right);
    Quaternion relative = q_left.inverse() * q_right;
    if (relative.w() < 0.0) relative = -relative;
    return 2.0 * Quaternion::log(relative).vec() / (right - left);
}

Eigen::Vector3d SpringQuaternionInterpolation::evaluate_acceleration(double t) const {
    t = check_time(t);
    const double left = std::max(t_min_, t - derivative_step_);
    const double right = std::min(t_max_, t + derivative_step_);
    if (right - left <= kEpsilon) return Eigen::Vector3d::Zero();
    return (evaluate_velocity(right) - evaluate_velocity(left)) /
           (right - left);
}

std::pair<std::vector<double>, std::vector<Quaternion>>
SpringQuaternionInterpolation::generate_trajectory(int num_points) const {
    if (num_points < 2) {
        throw std::invalid_argument("num_points must be at least 2");
    }
    std::vector<double> times(static_cast<std::size_t>(num_points));
    std::vector<Quaternion> trajectory;
    trajectory.reserve(static_cast<std::size_t>(num_points));
    const double denominator = static_cast<double>(num_points - 1);
    for (int index = 0; index < num_points; ++index) {
        const double fraction = static_cast<double>(index) / denominator;
        const double time = index == num_points - 1 ? t_max_ :
            t_min_ + fraction * (t_max_ - t_min_);
        times[static_cast<std::size_t>(index)] = time;
        trajectory.push_back(evaluate(time));
    }
    return {std::move(times), std::move(trajectory)};
}

}  // namespace interpolatecpp::quat
