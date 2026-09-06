#include <interpolatecpp/quat/spring_quaternion_interpolation.hpp>

#include "spring_energy.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
#include <utility>

namespace interpolatecpp::quat {

namespace {

struct RefinementInterval {
    std::size_t distance;
    std::size_t candidate;
    std::size_t left;
    std::size_t right;
};

struct RefinementIntervalPriority {
    bool operator()(const RefinementInterval& lhs,
                    const RefinementInterval& rhs) const noexcept {
        if (lhs.distance != rhs.distance) return lhs.distance < rhs.distance;
        if (lhs.candidate != rhs.candidate) return lhs.candidate > rhs.candidate;
        if (lhs.left != rhs.left) return lhs.left > rhs.left;
        return lhs.right > rhs.right;
    }
};

}  // namespace

std::vector<int>
SpringQuaternionInterpolation::create_refinement_sample_counts() const {
    const int keyframe_count = static_cast<int>(quaternions_.size());
    const int minimum_coarse_count =
        std::min(config_.num_samples, 2 * keyframe_count - 1);
    std::vector<int> descending = {config_.num_samples};
    for (int level = 1; level < config_.refinement_levels; ++level) {
        const double scaled = static_cast<double>(descending.back()) /
                              static_cast<double>(kRefinementFactor);
        const int next_count = std::max(
            minimum_coarse_count, static_cast<int>(std::nearbyint(scaled)));
        if (next_count >= descending.back()) break;
        descending.push_back(next_count);
    }
    std::reverse(descending.begin(), descending.end());
    return descending;
}

std::vector<SpringQuaternionInterpolation::Indices>
SpringQuaternionInterpolation::create_nested_level_indices() const {
    std::set<std::size_t> selected(keyframe_indices_.begin(),
                                   keyframe_indices_.end());
    std::priority_queue<RefinementInterval, std::vector<RefinementInterval>,
                        RefinementIntervalPriority>
        intervals;
    const auto add_interval = [&intervals](std::size_t left, std::size_t right) {
        if (right - left <= 1) return;
        const std::size_t candidate = (left + right) / 2;
        const std::size_t distance =
            std::min(candidate - left, right - candidate);
        intervals.push({distance, candidate, left, right});
    };

    for (std::size_t index = 1; index < keyframe_indices_.size(); ++index) {
        add_interval(keyframe_indices_[index - 1], keyframe_indices_[index]);
    }

    std::vector<Indices> levels;
    levels.reserve(refinement_sample_counts_.size());
    for (const int sample_count : refinement_sample_counts_) {
        while (selected.size() < static_cast<std::size_t>(sample_count)) {
            if (intervals.empty()) {
                throw std::runtime_error(
                    "sample count exceeds the final sample lattice");
            }
            const RefinementInterval interval = intervals.top();
            intervals.pop();
            selected.insert(interval.candidate);
            add_interval(interval.left, interval.candidate);
            add_interval(interval.candidate, interval.right);
        }
        levels.emplace_back(selected.begin(), selected.end());
    }
    return levels;
}

std::vector<int> SpringQuaternionInterpolation::create_iteration_budgets(
    int total_iterations, std::size_t stage_count) {
    std::vector<int> budgets(stage_count, 0);
    if (stage_count == 0) return budgets;

    double total_weight = 0.0;
    for (std::size_t index = 0; index < stage_count; ++index) {
        total_weight += static_cast<double>(stage_count + 1 - index);
    }
    std::vector<double> fractions(stage_count);
    int assigned = 0;
    for (std::size_t index = 0; index < stage_count; ++index) {
        const double weight = static_cast<double>(stage_count + 1 - index);
        const double exact = static_cast<double>(total_iterations) * weight /
                             total_weight;
        budgets[index] = static_cast<int>(std::floor(exact));
        fractions[index] = exact - static_cast<double>(budgets[index]);
        assigned += budgets[index];
    }

    std::vector<std::size_t> order(stage_count);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&fractions](std::size_t lhs, std::size_t rhs) {
                         return fractions[lhs] > fractions[rhs];
                     });
    const int remaining = total_iterations - assigned;
    for (int index = 0; index < remaining; ++index) {
        budgets[order[static_cast<std::size_t>(index)]] += 1;
    }
    return budgets;
}

std::vector<double> SpringQuaternionInterpolation::curvature_weights(
    const Indices& indices) const {
    std::vector<double> weights(indices.size() - 2, 1.0);
    for (std::size_t keyframe = 1; keyframe + 1 < keyframe_indices_.size();
         ++keyframe) {
        const auto position = std::lower_bound(indices.begin(), indices.end(),
                                               keyframe_indices_[keyframe]);
        const auto offset = static_cast<std::size_t>(position - indices.begin());
        weights[offset - 1] = config_.keyframe_curvature_weight;
    }
    return weights;
}

SpringQuaternionInterpolation::Frames
SpringQuaternionInterpolation::refine_initial_curve(
    const Indices& source_indices,
    const Frames& source_frames,
    const Indices& target_indices) {
    Frames refined;
    refined.reserve(target_indices.size());
    for (const std::size_t target_index : target_indices) {
        const auto upper = std::lower_bound(source_indices.begin(),
                                            source_indices.end(), target_index);
        const std::size_t upper_position =
            static_cast<std::size_t>(upper - source_indices.begin());
        if (upper != source_indices.end() && *upper == target_index) {
            refined.push_back(source_frames[upper_position]);
            continue;
        }

        const std::size_t lower_position = upper_position - 1;
        const double fraction =
            static_cast<double>(target_index - source_indices[lower_position]) /
            static_cast<double>(source_indices[upper_position] -
                                source_indices[lower_position]);
        const Quaternion lower =
            frame_to_quaternion(source_frames[lower_position].normalized());
        const Quaternion upper_quaternion =
            frame_to_quaternion(source_frames[upper_position].normalized());
        refined.push_back(quaternion_to_frame(
            Quaternion::slerp(lower, upper_quaternion, fraction)));
    }
    return refined;
}

SpringQuaternionInterpolation::EnergyGradient
SpringQuaternionInterpolation::curvature_energy_gradient(
    const Frames& frames,
    const std::vector<double>& curvature_weights,
    double norm_penalty,
    const Indices& sample_indices) {
    const detail::SpringEnergy model(frames.size(), curvature_weights, norm_penalty, sample_indices);
    EnergyGradient result{};
    result.energy = model.energy_gradient(frames, result.gradient);
    return result;
}

double SpringQuaternionInterpolation::curvature_energy(
    const Frames& frames,
    const std::vector<double>& curvature_weights) {
    return detail::SpringEnergy(frames.size(), curvature_weights, 0.0).energy(frames);
}

std::pair<SpringQuaternionInterpolation::Frames, std::vector<double>>
SpringQuaternionInterpolation::minimize(
    const Frames& initial_frames,
    const std::vector<bool>& fixed_mask,
    const std::vector<double>& curvature_weights,
    int iterations,
    const Indices& sample_indices) const {
    if (config_.solver == "gauss_newton" && initial_frames.size() == static_cast<std::size_t>(config_.num_samples)) {
        return minimize_gauss_newton(initial_frames, fixed_mask, curvature_weights, iterations, sample_indices);
    }
    Frames frames = initial_frames;
    const detail::SpringEnergy model(frames.size(), curvature_weights, config_.norm_penalty, sample_indices);
    EnergyGradient current{};
    current.energy = model.energy_gradient(frames, current.gradient);
    Frames candidate(frames.size());
    std::vector<double> history = {current.energy};

    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (std::size_t index = 0; index < current.gradient.size(); ++index) {
            if (fixed_mask[index]) current.gradient[index].setZero();
        }
        double squared_gradient_norm = 0.0;
        for (const Frame& value : current.gradient) {
            squared_gradient_norm += value.squaredNorm();
        }
        const double gradient_norm = std::sqrt(squared_gradient_norm);
        if (gradient_norm <= config_.tolerance) break;

        const double inverse_gradient_norm = 1.0 / gradient_norm;
        double step = config_.step_size;
        bool accepted = false;
        for (int backtrack = 0; backtrack < kMaxBacktracks; ++backtrack) {
            candidate = frames;
            for (std::size_t index = 0; index < candidate.size(); ++index) {
                candidate[index] -= step * inverse_gradient_norm *
                                    current.gradient[index];
                if (fixed_mask[index]) candidate[index] = initial_frames[index];
            }
            const double trial_energy = model.energy(candidate);
            if (trial_energy < current.energy) {
                frames.swap(candidate);
                current.energy = model.energy_gradient(frames, current.gradient);
                history.push_back(current.energy);
                accepted = true;
                break;
            }
            step *= 0.5;
            if (step < kMinBacktrackStep) break;
        }
        if (!accepted) break;
    }
    return {std::move(frames), std::move(history)};
}

SpringQuaternionInterpolation::Frames
SpringQuaternionInterpolation::optimize_levels(
    const Frames& final_initial_frames,
    const std::vector<Indices>& level_indices,
    const std::vector<int>& budgets) {
    // Test convergence on the target grid before introducing coarse-grid
    // truncation error into an already stationary curve.
    const auto target_weights = curvature_weights(level_indices.back());
    const double initial_curvature = curvature_energy(final_initial_frames, target_weights);
    auto initial = curvature_energy_gradient(
        final_initial_frames, target_weights, config_.norm_penalty);
    for (const std::size_t index : keyframe_indices_) initial.gradient[index].setZero();
    double squared_gradient_norm = 0.0;
    for (const Frame& value : initial.gradient) squared_gradient_norm += value.squaredNorm();
    const bool converged = std::sqrt(squared_gradient_norm) <= config_.tolerance;

    Indices previous_indices;
    Frames previous_frames;
    bool has_previous = false;
    stage_energy_history_.reserve(level_indices.size());

    for (std::size_t level = 0; level < level_indices.size(); ++level) {
        const Indices& current_indices = level_indices[level];
        Frames stage_initial;
        Indices fixed_final_indices;
        if (!has_previous || converged) {
            stage_initial.reserve(current_indices.size());
            for (const std::size_t index : current_indices) {
                stage_initial.push_back(final_initial_frames[index]);
            }
            fixed_final_indices = keyframe_indices_;
        } else {
            stage_initial = refine_initial_curve(previous_indices, previous_frames,
                                                 current_indices);
            fixed_final_indices = previous_indices;
        }

        if (has_previous && current_indices.size() == final_initial_frames.size()) {
            // Reject a coarse seed that increases curvature on the target grid.
            Frames normalized_initial = stage_initial;
            for (Frame& frame : normalized_initial) frame.normalize();
            if (curvature_energy(normalized_initial, target_weights) > initial_curvature) {
                stage_initial = final_initial_frames;
                fixed_final_indices = keyframe_indices_;
            }
        }

        std::vector<bool> fixed_mask(current_indices.size(), false);
        for (std::size_t index = 0; index < current_indices.size(); ++index) {
            fixed_mask[index] = std::binary_search(
                fixed_final_indices.begin(), fixed_final_indices.end(),
                current_indices[index]);
        }
        auto [stage_frames, history] = minimize(
            stage_initial, fixed_mask, curvature_weights(current_indices),
            converged ? 0 :
                (level + 1 == level_indices.size() && config_.final_iterations >= 0
                    ? config_.final_iterations : budgets[level]), current_indices);
        auto final = curvature_energy_gradient(stage_frames, curvature_weights(current_indices),
                                                config_.norm_penalty, current_indices);
        double squared_norm = 0.0;
        for (std::size_t index = 0; index < stage_frames.size(); ++index) {
            if (!fixed_mask[index]) squared_norm += final.gradient[index].squaredNorm();
        }
        stage_gradient_norms_.push_back(std::sqrt(squared_norm));
        stage_energy_history_.push_back(std::move(history));
        previous_indices = current_indices;
        previous_frames = std::move(stage_frames);
        has_previous = true;
    }
    if (!has_previous) {
        throw std::runtime_error("SPRING refinement produced no stages");
    }
    converged_ = converged || stage_gradient_norms_.back() <= config_.tolerance;
    return previous_frames;
}

}  // namespace interpolatecpp::quat
