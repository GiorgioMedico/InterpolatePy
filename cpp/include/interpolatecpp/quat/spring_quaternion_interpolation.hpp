#pragma once

#include <Eigen/Core>

#include <cstddef>
#include <utility>
#include <vector>

#include <interpolatecpp/config.hpp>
#include <interpolatecpp/quat/quaternion.hpp>

namespace interpolatecpp::quat {

/// Numerical settings for SPRING minimum-curvature interpolation.
struct SpringConfig {
    int num_samples = 101;
    int iterations = 300;
    int refinement_levels = 3;
    double step_size = 0.05;
    double norm_penalty = 100.0;
    double keyframe_curvature_weight = 1.2;
    double tolerance = 1e-9;
};

/// Spherical interpolation using numerical gradient descent (SPRING).
///
/// The sampled quaternion curve is initialized with piecewise SLERP and
/// relaxed to reduce discrete tangential curvature while retaining every
/// original keyframe. Optimization proceeds on nested coarse-to-fine grids.
class INTERPOLATECPP_API SpringQuaternionInterpolation {
  public:
    SpringQuaternionInterpolation(const std::vector<double>& time_points,
                                  const std::vector<Quaternion>& quaternions,
                                  SpringConfig config = {});

    [[nodiscard]] Quaternion evaluate(double t) const;
    [[nodiscard]] Eigen::Vector3d evaluate_velocity(double t) const;
    [[nodiscard]] Eigen::Vector3d evaluate_acceleration(double t) const;
    [[nodiscard]] std::pair<std::vector<double>, std::vector<Quaternion>>
    generate_trajectory(int num_points = 100) const;

    [[nodiscard]] double t_min() const noexcept { return t_min_; }
    [[nodiscard]] double t_max() const noexcept { return t_max_; }
    [[nodiscard]] const SpringConfig& config() const noexcept { return config_; }
    [[nodiscard]] const std::vector<double>& time_points() const noexcept {
        return time_points_;
    }
    [[nodiscard]] const std::vector<Quaternion>& quaternions() const noexcept {
        return quaternions_;
    }
    [[nodiscard]] const std::vector<double>& sample_times() const noexcept {
        return sample_times_;
    }
    [[nodiscard]] const std::vector<Quaternion>& samples() const noexcept {
        return samples_;
    }
    [[nodiscard]] const std::vector<std::size_t>& keyframe_indices() const noexcept {
        return keyframe_indices_;
    }
    [[nodiscard]] const std::vector<int>& refinement_sample_counts() const noexcept {
        return refinement_sample_counts_;
    }
    [[nodiscard]] const std::vector<std::vector<double>>& stage_energy_history()
        const noexcept {
        return stage_energy_history_;
    }
    [[nodiscard]] const std::vector<double>& energy_history() const noexcept {
        return stage_energy_history_.back();
    }
    [[nodiscard]] double initial_energy() const noexcept { return initial_energy_; }
    [[nodiscard]] double final_energy() const noexcept { return final_energy_; }
    [[nodiscard]] int iterations_run() const noexcept { return iterations_run_; }

  private:
    using Frame = Eigen::Vector4d;
    using Frames = std::vector<Frame>;
    using Indices = std::vector<std::size_t>;

    struct EnergyGradient {
        double energy;
        Frames gradient;
    };

    SpringConfig config_;
    std::vector<double> time_points_;
    std::vector<Quaternion> quaternions_;
    std::vector<double> sample_times_;
    std::vector<Quaternion> samples_;
    Indices keyframe_indices_;
    std::vector<int> refinement_sample_counts_;
    std::vector<std::vector<double>> stage_energy_history_;
    double initial_energy_ = 0.0;
    double final_energy_ = 0.0;
    int iterations_run_ = 0;
    double t_min_ = 0.0;
    double t_max_ = 0.0;
    double derivative_step_ = 0.0;

    static constexpr double kEpsilon = 1e-12;
    static constexpr double kMinBacktrackStep = 1e-12;
    static constexpr int kMaxBacktracks = 30;
    static constexpr int kRefinementFactor = 5;

    static Frame quaternion_to_frame(const Quaternion& quaternion);
    static Quaternion frame_to_quaternion(const Frame& frame);
    static void validate_config(const SpringConfig& config);
    void validate_and_prepare_keyframes(const std::vector<Quaternion>& quaternions);
    [[nodiscard]] std::vector<int> allocate_intervals() const;
    [[nodiscard]] Frames create_initial_curve(const std::vector<int>& intervals);
    [[nodiscard]] std::vector<int> create_refinement_sample_counts() const;
    [[nodiscard]] std::vector<Indices> create_nested_level_indices() const;
    [[nodiscard]] static std::vector<int> create_iteration_budgets(
        int total_iterations, std::size_t stage_count);
    [[nodiscard]] std::vector<double> curvature_weights(const Indices& indices) const;
    [[nodiscard]] static Frames refine_initial_curve(const Indices& source_indices,
                                                     const Frames& source_frames,
                                                     const Indices& target_indices);
    [[nodiscard]] static EnergyGradient curvature_energy_gradient(
        const Frames& frames, const std::vector<double>& curvature_weights,
        double norm_penalty, const Indices& sample_indices = {});
    [[nodiscard]] static double curvature_energy(
        const Frames& frames, const std::vector<double>& curvature_weights);
    [[nodiscard]] std::pair<Frames, std::vector<double>> minimize(
        const Frames& initial_frames, const std::vector<bool>& fixed_mask,
        const std::vector<double>& curvature_weights, int iterations,
        const Indices& sample_indices) const;
    [[nodiscard]] Frames optimize_levels(const Frames& final_initial_frames,
                                         const std::vector<Indices>& level_indices,
                                         const std::vector<int>& budgets);
    [[nodiscard]] double check_time(double t) const;
};

}  // namespace interpolatecpp::quat
