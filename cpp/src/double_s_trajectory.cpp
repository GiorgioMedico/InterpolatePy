#include <interpolatecpp/motion/double_s_trajectory.hpp>

#include <algorithm>
#include <cmath>

namespace interpolatecpp::motion {

DoubleSTrajectory::DoubleSTrajectory(const StateParams& state, const TrajectoryBounds& bounds)
    : state_(state),
      bounds_(bounds),
      T_(0.0),
      Ta_(0.0),
      Tv_(0.0),
      Td_(0.0),
      Tj_1_(0.0),
      Tj_2_(0.0),
      a_lim_a_(0.0),
      a_lim_d_(0.0),
      v_lim_(0.0),
      sigma_(1.0) {
    if (std::abs(state_.v_0) > bounds_.v_bound || std::abs(state_.v_1) > bounds_.v_bound) {
        throw std::invalid_argument(
            "Initial or final velocities exceed the velocity bound of " +
            std::to_string(bounds_.v_bound));
    }

    plan_trajectory();
}

void DoubleSTrajectory::plan_trajectory() {
    const double qd_0 = state_.q_0;
    const double qd_1 = state_.q_1;
    const double vd_0 = state_.v_0;
    const double vd_1 = state_.v_1;

    // BLOCK 1: Handle equal positions
    if (std::abs(qd_1 - qd_0) < kEpsilon) {
        if (std::abs(vd_1 - vd_0) < kEpsilon) {
            // Same position, same velocity: static trajectory
            T_ = 0.0;
            return;
        }
        // Same position but different velocities: minimal trajectory
        const double t_min = std::abs(vd_1 - vd_0) / bounds_.a_bound;
        T_ = std::max(t_min * 1.5, 0.1);
        return;
    }

    // Normal case: different positions
    sigma_ = (qd_1 > qd_0) ? 1.0 : -1.0;

    // Transform parameters based on direction
    const double q_0_t = sigma_ * qd_0;
    const double q_1_t = sigma_ * qd_1;
    const double v_0_t = sigma_ * vd_0;
    const double v_1_t = sigma_ * vd_1;

    // Set limits based on direction (sigma transform collapses to using positive bounds)
    const double v_max = ((sigma_ + 1.0) / 2.0) * bounds_.v_bound +
                         ((sigma_ - 1.0) / 2.0) * (-bounds_.v_bound);
    const double a_max_initial = ((sigma_ + 1.0) / 2.0) * bounds_.a_bound +
                                 ((sigma_ - 1.0) / 2.0) * (-bounds_.a_bound);
    const double j_max = ((sigma_ + 1.0) / 2.0) * bounds_.j_bound +
                         ((sigma_ - 1.0) / 2.0) * (-bounds_.j_bound);

    double a_max = a_max_initial;

    // Compute time intervals assuming v_max and a_max are reached

    // Acceleration part
    if ((v_max - v_0_t) * j_max < a_max * a_max) {
        Tj_1_ = std::sqrt(std::max((v_max - v_0_t) / j_max, 0.0));
        Ta_ = 2.0 * Tj_1_;
    } else {
        Tj_1_ = a_max / j_max;
        Ta_ = Tj_1_ + (v_max - v_0_t) / a_max;
    }

    // Deceleration part
    if ((v_max - v_1_t) * j_max < a_max * a_max) {
        Tj_2_ = std::sqrt(std::max((v_max - v_1_t) / j_max, 0.0));
        Td_ = 2.0 * Tj_2_;
    } else {
        Tj_2_ = a_max / j_max;
        Td_ = Tj_2_ + (v_max - v_1_t) / a_max;
    }

    // Determine the time duration of the constant velocity phase
    if (std::abs(v_max) < kEpsilon) {
        Tv_ = 0.0;
    } else {
        Tv_ = (q_1_t - q_0_t) / v_max -
              (Ta_ / 2.0) * (1.0 + v_0_t / v_max) -
              (Td_ / 2.0) * (1.0 + v_1_t / v_max);
    }

    // Check if Tv < 0 (v_max is not reached): binary search on gamma
    if (Tv_ < 0.0) {
        Tv_ = 0.0;

        double gamma_high = 1.0;
        double gamma_low = kMinGamma;
        double gamma_mid = 0.5;

        for (int iteration = 0; iteration < kMaxIterations; ++iteration) {
            gamma_mid = (gamma_high + gamma_low) / 2.0;

            const double a_max_test = gamma_mid * bounds_.a_bound;

            // Recalculate time intervals
            const double tj = a_max_test / j_max;
            const double delta =
                (a_max_test * a_max_test * a_max_test * a_max_test) / (j_max * j_max) +
                2.0 * (v_0_t * v_0_t + v_1_t * v_1_t) +
                a_max_test * (
                    4.0 * (q_1_t - q_0_t) -
                    2.0 * a_max_test / j_max * (v_0_t + v_1_t)
                );

            // Check if delta is negative (no solution with current gamma)
            if (delta < 0.0) {
                gamma_high = gamma_mid;
                continue;
            }

            const double sqrt_delta = std::sqrt(delta);

            double ta = (a_max_test * a_max_test / j_max - 2.0 * v_0_t + sqrt_delta) /
                        (2.0 * a_max_test);
            double td = (a_max_test * a_max_test / j_max - 2.0 * v_1_t + sqrt_delta) /
                        (2.0 * a_max_test);

            double tj_1_local = tj;
            double tj_2_local = tj;

            if (ta < 0.0) {
                if (std::abs(v_1_t + v_0_t) < kEpsilon) {
                    // Avoid division by zero
                    Ta_ = 0.0;
                    Td_ = 0.0;
                    Tj_1_ = 0.0;
                    Tj_2_ = 0.0;
                    break;
                }
                ta = 0.0;
                td = 2.0 * (q_1_t - q_0_t) / (v_1_t + v_0_t);
                const double inner = j_max * (
                    j_max * (q_1_t - q_0_t) * (q_1_t - q_0_t) +
                    (v_1_t + v_0_t) * (v_1_t + v_0_t) * (v_1_t - v_0_t)
                );
                const double tj_2_arg =
                    j_max * (q_1_t - q_0_t) - std::sqrt(inner);
                tj_1_local = 0.0;
                tj_2_local = (std::abs(tj_2_arg) > kEpsilon)
                    ? tj_2_arg / (j_max * (v_1_t + v_0_t))
                    : 0.0;

                // Validate and store
                if (tj_1_local >= 0.0 && tj_2_local >= 0.0 && ta >= 0.0 && td >= 0.0) {
                    a_max = a_max_test;
                    Tj_1_ = tj_1_local;
                    Tj_2_ = tj_2_local;
                    Ta_ = ta;
                    Td_ = td;
                    break;
                }
                gamma_high = gamma_mid;
            } else if (td < 0.0) {
                if (std::abs(v_1_t + v_0_t) < kEpsilon) {
                    // Avoid division by zero
                    Ta_ = 0.0;
                    Td_ = 0.0;
                    Tj_1_ = 0.0;
                    Tj_2_ = 0.0;
                    break;
                }
                td = 0.0;
                ta = 2.0 * (q_1_t - q_0_t) / (v_1_t + v_0_t);
                const double inner = j_max * (
                    j_max * (q_1_t - q_0_t) * (q_1_t - q_0_t) -
                    (v_1_t + v_0_t) * (v_1_t + v_0_t) * (v_1_t - v_0_t)
                );
                const double tj_1_arg =
                    j_max * (q_1_t - q_0_t) - std::sqrt(inner);
                tj_1_local = (std::abs(tj_1_arg) > kEpsilon)
                    ? tj_1_arg / (j_max * (v_1_t + v_0_t))
                    : 0.0;
                tj_2_local = 0.0;

                // Validate and store
                if (tj_1_local >= 0.0 && tj_2_local >= 0.0 && ta >= 0.0 && td >= 0.0) {
                    a_max = a_max_test;
                    Tj_1_ = tj_1_local;
                    Tj_2_ = tj_2_local;
                    Ta_ = ta;
                    Td_ = td;
                    break;
                }
                gamma_high = gamma_mid;
            } else if (ta > 2.0 * tj && td > 2.0 * tj) {
                // Valid solution found
                a_max = a_max_test;
                Tj_1_ = tj;
                Tj_2_ = tj;
                Ta_ = ta;
                Td_ = td;
                break;
            } else {
                // Need to reduce gamma further
                gamma_high = gamma_mid;
            }
        }
    }

    // Compute trajectory parameters
    a_lim_a_ = j_max * Tj_1_;
    a_lim_d_ = -j_max * Tj_2_;

    // Ensure non-negative time periods
    Ta_ = std::max(Ta_, 0.0);
    Td_ = std::max(Td_, 0.0);
    Tv_ = std::max(Tv_, 0.0);
    Tj_1_ = std::max(Tj_1_, 0.0);
    Tj_2_ = std::max(Tj_2_, 0.0);

    // Calculate v_lim safely
    if (Ta_ <= Tj_1_) {
        v_lim_ = v_0_t + j_max * Ta_ * Ta_ / 2.0;
    } else {
        v_lim_ = v_0_t + (Ta_ - Tj_1_) * a_lim_a_;
    }

    // Total trajectory time
    T_ = Ta_ + Tv_ + Td_;

    // Round final time to discrete ticks (milliseconds)
    T_ = std::round(T_ * 1000.0) / 1000.0;
}

FullTrajectoryResult DoubleSTrajectory::evaluate(double t) const {
    // Special case: equal positions with equal velocities
    if (std::abs(state_.q_1 - state_.q_0) < kEpsilon &&
        std::abs(state_.v_1 - state_.v_0) < kEpsilon) {
        return {state_.q_0, state_.v_0, 0.0, 0.0};
    }

    // Special case: equal positions with different velocities
    if (std::abs(state_.q_1 - state_.q_0) < kEpsilon &&
        std::abs(state_.v_1 - state_.v_0) >= kEpsilon) {
        const double t_norm = std::min(t / T_, 1.0);
        const double qp_val = state_.v_0 + t_norm * (state_.v_1 - state_.v_0);

        const double phase = 2.0 * M_PI * t_norm;
        const double amplitude = (state_.v_1 - state_.v_0) * T_ / (2.0 * M_PI);
        const double q_val = state_.q_0 + amplitude * std::sin(phase);

        const double qpp_val = (state_.v_1 - state_.v_0) / T_ +
                               amplitude * (2.0 * M_PI / T_) * std::cos(phase);
        const double qppp_val = -amplitude * (2.0 * M_PI / T_) * (2.0 * M_PI / T_) *
                                std::sin(phase);

        return {q_val, qp_val, qpp_val, qppp_val};
    }

    return evaluate_7phase(t);
}

FullTrajectoryResult DoubleSTrajectory::evaluate_7phase(double t) const {
    // Clamp t to [0, T]
    t = std::clamp(t, 0.0, T_);

    // Handle zero or near-zero duration trajectory
    if (T_ < kEpsilon) {
        return {state_.q_1, state_.v_1, 0.0, 0.0};
    }

    // Use transformed coordinates
    const double q_0 = sigma_ * state_.q_0;
    const double q_1 = sigma_ * state_.q_1;
    const double v_0_t = sigma_ * state_.v_0;
    const double v_1_t = sigma_ * state_.v_1;

    // j_max and j_min in transformed space
    const double j_max = ((sigma_ + 1.0) / 2.0) * bounds_.j_bound +
                         ((sigma_ - 1.0) / 2.0) * (-bounds_.j_bound);
    const double j_min = -j_max;

    double q_val = 0.0;
    double qp_val = 0.0;
    double qpp_val = 0.0;
    double qppp_val = 0.0;

    // Phase 1: t in [0, Tj_1] -- jerk ramp-up
    if (t <= Tj_1_ && Tj_1_ > 0.0) {
        q_val = q_0 + v_0_t * t + j_max * t * t * t / 6.0;
        qp_val = v_0_t + j_max * t * t / 2.0;
        qpp_val = j_max * t;
        qppp_val = j_max;
    }
    // Phase 2: t in [Tj_1, Ta - Tj_1] -- constant acceleration
    else if (t <= (Ta_ - Tj_1_) && Ta_ > Tj_1_) {
        q_val = q_0 + v_0_t * t +
                a_lim_a_ / 6.0 * (3.0 * t * t - 3.0 * Tj_1_ * t + Tj_1_ * Tj_1_);
        qp_val = v_0_t + a_lim_a_ * (t - Tj_1_ / 2.0);
        qpp_val = a_lim_a_;
        qppp_val = 0.0;
    }
    // Phase 3: t in [Ta - Tj_1, Ta] -- jerk ramp-down (end of acceleration)
    else if (t <= Ta_ && Ta_ > 0.0) {
        const double dt = Ta_ - t;
        q_val = q_0 + (v_lim_ + v_0_t) * Ta_ / 2.0 -
                v_lim_ * dt - j_min * dt * dt * dt / 6.0;
        qp_val = v_lim_ + j_min * dt * dt / 2.0;
        qpp_val = -j_min * dt;
        qppp_val = j_min;
    }
    // Phase 4: t in [Ta, Ta + Tv] -- constant velocity
    else if (t <= (Ta_ + Tv_) && Tv_ > 0.0) {
        q_val = q_0 + (v_lim_ + v_0_t) * Ta_ / 2.0 + v_lim_ * (t - Ta_);
        qp_val = v_lim_;
        qpp_val = 0.0;
        qppp_val = 0.0;
    }
    // Phase 5: t in [Ta + Tv, Ta + Tv + Tj_2] -- start of deceleration
    else if (t <= (Ta_ + Tv_ + Tj_2_) && Tj_2_ > 0.0) {
        const double dt = t - T_ + Td_;
        q_val = q_1 - (v_lim_ + v_1_t) * Td_ / 2.0 +
                v_lim_ * dt - j_max * dt * dt * dt / 6.0;
        qp_val = v_lim_ - j_max * dt * dt / 2.0;
        qpp_val = -j_max * dt;
        qppp_val = -j_max;
    }
    // Phase 6: t in [Ta + Tv + Tj_2, Ta + Tv + Td - Tj_2] -- constant deceleration
    else if (t <= (Ta_ + Tv_ + Td_ - Tj_2_) && Td_ > Tj_2_) {
        const double dt = t - T_ + Td_;
        q_val = q_1 - (v_lim_ + v_1_t) * Td_ / 2.0 +
                v_lim_ * dt +
                a_lim_d_ / 6.0 * (3.0 * dt * dt - 3.0 * Tj_2_ * dt + Tj_2_ * Tj_2_);
        qp_val = v_lim_ + a_lim_d_ * (dt - Tj_2_ / 2.0);
        qpp_val = a_lim_d_;
        qppp_val = 0.0;
    }
    // Phase 7: t in [T - Tj_2, T] -- final jerk ramp
    else if (t <= T_ && Td_ > 0.0) {
        const double dt = T_ - t;
        q_val = q_1 - v_1_t * dt - j_max * dt * dt * dt / 6.0;
        qp_val = v_1_t + j_max * dt * dt / 2.0;
        qpp_val = -j_max * dt;
        qppp_val = j_max;
    }
    // After end of trajectory or for empty phases
    else {
        q_val = q_1;
        qp_val = v_1_t;
        qpp_val = 0.0;
        qppp_val = 0.0;
    }

    // Transform back using sigma
    return {
        sigma_ * q_val,
        sigma_ * qp_val,
        sigma_ * qpp_val,
        sigma_ * qppp_val
    };
}

std::map<std::string, double> DoubleSTrajectory::phase_durations() const {
    return {
        {"total", T_},
        {"acceleration", Ta_},
        {"constant_velocity", Tv_},
        {"deceleration", Td_},
        {"jerk_acceleration", Tj_1_},
        {"jerk_deceleration", Tj_2_}
    };
}

}  // namespace interpolatecpp::motion
