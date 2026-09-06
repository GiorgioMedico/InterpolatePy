#include "shooting_solver.hpp"

#include <Eigen/Geometry>
#include <Eigen/SparseLU>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace interpolatecpp::quat::shooting_detail {
namespace {

using Sensitivity = Eigen::Matrix<double, 10, 9>;
using Parameters = Eigen::Matrix<double, 9, 1>;
using SparseMatrix = Eigen::SparseMatrix<double>;

Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d result;
    result << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
    return result;
}

Eigen::Vector4d values(const Quaternion& q) { return {q.w(), q.x(), q.y(), q.z()}; }

Eigen::Matrix4d left_matrix(const Eigen::Vector4d& q) {
    Eigen::Matrix4d result;
    result(0, 0) = q[0];
    result.block<1, 3>(0, 1) = -q.tail<3>().transpose();
    result.block<3, 1>(1, 0) = q.tail<3>();
    result.block<3, 3>(1, 1) = q[0] * Eigen::Matrix3d::Identity() + skew(q.tail<3>());
    return result;
}

std::pair<Eigen::Vector3d, Eigen::Matrix<double, 3, 4>>
rotation_residual(Eigen::Vector4d q) {
    const double sign = q[0] < 0.0 ? -1.0 : 1.0;
    q *= sign;
    const double scalar = q[0];
    const Eigen::Vector3d vector = q.tail<3>();
    const double radius = vector.norm();
    const double norm_squared = q.squaredNorm();
    double factor;
    double coefficient;
    if (radius < 1e-7) {
        factor = 1.0 / scalar - radius * radius / (3.0 * scalar * scalar * scalar);
        coefficient = -2.0 / (3.0 * scalar * scalar * scalar);
    } else {
        factor = std::atan2(radius, scalar) / radius;
        coefficient = (scalar / norm_squared - factor) / (radius * radius);
    }
    Eigen::Matrix<double, 3, 4> derivative;
    derivative.col(0) = -vector / norm_squared;
    derivative.rightCols<3>() = factor * Eigen::Matrix3d::Identity() + coefficient * vector * vector.transpose();
    return {factor * vector, sign * derivative};
}

State rhs(const State& state, const Eigen::Vector3d& constant,
          const Sensitivity* sensitivity, Sensitivity* derivative) {
    const Eigen::Vector4d q = state.head<4>();
    const Eigen::Vector3d v = state.segment<3>(4);
    const Eigen::Vector3d a = state.tail<3>();
    State result;
    result[0] = -q.tail<3>().dot(v);
    result.segment<3>(1) = q[0] * v + q.tail<3>().cross(v);
    result.segment<3>(4) = a;
    result.tail<3>() = constant - 2.0 * v.cross(a);
    if (sensitivity != nullptr) {
        Eigen::Matrix4d right = Eigen::Matrix4d::Zero();
        right.block<1, 3>(0, 1) = -v.transpose();
        right.block<3, 1>(1, 0) = v;
        right.block<3, 3>(1, 1) = -skew(v);
        derivative->topRows<4>() = right * sensitivity->topRows<4>() +
            left_matrix(q).rightCols<3>() * sensitivity->middleRows<3>(4);
        derivative->middleRows<3>(4) = sensitivity->bottomRows<3>();
        derivative->bottomRows<3>() = 2.0 * skew(a) * sensitivity->middleRows<3>(4) -
            2.0 * skew(v) * sensitivity->bottomRows<3>();
        derivative->block<3, 3>(7, 6) += Eigen::Matrix3d::Identity();
    }
    return result;
}

State rk4_step(const State& state, const Eigen::Vector3d& constant, double step,
               Sensitivity* sensitivity, double* energy) {
    Sensitivity d1, d2, d3, d4, temporary;
    const State k1 = rhs(state, constant, sensitivity, &d1);
    const State y2 = state + 0.5 * step * k1;
    if (sensitivity != nullptr) temporary = *sensitivity + 0.5 * step * d1;
    const State k2 = rhs(y2, constant, sensitivity == nullptr ? nullptr : &temporary, &d2);
    const State y3 = state + 0.5 * step * k2;
    if (sensitivity != nullptr) temporary = *sensitivity + 0.5 * step * d2;
    const State k3 = rhs(y3, constant, sensitivity == nullptr ? nullptr : &temporary, &d3);
    const State y4 = state + step * k3;
    if (sensitivity != nullptr) temporary = *sensitivity + step * d3;
    const State k4 = rhs(y4, constant, sensitivity == nullptr ? nullptr : &temporary, &d4);
    State result = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    const double norm = result.head<4>().norm();
    result.head<4>() /= norm;
    if (sensitivity != nullptr) {
        *sensitivity += (step / 6.0) * (d1 + 2.0 * d2 + 2.0 * d3 + d4);
        const Eigen::Matrix<double, 1, 9> radial = result.head<4>().transpose() * sensitivity->topRows<4>();
        sensitivity->topRows<4>() =
            (sensitivity->topRows<4>() - result.head<4>() * radial) / norm;
    }
    if (energy != nullptr) {
        *energy += (step / 6.0) * (state.tail<3>().squaredNorm() +
            2.0 * y2.tail<3>().squaredNorm() + 2.0 * y3.tail<3>().squaredNorm() + y4.tail<3>().squaredNorm());
    }
    return result;
}

struct Integration {
    State state;
    Sensitivity sensitivity;
    std::vector<State> nodes;
    double energy = 0.0;
};

Integration integrate(const Parameters& parameters, const Quaternion& start,
                       int steps, bool jacobian, bool store_nodes = false) {
    Integration result;
    result.state << values(start), parameters.head<6>();
    result.sensitivity.setZero();
    result.sensitivity.block<6, 6>(4, 0).setIdentity();
    if (store_nodes) {
        result.nodes.reserve(static_cast<std::size_t>(steps) + 1);
        result.nodes.push_back(result.state);
    }
    for (int index = 0; index < steps; ++index) {
        result.state = rk4_step(result.state, parameters.tail<3>(), 1.0 / static_cast<double>(steps),
                                jacobian ? &result.sensitivity : nullptr, store_nodes ? &result.energy : nullptr);
        if (!result.state.allFinite()) throw std::runtime_error("Shooting integration diverged");
        if (store_nodes) result.nodes.push_back(result.state);
    }
    return result;
}

struct System {
    Eigen::VectorXd residual;
    SparseMatrix jacobian;
};

System matching_system(const Eigen::MatrixXd& parameters, const std::vector<Quaternion>& quaternions,
                        const std::vector<double>& durations, int steps, bool jacobian) {
    const Eigen::Index count = parameters.rows();
    System result{Eigen::VectorXd::Zero(9 * count), SparseMatrix(9 * count, 9 * count)};
    std::vector<Eigen::Triplet<double>> triplets;
    const auto add_block = [&triplets](Eigen::Index row, Eigen::Index column, const auto& block) {
        for (Eigen::Index i = 0; i < block.rows(); ++i) {
            for (Eigen::Index j = 0; j < block.cols(); ++j) {
                triplets.emplace_back(static_cast<int>(row + i), static_cast<int>(column + j), block(i, j));
            }
        }
    };
    for (Eigen::Index index = 0; index < count; ++index) {
        const std::size_t segment = static_cast<std::size_t>(index);
        const auto end = integrate(parameters.row(index).transpose(), quaternions[segment], steps, jacobian);
        const Eigen::Matrix4d rotation = left_matrix(values(quaternions[segment + 1].conjugate()));
        const auto [orientation, log_derivative] = rotation_residual(rotation * end.state.head<4>());
        result.residual.segment<3>(3 * index) = orientation;
        if (jacobian) {
            const Eigen::Matrix<double, 3, 9> block = log_derivative * rotation * end.sensitivity.topRows<4>();
            add_block(3 * index, 9 * index, block);
        }
        if (index + 1 < count) {
            // Scale before forming harmonic-mean weights: physical duration
            // products can overflow or underflow for otherwise valid inputs.
            const double scale = std::max(durations[segment], durations[segment + 1]);
            const double h_left = durations[segment] / scale;
            const double h_right = durations[segment + 1] / scale;
            const double left = 2.0 * h_right / (h_left + h_right);
            const double right = 2.0 * h_left / (h_left + h_right);
            Eigen::Index row = 3 * count + 3 * index;
            result.residual.segment<3>(row) = left * end.state.segment<3>(4) -
                right * parameters.block<1, 3>(index + 1, 0).transpose();
            if (jacobian) {
                add_block(row, 9 * index, (left * end.sensitivity.middleRows<3>(4)).eval());
                add_block(row, 9 * (index + 1), (-right * Eigen::Matrix3d::Identity()).eval());
            }
            row += 3 * (count - 1);
            result.residual.segment<3>(row) = left * left * end.state.tail<3>() -
                right * right * parameters.block<1, 3>(index + 1, 3).transpose();
            if (jacobian) {
                add_block(row, 9 * index, (left * left * end.sensitivity.bottomRows<3>()).eval());
                add_block(row, 9 * (index + 1) + 3, (-right * right * Eigen::Matrix3d::Identity()).eval());
            }
        } else {
            result.residual.tail<3>() = end.state.tail<3>();
            if (jacobian) add_block(9 * count - 3, 9 * index, end.sensitivity.bottomRows<3>());
        }
    }
    result.residual.segment<3>(9 * count - 6) = parameters.block<1, 3>(0, 3).transpose();
    if (!result.residual.allFinite()) {
        throw std::runtime_error("Multiple shooting matching residual is not finite");
    }
    if (jacobian) {
        add_block(9 * count - 6, 3, Eigen::Matrix3d::Identity());
        result.jacobian.setFromTriplets(triplets.begin(), triplets.end());
    }
    return result;
}

int newton_solve(Eigen::MatrixXd& parameters, const std::vector<Quaternion>& quaternions,
                 const std::vector<double>& durations, int steps, double tolerance, int iterations) {
    Eigen::SparseLU<SparseMatrix> solver;
    for (int iteration = 0; iteration <= iterations; ++iteration) {
        const System current = matching_system(parameters, quaternions, durations, steps, true);
        if (current.residual.cwiseAbs().maxCoeff() <= tolerance) return iteration;
        if (iteration == iterations) break;
        solver.compute(current.jacobian);
        if (solver.info() != Eigen::Success) throw std::runtime_error("Multiple shooting Jacobian is singular");
        const Eigen::VectorXd solution = solver.solve(-current.residual);
        if (solver.info() != Eigen::Success || !solution.allFinite()) {
            throw std::runtime_error("Multiple shooting Jacobian solve failed");
        }
        Eigen::MatrixXd direction(parameters.rows(), 9);
        for (Eigen::Index index = 0; index < parameters.rows(); ++index) {
            direction.row(index) = solution.segment<9>(9 * index).transpose();
        }
        double step = 1.0;
        bool accepted = false;
        for (int backtrack = 0; backtrack < 20; ++backtrack) {
            Eigen::MatrixXd candidate = parameters + step * direction;
            try {
                const auto trial = matching_system(candidate, quaternions, durations, steps, false);
                if (trial.residual.norm() < (1.0 - 1e-4 * step) * current.residual.norm()) {
                    parameters = std::move(candidate);
                    accepted = true;
                    break;
                }
            } catch (const std::runtime_error&) {
                // A divergent trial integration is rejected by backtracking.
            }
            step *= 0.5;
        }
        if (!accepted) throw std::runtime_error("Multiple shooting line search failed to reduce the matching residual");
    }
    throw std::runtime_error("Multiple shooting did not converge within max_iterations");
}

}  // namespace

State evaluate_step(const State& state, const Eigen::Vector3d& constant, double step) {
    return rk4_step(state, constant, step, nullptr, nullptr);
}

Solution solve(const std::vector<Quaternion>& quaternions,
               const std::vector<double>& durations, const ShootingConfig& config) {
    const Eigen::Index count = static_cast<Eigen::Index>(durations.size());
    Solution result;
    result.parameters = Eigen::MatrixXd::Zero(count, 9);
    for (Eigen::Index index = 0; index < count; ++index) {
        const auto segment = static_cast<std::size_t>(index);
        const Eigen::Vector4d relative = left_matrix(values(quaternions[segment].conjugate())) * values(quaternions[segment + 1]);
        result.parameters.block<1, 3>(index, 0) = rotation_residual(relative).first.transpose();
    }
    int steps = config.integration_steps;
    while (steps < config.max_integration_steps) {
        result.iterations += newton_solve(result.parameters, quaternions, durations, steps,
                                          config.tolerance * 0.1, config.max_iterations - result.iterations);
        const int finer = steps > config.max_integration_steps / 2 ? config.max_integration_steps : 2 * steps;
        const auto verification = matching_system(result.parameters, quaternions, durations, finer, false);
        result.residual = verification.residual.cwiseAbs().maxCoeff();
        if (result.residual <= config.tolerance) {
            result.steps = finer;
            break;
        }
        steps = finer;
    }
    if (result.steps == 0) {
        throw std::runtime_error("Multiple shooting integration accuracy exceeds tolerance at max_integration_steps");
    }
    result.nodes.reserve(durations.size());
    for (Eigen::Index index = 0; index < count; ++index) {
        const auto segment = static_cast<std::size_t>(index);
        auto curve = integrate(result.parameters.row(index).transpose(), quaternions[segment], result.steps, false, true);
        result.nodes.push_back(std::move(curve.nodes));
        const double duration = durations[segment];
        result.energy += 4.0 * curve.energy / duration / duration / duration;
    }
    return result;
}

}  // namespace interpolatecpp::quat::shooting_detail
