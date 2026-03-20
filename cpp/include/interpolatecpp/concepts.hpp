#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <concepts>

namespace interpolatecpp {

/// Concept for scalar (1D) trajectory evaluation.
/// Conforming types: CubicSpline, CubicSmoothingSpline,
/// CubicSplineWithAcceleration1, CubicSplineWithAcceleration2
template <typename T>
concept ScalarTrajectory = requires(const T& traj, double t, const Eigen::VectorXd& tv) {
    { traj.evaluate(t) } -> std::convertible_to<double>;
    { traj.evaluate(tv) } -> std::convertible_to<Eigen::VectorXd>;
    { traj.evaluate_velocity(t) } -> std::convertible_to<double>;
    { traj.evaluate_velocity(tv) } -> std::convertible_to<Eigen::VectorXd>;
    { traj.evaluate_acceleration(t) } -> std::convertible_to<double>;
    { traj.evaluate_acceleration(tv) } -> std::convertible_to<Eigen::VectorXd>;
};

/// Concept for parametric curve evaluation with derivative support.
/// Conforming types: BSpline family (Phase 2)
template <typename T>
concept CurveEvaluator = requires(const T& curve, double u, int order) {
    { curve.evaluate(u) } -> std::convertible_to<Eigen::VectorXd>;
    { curve.evaluate_derivative(u, order) } -> std::convertible_to<Eigen::VectorXd>;
};

/// Concept for 3D geometric path evaluation by arc length.
/// Conforming types: LinearPath, CircularPath (Phase 5)
template <typename T>
concept GeometricPath = requires(const T& path, double s, const Eigen::VectorXd& sv) {
    { path.position(s) } -> std::convertible_to<Eigen::Vector3d>;
    { path.position(sv) } -> std::convertible_to<Eigen::MatrixXd>;
    { path.velocity(s) } -> std::convertible_to<Eigen::Vector3d>;
    { path.acceleration(s) } -> std::convertible_to<Eigen::Vector3d>;
};

/// Concept for quaternion-valued trajectory evaluation.
/// Conforming types: SquadC2, LogQuaternionInterpolation (Phase 4)
template <typename T>
concept QuaternionTrajectory = requires(const T& traj, double t) {
    { traj.evaluate(t) } -> std::convertible_to<Eigen::Quaterniond>;
    { traj.evaluate_velocity(t) } -> std::convertible_to<Eigen::Vector3d>;
    { traj.evaluate_acceleration(t) } -> std::convertible_to<Eigen::Vector3d>;
};

/// Concept for callable trajectory functions returning (pos, vel, acc).
/// Conforming types: Motion profile callables (Phase 3)
template <typename T>
concept TrajectoryFunction = requires(const T& func, double t) {
    { func(t) } -> std::convertible_to<std::tuple<double, double, double>>;
};

}  // namespace interpolatecpp
