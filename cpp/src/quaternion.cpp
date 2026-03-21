#include <interpolatecpp/quat/quaternion.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace interpolatecpp::quat {

Quaternion::Quaternion(double w, double x, double y, double z)
    : q_(w, x, y, z) {}

Quaternion::Quaternion(const Eigen::Quaterniond& eq) : q_(eq) {}

Quaternion Quaternion::identity() { return Quaternion(1.0, 0.0, 0.0, 0.0); }

Quaternion Quaternion::from_angle_axis(double angle, const Eigen::Vector3d& axis) {
    Eigen::Vector3d n = axis.normalized();
    double half = angle / 2.0;
    return Quaternion(std::cos(half), n.x() * std::sin(half), n.y() * std::sin(half),
                      n.z() * std::sin(half));
}

Quaternion Quaternion::from_euler_angles(double roll, double pitch, double yaw) {
    Eigen::Quaterniond eq =
        Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());
    return Quaternion(eq);
}

Quaternion Quaternion::operator*(const Quaternion& other) const {
    return Quaternion(q_ * other.q_);
}

Quaternion Quaternion::operator*(double scalar) const {
    return Quaternion(q_.w() * scalar, q_.x() * scalar, q_.y() * scalar,
                      q_.z() * scalar);
}

Quaternion Quaternion::operator+(const Quaternion& other) const {
    return Quaternion(q_.w() + other.q_.w(), q_.x() + other.q_.x(),
                      q_.y() + other.q_.y(), q_.z() + other.q_.z());
}

Quaternion Quaternion::operator-(const Quaternion& other) const {
    return Quaternion(q_.w() - other.q_.w(), q_.x() - other.q_.x(),
                      q_.y() - other.q_.y(), q_.z() - other.q_.z());
}

Quaternion Quaternion::operator-() const {
    return Quaternion(-q_.w(), -q_.x(), -q_.y(), -q_.z());
}

Quaternion Quaternion::conjugate() const {
    return Quaternion(q_.w(), -q_.x(), -q_.y(), -q_.z());
}

Quaternion Quaternion::inverse() const {
    double ns = norm_squared();
    if (ns < kEpsilon * kEpsilon) {
        return identity();
    }
    auto c = conjugate();
    return Quaternion(c.w() / ns, c.x() / ns, c.y() / ns, c.z() / ns);
}

Quaternion Quaternion::unit() const {
    double n = norm();
    if (n < kEpsilon) {
        return identity();
    }
    return Quaternion(q_.w() / n, q_.x() / n, q_.y() / n, q_.z() / n);
}

double Quaternion::norm() const noexcept { return q_.norm(); }

double Quaternion::norm_squared() const noexcept { return q_.squaredNorm(); }

double Quaternion::dot_product(const Quaternion& other) const {
    return q_.w() * other.q_.w() + q_.x() * other.q_.x() + q_.y() * other.q_.y() +
           q_.z() * other.q_.z();
}

Quaternion Quaternion::exp(const Quaternion& q) {
    Eigen::Vector3d v(q.x(), q.y(), q.z());
    double theta = v.norm();

    if (theta < kEpsilon) {
        return Quaternion(1.0, v.x(), v.y(), v.z());
    }

    double sinc = std::sin(theta) / theta;
    return Quaternion(std::cos(theta), v.x() * sinc, v.y() * sinc, v.z() * sinc);
}

Quaternion Quaternion::log(const Quaternion& q) {
    double s = std::clamp(q.w(), -1.0, 1.0);
    double theta = std::acos(s);
    Eigen::Vector3d v = q.vec();
    double sin_theta = std::sin(theta);

    if (std::abs(sin_theta) < kEpsilon) {
        return Quaternion(0.0, v.x(), v.y(), v.z());
    }

    double factor = theta / sin_theta;
    return Quaternion(0.0, v.x() * factor, v.y() * factor, v.z() * factor);
}

Quaternion Quaternion::power(const Quaternion& q, double t) {
    return exp(log(q) * t);
}

Quaternion Quaternion::slerp(const Quaternion& q0, const Quaternion& q1, double t) {
    // Handle double-cover
    Quaternion target = q1;
    if (q0.dot_product(q1) < 0.0) {
        target = -q1;
    }

    Quaternion rel = q0.inverse() * target;
    return q0 * power(rel, t);
}

Quaternion Quaternion::slerp_prime(const Quaternion& q0, const Quaternion& q1,
                                  double t) {
    Quaternion target = q1;
    if (q0.dot_product(q1) < 0.0) {
        target = -q1;
    }

    Quaternion rel = q0.inverse() * target;
    Quaternion log_rel = log(rel);
    Quaternion result = slerp(q0, target, t);
    return result * log_rel;
}

Quaternion Quaternion::squad(const Quaternion& p, const Quaternion& a,
                             const Quaternion& b, const Quaternion& q, double t) {
    Quaternion slerp_pq = slerp(p, q, t);
    Quaternion slerp_ab = slerp(a, b, t);
    return slerp(slerp_pq, slerp_ab, 2.0 * t * (1.0 - t));
}

Quaternion Quaternion::compute_intermediate_quaternion(const Quaternion& q_prev,
                                                       const Quaternion& q_curr,
                                                       const Quaternion& q_next) {
    Quaternion q_inv = q_curr.inverse();

    // Handle double-cover for both neighbors
    Quaternion next = q_next;
    if (q_curr.dot_product(q_next) < 0.0) next = -q_next;
    Quaternion prev = q_prev;
    if (q_curr.dot_product(q_prev) < 0.0) prev = -q_prev;

    Quaternion log_next = log(q_inv * next);
    Quaternion log_prev = log(q_inv * prev);

    Quaternion sum = (log_next + log_prev) * (-0.25);
    return q_curr * exp(sum);
}

Eigen::Matrix3d Quaternion::to_rotation_matrix() const {
    return q_.normalized().toRotationMatrix();
}

std::pair<Eigen::Vector3d, double> Quaternion::to_axis_angle() const {
    Eigen::AngleAxisd aa(q_.normalized());
    double angle = aa.angle();
    Eigen::Vector3d axis = aa.axis();

    if (angle < kEpsilon) {
        return {Eigen::Vector3d::UnitX(), 0.0};
    }
    return {axis, angle};
}

Eigen::Matrix4d Quaternion::to_transformation_matrix() const {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = to_rotation_matrix();
    return T;
}

std::tuple<double, double, double> Quaternion::to_euler_angles() const {
    double w = q_.w(), x = q_.x(), y = q_.y(), z = q_.z();

    // Roll (x-axis rotation)
    double sinr_cosp = 2.0 * (w * x + y * z);
    double cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
    double roll = std::atan2(sinr_cosp, cosr_cosp);

    // Pitch (y-axis rotation)
    double sinp = 2.0 * (w * y - z * x);
    double pitch = (std::abs(sinp) >= 1.0)
                       ? std::copysign(M_PI / 2.0, sinp)
                       : std::asin(sinp);

    // Yaw (z-axis rotation)
    double siny_cosp = 2.0 * (w * z + x * y);
    double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
    double yaw = std::atan2(siny_cosp, cosy_cosp);

    return {roll, pitch, yaw};
}

Quaternion Quaternion::from_rotation_matrix(const Eigen::Matrix3d& rotation_matrix) {
    Eigen::Quaterniond eq(rotation_matrix);
    eq.normalize();
    return Quaternion(eq);
}

Eigen::Matrix3d Quaternion::E(int sign) const {
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Vector3d v = vec();

    // Skew-symmetric matrix of v
    Eigen::Matrix3d S;
    S << 0, -v.z(), v.y(),
         v.z(), 0, -v.x(),
         -v.y(), v.x(), 0;

    if (sign == 1) {
        // Body frame: E = sI + S(v)
        return w() * I + S;
    }
    // Base frame: E = sI - S(v)
    return w() * I - S;
}

Quaternion Quaternion::dot(const Eigen::Vector3d& omega, int sign) const {
    // Scalar derivative: s_dot = -0.5 * v^T * omega
    double s_dot = -0.5 * vec().dot(omega);

    // Vector derivative: v_dot = 0.5 * E(sign) * omega
    Eigen::Vector3d v_dot = 0.5 * E(sign) * omega;

    return Quaternion(s_dot, v_dot.x(), v_dot.y(), v_dot.z());
}

Eigen::Vector3d Quaternion::Omega(const Quaternion& q, const Quaternion& q_dot) {
    Eigen::Matrix3d e_matrix = 0.5 * q.E(0);  // Base frame
    Eigen::Vector3d v_dot(q_dot.x(), q_dot.y(), q_dot.z());

    double det = e_matrix.determinant();
    if (std::abs(det) > kEpsilon) {
        return e_matrix.inverse() * v_dot;
    }
    // Fallback to pseudo-inverse for singular case
    return e_matrix.completeOrthogonalDecomposition().solve(v_dot);
}

}  // namespace interpolatecpp::quat
