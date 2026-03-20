#include <interpolatecpp/path/frenet_frame.hpp>

#include <Eigen/Geometry>
#include <cmath>

namespace interpolatecpp::path {

std::vector<FrenetFrame> compute_frenet_frames(
    const std::function<std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>(
        double)>& curve,
    const Eigen::VectorXd& s_values) {
    std::vector<FrenetFrame> frames;
    frames.reserve(s_values.size());

    for (Eigen::Index i = 0; i < s_values.size(); ++i) {
        auto [pos, vel, acc] = curve(s_values[i]);

        FrenetFrame frame;
        double vel_norm = vel.norm();

        if (vel_norm < 1e-10) {
            frame.tangent = Eigen::Vector3d::UnitX();
            frame.normal = Eigen::Vector3d::UnitY();
            frame.binormal = Eigen::Vector3d::UnitZ();
            frame.curvature = 0.0;
            frame.torsion = 0.0;
        } else {
            frame.tangent = vel / vel_norm;

            Eigen::Vector3d cross = vel.cross(acc);
            double cross_norm = cross.norm();

            frame.curvature = cross_norm / (vel_norm * vel_norm * vel_norm);

            if (cross_norm < 1e-10) {
                // Straight line - pick arbitrary normal
                Eigen::Vector3d arbitrary =
                    (std::abs(frame.tangent.dot(Eigen::Vector3d::UnitX())) < 0.9)
                        ? Eigen::Vector3d::UnitX()
                        : Eigen::Vector3d::UnitY();
                frame.binormal = frame.tangent.cross(arbitrary).normalized();
                frame.normal = frame.binormal.cross(frame.tangent);
            } else {
                frame.binormal = cross / cross_norm;
                frame.normal = frame.binormal.cross(frame.tangent);
            }

            frame.torsion = 0.0;  // Requires third derivative for torsion
        }

        frames.push_back(frame);
    }

    return frames;
}

std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>
circular_trajectory_with_derivatives(double u, double r) {
    Eigen::Vector3d p(r * std::cos(u), r * std::sin(u), 0.0);
    Eigen::Vector3d dp(-r * std::sin(u), r * std::cos(u), 0.0);
    Eigen::Vector3d d2p(-r * std::cos(u), -r * std::sin(u), 0.0);
    return {p, dp, d2p};
}

std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>
helicoidal_trajectory_with_derivatives(double u, double r, double d) {
    Eigen::Vector3d p(r * std::cos(u), r * std::sin(u), d * u);
    Eigen::Vector3d dp(-r * std::sin(u), r * std::cos(u), d);
    Eigen::Vector3d d2p(-r * std::cos(u), -r * std::sin(u), 0.0);
    return {p, dp, d2p};
}

}  // namespace interpolatecpp::path
