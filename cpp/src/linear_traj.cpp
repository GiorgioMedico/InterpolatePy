#include <interpolatecpp/path/linear_traj.hpp>

namespace interpolatecpp::path {

LinearTrajResult linear_traj(const Eigen::VectorXd& p0, const Eigen::VectorXd& p1,
                             double t0, double t1, int num_points) {
    const int dim = static_cast<int>(p0.size());
    const double dt = (num_points > 1) ? (t1 - t0) / (num_points - 1) : 0.0;
    Eigen::VectorXd vel;
    if (t1 > t0) {
        vel = (p1 - p0) / (t1 - t0);
    } else {
        vel = Eigen::VectorXd::Zero(dim);
    }

    LinearTrajResult result;
    result.positions = Eigen::MatrixXd(num_points, dim);
    result.velocities = Eigen::MatrixXd(num_points, dim);
    result.accelerations = Eigen::MatrixXd::Zero(num_points, dim);

    for (int i = 0; i < num_points; ++i) {
        double t = t0 + i * dt;
        double u = (t1 > t0) ? (t - t0) / (t1 - t0) : 0.0;
        result.positions.row(i) = (p0 + u * (p1 - p0)).transpose();
        result.velocities.row(i) = vel.transpose();
    }

    return result;
}

}  // namespace interpolatecpp::path
