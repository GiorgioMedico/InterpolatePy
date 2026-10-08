#include <catch2/catch_test_macros.hpp>
#include <interpolatecpp/quat/log_quaternion_interpolation.hpp>
#include <interpolatecpp/quat/modified_log_quaternion_interpolation.hpp>

using namespace interpolatecpp::quat;

namespace {

std::vector<Quaternion> quats() {
    return {Quaternion(1, 0, 0, 0), Quaternion(0.7071067811865476, 0, 0.7071067811865476, 0),
            Quaternion(0, 0, 1, 0)};
}

const std::vector<double> kTimes{0.0, 1.0, 2.0};

}  // namespace

TEST_CASE("log quat exposes its B-spline", "[quat][accessor]") {
    const Eigen::VectorXd zero = Eigen::VectorXd::Zero(3);
    LogQuaternionInterpolation interp(kTimes, quats(), 3, zero, zero);
    const auto& spline = interp.bspline_interpolator();
    REQUIRE(spline.degree() == 3);
    REQUIRE(spline.knots().size() > 0);
}

TEST_CASE("modified log quat angular velocity is zero at rest ends", "[quat][accessor]") {
    const Eigen::VectorXd zero = Eigen::VectorXd::Zero(4);
    ModifiedLogQuaternionInterpolation interp(kTimes, quats(), 3, true, zero, zero);
    REQUIRE(interp.angular_velocity(0.0).norm() < 1e-9);
    REQUIRE(interp.angular_velocity(2.0).norm() < 1e-9);
    REQUIRE(interp.angular_velocity(1.0).norm() > 0.1);
}
