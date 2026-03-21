#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <interpolatecpp/spline/cubic_spline.hpp>

#include "test_data.hpp"

#include <cmath>
#include <vector>

using namespace interpolatecpp::spline;
using namespace interpolatecpp::test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("CubicSpline construction", "[cubic_spline]") {
    SECTION("Basic construction") {
        std::vector<double> t = {0.0, 1.0, 2.0, 3.0};
        std::vector<double> q = {0.0, 1.0, 0.0, 1.0};
        CubicSpline spline(t, q);

        REQUIRE(spline.n_segments() == 3);
        REQUIRE(spline.t_points().size() == 4);
        REQUIRE(spline.q_points().size() == 4);
        REQUIRE(spline.velocities().size() == 4);
        REQUIRE(spline.coefficients().rows() == 3);
        REQUIRE(spline.coefficients().cols() == 4);
    }

    SECTION("With boundary velocities") {
        std::vector<double> t = {0.0, 1.0, 2.0, 3.0};
        std::vector<double> q = {0.0, 1.0, 0.0, 1.0};
        CubicSpline spline(t, q, 1.0, -1.0);

        REQUIRE_THAT(spline.velocities()(0), WithinAbs(1.0, kRegularAtol));
        REQUIRE_THAT(spline.velocities()(3), WithinAbs(-1.0, kRegularAtol));
    }

    SECTION("Mismatched lengths throw") {
        std::vector<double> t = {0.0, 1.0, 2.0};
        std::vector<double> q = {0.0, 1.0};
        REQUIRE_THROWS_AS(CubicSpline(t, q), std::invalid_argument);
    }

    SECTION("Non-increasing time throws") {
        std::vector<double> t = {0.0, 2.0, 1.0, 3.0};
        std::vector<double> q = {0.0, 1.0, 0.0, 1.0};
        REQUIRE_THROWS_AS(CubicSpline(t, q), std::invalid_argument);
    }

    SECTION("Repeated time throws") {
        std::vector<double> t = {0.0, 1.0, 1.0, 3.0};
        std::vector<double> q = {0.0, 1.0, 0.0, 1.0};
        REQUIRE_THROWS_AS(CubicSpline(t, q), std::invalid_argument);
    }

    SECTION("Single segment (2 points)") {
        std::vector<double> t = {0.0, 1.0};
        std::vector<double> q = {0.0, 1.0};
        CubicSpline spline(t, q);

        REQUIRE(spline.n_segments() == 1);
        REQUIRE_THAT(spline.evaluate(0.0), WithinAbs(0.0, kRegularAtol));
        REQUIRE_THAT(spline.evaluate(1.0), WithinAbs(1.0, kRegularAtol));
    }

    SECTION("Two segments (3 points)") {
        std::vector<double> t = {0.0, 1.0, 2.0};
        std::vector<double> q = {0.0, 1.0, 0.0};
        CubicSpline spline(t, q);

        REQUIRE(spline.n_segments() == 2);
    }
}

TEST_CASE("CubicSpline mathematical accuracy", "[cubic_spline]") {
    SECTION("Linear function is exact") {
        // y = 2x with v0=2, vn=2
        std::vector<double> t = {0.0, 1.0, 2.0, 3.0};
        std::vector<double> q = {0.0, 2.0, 4.0, 6.0};
        CubicSpline spline(t, q, 2.0, 2.0);

        for (double ti = 0.0; ti <= 3.0; ti += 0.25) {
            REQUIRE_THAT(spline.evaluate(ti), WithinAbs(2.0 * ti, kRegularAtol));
            REQUIRE_THAT(spline.evaluate_velocity(ti), WithinAbs(2.0, kNumericalAtol));
            REQUIRE_THAT(spline.evaluate_acceleration(ti), WithinAbs(0.0, kNumericalAtol));
        }
    }

    SECTION("Quadratic function") {
        // y = x^2, v0=0, vn=6
        std::vector<double> t = {0.0, 1.0, 2.0, 3.0};
        std::vector<double> q = {0.0, 1.0, 4.0, 9.0};
        CubicSpline spline(t, q, 0.0, 6.0);

        for (double ti = 0.0; ti <= 3.0; ti += 0.5) {
            REQUIRE_THAT(spline.evaluate(ti), WithinAbs(ti * ti, kNumericalAtol));
        }
    }

    SECTION("Waypoint interpolation exactness") {
        std::vector<double> t = {0.0, 1.0, 2.0, 3.0};
        std::vector<double> q = {0.0, 1.0, 0.0, 1.0};
        CubicSpline spline(t, q);

        // Must pass exactly through waypoints
        for (int i = 0; i < 4; ++i) {
            REQUIRE_THAT(spline.evaluate(t[static_cast<size_t>(i)]),
                         WithinAbs(q[static_cast<size_t>(i)], kRegularAtol));
        }
    }

    SECTION("Symmetric trajectory") {
        std::vector<double> t = {0.0, 1.0, 2.0, 3.0, 4.0};
        std::vector<double> q = {0.0, 1.0, 2.0, 1.0, 0.0};
        CubicSpline spline(t, q);

        // Symmetric about midpoint
        REQUIRE_THAT(spline.evaluate(1.0), WithinAbs(spline.evaluate(3.0), kNumericalAtol));
        REQUIRE_THAT(spline.evaluate(0.5), WithinAbs(spline.evaluate(3.5), kNumericalAtol));
    }
}

TEST_CASE("CubicSpline C2 continuity", "[cubic_spline]") {
    std::vector<double> t = {0.0, 1.0, 2.0, 3.0};
    std::vector<double> q = {0.0, 1.0, 0.0, 1.0};
    CubicSpline spline(t, q);
    double eps = 1e-8;

    SECTION("C0 continuity (position)") {
        for (int i = 1; i < 3; ++i) {
            double ti = t[static_cast<size_t>(i)];
            double left = spline.evaluate(ti - eps);
            double right = spline.evaluate(ti + eps);
            REQUIRE_THAT(left, WithinAbs(right, kNumericalAtol));
        }
    }

    SECTION("C1 continuity (velocity)") {
        for (int i = 1; i < 3; ++i) {
            double ti = t[static_cast<size_t>(i)];
            double left = spline.evaluate_velocity(ti - eps);
            double right = spline.evaluate_velocity(ti + eps);
            REQUIRE_THAT(left, WithinAbs(right, kNumericalAtol));
        }
    }

    SECTION("C2 continuity (acceleration)") {
        for (int i = 1; i < 3; ++i) {
            double ti = t[static_cast<size_t>(i)];
            double left = spline.evaluate_acceleration(ti - eps);
            double right = spline.evaluate_acceleration(ti + eps);
            REQUIRE_THAT(left, WithinAbs(right, kNumericalAtol));
        }
    }
}

TEST_CASE("CubicSpline evaluation methods", "[cubic_spline]") {
    std::vector<double> t = {0.0, 1.0, 2.0, 3.0};
    std::vector<double> q = {0.0, 1.0, 0.0, 1.0};
    CubicSpline spline(t, q);

    SECTION("Vectorized evaluation matches scalar") {
        Eigen::VectorXd tv = Eigen::VectorXd::LinSpaced(20, 0.0, 3.0);
        Eigen::VectorXd result = spline.evaluate(tv);

        for (Eigen::Index i = 0; i < tv.size(); ++i) {
            REQUIRE_THAT(result(i), WithinAbs(spline.evaluate(tv(i)), kRegularAtol));
        }
    }

    SECTION("Vectorized velocity matches scalar") {
        Eigen::VectorXd tv = Eigen::VectorXd::LinSpaced(20, 0.0, 3.0);
        Eigen::VectorXd result = spline.evaluate_velocity(tv);

        for (Eigen::Index i = 0; i < tv.size(); ++i) {
            REQUIRE_THAT(result(i), WithinAbs(spline.evaluate_velocity(tv(i)), kRegularAtol));
        }
    }

    SECTION("Boundary extrapolation (clamp)") {
        // Before start: clamps to segment 0, tau=0
        double before = spline.evaluate(-1.0);
        double at_start = spline.evaluate(0.0);
        REQUIRE_THAT(before, WithinAbs(at_start, kRegularAtol));

        // After end: clamps to last segment end
        double after = spline.evaluate(4.0);
        double at_end = spline.evaluate(3.0);
        REQUIRE_THAT(after, WithinAbs(at_end, kRegularAtol));
    }

    SECTION("Evaluation consistency via finite differences") {
        double t_test = 1.5;
        double h = 1e-7;
        double pos = spline.evaluate(t_test);
        double pos_h = spline.evaluate(t_test + h);
        double vel = spline.evaluate_velocity(t_test);
        double numerical_vel = (pos_h - pos) / h;

        REQUIRE_THAT(vel, WithinAbs(numerical_vel, kLooseAtol));
    }
}

TEST_CASE("CubicSpline edge cases", "[cubic_spline]") {
    SECTION("Identical positions") {
        std::vector<double> t = {0.0, 1.0, 2.0, 3.0};
        std::vector<double> q = {5.0, 5.0, 5.0, 5.0};
        CubicSpline spline(t, q);

        REQUIRE_THAT(spline.evaluate(1.5), WithinAbs(5.0, kNumericalAtol));
        REQUIRE_THAT(spline.evaluate_velocity(1.5), WithinAbs(0.0, kNumericalAtol));
    }

    SECTION("Negative time values") {
        std::vector<double> t = {-2.0, -1.0, 0.0, 1.0};
        std::vector<double> q = {0.0, 1.0, 0.0, 1.0};
        CubicSpline spline(t, q);

        REQUIRE(spline.n_segments() == 3);
        REQUIRE_THAT(spline.evaluate(-2.0), WithinAbs(0.0, kRegularAtol));
        REQUIRE_THAT(spline.evaluate(1.0), WithinAbs(1.0, kRegularAtol));
    }

    SECTION("Large time intervals") {
        std::vector<double> t = {0.0, 100.0, 200.0, 300.0};
        std::vector<double> q = {0.0, 1.0, 0.0, 1.0};
        CubicSpline spline(t, q);

        REQUIRE(std::isfinite(spline.evaluate(150.0)));
        REQUIRE(std::isfinite(spline.evaluate_velocity(150.0)));
    }

    SECTION("Zero boundary velocities") {
        std::vector<double> t = {0.0, 1.0, 2.0, 3.0};
        std::vector<double> q = {0.0, 1.0, 0.0, 1.0};
        CubicSpline spline(t, q, 0.0, 0.0);

        REQUIRE_THAT(spline.evaluate_velocity(0.0), WithinAbs(0.0, kRegularAtol));
        REQUIRE_THAT(spline.evaluate_velocity(3.0), WithinAbs(0.0, kRegularAtol));
    }
}

TEST_CASE("CubicSpline numerical stability", "[cubic_spline]") {
    SECTION("Convergence with increasing points") {
        // Interpolate sin(x) on [0, pi] — error should decrease with more points
        auto make_spline = [](int n) {
            std::vector<double> t(static_cast<size_t>(n));
            std::vector<double> q(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) {
                t[static_cast<size_t>(i)] = M_PI * i / (n - 1);
                q[static_cast<size_t>(i)] = std::sin(t[static_cast<size_t>(i)]);
            }
            return CubicSpline(t, q, std::cos(0.0), std::cos(M_PI));
        };

        auto spline_10 = make_spline(10);
        auto spline_50 = make_spline(50);

        // Evaluate at midpoint
        double error_10 = std::abs(spline_10.evaluate(M_PI / 2.0) - 1.0);
        double error_50 = std::abs(spline_50.evaluate(M_PI / 2.0) - 1.0);

        REQUIRE(error_50 < error_10);
    }
}
