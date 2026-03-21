#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <interpolatecpp/tridiagonal.hpp>

#include "test_data.hpp"

using namespace interpolatecpp;
using namespace interpolatecpp::test;

TEST_CASE("Tridiagonal solver", "[tridiagonal]") {
    SECTION("4x4 known-answer test") {
        // System from Python docstring example:
        // a = [0, 1, 2, 3], b = [2, 3, 4, 5], c = [1, 2, 3, 0], d = [1, 2, 3, 4]
        Eigen::VectorXd lower = (Eigen::VectorXd(4) << 0, 1, 2, 3).finished();
        Eigen::VectorXd main = (Eigen::VectorXd(4) << 2, 3, 4, 5).finished();
        Eigen::VectorXd upper = (Eigen::VectorXd(4) << 1, 2, 3, 0).finished();
        Eigen::VectorXd rhs = (Eigen::VectorXd(4) << 1, 2, 3, 4).finished();

        auto x = solve_tridiagonal(lower, main, upper, rhs);

        // Verify: A*x should equal d
        // Manually reconstruct A*x for verification
        Eigen::VectorXd ax(4);
        ax(0) = main(0) * x(0) + upper(0) * x(1);
        ax(1) = lower(1) * x(0) + main(1) * x(1) + upper(1) * x(2);
        ax(2) = lower(2) * x(1) + main(2) * x(2) + upper(2) * x(3);
        ax(3) = lower(3) * x(2) + main(3) * x(3);

        for (int i = 0; i < 4; ++i) {
            REQUIRE_THAT(ax(i), Catch::Matchers::WithinAbs(rhs(i), kRegularAtol));
        }
    }

    SECTION("Identity-like system (diagonal = 1)") {
        int n = 5;
        Eigen::VectorXd lower = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd main = Eigen::VectorXd::Ones(n);
        Eigen::VectorXd upper = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd rhs = (Eigen::VectorXd(n) << 1, 2, 3, 4, 5).finished();

        auto x = solve_tridiagonal(lower, main, upper, rhs);

        for (int i = 0; i < n; ++i) {
            REQUIRE_THAT(x(i), Catch::Matchers::WithinAbs(rhs(i), kRegularAtol));
        }
    }

    SECTION("2x2 system") {
        Eigen::VectorXd lower = (Eigen::VectorXd(2) << 0, 1).finished();
        Eigen::VectorXd main = (Eigen::VectorXd(2) << 2, 3).finished();
        Eigen::VectorXd upper = (Eigen::VectorXd(2) << 1, 0).finished();
        Eigen::VectorXd rhs = (Eigen::VectorXd(2) << 5, 7).finished();

        auto x = solve_tridiagonal(lower, main, upper, rhs);

        // Verify: [2 1; 1 3] * x = [5; 7]
        REQUIRE_THAT(main(0) * x(0) + upper(0) * x(1),
                     Catch::Matchers::WithinAbs(rhs(0), kRegularAtol));
        REQUIRE_THAT(lower(1) * x(0) + main(1) * x(1),
                     Catch::Matchers::WithinAbs(rhs(1), kRegularAtol));
    }

    SECTION("1x1 system") {
        Eigen::VectorXd lower = (Eigen::VectorXd(1) << 0).finished();
        Eigen::VectorXd main = (Eigen::VectorXd(1) << 4).finished();
        Eigen::VectorXd upper = (Eigen::VectorXd(1) << 0).finished();
        Eigen::VectorXd rhs = (Eigen::VectorXd(1) << 8).finished();

        auto x = solve_tridiagonal(lower, main, upper, rhs);

        REQUIRE_THAT(x(0), Catch::Matchers::WithinAbs(2.0, kRegularAtol));
    }

    SECTION("Throws on zero pivot") {
        Eigen::VectorXd lower = (Eigen::VectorXd(2) << 0, 1).finished();
        Eigen::VectorXd main = (Eigen::VectorXd(2) << 0, 3).finished();  // zero pivot
        Eigen::VectorXd upper = (Eigen::VectorXd(2) << 1, 0).finished();
        Eigen::VectorXd rhs = (Eigen::VectorXd(2) << 5, 7).finished();

        REQUIRE_THROWS_AS(solve_tridiagonal(lower, main, upper, rhs), std::invalid_argument);
    }

    SECTION("Symmetric positive definite system") {
        // SPD tridiagonal: main = [4,4,4], off-diag = [1,1]
        Eigen::VectorXd lower = (Eigen::VectorXd(3) << 0, 1, 1).finished();
        Eigen::VectorXd main = (Eigen::VectorXd(3) << 4, 4, 4).finished();
        Eigen::VectorXd upper = (Eigen::VectorXd(3) << 1, 1, 0).finished();
        Eigen::VectorXd rhs = (Eigen::VectorXd(3) << 6, 12, 10).finished();

        auto x = solve_tridiagonal(lower, main, upper, rhs);

        // Verify A*x = rhs
        Eigen::VectorXd ax(3);
        ax(0) = main(0) * x(0) + upper(0) * x(1);
        ax(1) = lower(1) * x(0) + main(1) * x(1) + upper(1) * x(2);
        ax(2) = lower(2) * x(1) + main(2) * x(2);

        for (int i = 0; i < 3; ++i) {
            REQUIRE_THAT(ax(i), Catch::Matchers::WithinAbs(rhs(i), kRegularAtol));
        }
    }
}
