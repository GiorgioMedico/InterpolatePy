#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <interpolatecpp/bspline/bspline.hpp>
#include "test_data.hpp"

using namespace interpolatecpp::bspline;
using namespace interpolatecpp::test;
using Catch::Matchers::WithinAbs;

// --- Construction ---

TEST_CASE("BSpline construction", "[bspline]") {
    SECTION("Basic 1D construction") {
        std::vector<double> knots = {0, 0, 0, 1, 2, 3, 3, 3};
        Eigen::MatrixXd cp(5, 1);
        cp << 1, 2, 3, 4, 5;
        BSpline bs(2, knots, cp);

        REQUIRE(bs.degree() == 2);
        REQUIRE(bs.knots().size() == 8);
        REQUIRE(bs.n_control_points() == 5);
        REQUIRE(bs.dimension() == 1);
        REQUIRE_THAT(bs.u_min(), WithinAbs(0.0, kRegularAtol));
        REQUIRE_THAT(bs.u_max(), WithinAbs(3.0, kRegularAtol));
    }

    SECTION("Basic 2D construction") {
        std::vector<double> knots = {0, 0, 0, 1, 2, 3, 3, 3};
        Eigen::MatrixXd cp(5, 2);
        cp << 0, 0, 1, 1, 2, 0, 3, 1, 4, 0;
        BSpline bs(2, knots, cp);

        REQUIRE(bs.dimension() == 2);
        REQUIRE(bs.control_points().rows() == 5);
        REQUIRE(bs.control_points().cols() == 2);
    }

    SECTION("Basic 3D construction") {
        std::vector<double> knots = {0, 0, 1, 2, 2};
        Eigen::MatrixXd cp(3, 3);
        cp << 0, 0, 0, 1, 1, 1, 2, 0, 2;
        BSpline bs(1, knots, cp);

        REQUIRE(bs.dimension() == 3);
        REQUIRE(bs.control_points().rows() == 3);
    }

    SECTION("Degree zero construction") {
        std::vector<double> knots = {0, 1, 2, 3};
        Eigen::MatrixXd cp(3, 1);
        cp << 1, 2, 3;
        BSpline bs(0, knots, cp);

        REQUIRE(bs.degree() == 0);
        REQUIRE_THAT(bs.u_min(), WithinAbs(0.0, kRegularAtol));
        REQUIRE_THAT(bs.u_max(), WithinAbs(3.0, kRegularAtol));
    }

    SECTION("High degree construction") {
        std::vector<double> knots = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
        Eigen::MatrixXd cp(6, 2);
        for (int i = 0; i < 6; ++i) {
            cp(i, 0) = static_cast<double>(i);
            cp(i, 1) = static_cast<double>(i * i);
        }
        BSpline bs(5, knots, cp);

        REQUIRE(bs.degree() == 5);
        REQUIRE(bs.dimension() == 2);
    }
}

TEST_CASE("BSpline construction validation", "[bspline]") {
    SECTION("Negative degree") {
        std::vector<double> knots = {0, 0, 1, 1};
        Eigen::MatrixXd cp(2, 1);
        cp << 1, 2;
        REQUIRE_THROWS_AS(BSpline(-1, knots, cp), std::invalid_argument);
    }

    SECTION("Non-decreasing knots") {
        std::vector<double> knots = {0, 1, 0.5, 1};
        Eigen::MatrixXd cp(2, 1);
        cp << 1, 2;
        REQUIRE_THROWS_AS(BSpline(1, knots, cp), std::invalid_argument);
    }

    SECTION("Invalid knot relationship") {
        std::vector<double> knots = {0, 0, 0, 1, 1, 1};
        Eigen::MatrixXd cp(4, 1);
        cp << 1, 2, 3, 4;
        REQUIRE_THROWS_AS(BSpline(2, knots, cp), std::invalid_argument);
    }
}

// --- Knot span ---

TEST_CASE("BSpline knot span", "[bspline]") {
    std::vector<double> knots = {0, 0, 0, 1, 2, 3, 3, 3};
    Eigen::MatrixXd cp(5, 1);
    cp << 1, 2, 3, 4, 5;
    BSpline bs(2, knots, cp);

    SECTION("Basic knot span finding") {
        REQUIRE(bs.find_knot_span(0.0) == 2);
        REQUIRE(bs.find_knot_span(0.5) == 2);
        REQUIRE(bs.find_knot_span(1.0) == 3);
        REQUIRE(bs.find_knot_span(1.5) == 3);
        REQUIRE(bs.find_knot_span(2.0) == 4);
        REQUIRE(bs.find_knot_span(3.0) == 4);
    }

    SECTION("Boundary conditions") {
        REQUIRE(bs.find_knot_span(bs.u_min()) == bs.degree());
        REQUIRE(bs.find_knot_span(bs.u_max()) ==
                static_cast<int>(bs.knots().size()) - bs.degree() - 2);
    }

    SECTION("Out of range") {
        REQUIRE_THROWS_AS(bs.find_knot_span(-1.0), std::invalid_argument);
        REQUIRE_THROWS_AS(bs.find_knot_span(4.0), std::invalid_argument);
    }
}

// --- Basis functions ---

TEST_CASE("BSpline basis functions", "[bspline]") {
    SECTION("Degree zero") {
        std::vector<double> knots = {0, 1, 2, 3};
        Eigen::MatrixXd cp(3, 1);
        cp << 1, 2, 3;
        BSpline bs(0, knots, cp);

        auto basis = bs.basis_functions(0.5, bs.find_knot_span(0.5));
        REQUIRE(basis.size() == 1);
        REQUIRE_THAT(basis.sum(), WithinAbs(1.0, kNumericalAtol));
    }

    SECTION("Degree one") {
        std::vector<double> knots = {0, 0, 1, 2, 2};
        Eigen::MatrixXd cp(3, 1);
        cp << 1, 2, 3;
        BSpline bs(1, knots, cp);

        auto basis = bs.basis_functions(0.5, bs.find_knot_span(0.5));
        REQUIRE(basis.size() == 2);
        REQUIRE_THAT(basis.sum(), WithinAbs(1.0, kNumericalAtol));
    }

    SECTION("Degree two") {
        std::vector<double> knots = {0, 0, 0, 1, 2, 2, 2};
        Eigen::MatrixXd cp(4, 1);
        cp << 1, 2, 3, 4;
        BSpline bs(2, knots, cp);

        auto basis = bs.basis_functions(0.5, bs.find_knot_span(0.5));
        REQUIRE(basis.size() == 3);
        REQUIRE_THAT(basis.sum(), WithinAbs(1.0, kNumericalAtol));
        for (int i = 0; i < basis.size(); ++i) {
            REQUIRE(basis[i] >= -kNumericalAtol);
        }
    }

    SECTION("Partition of unity") {
        std::vector<double> knots = {0, 0, 0, 0, 1, 2, 3, 3, 3, 3};
        Eigen::MatrixXd cp(6, 1);
        cp << 1, 2, 3, 4, 5, 6;
        BSpline bs(3, knots, cp);

        std::vector<double> test_u = {0.0, 0.3, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0};
        for (double u : test_u) {
            int span = bs.find_knot_span(u);
            auto basis = bs.basis_functions(u, span);
            REQUIRE_THAT(basis.sum(), WithinAbs(1.0, kNumericalAtol));
            for (int i = 0; i < basis.size(); ++i) {
                REQUIRE(basis[i] >= -kNumericalAtol);
            }
        }
    }

    SECTION("Basis function derivatives") {
        std::vector<double> knots = {0, 0, 0, 1, 2, 2, 2};
        Eigen::MatrixXd cp(4, 1);
        cp << 1, 2, 3, 4;
        BSpline bs(2, knots, cp);

        int span = bs.find_knot_span(0.5);
        auto ders = bs.basis_function_derivatives(0.5, span, 2);

        REQUIRE(ders.rows() == 3);  // orders 0, 1, 2
        REQUIRE(ders.cols() == 3);  // degree+1 basis functions

        // Zero-th derivative row should equal basis functions
        auto basis = bs.basis_functions(0.5, span);
        for (int j = 0; j < 3; ++j) {
            REQUIRE_THAT(ders(0, j), WithinAbs(basis[j], kNumericalAtol));
        }
    }

    SECTION("First derivatives sum to zero") {
        std::vector<double> knots = {0, 0, 0, 1, 2, 3, 3, 3};
        Eigen::MatrixXd cp(5, 1);
        cp << 1, 2, 3, 4, 5;
        BSpline bs(2, knots, cp);

        int span = bs.find_knot_span(1.5);
        auto ders = bs.basis_function_derivatives(1.5, span, 1);

        double sum_d1 = 0.0;
        for (int j = 0; j <= bs.degree(); ++j) {
            sum_d1 += ders(1, j);
        }
        REQUIRE_THAT(std::abs(sum_d1), WithinAbs(0.0, kNumericalAtol));
    }
}

// --- Evaluation ---

TEST_CASE("BSpline evaluation", "[bspline]") {
    SECTION("1D linear") {
        std::vector<double> knots = {0, 0, 1, 1};
        Eigen::MatrixXd cp(2, 1);
        cp << 0, 2;
        BSpline bs(1, knots, cp);

        REQUIRE_THAT(bs.evaluate(0.0)(0), WithinAbs(0.0, kNumericalAtol));
        REQUIRE_THAT(bs.evaluate(0.5)(0), WithinAbs(1.0, kNumericalAtol));
        REQUIRE_THAT(bs.evaluate(1.0)(0), WithinAbs(2.0, kNumericalAtol));
    }

    SECTION("2D curve endpoints") {
        std::vector<double> knots = {0, 0, 0, 1, 1, 1};
        Eigen::MatrixXd cp(3, 2);
        cp << 0, 0, 1, 1, 2, 0;
        BSpline bs(2, knots, cp);

        auto p_min = bs.evaluate(bs.u_min());
        auto p_max = bs.evaluate(bs.u_max());
        REQUIRE_THAT(p_min(0), WithinAbs(0.0, kRegularAtol));
        REQUIRE_THAT(p_min(1), WithinAbs(0.0, kRegularAtol));
        REQUIRE_THAT(p_max(0), WithinAbs(2.0, kRegularAtol));
        REQUIRE_THAT(p_max(1), WithinAbs(0.0, kRegularAtol));
    }

    SECTION("3D curve evaluation") {
        std::vector<double> knots = {0, 0, 1, 2, 2};
        Eigen::MatrixXd cp(3, 3);
        cp << 0, 0, 0, 1, 1, 1, 2, 0, 2;
        BSpline bs(1, knots, cp);

        auto pt = bs.evaluate(0.5);
        REQUIRE(pt.size() == 3);
        REQUIRE(pt.allFinite());
    }

    SECTION("Derivative order zero equals evaluate") {
        std::vector<double> knots = {0, 0, 0, 1, 2, 2, 2};
        Eigen::MatrixXd cp(4, 2);
        cp << 0, 0, 1, 1, 2, 0, 3, 1;
        BSpline bs(2, knots, cp);

        auto val = bs.evaluate(0.7);
        auto deriv0 = bs.evaluate_derivative(0.7, 0);
        for (int i = 0; i < val.size(); ++i) {
            REQUIRE_THAT(val(i), WithinAbs(deriv0(i), kNumericalAtol));
        }
    }

    SECTION("Derivative finite") {
        std::vector<double> knots = {0, 0, 0, 1, 1, 1};
        Eigen::MatrixXd cp(3, 2);
        cp << 0, 0, 1, 1, 2, 0;
        BSpline bs(2, knots, cp);

        auto d1 = bs.evaluate_derivative(0.5, 1);
        REQUIRE(d1.size() == 2);
        REQUIRE(d1.allFinite());
    }

    SECTION("Derivative order exceeds degree") {
        std::vector<double> knots = {0, 0, 0, 1, 1, 1};
        Eigen::MatrixXd cp(3, 1);
        cp << 1, 2, 3;
        BSpline bs(2, knots, cp);

        REQUIRE_THROWS_AS(bs.evaluate_derivative(0.5, 3), std::invalid_argument);
    }

    SECTION("Finite difference vs exact derivative") {
        std::vector<double> knots = {0, 0, 0, 1, 2, 2, 2};
        Eigen::MatrixXd cp(4, 1);
        cp << 1, 2, 3, 4;
        BSpline bs(2, knots, cp);

        // Use a safe interior point (away from knots and boundaries)
        double u = 0.5;
        double h = 1e-5;
        auto d_exact = bs.evaluate_derivative(u, 1);
        auto val_plus = bs.evaluate(u + h);
        auto val_minus = bs.evaluate(u - h);
        double d_approx = (val_plus(0) - val_minus(0)) / (2.0 * h);

        REQUIRE_THAT(d_exact(0), WithinAbs(d_approx, 1e-3));
    }
}

// --- Edge cases ---

TEST_CASE("BSpline edge cases", "[bspline]") {
    SECTION("Single control point degree 0") {
        std::vector<double> knots = {0, 1};
        Eigen::MatrixXd cp(1, 1);
        cp << 5.0;
        BSpline bs(0, knots, cp);

        REQUIRE_THAT(bs.evaluate(0.0)(0), WithinAbs(5.0, kNumericalAtol));
        REQUIRE_THAT(bs.evaluate(0.5)(0), WithinAbs(5.0, kNumericalAtol));
        REQUIRE_THAT(bs.evaluate(1.0)(0), WithinAbs(5.0, kNumericalAtol));
    }

    SECTION("Identical control points") {
        std::vector<double> knots = {0, 0, 0, 1, 1, 1};
        Eigen::MatrixXd cp(3, 2);
        cp << 1, 2, 1, 2, 1, 2;
        BSpline bs(2, knots, cp);

        auto pt = bs.evaluate(0.5);
        REQUIRE_THAT(pt(0), WithinAbs(1.0, kNumericalAtol));
        REQUIRE_THAT(pt(1), WithinAbs(2.0, kNumericalAtol));
    }

    SECTION("Repeated internal knots") {
        std::vector<double> knots = {0, 0, 0, 0.5, 0.5, 1, 1, 1};
        Eigen::MatrixXd cp(5, 1);
        cp << 1, 2, 3, 4, 5;
        BSpline bs(2, knots, cp);

        auto pt = bs.evaluate(0.5);
        REQUIRE(pt.allFinite());
    }
}

// --- Knot vector creation ---

TEST_CASE("BSpline uniform knots", "[bspline]") {
    SECTION("Basic creation") {
        auto knots = BSpline::create_uniform_knots(2, 5);
        REQUIRE(knots.size() == 8);

        // Non-decreasing
        for (int i = 1; i < knots.size(); ++i) {
            REQUIRE(knots[i] >= knots[i - 1]);
        }

        // First and last clamped
        for (int i = 0; i < 3; ++i) {
            REQUIRE_THAT(knots[i], WithinAbs(0.0, kRegularAtol));
        }
        for (int i = 5; i < 8; ++i) {
            REQUIRE_THAT(knots[i], WithinAbs(1.0, kRegularAtol));
        }
    }

    SECTION("Custom domain") {
        auto knots = BSpline::create_uniform_knots(1, 3, -2.0, 5.0);
        REQUIRE_THAT(knots[0], WithinAbs(-2.0, kRegularAtol));
        REQUIRE_THAT(knots[knots.size() - 1], WithinAbs(5.0, kRegularAtol));
    }

    SECTION("Validation") {
        REQUIRE_THROWS_AS(BSpline::create_uniform_knots(-1, 5), std::invalid_argument);
        REQUIRE_THROWS_AS(BSpline::create_uniform_knots(3, 3), std::invalid_argument);
    }
}

TEST_CASE("BSpline periodic knots", "[bspline]") {
    SECTION("Basic creation") {
        auto knots = BSpline::create_periodic_knots(2, 4);
        REQUIRE(knots.size() == 7);

        // Non-decreasing
        for (int i = 1; i < knots.size(); ++i) {
            REQUIRE(knots[i] >= knots[i - 1]);
        }
    }

    SECTION("Validation") {
        REQUIRE_THROWS_AS(BSpline::create_periodic_knots(-1, 3), std::invalid_argument);
        REQUIRE_THROWS_AS(BSpline::create_periodic_knots(3, 2), std::invalid_argument);
    }
}

// --- Curve generation ---

TEST_CASE("BSpline curve generation", "[bspline]") {
    std::vector<double> knots = {0, 0, 0, 1, 1, 1};
    Eigen::MatrixXd cp(3, 2);
    cp << 0, 0, 1, 1, 2, 0;
    BSpline bs(2, knots, cp);

    SECTION("Basic generation") {
        auto [u_vals, pts] = bs.generate_curve_points(10);
        REQUIRE(u_vals.size() == 10);
        REQUIRE(pts.rows() == 10);
        REQUIRE(pts.cols() == 2);
        REQUIRE_THAT(u_vals[0], WithinAbs(bs.u_min(), kRegularAtol));
        REQUIRE_THAT(u_vals[9], WithinAbs(bs.u_max(), kRegularAtol));
    }

    SECTION("Different counts") {
        for (int n : {5, 50, 100}) {
            auto [u_vals, pts] = bs.generate_curve_points(n);
            REQUIRE(u_vals.size() == n);
            REQUIRE(pts.rows() == n);
        }
    }
}

// --- Numerical stability ---

TEST_CASE("BSpline numerical stability", "[bspline]") {
    SECTION("Nearly coincident knots") {
        std::vector<double> knots = {0, 0, 0, 0.5, 0.5 + 1e-12, 1, 1, 1};
        Eigen::MatrixXd cp(5, 1);
        cp << 1, 2, 3, 4, 5;
        BSpline bs(2, knots, cp);

        auto pt = bs.evaluate(0.5);
        REQUIRE(pt.allFinite());
    }

    SECTION("High degree stability") {
        auto knots = BSpline::create_uniform_knots(7, 10);
        Eigen::MatrixXd cp(10, 2);
        for (int i = 0; i < 10; ++i) {
            cp(i, 0) = static_cast<double>(i);
            cp(i, 1) = std::sin(static_cast<double>(i));
        }
        BSpline bs(7, std::span<const double>(knots.data(), knots.size()), cp);

        int num_test = 20;
        auto u_test = Eigen::VectorXd::LinSpaced(num_test, bs.u_min(), bs.u_max());
        for (int i = 0; i < num_test; ++i) {
            auto pt = bs.evaluate(u_test[i]);
            REQUIRE(pt.allFinite());
        }
    }
}
