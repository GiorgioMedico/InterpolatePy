#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <interpolatecpp/bspline/approximation_bspline.hpp>
#include <interpolatecpp/bspline/bspline_interpolator.hpp>
#include <interpolatecpp/bspline/bspline_parameters.hpp>
#include <interpolatecpp/bspline/cubic_bspline_interpolation.hpp>
#include <interpolatecpp/bspline/smoothing_cubic_bspline.hpp>
#include "test_data.hpp"

#include <cmath>

using namespace interpolatecpp::bspline;
using namespace interpolatecpp::test;
using Catch::Matchers::WithinAbs;

// Helper: create quadratic 2D points [[i, i^2]]
static Eigen::MatrixXd make_quadratic_2d(int n) {
    Eigen::MatrixXd pts(n, 2);
    for (int i = 0; i < n; ++i) {
        pts(i, 0) = static_cast<double>(i);
        pts(i, 1) = static_cast<double>(i * i);
    }
    return pts;
}

// Helper: create sine 2D points
static Eigen::MatrixXd make_sine_2d(int n, double x_max = 2.0 * M_PI) {
    Eigen::MatrixXd pts(n, 2);
    for (int i = 0; i < n; ++i) {
        double x = x_max * i / (n - 1);
        pts(i, 0) = x;
        pts(i, 1) = std::sin(x);
    }
    return pts;
}

// ===== CubicBSplineInterpolation =====

TEST_CASE("CubicBSplineInterpolation construction", "[cubic_bspline_interp]") {
    SECTION("Basic construction") {
        Eigen::MatrixXd pts(4, 2);
        pts << 0, 0, 1, 1, 2, 4, 3, 9;
        CubicBSplineInterpolation cbi(pts);

        REQUIRE(cbi.degree() == 3);
        REQUIRE(cbi.dimension() == 2);
    }

    SECTION("1D interpolation") {
        Eigen::MatrixXd pts(5, 1);
        pts << 0, 1, 4, 9, 16;
        CubicBSplineInterpolation cbi(pts);

        REQUIRE(cbi.dimension() == 1);
        auto pt = cbi.evaluate(0.5);
        REQUIRE(pt.allFinite());
    }

    SECTION("3D interpolation") {
        Eigen::MatrixXd pts(4, 3);
        pts << 0, 0, 0, 1, 1, 1, 2, 4, 8, 3, 9, 27;
        CubicBSplineInterpolation cbi(pts);

        REQUIRE(cbi.dimension() == 3);
        auto pt = cbi.evaluate(0.5);
        REQUIRE(pt.size() == 3);
        REQUIRE(pt.allFinite());
    }
}

TEST_CASE("CubicBSplineInterpolation endpoint accuracy", "[cubic_bspline_interp]") {
    Eigen::MatrixXd pts(4, 2);
    pts << 0, 0, 1, 1, 2, 4, 3, 9;
    CubicBSplineInterpolation cbi(pts);

    auto p_start = cbi.evaluate(cbi.u_min());
    auto p_end = cbi.evaluate(cbi.u_max());

    REQUIRE_THAT(p_start(0), WithinAbs(0.0, kRegularAtol));
    REQUIRE_THAT(p_start(1), WithinAbs(0.0, kRegularAtol));
    REQUIRE_THAT(p_end(0), WithinAbs(3.0, kRegularAtol));
    REQUIRE_THAT(p_end(1), WithinAbs(9.0, kRegularAtol));
}

TEST_CASE("CubicBSplineInterpolation parameterization methods", "[cubic_bspline_interp]") {
    Eigen::MatrixXd pts(4, 2);
    pts << 0, 0, 1, 1, 2, 0, 3, 1;

    for (auto method :
         {Parameterization::EquallySpaced, Parameterization::ChordLength,
          Parameterization::Centripetal}) {
        CubicBSplineInterpolation cbi(pts, std::nullopt, std::nullopt, method);

        REQUIRE(cbi.degree() == 3);
        auto pt = cbi.evaluate(0.5);
        REQUIRE(pt.allFinite());
    }
}

TEST_CASE("CubicBSplineInterpolation derivative constraints", "[cubic_bspline_interp]") {
    Eigen::MatrixXd pts(4, 2);
    pts << 0, 0, 1, 1, 2, 4, 3, 9;
    Eigen::VectorXd v0(2);
    v0 << 1.0, 1.0;
    Eigen::VectorXd vn(2);
    vn << 1.0, 6.0;

    CubicBSplineInterpolation cbi(pts, v0, vn);

    REQUIRE(cbi.start_derivative().isApprox(v0));
    REQUIRE(cbi.end_derivative().isApprox(vn));
}

TEST_CASE("CubicBSplineInterpolation auto derivatives", "[cubic_bspline_interp]") {
    Eigen::MatrixXd pts(4, 2);
    pts << 0, 0, 1, 1, 2, 4, 3, 9;

    SECTION("With auto") {
        CubicBSplineInterpolation cbi(pts, std::nullopt, std::nullopt,
                                      Parameterization::ChordLength, true);
        REQUIRE(cbi.start_derivative().allFinite());
        REQUIRE(cbi.end_derivative().allFinite());
    }

    SECTION("Without auto") {
        CubicBSplineInterpolation cbi(pts, std::nullopt, std::nullopt,
                                      Parameterization::ChordLength, false);
        REQUIRE(cbi.start_derivative().allFinite());
    }
}

TEST_CASE("CubicBSplineInterpolation collinear points", "[cubic_bspline_interp]") {
    Eigen::MatrixXd pts(4, 2);
    pts << 0, 0, 1, 1, 2, 2, 3, 3;
    CubicBSplineInterpolation cbi(pts);

    auto pt = cbi.evaluate(0.5);
    REQUIRE(pt.allFinite());
}

// ===== BSplineInterpolator =====

TEST_CASE("BSplineInterpolator construction", "[bspline_interpolator]") {
    SECTION("Basic degree 3") {
        Eigen::MatrixXd pts(5, 2);
        pts << 0, 0, 1, 1, 2, 4, 3, 9, 4, 16;
        BSplineInterpolator bsi(3, pts);

        REQUIRE(bsi.degree() == 3);
        auto pt = bsi.evaluate(0.5);
        REQUIRE(pt.size() == 2);
        REQUIRE(pt.allFinite());
    }

    SECTION("With times") {
        Eigen::MatrixXd pts(5, 2);
        pts << 0, 0, 1, 1, 2, 4, 3, 9, 4, 16;
        Eigen::VectorXd times(5);
        times << 0, 1, 2, 3, 4;
        BSplineInterpolator bsi(3, pts, times);

        REQUIRE(bsi.degree() == 3);
    }
}

TEST_CASE("BSplineInterpolator degree validation", "[bspline_interpolator]") {
    Eigen::MatrixXd pts(5, 2);
    pts << 0, 0, 1, 1, 2, 4, 3, 9, 4, 16;

    SECTION("Invalid degree") {
        REQUIRE_THROWS_AS(BSplineInterpolator(2, pts), std::invalid_argument);
    }

    SECTION("Too few points") {
        Eigen::MatrixXd few_pts(3, 2);
        few_pts << 0, 0, 1, 1, 2, 4;
        REQUIRE_THROWS_AS(BSplineInterpolator(5, few_pts), std::invalid_argument);
    }
}

TEST_CASE("BSplineInterpolator end conditions", "[bspline_interpolator]") {
    Eigen::MatrixXd pts(4, 2);
    pts << 0, 0, 1, 1, 2, 0, 3, 1;
    BSplineInterpolator bsi(3, pts);

    auto p0 = bsi.evaluate(bsi.u_min());
    auto pn = bsi.evaluate(bsi.u_max());
    REQUIRE_THAT(p0(0), WithinAbs(0.0, kNumericalAtol));
    REQUIRE_THAT(p0(1), WithinAbs(0.0, kNumericalAtol));
}

TEST_CASE("BSplineInterpolator boundary conditions", "[bspline_interpolator]") {
    Eigen::MatrixXd pts = make_quadratic_2d(10);
    Eigen::VectorXd v0(2);
    v0 << 1.0, 2.0;
    Eigen::VectorXd vn(2);
    vn << 1.0, 2.0;

    BSplineInterpolator bsi(3, pts, std::nullopt, v0, vn);

    auto pt = bsi.evaluate(0.5);
    REQUIRE(pt.allFinite());
}

TEST_CASE("BSplineInterpolator cyclic", "[bspline_interpolator]") {
    Eigen::MatrixXd pts(4, 2);
    pts << 0, 0, 1, 0, 1, 1, 0, 1;
    BSplineInterpolator bsi(3, pts, std::nullopt, std::nullopt, std::nullopt,
                            std::nullopt, std::nullopt, true);

    auto u_test = Eigen::VectorXd::LinSpaced(5, bsi.u_min(), bsi.u_max());
    for (int i = 0; i < 5; ++i) {
        auto pt = bsi.evaluate(u_test[i]);
        REQUIRE(pt.allFinite());
    }
}

TEST_CASE("BSplineInterpolator 1D points", "[bspline_interpolator]") {
    Eigen::MatrixXd pts(5, 1);
    pts << 0, 1, 4, 9, 16;
    BSplineInterpolator bsi(3, pts);

    auto pt = bsi.evaluate(0.5);
    REQUIRE(pt.size() == 1);
    REQUIRE(pt.allFinite());
}

TEST_CASE("BSplineInterpolator different degrees", "[bspline_interpolator]") {
    // Use enough points for all degrees (degree 5 needs at least 6 points,
    // but rank issues require more for boundary conditions)
    Eigen::MatrixXd pts = make_quadratic_2d(20);

    for (int deg : {3, 4, 5}) {
        BSplineInterpolator bsi(deg, pts);
        REQUIRE(bsi.degree() == deg);

        auto u_test = Eigen::VectorXd::LinSpaced(15, bsi.u_min(), bsi.u_max());
        for (int i = 0; i < 15; ++i) {
            auto pt = bsi.evaluate(u_test[i]);
            REQUIRE(pt.allFinite());
        }
    }
}

// ===== ApproximationBSpline =====

TEST_CASE("ApproximationBSpline construction", "[approx_bspline]") {
    SECTION("Basic construction") {
        Eigen::MatrixXd pts(7, 2);
        pts << 0, 0, 0.5, 0.25, 1, 1, 1.5, 2.25, 2, 4, 2.5, 6.25, 3, 9;
        ApproximationBSpline abs(pts, 5, 3);

        REQUIRE(abs.degree() == 3);
    }

    SECTION("Sine approximation") {
        auto pts = make_sine_2d(15);
        ApproximationBSpline abs(pts, 8, 3);

        auto u_test = Eigen::VectorXd::LinSpaced(20, abs.u_min(), abs.u_max());
        for (int i = 0; i < 20; ++i) {
            auto pt = abs.evaluate(u_test[i]);
            REQUIRE(pt.allFinite());
            REQUIRE(pt.size() == 2);
        }
    }
}

TEST_CASE("ApproximationBSpline validation", "[approx_bspline]") {
    auto pts = make_quadratic_2d(10);

    SECTION("Degree too low") {
        REQUIRE_THROWS_AS(ApproximationBSpline(pts, 5, 0), std::invalid_argument);
    }

    SECTION("Too few control points") {
        REQUIRE_THROWS_AS(ApproximationBSpline(pts, 2, 3), std::invalid_argument);
    }

    SECTION("More control points than data") {
        Eigen::MatrixXd few(3, 2);
        few << 0, 0, 1, 1, 2, 4;
        REQUIRE_THROWS_AS(ApproximationBSpline(few, 5, 3), std::invalid_argument);
    }
}

TEST_CASE("ApproximationBSpline parameterization methods", "[approx_bspline]") {
    auto pts = make_quadratic_2d(10);

    for (auto method :
         {Parameterization::EquallySpaced, Parameterization::ChordLength,
          Parameterization::Centripetal}) {
        ApproximationBSpline abs(pts, 6, 3, std::nullopt, method);
        auto pt = abs.evaluate(0.5);
        REQUIRE(pt.allFinite());
    }
}

TEST_CASE("ApproximationBSpline error calculation", "[approx_bspline]") {
    auto pts = make_sine_2d(25);
    ApproximationBSpline abs(pts, 12, 3);

    double error = abs.calculate_approximation_error();
    REQUIRE(std::isfinite(error));
    REQUIRE(error >= 0.0);
}

TEST_CASE("ApproximationBSpline original data storage", "[approx_bspline]") {
    auto pts = make_quadratic_2d(10);
    ApproximationBSpline abs(pts, 6, 3);

    REQUIRE(abs.original_points().rows() == 10);
    REQUIRE(abs.original_parameters().size() == 10);

    // Parameters should be in [0, 1]
    for (int i = 0; i < abs.original_parameters().size(); ++i) {
        REQUIRE(abs.original_parameters()[i] >= 0.0);
        REQUIRE(abs.original_parameters()[i] <= 1.0);
    }
}

TEST_CASE("ApproximationBSpline knot vector properties", "[approx_bspline]") {
    auto pts = make_quadratic_2d(15);
    ApproximationBSpline abs(pts, 8, 3);

    // n_knots = n_control + degree + 1
    REQUIRE(abs.knots().size() == abs.n_control_points() + abs.degree() + 1);

    // Non-decreasing
    for (int i = 1; i < abs.knots().size(); ++i) {
        REQUIRE(abs.knots()[i] >= abs.knots()[i - 1]);
    }
}

TEST_CASE("ApproximationBSpline different degrees", "[approx_bspline]") {
    auto pts = make_quadratic_2d(12);

    for (int deg : {1, 2, 3, 4}) {
        int n_ctrl = deg + 2;
        if (n_ctrl >= static_cast<int>(pts.rows())) continue;
        ApproximationBSpline abs(pts, n_ctrl, deg);
        auto pt = abs.evaluate(0.5);
        REQUIRE(pt.allFinite());
    }
}

// ===== SmoothingCubicBSpline =====

TEST_CASE("SmoothingCubicBSpline construction", "[smoothing_bspline]") {
    SECTION("Basic construction") {
        Eigen::MatrixXd pts(5, 2);
        pts << 0, 0, 1, 1, 2, 4, 3, 9, 4, 16;
        SmoothingCubicBSpline sbs(pts);

        REQUIRE(sbs.degree() == 3);
    }

    SECTION("With custom params") {
        Eigen::MatrixXd pts(5, 2);
        pts << 0, 0, 1, 1, 2, 4, 3, 9, 4, 16;
        BSplineParams params;
        params.mu = 0.7;
        params.method = Parameterization::Centripetal;
        SmoothingCubicBSpline sbs(pts, params);

        REQUIRE(sbs.degree() == 3);
    }
}

TEST_CASE("SmoothingCubicBSpline smoothing parameters", "[smoothing_bspline]") {
    auto pts = make_sine_2d(20);

    for (double mu : {0.1, 0.5, 0.9}) {
        BSplineParams params;
        params.mu = mu;
        params.method = Parameterization::ChordLength;
        SmoothingCubicBSpline sbs(pts, params);

        REQUIRE_THAT(sbs.mu(), WithinAbs(mu, kRegularAtol));
    }
}

TEST_CASE("SmoothingCubicBSpline smoothing effect", "[smoothing_bspline]") {
    // Noisy data
    Eigen::MatrixXd pts(15, 2);
    for (int i = 0; i < 15; ++i) {
        double x = 4.0 * i / 14.0;
        pts(i, 0) = x;
        pts(i, 1) = x * x + 0.1 * std::sin(10.0 * x);  // Quadratic + noise
    }

    BSplineParams params;
    params.mu = 0.5;
    SmoothingCubicBSpline sbs(pts, params);

    auto u_test = Eigen::VectorXd::LinSpaced(10, sbs.u_min(), sbs.u_max());
    for (int i = 0; i < 10; ++i) {
        auto pt = sbs.evaluate(u_test[i]);
        REQUIRE(pt.allFinite());
        REQUIRE(pt.size() == 2);
    }
}

TEST_CASE("SmoothingCubicBSpline parameterization methods", "[smoothing_bspline]") {
    auto pts = make_quadratic_2d(15);

    for (auto method :
         {Parameterization::EquallySpaced, Parameterization::ChordLength,
          Parameterization::Centripetal}) {
        BSplineParams params;
        params.method = method;
        SmoothingCubicBSpline sbs(pts, params);

        auto u_test = Eigen::VectorXd::LinSpaced(10, sbs.u_min(), sbs.u_max());
        for (int i = 0; i < 10; ++i) {
            auto pt = sbs.evaluate(u_test[i]);
            REQUIRE(pt.allFinite());
        }
    }
}

TEST_CASE("SmoothingCubicBSpline endpoint enforcement", "[smoothing_bspline]") {
    auto pts = make_quadratic_2d(10);

    SECTION("Without enforcement") {
        SmoothingCubicBSpline sbs(pts);
        auto pt = sbs.evaluate(0.5);
        REQUIRE(pt.allFinite());
    }

    SECTION("With enforcement") {
        BSplineParams params;
        params.enforce_endpoints = true;
        SmoothingCubicBSpline sbs(pts, params);
        auto pt = sbs.evaluate(0.5);
        REQUIRE(pt.allFinite());
    }
}

TEST_CASE("SmoothingCubicBSpline error calculation", "[smoothing_bspline]") {
    auto pts = make_sine_2d(20);
    SmoothingCubicBSpline sbs(pts);

    auto errors = sbs.calculate_approximation_error();
    REQUIRE(errors.size() == 20);
    REQUIRE(errors.allFinite());

    double total = sbs.calculate_total_error();
    REQUIRE(std::isfinite(total));
    REQUIRE(total >= 0.0);
}

TEST_CASE("SmoothingCubicBSpline mu effects differ", "[smoothing_bspline]") {
    // Use noisy data to make smoothing effects more pronounced
    Eigen::MatrixXd pts(25, 2);
    for (int i = 0; i < 25; ++i) {
        double x = 2.0 * M_PI * i / 24.0;
        pts(i, 0) = x;
        pts(i, 1) = std::sin(x) + 0.3 * std::sin(10.0 * x);  // Signal + noise
    }

    BSplineParams params_tight;
    params_tight.mu = 0.99;
    SmoothingCubicBSpline sbs_tight(pts, params_tight);

    BSplineParams params_smooth;
    params_smooth.mu = 0.01;
    SmoothingCubicBSpline sbs_smooth(pts, params_smooth);

    // Check multiple evaluation points — at least one should differ
    double max_diff = 0.0;
    auto u_test = Eigen::VectorXd::LinSpaced(10, sbs_tight.u_min(), sbs_tight.u_max());
    for (int i = 0; i < 10; ++i) {
        auto pt_t = sbs_tight.evaluate(u_test[i]);
        auto pt_s = sbs_smooth.evaluate(u_test[i]);
        max_diff = std::max(max_diff, (pt_t - pt_s).norm());
    }
    REQUIRE(max_diff > 1e-3);
}

TEST_CASE("SmoothingCubicBSpline constant data", "[smoothing_bspline]") {
    Eigen::MatrixXd pts(8, 2);
    for (int i = 0; i < 8; ++i) {
        pts(i, 0) = static_cast<double>(i);
        pts(i, 1) = 5.0;
    }

    SmoothingCubicBSpline sbs(pts);
    auto pt = sbs.evaluate(0.5);
    REQUIRE(pt.allFinite());
    REQUIRE_THAT(pt(1), WithinAbs(5.0, 0.5));
}

TEST_CASE("SmoothingCubicBSpline auto derivatives", "[smoothing_bspline]") {
    // Circle-like points
    Eigen::MatrixXd pts(12, 2);
    for (int i = 0; i < 12; ++i) {
        double angle = 2.0 * M_PI * i / 12.0;
        pts(i, 0) = std::cos(angle);
        pts(i, 1) = std::sin(angle);
    }

    SECTION("With auto derivatives") {
        BSplineParams params;
        params.enforce_endpoints = true;
        params.auto_derivatives = true;
        SmoothingCubicBSpline sbs(pts, params);
        auto pt = sbs.evaluate(0.3);
        REQUIRE(pt.allFinite());
    }

    SECTION("Without auto derivatives") {
        BSplineParams params;
        params.enforce_endpoints = true;
        params.auto_derivatives = false;
        SmoothingCubicBSpline sbs(pts, params);
        auto pt = sbs.evaluate(0.3);
        REQUIRE(pt.allFinite());
    }
}

// ===== Variant inheritance =====

TEST_CASE("BSpline variant inheritance", "[bspline_variants]") {
    Eigen::MatrixXd pts(5, 2);
    pts << 0, 0, 1, 1, 2, 0, 3, 1, 4, 0;

    auto extended_pts = make_quadratic_2d(8);

    SECTION("CubicBSplineInterpolation inherits BSpline") {
        CubicBSplineInterpolation cbi(pts);
        const BSpline& base = cbi;
        REQUIRE(base.degree() == 3);
        REQUIRE(base.evaluate(0.5).allFinite());
    }

    SECTION("BSplineInterpolator inherits BSpline") {
        BSplineInterpolator bsi(3, pts);
        const BSpline& base = bsi;
        REQUIRE(base.degree() == 3);
        REQUIRE(base.evaluate(0.5).allFinite());
    }

    SECTION("ApproximationBSpline inherits BSpline") {
        ApproximationBSpline abs(extended_pts, 5, 3);
        const BSpline& base = abs;
        REQUIRE(base.degree() == 3);
        REQUIRE(base.evaluate(0.5).allFinite());
    }

    SECTION("SmoothingCubicBSpline inherits BSpline") {
        SmoothingCubicBSpline sbs(pts);
        const BSpline& base = sbs;
        REQUIRE(base.degree() == 3);
        REQUIRE(base.evaluate(0.5).allFinite());
    }
}
