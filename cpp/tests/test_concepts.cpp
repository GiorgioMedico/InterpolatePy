#include <catch2/catch_test_macros.hpp>

#include <interpolatecpp/concepts.hpp>
#include <interpolatecpp/spline/cubic_smoothing_spline.hpp>
#include <interpolatecpp/spline/cubic_spline.hpp>
#include <interpolatecpp/spline/cubic_spline_with_acc1.hpp>
#include <interpolatecpp/spline/cubic_spline_with_acc2.hpp>

using namespace interpolatecpp;
using namespace interpolatecpp::spline;

// Compile-time concept conformance checks
static_assert(ScalarTrajectory<CubicSpline>,
              "CubicSpline must satisfy ScalarTrajectory concept");
static_assert(ScalarTrajectory<CubicSmoothingSpline>,
              "CubicSmoothingSpline must satisfy ScalarTrajectory concept");
static_assert(ScalarTrajectory<CubicSplineWithAcceleration1>,
              "CubicSplineWithAcceleration1 must satisfy ScalarTrajectory concept");
static_assert(ScalarTrajectory<CubicSplineWithAcceleration2>,
              "CubicSplineWithAcceleration2 must satisfy ScalarTrajectory concept");

TEST_CASE("Concept conformance compiles", "[concepts]") {
    // The static_asserts above are the real test — this just ensures
    // the test file is linked and the assertions are evaluated.
    REQUIRE(true);
}
