#include <catch2/catch_test_macros.hpp>

#include <interpolatecpp/concepts.hpp>
#include <interpolatecpp/spline/cubic_smoothing_spline.hpp>
#include <interpolatecpp/spline/cubic_spline.hpp>
#include <interpolatecpp/spline/cubic_spline_with_acc1.hpp>
#include <interpolatecpp/spline/cubic_spline_with_acc2.hpp>
// Phase 2
#include <interpolatecpp/bspline/approximation_bspline.hpp>
#include <interpolatecpp/bspline/bspline.hpp>
#include <interpolatecpp/bspline/bspline_interpolator.hpp>
#include <interpolatecpp/bspline/cubic_bspline_interpolation.hpp>
#include <interpolatecpp/bspline/smoothing_cubic_bspline.hpp>
// Phase 4
#include <interpolatecpp/quat/log_quaternion_interpolation.hpp>
#include <interpolatecpp/quat/quaternion_spline.hpp>
#include <interpolatecpp/quat/squad_c2.hpp>
// Phase 5
#include <interpolatecpp/path/circular_path.hpp>
#include <interpolatecpp/path/linear_path.hpp>

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

// Phase 2: B-spline family
static_assert(CurveEvaluator<bspline::BSpline>,
              "BSpline must satisfy CurveEvaluator concept");
static_assert(CurveEvaluator<bspline::CubicBSplineInterpolation>,
              "CubicBSplineInterpolation must satisfy CurveEvaluator concept");
static_assert(CurveEvaluator<bspline::BSplineInterpolator>,
              "BSplineInterpolator must satisfy CurveEvaluator concept");
static_assert(CurveEvaluator<bspline::ApproximationBSpline>,
              "ApproximationBSpline must satisfy CurveEvaluator concept");
static_assert(CurveEvaluator<bspline::SmoothingCubicBSpline>,
              "SmoothingCubicBSpline must satisfy CurveEvaluator concept");

// Phase 4: Quaternion trajectory
static_assert(QuaternionTrajectory<quat::SquadC2>,
              "SquadC2 must satisfy QuaternionTrajectory concept");
static_assert(QuaternionTrajectory<quat::LogQuaternionInterpolation>,
              "LogQuaternionInterpolation must satisfy QuaternionTrajectory concept");
static_assert(QuaternionTrajectory<quat::QuaternionSpline>,
              "QuaternionSpline must satisfy QuaternionTrajectory concept");
// Note: ModifiedLogQuaternionInterpolation uses 4D velocity/acceleration
// so it deliberately does NOT satisfy QuaternionTrajectory (3D).

// Phase 5: Path primitives
static_assert(GeometricPath<path::LinearPath>,
              "LinearPath must satisfy GeometricPath concept");
static_assert(GeometricPath<path::CircularPath>,
              "CircularPath must satisfy GeometricPath concept");

TEST_CASE("Concept conformance compiles", "[concepts]") {
    // The static_asserts above are the real test — this just ensures
    // the test file is linked and the assertions are evaluated.
    REQUIRE(true);
}
