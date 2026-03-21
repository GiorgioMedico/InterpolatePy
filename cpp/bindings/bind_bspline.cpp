#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <interpolatecpp/bspline/approximation_bspline.hpp>
#include <interpolatecpp/bspline/bspline.hpp>
#include <interpolatecpp/bspline/bspline_interpolator.hpp>
#include <interpolatecpp/bspline/bspline_parameters.hpp>
#include <interpolatecpp/bspline/cubic_bspline_interpolation.hpp>
#include <interpolatecpp/bspline/smoothing_cubic_bspline.hpp>

namespace py = pybind11;
using namespace interpolatecpp::bspline;

void bind_bspline(py::module_& m) {
    auto bspline_mod = m.def_submodule("bspline", "B-spline interpolation algorithms");

    // Parameterization enum
    py::enum_<Parameterization>(bspline_mod, "Parameterization")
        .value("EquallySpaced", Parameterization::EquallySpaced)
        .value("ChordLength", Parameterization::ChordLength)
        .value("Centripetal", Parameterization::Centripetal);

    // BSpline base class
    py::class_<BSpline>(bspline_mod, "BSpline")
        .def(py::init([](int degree, std::vector<double> knots,
                         const Eigen::MatrixXd& control_points) {
                 return BSpline(degree, std::span<const double>(knots), control_points);
             }),
             py::arg("degree"), py::arg("knots"), py::arg("control_points"))
        .def("evaluate", &BSpline::evaluate, py::arg("u"))
        .def("evaluate_derivative", &BSpline::evaluate_derivative, py::arg("u"),
             py::arg("order") = 1)
        .def("generate_curve_points", &BSpline::generate_curve_points,
             py::arg("num_points") = 100)
        .def("find_knot_span", &BSpline::find_knot_span, py::arg("u"))
        .def_property_readonly("degree", &BSpline::degree)
        .def_property_readonly("knots", &BSpline::knots)
        .def_property_readonly("control_points", &BSpline::control_points)
        .def_property_readonly("u_min", &BSpline::u_min)
        .def_property_readonly("u_max", &BSpline::u_max)
        .def_property_readonly("dimension", &BSpline::dimension)
        .def_property_readonly("n_control_points", &BSpline::n_control_points)
        .def_static("create_uniform_knots", &BSpline::create_uniform_knots, py::arg("degree"),
                    py::arg("num_control_points"), py::arg("domain_min") = 0.0,
                    py::arg("domain_max") = 1.0)
        .def_static("create_periodic_knots", &BSpline::create_periodic_knots,
                    py::arg("degree"), py::arg("num_control_points"),
                    py::arg("domain_min") = 0.0, py::arg("domain_max") = 1.0)
        .def("basis_functions", &BSpline::basis_functions, py::arg("u"),
             py::arg("span_index"))
        .def("basis_function_derivatives", &BSpline::basis_function_derivatives,
             py::arg("u"), py::arg("span_index"), py::arg("order"));

    // CubicBSplineInterpolation
    py::class_<CubicBSplineInterpolation, BSpline>(bspline_mod, "CubicBSplineInterpolation")
        .def(py::init<const Eigen::MatrixXd&, const std::optional<Eigen::VectorXd>&,
                      const std::optional<Eigen::VectorXd>&, Parameterization, bool>(),
             py::arg("points"), py::arg("v0") = std::nullopt, py::arg("vn") = std::nullopt,
             py::arg("method") = Parameterization::ChordLength,
             py::arg("auto_derivatives") = false)
        .def_property_readonly("interpolation_points",
                               &CubicBSplineInterpolation::interpolation_points)
        .def_property_readonly("u_bars", &CubicBSplineInterpolation::u_bars);

    // BSplineInterpolator
    py::class_<BSplineInterpolator, BSpline>(bspline_mod, "BSplineInterpolator")
        .def(py::init<int, const Eigen::MatrixXd&, const std::optional<Eigen::VectorXd>&,
                      const std::optional<Eigen::VectorXd>&,
                      const std::optional<Eigen::VectorXd>&,
                      const std::optional<Eigen::VectorXd>&,
                      const std::optional<Eigen::VectorXd>&, bool>(),
             py::arg("degree"), py::arg("points"), py::arg("times") = std::nullopt,
             py::arg("initial_velocity") = std::nullopt,
             py::arg("final_velocity") = std::nullopt,
             py::arg("initial_acceleration") = std::nullopt,
             py::arg("final_acceleration") = std::nullopt, py::arg("cyclic") = false)
        .def_property_readonly("interp_points", &BSplineInterpolator::interp_points)
        .def_property_readonly("times", &BSplineInterpolator::times);

    // ApproximationBSpline
    py::class_<ApproximationBSpline, BSpline>(bspline_mod, "ApproximationBSpline")
        .def(py::init<const Eigen::MatrixXd&, int, int, const std::optional<Eigen::VectorXd>&,
                      Parameterization, bool>(),
             py::arg("points"), py::arg("num_control_points"), py::arg("degree") = 3,
             py::arg("weights") = std::nullopt,
             py::arg("method") = Parameterization::ChordLength, py::arg("debug") = false)
        .def("calculate_approximation_error", &ApproximationBSpline::calculate_approximation_error)
        .def_property_readonly("original_points", &ApproximationBSpline::original_points)
        .def_property_readonly("original_parameters",
                               &ApproximationBSpline::original_parameters);

    // BSplineParams (for SmoothingCubicBSpline config)
    py::class_<BSplineParams>(bspline_mod, "BSplineParams")
        .def(py::init<>())
        .def_readwrite("mu", &BSplineParams::mu)
        .def_readwrite("weights", &BSplineParams::weights)
        .def_readwrite("v0", &BSplineParams::v0)
        .def_readwrite("vn", &BSplineParams::vn)
        .def_readwrite("method", &BSplineParams::method)
        .def_readwrite("enforce_endpoints", &BSplineParams::enforce_endpoints)
        .def_readwrite("auto_derivatives", &BSplineParams::auto_derivatives);

    // SmoothingCubicBSpline
    py::class_<SmoothingCubicBSpline, BSpline>(bspline_mod, "SmoothingCubicBSpline")
        .def(py::init<const Eigen::MatrixXd&, const BSplineParams&>(), py::arg("points"),
             py::arg("params") = BSplineParams{})
        .def("calculate_approximation_error",
             &SmoothingCubicBSpline::calculate_approximation_error)
        .def("calculate_total_error", &SmoothingCubicBSpline::calculate_total_error)
        .def_property_readonly("approximation_points",
                               &SmoothingCubicBSpline::approximation_points)
        .def_property_readonly("u_bars", &SmoothingCubicBSpline::u_bars)
        .def_property_readonly("mu", &SmoothingCubicBSpline::mu)
        .def_property_readonly("lambda_param", &SmoothingCubicBSpline::lambda_param);
}
