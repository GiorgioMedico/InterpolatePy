#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <interpolatecpp/spline/cubic_smoothing_spline.hpp>

#include <optional>

namespace py = pybind11;
using namespace interpolatecpp::spline;

void bind_smoothing_spline(py::module_& m) {
    py::class_<CubicSmoothingSpline>(m, "CubicSmoothingSpline")
        .def(py::init([](std::vector<double> t, std::vector<double> q, double mu,
                         std::optional<std::vector<double>> weights, double v0, double vn,
                         bool debug) {
                 std::optional<std::span<const double>> w_span;
                 if (weights.has_value()) {
                     w_span = std::span<const double>(weights->data(), weights->size());
                 }
                 return CubicSmoothingSpline(t, q, mu, w_span, v0, vn, debug);
             }),
             py::arg("t_points"), py::arg("q_points"), py::arg("mu") = 0.5,
             py::arg("weights") = std::nullopt, py::arg("v0") = 0.0, py::arg("vn") = 0.0,
             py::arg("debug") = false)
        .def("evaluate",
             py::overload_cast<double>(&CubicSmoothingSpline::evaluate, py::const_),
             py::arg("t"))
        .def("evaluate",
             py::overload_cast<const Eigen::VectorXd&>(&CubicSmoothingSpline::evaluate,
                                                         py::const_),
             py::arg("t"))
        .def("evaluate_velocity",
             py::overload_cast<double>(&CubicSmoothingSpline::evaluate_velocity, py::const_),
             py::arg("t"))
        .def("evaluate_velocity",
             py::overload_cast<const Eigen::VectorXd&>(
                 &CubicSmoothingSpline::evaluate_velocity, py::const_),
             py::arg("t"))
        .def("evaluate_acceleration",
             py::overload_cast<double>(&CubicSmoothingSpline::evaluate_acceleration, py::const_),
             py::arg("t"))
        .def("evaluate_acceleration",
             py::overload_cast<const Eigen::VectorXd&>(
                 &CubicSmoothingSpline::evaluate_acceleration, py::const_),
             py::arg("t"))
        .def_property_readonly("t_points", &CubicSmoothingSpline::t_points)
        .def_property_readonly("q_points", &CubicSmoothingSpline::q_points)
        .def_property_readonly("s_points", &CubicSmoothingSpline::s_points)
        .def_property_readonly("mu", &CubicSmoothingSpline::mu)
        .def_property_readonly("coefficients", &CubicSmoothingSpline::coefficients);
}
