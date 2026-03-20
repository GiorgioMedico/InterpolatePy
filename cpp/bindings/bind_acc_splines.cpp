#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <interpolatecpp/spline/cubic_spline_with_acc1.hpp>
#include <interpolatecpp/spline/cubic_spline_with_acc2.hpp>

namespace py = pybind11;
using namespace interpolatecpp::spline;

void bind_acc_splines(py::module_& m) {
    // SplineParameters
    py::class_<SplineParameters>(m, "SplineParameters")
        .def(py::init([](double v0, double vn, std::optional<double> a0, std::optional<double> an,
                         bool debug) {
                 return SplineParameters{v0, vn, a0, an, debug};
             }),
             py::arg("v0") = 0.0, py::arg("vn") = 0.0, py::arg("a0") = std::nullopt,
             py::arg("an") = std::nullopt, py::arg("debug") = false)
        .def_readwrite("v0", &SplineParameters::v0)
        .def_readwrite("vn", &SplineParameters::vn)
        .def_readwrite("a0", &SplineParameters::a0)
        .def_readwrite("an", &SplineParameters::an)
        .def_readwrite("debug", &SplineParameters::debug);

    // CubicSplineWithAcceleration1
    py::class_<CubicSplineWithAcceleration1>(m, "CubicSplineWithAcceleration1")
        .def(py::init([](std::vector<double> t, std::vector<double> q, double v0, double vn,
                         double a0, double an, bool debug) {
                 return CubicSplineWithAcceleration1(t, q, v0, vn, a0, an, debug);
             }),
             py::arg("t_points"), py::arg("q_points"), py::arg("v0") = 0.0,
             py::arg("vn") = 0.0, py::arg("a0") = 0.0, py::arg("an") = 0.0,
             py::arg("debug") = false)
        .def("evaluate",
             py::overload_cast<double>(&CubicSplineWithAcceleration1::evaluate, py::const_),
             py::arg("t"))
        .def("evaluate",
             py::overload_cast<const Eigen::VectorXd&>(
                 &CubicSplineWithAcceleration1::evaluate, py::const_),
             py::arg("t"))
        .def("evaluate_velocity",
             py::overload_cast<double>(&CubicSplineWithAcceleration1::evaluate_velocity,
                                        py::const_),
             py::arg("t"))
        .def("evaluate_velocity",
             py::overload_cast<const Eigen::VectorXd&>(
                 &CubicSplineWithAcceleration1::evaluate_velocity, py::const_),
             py::arg("t"))
        .def("evaluate_acceleration",
             py::overload_cast<double>(&CubicSplineWithAcceleration1::evaluate_acceleration,
                                        py::const_),
             py::arg("t"))
        .def("evaluate_acceleration",
             py::overload_cast<const Eigen::VectorXd&>(
                 &CubicSplineWithAcceleration1::evaluate_acceleration, py::const_),
             py::arg("t"))
        .def_property_readonly("t_points", &CubicSplineWithAcceleration1::t_points)
        .def_property_readonly("q_points", &CubicSplineWithAcceleration1::q_points)
        .def_property_readonly("omega", &CubicSplineWithAcceleration1::omega)
        .def_property_readonly("n_points", &CubicSplineWithAcceleration1::n_points)
        .def_property_readonly("n_orig", &CubicSplineWithAcceleration1::n_orig);

    // CubicSplineWithAcceleration2
    py::class_<CubicSplineWithAcceleration2, CubicSpline>(m, "CubicSplineWithAcceleration2")
        .def(py::init([](std::vector<double> t, std::vector<double> q, SplineParameters params) {
                 return CubicSplineWithAcceleration2(t, q, params);
             }),
             py::arg("t_points"), py::arg("q_points"),
             py::arg("params") = SplineParameters{})
        .def("evaluate",
             py::overload_cast<double>(&CubicSplineWithAcceleration2::evaluate, py::const_),
             py::arg("t"))
        .def("evaluate",
             py::overload_cast<const Eigen::VectorXd&>(
                 &CubicSplineWithAcceleration2::evaluate, py::const_),
             py::arg("t"))
        .def("evaluate_velocity",
             py::overload_cast<double>(&CubicSplineWithAcceleration2::evaluate_velocity,
                                        py::const_),
             py::arg("t"))
        .def("evaluate_velocity",
             py::overload_cast<const Eigen::VectorXd&>(
                 &CubicSplineWithAcceleration2::evaluate_velocity, py::const_),
             py::arg("t"))
        .def("evaluate_acceleration",
             py::overload_cast<double>(&CubicSplineWithAcceleration2::evaluate_acceleration,
                                        py::const_),
             py::arg("t"))
        .def("evaluate_acceleration",
             py::overload_cast<const Eigen::VectorXd&>(
                 &CubicSplineWithAcceleration2::evaluate_acceleration, py::const_),
             py::arg("t"))
        .def_property_readonly("has_quintic_first", &CubicSplineWithAcceleration2::has_quintic_first)
        .def_property_readonly("has_quintic_last", &CubicSplineWithAcceleration2::has_quintic_last);
}
