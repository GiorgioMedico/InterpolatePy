#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <interpolatecpp/spline/cubic_spline.hpp>

namespace py = pybind11;
using namespace interpolatecpp::spline;

void bind_cubic_spline(py::module_& m) {
    py::class_<CubicSpline>(m, "CubicSpline")
        .def(py::init([](std::vector<double> t, std::vector<double> q, double v0, double vn,
                         bool debug) { return CubicSpline(t, q, v0, vn, debug); }),
             py::arg("t_points"), py::arg("q_points"), py::arg("v0") = 0.0,
             py::arg("vn") = 0.0, py::arg("debug") = false)
        .def("evaluate", py::overload_cast<double>(&CubicSpline::evaluate, py::const_),
             py::arg("t"))
        .def("evaluate",
             py::overload_cast<const Eigen::VectorXd&>(&CubicSpline::evaluate, py::const_),
             py::arg("t"))
        .def("evaluate_velocity",
             py::overload_cast<double>(&CubicSpline::evaluate_velocity, py::const_), py::arg("t"))
        .def("evaluate_velocity",
             py::overload_cast<const Eigen::VectorXd&>(&CubicSpline::evaluate_velocity,
                                                         py::const_),
             py::arg("t"))
        .def("evaluate_acceleration",
             py::overload_cast<double>(&CubicSpline::evaluate_acceleration, py::const_),
             py::arg("t"))
        .def("evaluate_acceleration",
             py::overload_cast<const Eigen::VectorXd&>(&CubicSpline::evaluate_acceleration,
                                                         py::const_),
             py::arg("t"))
        .def_property_readonly("t_points", &CubicSpline::t_points)
        .def_property_readonly("q_points", &CubicSpline::q_points)
        .def_property_readonly("velocities", &CubicSpline::velocities)
        .def_property_readonly("coefficients", &CubicSpline::coefficients)
        .def_property_readonly("n_segments", &CubicSpline::n_segments);
}
