#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <interpolatecpp/path/circular_path.hpp>
#include <interpolatecpp/path/frenet_frame.hpp>
#include <interpolatecpp/path/linear_path.hpp>
#include <interpolatecpp/path/linear_traj.hpp>

namespace py = pybind11;
using namespace interpolatecpp::path;

void bind_paths(py::module_& m) {
    auto path_mod = m.def_submodule("path", "Geometric path algorithms");

    // LinearPath
    py::class_<LinearPath>(path_mod, "LinearPath")
        .def(py::init<const Eigen::Vector3d&, const Eigen::Vector3d&>(), py::arg("pi"),
             py::arg("pf"))
        .def("position",
             py::overload_cast<double>(&LinearPath::position, py::const_), py::arg("s"))
        .def("position",
             py::overload_cast<const Eigen::VectorXd&>(&LinearPath::position, py::const_),
             py::arg("s"))
        .def("velocity", &LinearPath::velocity, py::arg("s"))
        .def("acceleration", &LinearPath::acceleration, py::arg("s"))
        .def_property_readonly("length", &LinearPath::length);

    // CircularPath
    py::class_<CircularPath>(path_mod, "CircularPath")
        .def(py::init<const Eigen::Vector3d&, const Eigen::Vector3d&,
                      const Eigen::Vector3d&>(),
             py::arg("axis"), py::arg("axis_point"), py::arg("circle_point"))
        .def("position",
             py::overload_cast<double>(&CircularPath::position, py::const_), py::arg("s"))
        .def("position",
             py::overload_cast<const Eigen::VectorXd&>(&CircularPath::position, py::const_),
             py::arg("s"))
        .def("velocity", &CircularPath::velocity, py::arg("s"))
        .def("acceleration", &CircularPath::acceleration, py::arg("s"))
        .def_property_readonly("radius", &CircularPath::radius)
        .def_property_readonly("center", &CircularPath::center);

    // FrenetFrame
    py::class_<FrenetFrame>(path_mod, "FrenetFrame")
        .def_readonly("tangent", &FrenetFrame::tangent)
        .def_readonly("normal", &FrenetFrame::normal)
        .def_readonly("binormal", &FrenetFrame::binormal)
        .def_readonly("curvature", &FrenetFrame::curvature)
        .def_readonly("torsion", &FrenetFrame::torsion);

    // compute_frenet_frames
    path_mod.def("compute_frenet_frames", &compute_frenet_frames, py::arg("curve"),
                 py::arg("s_values"));

    // Helper trajectory functions
    path_mod.def("circular_trajectory_with_derivatives",
                 &circular_trajectory_with_derivatives, py::arg("u"), py::arg("r") = 2.0);
    path_mod.def("helicoidal_trajectory_with_derivatives",
                 &helicoidal_trajectory_with_derivatives, py::arg("u"), py::arg("r") = 2.0,
                 py::arg("d") = 0.5);

    // LinearTrajResult
    py::class_<LinearTrajResult>(path_mod, "LinearTrajResult")
        .def_readonly("positions", &LinearTrajResult::positions)
        .def_readonly("velocities", &LinearTrajResult::velocities)
        .def_readonly("accelerations", &LinearTrajResult::accelerations);

    // linear_traj
    path_mod.def("linear_traj", &linear_traj, py::arg("p0"), py::arg("p1"), py::arg("t0"),
                 py::arg("t1"), py::arg("num_points") = 100);
}
