#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <interpolatecpp/quat/log_quaternion_interpolation.hpp>
#include <interpolatecpp/quat/modified_log_quaternion_interpolation.hpp>
#include <interpolatecpp/quat/quaternion.hpp>
#include <interpolatecpp/quat/quaternion_spline.hpp>
#include <interpolatecpp/quat/squad_c2.hpp>

namespace py = pybind11;
using namespace interpolatecpp::quat;

void bind_quaternion(py::module_& m) {
    auto quat_mod = m.def_submodule("quat", "Quaternion interpolation algorithms");

    // Quaternion
    py::class_<Quaternion>(quat_mod, "Quaternion")
        .def(py::init<double, double, double, double>(), py::arg("w") = 1.0,
             py::arg("x") = 0.0, py::arg("y") = 0.0, py::arg("z") = 0.0)
        .def_static("identity", &Quaternion::identity)
        .def_static("from_angle_axis", &Quaternion::from_angle_axis, py::arg("angle"),
                    py::arg("axis"))
        .def_static("from_euler_angles", &Quaternion::from_euler_angles, py::arg("roll"),
                    py::arg("pitch"), py::arg("yaw"))
        .def_property_readonly("w", &Quaternion::w)
        .def_property_readonly("x", &Quaternion::x)
        .def_property_readonly("y", &Quaternion::y)
        .def_property_readonly("z", &Quaternion::z)
        .def_property_readonly("vec", &Quaternion::vec)
        .def("__mul__",
             py::overload_cast<const Quaternion&>(&Quaternion::operator*, py::const_))
        .def("__mul__",
             py::overload_cast<double>(&Quaternion::operator*, py::const_))
        .def("__rmul__",
             [](const Quaternion& q, double s) { return q * s; })
        .def("__add__",
             py::overload_cast<const Quaternion&>(&Quaternion::operator+, py::const_))
        .def("__sub__",
             py::overload_cast<const Quaternion&>(&Quaternion::operator-, py::const_))
        .def("__neg__", [](const Quaternion& q) { return -q; })
        .def("conjugate", &Quaternion::conjugate)
        .def("inverse", &Quaternion::inverse)
        .def("unit", &Quaternion::unit)
        .def("norm", &Quaternion::norm)
        .def("norm_squared", &Quaternion::norm_squared)
        .def("dot_product", &Quaternion::dot_product, py::arg("other"))
        .def("to_rotation_matrix", &Quaternion::to_rotation_matrix)
        .def("to_transformation_matrix", &Quaternion::to_transformation_matrix)
        .def("to_axis_angle", &Quaternion::to_axis_angle)
        .def("to_euler_angles", &Quaternion::to_euler_angles)
        .def_static("from_rotation_matrix", &Quaternion::from_rotation_matrix,
                    py::arg("rotation_matrix"))
        // Dynamics
        .def("E", &Quaternion::E, py::arg("sign"))
        .def("dot", &Quaternion::dot, py::arg("omega"), py::arg("sign"))
        .def_static("Omega", &Quaternion::Omega, py::arg("q"), py::arg("q_dot"))
        .def_static("slerp", &Quaternion::slerp, py::arg("q0"), py::arg("q1"), py::arg("t"))
        .def_static("slerp_prime", &Quaternion::slerp_prime, py::arg("q0"),
                    py::arg("q1"), py::arg("t"))
        .def_static("squad", &Quaternion::squad, py::arg("p"), py::arg("a"), py::arg("b"),
                    py::arg("q"), py::arg("t"))
        .def_static("compute_intermediate_quaternion",
                    &Quaternion::compute_intermediate_quaternion, py::arg("q_prev"),
                    py::arg("q_curr"), py::arg("q_next"))
        .def_static("exp", &Quaternion::exp, py::arg("q"))
        .def_static("log", &Quaternion::log, py::arg("q"))
        .def_static("power", &Quaternion::power, py::arg("q"), py::arg("t"));

    // QuaternionSpline
    py::enum_<QuaternionSpline::Method>(quat_mod, "QuaternionSplineMethod")
        .value("Slerp", QuaternionSpline::Method::Slerp)
        .value("Squad", QuaternionSpline::Method::Squad)
        .value("Auto", QuaternionSpline::Method::Auto);

    py::class_<QuaternionSpline>(quat_mod, "QuaternionSpline")
        .def(py::init<const std::vector<double>&, const std::vector<Quaternion>&,
                      QuaternionSpline::Method>(),
             py::arg("time_points"), py::arg("quaternions"),
             py::arg("method") = QuaternionSpline::Method::Auto)
        .def("evaluate", &QuaternionSpline::evaluate, py::arg("t"))
        .def("evaluate_velocity", &QuaternionSpline::evaluate_velocity, py::arg("t"))
        .def("evaluate_acceleration", &QuaternionSpline::evaluate_acceleration, py::arg("t"))
        .def_property_readonly("t_min", &QuaternionSpline::t_min)
        .def_property_readonly("t_max", &QuaternionSpline::t_max);

    // SquadC2Config
    py::class_<SquadC2Config>(quat_mod, "SquadC2Config")
        .def(py::init<>())
        .def_readwrite("time_points", &SquadC2Config::time_points)
        .def_readwrite("quaternions", &SquadC2Config::quaternions)
        .def_readwrite("normalize_quaternions", &SquadC2Config::normalize_quaternions)
        .def_readwrite("validate_continuity", &SquadC2Config::validate_continuity);

    // SquadC2
    py::class_<SquadC2>(quat_mod, "SquadC2")
        .def(py::init<const std::vector<double>&, const std::vector<Quaternion>&, bool, bool>(),
             py::arg("time_points"), py::arg("quaternions"),
             py::arg("normalize_quaternions") = true,
             py::arg("validate_continuity") = true)
        .def(py::init<const SquadC2Config&>(), py::arg("config"))
        .def("evaluate", &SquadC2::evaluate, py::arg("t"))
        .def("evaluate_velocity", &SquadC2::evaluate_velocity, py::arg("t"))
        .def("evaluate_acceleration", &SquadC2::evaluate_acceleration, py::arg("t"))
        .def_property_readonly("t_min", &SquadC2::t_min)
        .def_property_readonly("t_max", &SquadC2::t_max)
        .def_property_readonly("validate_continuity", &SquadC2::validate_continuity);

    // LogQuaternionInterpolation
    py::class_<LogQuaternionInterpolation>(quat_mod, "LogQuaternionInterpolation")
        .def(py::init<const std::vector<double>&, const std::vector<Quaternion>&, int,
                      const std::optional<Eigen::VectorXd>&,
                      const std::optional<Eigen::VectorXd>&>(),
             py::arg("time_points"), py::arg("quaternions"), py::arg("degree") = 3,
             py::arg("initial_velocity") = std::nullopt,
             py::arg("final_velocity") = std::nullopt)
        .def("evaluate", &LogQuaternionInterpolation::evaluate, py::arg("t"))
        .def("evaluate_velocity", &LogQuaternionInterpolation::evaluate_velocity, py::arg("t"))
        .def("evaluate_acceleration", &LogQuaternionInterpolation::evaluate_acceleration,
             py::arg("t"))
        .def_property_readonly("t_min", &LogQuaternionInterpolation::t_min)
        .def_property_readonly("t_max", &LogQuaternionInterpolation::t_max);

    // ModifiedLogQuaternionInterpolation
    py::class_<ModifiedLogQuaternionInterpolation>(quat_mod,
                                                    "ModifiedLogQuaternionInterpolation")
        .def(py::init<const std::vector<double>&, const std::vector<Quaternion>&, int, bool,
                      const std::optional<Eigen::VectorXd>&,
                      const std::optional<Eigen::VectorXd>&>(),
             py::arg("time_points"), py::arg("quaternions"), py::arg("degree") = 3,
             py::arg("normalize_axis") = true, py::arg("initial_velocity") = std::nullopt,
             py::arg("final_velocity") = std::nullopt)
        .def("evaluate", &ModifiedLogQuaternionInterpolation::evaluate, py::arg("t"))
        .def("evaluate_velocity", &ModifiedLogQuaternionInterpolation::evaluate_velocity,
             py::arg("t"))
        .def("evaluate_acceleration",
             &ModifiedLogQuaternionInterpolation::evaluate_acceleration, py::arg("t"))
        .def_property_readonly("t_min", &ModifiedLogQuaternionInterpolation::t_min)
        .def_property_readonly("t_max", &ModifiedLogQuaternionInterpolation::t_max)
        .def_property_readonly("normalize_axis",
                               &ModifiedLogQuaternionInterpolation::normalize_axis);
}
