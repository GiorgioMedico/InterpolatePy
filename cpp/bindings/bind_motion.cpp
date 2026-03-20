#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <interpolatecpp/motion/double_s_trajectory.hpp>
#include <interpolatecpp/motion/motion_types.hpp>
#include <interpolatecpp/motion/parabolic_blend_trajectory.hpp>
#include <interpolatecpp/motion/polynomial_trajectory.hpp>
#include <interpolatecpp/motion/trapezoidal_trajectory.hpp>

namespace py = pybind11;
using namespace interpolatecpp::motion;

void bind_motion(py::module_& m) {
    auto motion_mod = m.def_submodule("motion", "Motion profile algorithms");

    // Result types
    py::class_<TrajectoryResult>(motion_mod, "TrajectoryResult")
        .def_readonly("position", &TrajectoryResult::position)
        .def_readonly("velocity", &TrajectoryResult::velocity)
        .def_readonly("acceleration", &TrajectoryResult::acceleration);

    py::class_<FullTrajectoryResult>(motion_mod, "FullTrajectoryResult")
        .def_readonly("position", &FullTrajectoryResult::position)
        .def_readonly("velocity", &FullTrajectoryResult::velocity)
        .def_readonly("acceleration", &FullTrajectoryResult::acceleration)
        .def_readonly("jerk", &FullTrajectoryResult::jerk);

    // BoundaryCondition
    py::class_<BoundaryCondition>(motion_mod, "BoundaryCondition")
        .def(py::init<>())
        .def_readwrite("position", &BoundaryCondition::position)
        .def_readwrite("velocity", &BoundaryCondition::velocity)
        .def_readwrite("acceleration", &BoundaryCondition::acceleration)
        .def_readwrite("jerk", &BoundaryCondition::jerk);

    // TimeInterval
    py::class_<TimeInterval>(motion_mod, "TimeInterval")
        .def(py::init<>())
        .def_readwrite("start", &TimeInterval::start)
        .def_readwrite("end", &TimeInterval::end)
        .def("duration", &TimeInterval::duration);

    // StateParams
    py::class_<StateParams>(motion_mod, "StateParams")
        .def(py::init([](double q0, double q1, double v0, double v1) {
                 return StateParams{q0, q1, v0, v1};
             }),
             py::arg("q_0"), py::arg("q_1"), py::arg("v_0") = 0.0, py::arg("v_1") = 0.0)
        .def_readonly("q_0", &StateParams::q_0)
        .def_readonly("q_1", &StateParams::q_1)
        .def_readonly("v_0", &StateParams::v_0)
        .def_readonly("v_1", &StateParams::v_1);

    // TrajectoryBounds
    py::class_<TrajectoryBounds>(motion_mod, "TrajectoryBounds")
        .def(py::init<double, double, double>(), py::arg("v_bound"), py::arg("a_bound"),
             py::arg("j_bound"))
        .def_readonly("v_bound", &TrajectoryBounds::v_bound)
        .def_readonly("a_bound", &TrajectoryBounds::a_bound)
        .def_readonly("j_bound", &TrajectoryBounds::j_bound);

    // PolynomialTrajectory
    py::class_<PolynomialTrajectory>(motion_mod, "PolynomialTrajectory")
        .def(py::init<const BoundaryCondition&, const BoundaryCondition&, const TimeInterval&,
                      int>(),
             py::arg("bc_start"), py::arg("bc_end"), py::arg("interval"), py::arg("order"))
        .def("evaluate", &PolynomialTrajectory::evaluate, py::arg("t"))
        .def_property_readonly("order", &PolynomialTrajectory::order)
        .def_property_readonly("t_start", &PolynomialTrajectory::t_start)
        .def_property_readonly("t_end", &PolynomialTrajectory::t_end)
        .def_property_readonly("duration", &PolynomialTrajectory::duration)
        .def_property_readonly("coefficients", &PolynomialTrajectory::coefficients)
        .def_static("heuristic_velocities", &PolynomialTrajectory::heuristic_velocities,
                    py::arg("points"), py::arg("times"))
        .def_static("multipoint_trajectory", &PolynomialTrajectory::multipoint_trajectory,
                    py::arg("points"), py::arg("times"), py::arg("order") = 3,
                    py::arg("v0") = 0.0, py::arg("vn") = 0.0)
        .def_static("evaluate_multipoint", &PolynomialTrajectory::evaluate_multipoint,
                    py::arg("segments"), py::arg("t"));

    // DoubleSTrajectory
    py::class_<DoubleSTrajectory>(motion_mod, "DoubleSTrajectory")
        .def(py::init<const StateParams&, const TrajectoryBounds&>(), py::arg("state"),
             py::arg("bounds"))
        .def("evaluate", &DoubleSTrajectory::evaluate, py::arg("t"))
        .def_property_readonly("duration", &DoubleSTrajectory::duration)
        .def("phase_durations", &DoubleSTrajectory::phase_durations);

    // TrapezoidalTrajectory
    py::class_<TrapezoidalTrajectory>(motion_mod, "TrapezoidalTrajectory")
        .def(py::init<double, double, double, double, double, double, double>(),
             py::arg("q0"), py::arg("q1"), py::arg("amax"), py::arg("vmax"),
             py::arg("v0") = 0.0, py::arg("v1") = 0.0, py::arg("t0") = 0.0)
        .def("evaluate", &TrapezoidalTrajectory::evaluate, py::arg("t"))
        .def_property_readonly("duration", &TrapezoidalTrajectory::duration)
        .def_property_readonly("t_start", &TrapezoidalTrajectory::t_start)
        .def_property_readonly("t_end", &TrapezoidalTrajectory::t_end)
        .def_static("heuristic_velocities", &TrapezoidalTrajectory::heuristic_velocities,
                    py::arg("points"), py::arg("times"), py::arg("vmax"))
        .def_static("interpolate_waypoints", &TrapezoidalTrajectory::interpolate_waypoints,
                    py::arg("points"), py::arg("amax"), py::arg("vmax"), py::arg("v0") = 0.0,
                    py::arg("vn") = 0.0, py::arg("times") = std::vector<double>{},
                    py::arg("velocities") = std::vector<double>{})
        .def_static("evaluate_multipoint", &TrapezoidalTrajectory::evaluate_multipoint,
                    py::arg("segments"), py::arg("t"));

    // ParabolicBlendTrajectory
    py::class_<ParabolicBlendTrajectory>(motion_mod, "ParabolicBlendTrajectory")
        .def(py::init<const std::vector<double>&, const std::vector<double>&,
                      const std::vector<double>&>(),
             py::arg("q"), py::arg("t"), py::arg("dt_blend"))
        .def("evaluate", &ParabolicBlendTrajectory::evaluate, py::arg("t"))
        .def_property_readonly("duration", &ParabolicBlendTrajectory::duration)
        .def_property_readonly("n_waypoints", &ParabolicBlendTrajectory::n_waypoints);
}
