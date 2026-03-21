#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <interpolatecpp/spline/smoothing_search.hpp>

namespace py = pybind11;
using namespace interpolatecpp::spline;

void bind_smoothing_search(py::module_& m) {
    // SplineConfig
    py::class_<SplineConfig>(m, "SplineConfig")
        .def(py::init([](std::optional<Eigen::VectorXd> weights, double v0, double vn,
                         int max_iterations, bool debug) {
                 return SplineConfig{weights, v0, vn, max_iterations, debug};
             }),
             py::arg("weights") = std::nullopt, py::arg("v0") = 0.0, py::arg("vn") = 0.0,
             py::arg("max_iterations") = 50, py::arg("debug") = false)
        .def_readwrite("weights", &SplineConfig::weights)
        .def_readwrite("v0", &SplineConfig::v0)
        .def_readwrite("vn", &SplineConfig::vn)
        .def_readwrite("max_iterations", &SplineConfig::max_iterations)
        .def_readwrite("debug", &SplineConfig::debug);

    // SmoothingSearchResult
    py::class_<SmoothingSearchResult>(m, "SmoothingSearchResult")
        .def_readonly("spline", &SmoothingSearchResult::spline)
        .def_readonly("mu", &SmoothingSearchResult::mu)
        .def_readonly("max_error", &SmoothingSearchResult::max_error)
        .def_readonly("iterations", &SmoothingSearchResult::iterations);

    // Free function
    m.def(
        "smoothing_spline_with_tolerance",
        [](std::vector<double> t, std::vector<double> q, double tolerance,
           const SplineConfig& config) {
            return smoothing_spline_with_tolerance(t, q, tolerance, config);
        },
        py::arg("t_points"), py::arg("q_points"), py::arg("tolerance"), py::arg("config"));
}
