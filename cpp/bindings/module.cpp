#include <pybind11/pybind11.h>

namespace py = pybind11;

// Phase 1: Cubic Splines
void bind_tridiagonal(py::module_& m);
void bind_cubic_spline(py::module_& m);
void bind_smoothing_spline(py::module_& m);
void bind_acc_splines(py::module_& m);
void bind_smoothing_search(py::module_& m);

// Phase 2: B-Splines
void bind_bspline(py::module_& m);

// Phase 3: Motion Profiles
void bind_motion(py::module_& m);

// Phase 4: Quaternion Interpolation
void bind_quaternion(py::module_& m);

// Phase 5: Geometric Paths
void bind_paths(py::module_& m);

PYBIND11_MODULE(interpolatecpp_py, m) {
    m.doc() = "C++ backend for InterpolatePy trajectory planning library";

    // Phase 1
    bind_tridiagonal(m);
    bind_cubic_spline(m);
    bind_smoothing_spline(m);
    bind_acc_splines(m);
    bind_smoothing_search(m);

    // Phase 2
    bind_bspline(m);

    // Phase 3
    bind_motion(m);

    // Phase 4
    bind_quaternion(m);

    // Phase 5
    bind_paths(m);
}
