#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_tridiagonal(py::module_& m);
void bind_cubic_spline(py::module_& m);
void bind_smoothing_spline(py::module_& m);
void bind_acc_splines(py::module_& m);
void bind_smoothing_search(py::module_& m);

PYBIND11_MODULE(interpolatecpp_py, m) {
    m.doc() = "C++ backend for InterpolatePy trajectory planning library";

    bind_tridiagonal(m);
    bind_cubic_spline(m);
    bind_smoothing_spline(m);
    bind_acc_splines(m);
    bind_smoothing_search(m);
}
