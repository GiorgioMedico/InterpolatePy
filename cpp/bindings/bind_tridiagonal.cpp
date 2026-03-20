#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <interpolatecpp/tridiagonal.hpp>

namespace py = pybind11;

void bind_tridiagonal(py::module_& m) {
    m.def("solve_tridiagonal", &interpolatecpp::solve_tridiagonal,
          py::arg("lower_diagonal"), py::arg("main_diagonal"),
          py::arg("upper_diagonal"), py::arg("right_hand_side"),
          "Solve a tridiagonal system using the Thomas algorithm");
}
