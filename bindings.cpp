#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "solver.h"
using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(dxslv, m)
{
    m.doc() = "direct solver (cudss) python bindings";

    py::class_<CUSolver>(m, "CUSolver")
        .def(py::init<const Veci &, const Veci &, const Vec &>())
        .def(py::init<const Eigen::SparseMatrix<double> &>())
        .def("analyze_pattern", &CUSolver::analyze_pattern)
        .def("factorize", &CUSolver::factorize)
        .def("solve", &CUSolver::solve);
}