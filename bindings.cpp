#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(dxslv, m)
{
    m.doc() = "direct solver (cudss) python bindings";

    py::class_<SolverBase>(m, "SolverBase")
        .def(py::init<Eigen::VectorXi, Eigen::VectorXi, Eigen::VectorXd>())
}