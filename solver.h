#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "cudss.h"

using Vec = Eigen::VectorXd;
using Veci = Eigen::VectorXi;

struct SolverBase
{
    Veci outers;
    Veci indices;
    Vec values;
    int nnz, n;
    SolverBase(const Veci &outers, const Veci &indices, const Vec &values)
        : outers(outers), indices(indices), values(values), n(outers.rows() - 1), nnz(indices.rows()) {}
    SolverBase(const Eigen::SparseMatrix<double> &A) : n(A.outerSize()), nnz(A.nonZeros())
    {
        const int *outer_ptr = A.outerIndexPtr();
        const int *indices_ptr = A.innerIndexPtr();
        const double *values_ptr = A.valuePtr();

        outers.resize(A.outerSize() + 1);
        indices.resize(A.nonZeros());
        values.resize(A.nonZeros());

        memcpy(outers.data(), outer_ptr, (A.outerSize() + 1) * sizeof(int));
        memcpy(indices.data(), indices_ptr, A.nonZeros() * sizeof(int));
        memcpy(values.data(), values_ptr, A.nonZeros() * sizeof(double));
    }
    virtual void analyze_pattern() = 0;
    virtual void factorize() = 0;
    virtual Vec solve(const Vec &b) const = 0;
};

struct CUSolver : SolverBase
{
    CUSolver(const Veci &outers, const Veci &indices, const Vec &values);
    CUSolver(const Eigen::SparseMatrix<double> &A);

    cudssHandle_t handle;
    cudssConfig_t solver_config;
    cudssData_t solver_data;
    cudssMatrix_t A, x, b;

    void analyze_pattern() override;
    void factorize() override;
    Vec solve(const Vec &b) const override;
    int stage = 0; // 0: not analyzed, 1: analyzed, 2: factorized

    void init();
    ~CUSolver()
    {
        cudssMatrixDestroy(A);
        cudssMatrixDestroy(b);
        cudssMatrixDestroy(x);
        cudaFree(outers_d);
        cudaFree(indices_d);
        cudaFree(values_d);
        cudaFree(b_d);
        cudaFree(x_d);
        cudssDataDestroy(handle, solver_data);
        cudssConfigDestroy(solver_config);
        cudssDestroy(handle);
    }
    int *outers_d, *indices_d;
    double *values_d, *b_d, *x_d;
};
