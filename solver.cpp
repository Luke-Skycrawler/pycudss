#include "solver.h"
CUSolver::CUSolver(const Veci &outers, const Veci &indices, const Vec &values)
    : SolverBase(outers, indices, values)
{
    init();
}
CUSolver::CUSolver(const Eigen::SparseMatrix<double> &A)
    : SolverBase(A)
{
    init();
}
void CUSolver::init()
{
    cudssCreate(&handle);
    cudssConfigCreate(&solver_config);
    cudssDataCreate(handle, &solver_data);

    cudaMalloc((void **)&outers_d, (n + 1) * sizeof(int));
    cudaMalloc((void **)&indices_d, nnz * sizeof(int));
    cudaMalloc((void **)&values_d, nnz * sizeof(double));
    cudaMalloc((void **)&b_d, n * sizeof(double));
    cudaMalloc((void **)&x_d, n * sizeof(double));

    cudaMemcpy(outers_d, outers.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(indices_d, indices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(values_d, values.data(), nnz * sizeof(double), cudaMemcpyHostToDevice);

    int64_t nrows = n, ncols = n;
    int ldb = ncols, ldx = nrows;
    int nrhs = 1;
    cudssMatrixCreateDn(&b, ncols, nrhs, ldb, b_d, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR);
    cudssMatrixCreateDn(&x, nrows, nrhs, ldx, x_d, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR);

    /* Create a matrix object for the sparse input matrix. */
    cudssMatrixType_t mtype = CUDSS_MTYPE_SPD;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_UPPER;
    cudssIndexBase_t base = CUDSS_BASE_ZERO;
    cudssMatrixCreateCsr(&A, nrows, ncols, nnz, outers_d, NULL, indices_d, values_d, CUDA_R_32I, CUDA_R_64F, mtype, mview, base);
}

void CUSolver::factorize()
{
    assert(stage >= 1);
    /* Factorization */
    cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solver_config, solver_data, A, x, b);
    stage = 2;
}

void CUSolver::refactorize(const Vec &new_values) {
    assert(stage >= 1);
    cudaMemcpy(values_d, new_values.data(), nnz * sizeof(double), cudaMemcpyHostToDevice);

    cudssExecute(handle, CUDSS_PHASE_REFACTORIZATION, solver_config, solver_data, A, x, b);
    stage = 2;
}

Vec CUSolver::solve(const Vec &bb) const
{
    Vec xe;
    xe.resize(n);
    assert(stage == 2);
    int nrhs = 1;

    cudaMemcpy(b_d, bb.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    /* Solving */
    cudssExecute(handle, CUDSS_PHASE_SOLVE, solver_config, solver_data, A, x, b);
    cudaMemcpy(xe.data(), x_d, nrhs * n * sizeof(double), cudaMemcpyDeviceToHost);
    return xe;
}

void CUSolver::analyze_pattern()
{
    cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solver_config, solver_data, A, x, b);
    stage = 1;
}
