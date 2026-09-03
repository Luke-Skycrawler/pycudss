import numpy as np
from scipy.sparse import csr_matrix

from dxslv import CUSolver, CUSolverDevice


DIM = 100
TOLERANCE = 1.0e-10


def make_problem(dim=DIM):
    rng = np.random.default_rng(0)
    x = rng.random(dim)

    a = rng.random((dim, dim))
    a = a + a.T
    a[np.diag_indices(dim)] += np.sum(a, axis=1)

    a_refactored = rng.random((dim, dim))
    a_refactored = a_refactored + a_refactored.T
    a_refactored[np.diag_indices(dim)] += np.sum(a_refactored, axis=1)

    return x, csr_matrix(a), csr_matrix(a_refactored)


def test_cusolver():
    x, a, a_refactored = make_problem()
    b = a @ x

    solver = CUSolver(a.indptr, a.indices, a.data)
    solver.analyze_pattern()
    solver.factorize()

    x_solved = solver.solve(b)
    np.testing.assert_allclose(x_solved, x, rtol=0.0, atol=TOLERANCE)
    np.testing.assert_allclose(a @ x_solved, b, rtol=0.0, atol=TOLERANCE)

    b_refactored = a_refactored @ x
    solver.refactorize(a_refactored.data)
    x_solved = solver.solve(b_refactored)
    np.testing.assert_allclose(x_solved, x, rtol=0.0, atol=TOLERANCE)
    np.testing.assert_allclose(
        a_refactored @ x_solved, b_refactored, rtol=0.0, atol=TOLERANCE
    )


def test_cusolver_device():
    import warp as wp

    x, a, a_refactored = make_problem()
    b = a @ x
    n = a.shape[0]

    outers_d = wp.array(a.indptr, dtype=wp.int32, device="cuda")
    indices_d = wp.array(a.indices, dtype=wp.int32, device="cuda")
    values_d = wp.array(a.data, dtype=wp.float64, device="cuda")
    b_d = wp.array(b, dtype=wp.float64, device="cuda")
    x_d = wp.zeros(n, dtype=wp.float64, device="cuda")
    wp.synchronize()

    solver = CUSolverDevice(
        outers_d.ptr, indices_d.ptr, values_d.ptr, n, a.nnz
    )
    solver.analyze_pattern()
    solver.factorize()
    solver.solve(b_d.ptr, x_d.ptr)
    wp.synchronize()

    x_solved = x_d.numpy()
    np.testing.assert_allclose(x_solved, x, rtol=0.0, atol=TOLERANCE)
    np.testing.assert_allclose(a @ x_solved, b, rtol=0.0, atol=TOLERANCE)

    b_refactored = a_refactored @ x
    values_refactored_d = wp.array(
        a_refactored.data, dtype=wp.float64, device="cuda"
    )
    b_refactored_d = wp.array(b_refactored, dtype=wp.float64, device="cuda")
    solver.refactorize(values_refactored_d.ptr)
    solver.solve(b_refactored_d.ptr, x_d.ptr)
    wp.synchronize()

    x_solved = x_d.numpy()
    np.testing.assert_allclose(x_solved, x, rtol=0.0, atol=TOLERANCE)
    np.testing.assert_allclose(
        a_refactored @ x_solved, b_refactored, rtol=0.0, atol=TOLERANCE
    )


if __name__ == "__main__":
    # test_cusolver()
    test_cusolver_device()
    print("All solver tests passed.")
