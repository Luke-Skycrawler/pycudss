import numpy as np 
from dxslv import CUSolver
from scipy.sparse import csr_matrix, csc_matrix, diags
dim = 100
x = np.random.rand(dim)
a = np.random.rand(dim, dim)
a = a + a.T
# a = np.identity(dim)
# a = diags(np.random.rand(dim))
for i in range(dim):
    a[i, i] += np.sum(a[i])
print("a.T - a = ", np.max(np.abs(a.T - a)))

b = a @ x 
a_sparse = csc_matrix(a)
print(a_sparse.indices, a_sparse.indptr)
# solver = CUSolver(a_sparse)
solver = CUSolver(a_sparse.indptr[:], a_sparse.indices, a_sparse.data)
solver.analyze_pattern()
solver.factorize()
# b[:] = 0.0
x_solved = solver.solve(b)
diff = x_solved - x
print("x solved = ", x_solved)
print("x original = ", x)
# print("b = ", b)
print(f"Max difference: {np.max(np.abs(diff))}")
rhs_diff = a_sparse @ x_solved - b
print("Max difference in rhs: ", np.max(np.abs(rhs_diff)))
