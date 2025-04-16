import caboose
import numpy as np
from scipy.sparse import csr_matrix

A = np.array(
    [[1, 1, 1, 0, 1],
     [0, 1, 0, 1, 0],
     [0, 1, 1, 0, 1],
     [0, 0, 0, 1, 0]],
    dtype='float64')

A_sparse = csr_matrix(A)

k = 2
num_rows, num_cols = A_sparse.shape
index = caboose.Index(num_rows, num_cols, A_sparse.indptr, A_sparse.indices, A_sparse.data, k)

print(index.topk(0))
print(index.topk(3))
index.forget(0, 1)
print(index.topk(0))
print(index.topk(3))