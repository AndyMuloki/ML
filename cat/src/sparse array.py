import numpy as np
from scipy import sparse

example = np.array(
    [
        [0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0 ,0, 0]
    ]
)

print(f"Size of dense array: {example.nbytes}")

# convert numpy array to sparse CSR matrix
sparse_example = sparse.csr_matrix(example)

# print size of this sparse matrix
print(f"Size of sparse array: {sparse_example.data.nbytes}")

full_size = (
    sparse_example.data.nbytes + 
    sparse_example.indptr.nbytes + 
    sparse_example.indices.nbytes 
)

print(f"Full size of sparse array: {full_size}")