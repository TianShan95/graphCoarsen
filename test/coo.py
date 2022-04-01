import numpy as np
from scipy.sparse import csr_matrix

row  = np.array([0, 0, 1, 3, 1, 0, 0])
col  = np.array([0, 2, 1, 3, 1, 0, 0])
data = np.array([1, 1, 1, 1, 1, 1, 1])
csr_mat=csr_matrix((data, (row, col)), shape=(4, 4)).toarray()
print(csr_mat)