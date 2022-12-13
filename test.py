import numpy as np
import time
import numba as nb


@nb.njit(nb.types.Array(nb.float32,2,'C')(nb.types.Array(nb.float32,2,'C'),nb.types.Array(nb.float32,2,'C')), cache=True)
def nb_matmul(a,b):
    return a@b

n = 8192

a = np.ascontiguousarray(np.random.random((n,n)), np.float32)
b = np.ascontiguousarray(np.random.random((n,n)), np.float32)
tic = time.time()
c = nb_matmul(a, b)
toc = time.time()
print(toc-tic)