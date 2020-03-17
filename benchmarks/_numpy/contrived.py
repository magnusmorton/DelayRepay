import numpy as np
from benchmarks.common import JACOBI_SIZE

arr = np.ones(JACOBI_SIZE, dtype=np.float32)


print(arr + arr + arr + arr + arr)



