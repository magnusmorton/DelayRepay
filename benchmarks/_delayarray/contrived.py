import delayrepay as np
from benchmarks.common import JACOBI_SIZE
import numpy as num

arr = np.ones(JACOBI_SIZE, dtype=num.float32)


print(arr + arr + arr + arr + arr)
