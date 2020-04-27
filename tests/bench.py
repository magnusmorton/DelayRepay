import delayarray as cupy
import numpy as np

LAPTOP_MAX = 83361790

size = LAPTOP_MAX
data = cupy.random.random((size,))

#@cupy.fuse(kernel_name='func')
def func(arg):
    return np.sin(arg) ** 2 + np.cos(arg) ** 2

print(func(data))
