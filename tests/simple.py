import cupy
import timeit as tt
import delayarray as dr
data = cupy.ones((300000000, ), dtype='float32')

#@cupy.fuse(kernel_name='func')
def func(arg):
    return arg + 2

func(data).get()


