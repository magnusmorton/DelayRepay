import cupy
data = cupy.random.random((1000, 1000))

#@cupy.fuse(kernel_name='func')
def func(arg):
    return cupy.sin(arg) ** 2 + cupy.cos(arg) ** 2

print(func(data))
