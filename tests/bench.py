import delayarray as cupy
data = cupy.random.random((83361790,))

#@cupy.fuse(kernel_name='func')
def func(arg):
    return cupy.sin(arg) ** 2 + cupy.cos(arg) ** 2

print(func(data))
