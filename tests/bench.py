import sys
import importlib

pkg = sys.argv[1]
np = importlib.import_module(pkg)


LAPTOP_MAX = 83361790

size = LAPTOP_MAX
data = np.random.random((size,)).astype("float32")

#@cupy.fuse(kernel_name='func')
def func(arg):
    return np.sin(arg) ** 2.0 + np.cos(arg) ** 2.0

print(func(data))
