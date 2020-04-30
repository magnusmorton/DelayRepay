import sys
import importlib
from time import perf_counter

pkg = sys.argv[1]
np = importlib.import_module(pkg)


LAPTOP_MAX = 83361790

size = LAPTOP_MAX
data = np.random.random((size,))

#@cupy.fuse(kernel_name='func')
def func(arg):
    return np.sin(arg) ** 2 + np.cos(arg) ** 2

#then = perf_counter()
print(func(data))
#now = perf_counter()

#rint(f"time: {now-then}")
