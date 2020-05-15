import sys
import importlib
from time import perf_counter

pkg = sys.argv[1]
np = importlib.import_module(pkg)


LAPTOP_MAX = 19999000

size = LAPTOP_MAX
data = np.random.random((size,))
x = np.random.random((size,))
y = np.random.random((size,))
z = np.random.random((size,))
xx = np.random.random((size,))
yy = np.random.random((size,))
zz = np.random.random((size,))

#@cupy.fuse(kernel_name='func')
res = data * x * y * z * xx + yy/zz * data * 2
#then = perf_counter()
print(res)
#now = perf_counter()

#rint(f"time: {now-then}")
