import sys
import importlib
import timeit

pkg = sys.argv[1]
np = importlib.import_module(pkg)


LAPTOP_MAX = 83361790

SPA_MAX = 153000000

size = LAPTOP_MAX
data = np.random.random((size,))

print(np.sin(data) ** 2 + np.cos(data) ** 2)
