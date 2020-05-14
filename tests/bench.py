import sys
import importlib
import timeit

pkg = sys.argv[1]
np = importlib.import_module(pkg)


LAPTOP_MAX = 83361790

SPA_MAX = 153000000

size = LAPTOP_MAX
data = np.random.random((size,))

def func():
    print(np.sin(data) ** 2 + np.cos(data) ** 2)
    # if np.__name__ == "cuda":
    #     np.cuda.Device().synchronize()


print(min(timeit.repeat(func, repeat=100, number=1)))

#rint(f"time: {now-then}")
