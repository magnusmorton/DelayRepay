import sys
import importlib
import timeit

pkg = sys.argv[1]
np = importlib.import_module(pkg)


LAPTOP_MAX = 8336179

SPA_MAX = 153000000

size = LAPTOP_MAX
data = np.random.random((size,))

def func():
     temp = np.exp(2j * np.pi * data)
     print(temp)
     np.fft.fft(temp)



print(min(timeit.repeat(func, repeat=100, number=1)))

#rint(f"time: {now-then}")
