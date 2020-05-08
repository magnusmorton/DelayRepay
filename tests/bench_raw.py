import sys
import importlib
import timeit

pkg = sys.argv[1]
np = importlib.import_module(pkg)


LAPTOP_MAX = 83361790

DELAY_MAX = 210000000

size = DELAY_MAX
data = np.random.random((size,))

kern = np.ElementwiseKernel(
    'T x',
    'T z',
    '''
    T s = sin(x);
    T c = cos(x);
    z = s * s + c * c
    ''',
    'kern1234'
)

def func():
    kern(data)
    np.cuda.Device().synchronize()


#then = perf_counter()
print(min(timeit.repeat(func, repeat=1,number=1)))
#now = perf_counter()

#rint(f"time: {now-then}")
