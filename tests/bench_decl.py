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
    z = s * c
    ''',
    'kern1234'
)


kern2 = np.ElementwiseKernel(
    'T x',
    'T z',
    '''
    z = sin(x) * cos(x)
    ''',
    'kern12345'
)
def func():
    kern(data)
    np.cuda.Device().synchronize()

def func2():
    kern2(data)
    np.cuda.Device().synchronize()

#then = perf_counter()
print(min(timeit.repeat(func, repeat=500,number=1)))
#now = perf_counter()

print(min(timeit.repeat(func2, repeat=500,number=1)))

#rint(f"time: {now-then}")
