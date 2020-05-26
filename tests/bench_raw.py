import sys
import importlib
import timeit

pkg = sys.argv[1]
np = importlib.import_module(pkg)


LAPTOP_MAX = 83361790

DELAY_MAX = 210000000

size = LAPTOP_MAX
data = np.random.random((size,))

kern = np.ElementwiseKernel(
    'T x',
    'T z',
    '''
    T s = sin(x);
    T c = cos(x);
    T l = s * s;
    T r = c * c;
    z = l + r;
    ''',
    'kern1234'
)


# kern = np.ElementwiseKernel(
#     'T x',
#     'T z',
#     '''
#     z = sin(x) * sin(x) + cos(x) * cos(x)
#     ''',
#     'kern12345'
# )
def func():
    kern(data)
    np.cuda.Device().synchronize()


#then = perf_counter()
print(min(timeit.repeat(func, repeat=100,number=1)))
#now = perf_counter()

#rint(f"time: {now-then}")
