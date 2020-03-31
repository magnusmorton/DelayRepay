'''Delay array and related stuff'''

from typing import Any
import numpy as np
import numpy.lib.mixins
import delayarray.num as num
from .cl import run_gpu


def cast(func):
    '''cast to Delay array decorator'''
    def wrapper(*args, **kwargs):
        arr = func(*args, **kwargs)
        if not isinstance(arr, DelayArray):
            arr = DelayArray(arr.shape, buffer=arr)
        return arr
    return wrapper


def calc_shape(func, shape1, shape2):
    if len(shape1) == 1:
        shape1 = (1,) + shape1
    if len(shape2) == 1:
        shape2 = (1,) + shape2
    if func == 'multiply':
        return (shape1[0], shape2[1])
    if func == 'add':
        return shape1
    if func == 'dot':
        if shape1[0] == 1:
            return (1,)
        else:
            return (shape1[0], shape2[1])


def calc_type(func, type1, type2):
    if 'float64' in (type1, type2):
        return 'float64'
    elif 'float32' in (type1, type2):
        return 'float32'
    elif 'int64' in (type1, type2):
        return 'int64'
    else:
        return type1


HANDLED_FUNCTIONS = {}


def implements(np_function):
    "Register an __array_function__ implementation for DiagonalArray objects."
    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func
    return decorator


class DelayArray(numpy.lib.mixins.NDArrayOperatorsMixin):
    def __new__(cls, shape, dtype='float32', buffer=None, offset=0,
                strides=None, order=None, parent=None, ops=None, ex=None):
        self = super(DelayArray, cls).__new__(cls)
        if buffer is not None:
            self._ndarray = buffer
            self.ex = num.NPArray(buffer)
        elif ops is None:
            self._ndarray = np.ndarray(shape, dtype, buffer, offset, strides,
                                       order='C')
            self.ex = num.NPArray(self._ndarray)
        elif ex is not None:
            self.ex = ex
        else:
            # do type inference
            pass

        self.shape = shape
        self.dtype = dtype
        self.parent = parent
        self.ops = ops
        return self

    def __repr__(self):
        return str(self.__array__())

    def child(self, ops):
        return DelayArray(self.shape,  ops=ops)

    def walk(self):
        walker = num.NumpyWalker()
        return walker.walk(self)

    def __array__(self):
        # return NumpyFunction(self.ex)()
        if isinstance(self.ex, num.NPArray):
            return self.ex.array
        return run_gpu(self.ex)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc.__name__ == 'matmul':
            return self._dot(inputs, kwargs)
        # cls = func_to_numpy_ex(ufunc)
        args = [arg_to_numpy_ex(arg) for arg in inputs]
        return DelayArray(self.shape, ops=(ufunc, inputs, kwargs),
                          ex=num.create_ex(*args, ufunc))

    def _dot_mv(self, args, kwargs):
        return DelayArray((args[0].array.shape[0], ),
                          ops=(np.dot, args, kwargs),
                          ex=num.MVEx(args[0], args[1]))

    def _dot_mm(self, args, kwargs):
        return DelayArray((args[0].array.shape[0], args[1].array.shape[1]),
                          ops=(np.dot, args, kwargs),
                          ex=num.MMEx(args[0], args[1]))

    def _dot(self, args, kwargs):
        # scalar result dot
        args = [arg_to_numpy_ex(arg) for arg in args]
        if num.is_matrix_matrix(args[0].shape, args[1].shape):
            return self._dot_mm(args, kwargs)
        if num.is_matrix_vector(args[0].shape, args[1].shape):
            return self._dot_mv(args, kwargs)
        res = np.array(DelayArray(self.shape, ops=(np.dot, args, kwargs),
                                  ex=num.DotEx(args[0], args[1])))
        return np.sum(res)

    def __array_function__(self, func, types, args, kwargs):
        if func.__name__ == "dot":
            return self._dot(args, kwargs)
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def astype(self, *args, **kwargs):
        self._ndarray = self._ndarray.astype(*args, **kwargs)
        if isinstance(self.ex, num.NPArray):
            self.ex = num.NPArray(self._ndarray)
        else:
            raise Exception("Dont call astype here")
        return self

    def dot(self, other, out=None):
        return np.dot(self, other)


def arg_to_numpy_ex(arg: Any) -> num.NumpyEx:
    from numbers import Number
    if isinstance(arg, DelayArray):
        return arg.ex
    elif isinstance(arg, Number):
        return num.Scalar(arg)
    else:
        print(arg)
        print(type(arg))
        raise NotImplementedError


def func_to_numpy_ex(func):
    return {
        'matmul': num.Matmul,
        'add': num.Add,
        'multiply': num.Multiply
        }[func.__name__]


@implements(np.diag)
def diag(arr, k=0):
    if isinstance(arr.ex, num.NPArray):
        arr._ndarray = np.ascontiguousarray(np.diag(arr._ndarray, k))
        assert(arr._ndarray.flags['C_CONTIGUOUS'])
        arr.ex = num.NPArray(arr._ndarray)
        return arr
    else:
        return NotImplemented


@implements(np.diagflat)
@cast
def diagflat(arr, k=0):
    # keep it simple for now
    return np.diagflat(np.asarray(arr, order='C'))


# Ones and zeros
empty = cast(np.empty)
empty_like = cast(np.empty_like)
eye = cast(np.eye)
identity = cast(np.identity)
ones = cast(np.ones)
ones_like = cast(np.ones_like)
zeros = cast(np.zeros)
zeros_like = cast(np.zeros_like)
full = cast(np.full)
full_like = cast(np.full_like)


# From existing data

array = cast(np.array)
asarray = cast(np.asarray)
asanyarray = cast(np.asanyarray)
ascontiguousarray = cast(np.ascontiguousarray)
asmatrix = cast(np.asmatrix)
copy = cast(np.copy)
frombuffer = cast(np.frombuffer)
fromfile = cast(np.fromfile)
fromfunction = cast(np.fromfunction)
fromiter = cast(np.fromiter)
fromstring = cast(np.fromstring)
loadtxt = cast(np.loadtxt)

# Numerical ranges
arange = cast(np.arange)
linspace = cast(np.linspace)
logspace = cast(np.logspace)
geomspace = cast(np.geomspace)


# Building matrices
tri = cast(np.tri)
tril = cast(np.tril)
triu = cast(np.triu)
vander = cast(np.vander)
