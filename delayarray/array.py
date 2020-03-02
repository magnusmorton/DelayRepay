'''Delay array and related stuff'''

import logging
from dataclasses import dataclass
from typing import List, Any
import numpy as np
import numpy.lib.mixins
from .num import *
from .cl import run_gpu

def cast(func):
    '''cast to Delay array decorator'''
    def wrapper(*args, **kwargs):
        arr = func(*args, **kwargs)
        if not isinstance(arr,DelayArray):
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
    if 'float64' in (type1,type2):
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
            self.ex = NPArray(buffer)
        elif ops is None:
            self._ndarray = np.ndarray(shape, dtype, buffer, offset, strides, order='C')
            self.ex = NPArray(self._ndarray)
        elif ex is not None:
            self.ex = ex
        else:
            # do type inference
            pass
       
        self.shape = shape
        self.dtype = dtype
        self.parent = parent
        self.ops = ops
        self._logger = logging.getLogger("delayRepay.arr")
        return self

    def __repr__(self):
        return str(self.__array__())


    def child(self, ops):
        return DelayArray(self.shape,  ops=ops)
    
    def walk(self):
        walker = NumpyWalker()
        return walker.walk(self)

    def __array__(self):
        # return NumpyFunction(self.ex)()
        if isinstance(self.ex, NPArray):
            return self.ex.array
        return run_gpu(self.ex)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        self._logger.debug("func: {}".format(ufunc))
        self._logger.debug(method)
        self._logger.debug(inputs)
        print("not dot")
        if ufunc.__name__ == 'multiply':
            self._logger.debug("FOO")
        if ufunc.__name__ == 'matmul':
            return self._dot(inputs, kwargs)
        # cls = func_to_numpy_ex(ufunc)
        args = [arg_to_numpy_ex(arg) for arg in inputs]
        return DelayArray(self.shape, ops=(ufunc, inputs, kwargs), ex=BinaryNumpyEx(args[0], args[1], ufunc))

    def _dot(self, args, kwargs):
        args = [arg_to_numpy_ex(arg) for arg in args]
        res = np.array(DelayArray(self.shape, ops=(np.dot, args, kwargs), ex=DotEx(args[0], args[1])))
        return np.sum(res)


    def __array_function__(self, func, types, args, kwargs):
        print(func)
        self._logger.debug("array_function")
        self._logger.debug("func: {}".format(func))
        self._logger.debug("args: {}".format(type(args)))
        if func.__name__ == "dot":
            return self._dot(args, kwargs)
        elif func in HANDLED_FUNCTIONS:
            return HANDLED_FUNCTIONS[func](*args, **kwargs)
        else:
            return NotImplemented

    def astype(self, *args, **kwargs):
        self._ndarray = self._ndarray.astype(*args, **kwargs)
        if isinstance(self.ex, NPArray):
            self.ex = NPArray(self._ndarray)
        else:
            raise Exception("Dont call astype here")
        return self

    def dot(self, other, out=None):
        return np.dot(self, other)


def arg_to_numpy_ex(arg:Any) -> NumpyEx:
    from numbers import Number
    if isinstance(arg, DelayArray):
        return arg.ex
    elif isinstance(arg, Number):
        return Scalar(arg)
    else:
        print(arg)
        print(type(arg))
        raise NotImplementedError


def func_to_numpy_ex(func):
    return {
        'matmul': Matmul,
        'add': Add,
        'multiply': Multiply
        }[func.__name__]

@implements(np.diag)
def diag(arr, k=0):
    if isinstance(arr.ex, NPArray):
        arr._ndarray = np.diag(arr._ndarray, k)
        arr.ex = NPArray(arr._ndarray)
        return arr
    else:
        return NotImplemented

@implements(np.diagflat)
@cast
def diagflat(arr, k=0):
    #keep it simple for now
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
