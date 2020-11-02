""" CUDA numpy array with delayed-evaluation semantics

Needs split up

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""
# Copyright (C) 2020 by Univeristy of Edinburgh

from numbers import Number
from typing import Any, List, Dict, Tuple, Optional, Union, Set
import numpy as np  # type: ignore
import numpy.lib.mixins  # type: ignore
import delayrepay.cuda as backend

Shape = Tuple[int, int]


def cast(func):
    """cast to Delay array decorator"""

    def wrapper(*args, **kwargs):
        arr = func(*args, **kwargs)
        if not isinstance(arr, DelayArray):
            arr = NPArray(arr)
        return arr

    return wrapper


class DelayArray(numpy.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, *args, **kwargs):
        self._memo = None

    def __repr__(self):
        return str(self.__array__())

    def __array__(self):
        # return NumpyFunction(self.ex)()
        try:
            return self.array
        except AttributeError:
            self.array = backend.run_gpu(self)
            return self.array

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if len(inputs) > 1:
            left = inputs[0]
            right = inputs[1]
            if not isinstance(left, Number) and not isinstance(right, Number):
                if left.shape != right.shape:
                    if left.shape != (0,) and right.shape != (0,):
                        return backend.ufunc_lookup[ufunc.__name__](
                            left.__array__(), right.__array__()
                        )
        if ufunc.__name__ == "matmul":
            return self._dot(inputs, kwargs)
        # cls = func_to_numpy_ex(ufunc)
        args = [arg_to_numpy_ex(arg) for arg in inputs]
        return create_ex(ufunc, args)

    def _dot_mv(self, args, kwargs):
        return MVEx(args[0], args[1])

    def _dot_mm(self, args, kwargs):
        return MMEx(args[0], args[1])

    @cast
    def _dot(self, args, kwargs):
        # scalar result dot
        args = [arg_to_numpy_ex(arg) for arg in args]
        # if is_matrix_matrix(args[0].shape, args[1].shape):
        #     return self._dot_mm(args, kwargs)
        # if is_matrix_vector(args[0].shape, args[1].shape):
        #     return self._dot_mv(args, kwargs)

        left = args[0].__array__()
        right = args[1].__array__()

        # TODO: independent fallback mechanism
        return np.dot(left, right)

    def __array_function__(self, func, types, args, kwargs):
        if func.__name__ == "dot":
            return self._dot(args, kwargs)
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __gt__(self, other):
        return greater(self, other)

    def __lt__(self, other):
        return less(self, other)

    def dot(self, other, out=None):
        return np.dot(self, other)

    def get(self):
        return self.__array__().get()

    def run(self):
        self.__array__()

    def reshape(self, *args, **kwargs):
        return NPArray(self.__array__().reshape(*args, **kwargs))

    def __setitem__(self, key, item):
        arr = self.__array__()
        if isinstance(key, DelayArray):
            key = key.__array__()
        if isinstance(item, DelayArray):
            item = item.__array__()

        arr[key] = item

    @cast
    def __getitem__(self, key):
        if isinstance(key, DelayArray):
            key = key.__array__()
        arr = self.__array__()
        return arr[key]

    def var(self, *args, **kwargs):
        return np.var(self, *args, **kwargs)

    def sum(self, *args, **kwargs):
        return np.sum(self, *args, **kwargs)

    def __len__(self):
        return self.shape[0]

    @property
    def T(self):
        if len(self.shape) == 1:
            return self
        return np.transpose(self)

    def repeat(self, *args, **kwargs):
        return repeat(self, *args, **kwargs)


HANDLED_FUNCTIONS = {}


def implements(np_function):
    "Register an __array_function__ implementation for DiagonalArray objects."
    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func
    return decorator


@implements(np.diag)
def diag(arr, k=0):
    if isinstance(arr.ex, NPArray):
        arr._ndarray = np.ascontiguousarray(np.diag(arr._ndarray, k))
        assert arr._ndarray.flags["C_CONTIGUOUS"]
        arr.ex = NPArray(arr._ndarray)
        return arr
    else:
        return NotImplemented


@implements(np.diagflat)
@cast
def diagflat(arr, k=0):
    # keep it simple for now
    return np.diagflat(np.asarray(arr, order="C"))


@implements(np.var)
def var(arr, *args, **kwargs):
    return backend.fallback.var(arr.__array__(), *args, **kwargs)


@implements(np.sum)
def sum(arr, *args, **kwargs):
    return backend.fallback.sum(arr.__array__(), *args, **kwargs)


@implements(np.transpose)
@cast
def transpose(arr, *args, **kwargs):
    return backend.fallback.transpose(arr.__array__(), *args, **kwargs)


@implements(np.roll)
@cast
def roll(arr, *args, **kwargs):
    return backend.fallback.roll(arr.__array__(), *args, **kwargs)


@implements(np.max)
def max(arr, *args, **kwargs):
    return backend.fallback.max(arr.__array__(), *args, **kwargs)


@cast
@implements(np.maximum)
def maximum(arr, *args, **kwargs):
    return backend.fallback.maximum(arr.__array__(), *args, **kwargs)


@implements(np.average)
def average(arr, *args, **kwargs):
    return backend.fallback.average(arr.__array__(), *args, **kwargs)


@implements(np.repeat)
@cast
def repeat(arr, *args, **kwargs):
    return backend.fallback.repeat(arr.__array__(), *args, **kwargs)


@cast
@implements(np.cumsum)
def cumsum(arr, *args, **kwargs):
    return backend.fallback.cumsum(arr.__array__(), *args, **kwargs)


@implements(np.greater)
def greater(arr1, arr2, *args, **kwargs):
    return backend.fallback.greater(arr1.__array__(), arr2, *args, **kwargs)


@implements(np.less)
def less(arr1, arr2, *args, **kwargs):
    return backend.fallback.less(arr1.__array__(), arr2, *args, **kwargs)


add = np.add
multiply = np.multiply
dot = np.dot
cos = np.cos
sin = np.sin
tan = np.tan
tanh = np.tanh
sinh = np.sinh
cosh = np.cosh
arctan2 = np.arctan2
subtract = np.subtract
exp = np.exp
log = np.log
power = np.power
sqrt = np.sqrt
square = np.square
abs = np.abs
newaxis = backend.fallback.newaxis

# dtypes etc.
double = np.double
float32 = np.float32
uint32 = np.uint32

# Ones and zeros
empty = cast(backend.fallback.empty)
empty_like = cast(backend.fallback.empty_like)
eye = cast(backend.fallback.eye)
identity = cast(backend.fallback.identity)
ones = cast(backend.fallback.ones)
ones_like = cast(backend.fallback.ones_like)
zeros = cast(backend.fallback.zeros)
zeros_like = cast(backend.fallback.zeros_like)
full = cast(backend.fallback.full)
full_like = cast(backend.fallback.full_like)


@implements(np.tile)
@cast
def tile(arr, *args, **kwargs):

    if isinstance(arr, DelayArray):
        temp = np.array(arr.__array__().get())
        print(type(temp))
    return backend.fallback.tile(temp, *args, **kwargs)


# From existing data

array = cast(backend.fallback.array)
asarray = cast(backend.fallback.asarray)
asanyarray = cast(backend.fallback.asanyarray)
ascontiguousarray = cast(backend.fallback.ascontiguousarray)
asmatrix = cast(np.asmatrix)
copy = cast(backend.fallback.copy)
frombuffer = cast(np.frombuffer)
fromfile = cast(np.fromfile)
fromfunction = cast(np.fromfunction)
fromiter = cast(np.fromiter)
fromstring = cast(np.fromstring)
loadtxt = cast(np.loadtxt)

# Numerical ranges
arange = cast(backend.fallback.arange)
linspace = cast(backend.fallback.linspace)
logspace = cast(backend.fallback.logspace)
geomspace = cast(np.geomspace)


# Building matrices
tri = cast(backend.fallback.tri)
tril = cast(backend.fallback.tril)
triu = cast(backend.fallback.triu)
vander = cast(np.vander)
