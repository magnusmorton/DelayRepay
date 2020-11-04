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
from typing import Any, List, Tuple
import numpy as np  # type: ignore
import numpy.lib.mixins  # type: ignore
import delayrepay.cuda as backend


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
                        return ufunc_lookup[ufunc.__name__](
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
        return backend.fallback.dot(left, right)

    def __array_function__(self, func, types, args, kwargs):
        if func.__name__ == "dot":
            return self._dot(args, kwargs)
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __gt__(self, other):
        return greater(self, other)

    def __lt__(self, other):
        return less(self, other)

    def dot(self, other, out=None):
        return self._dot(other, out)

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


Shape = Tuple[int, int]


OPS = {
    "matmul": "@",
    "add": "+",
    "multiply": "*",
    "subtract": "-",
    "true_divide": "/",
}


FUNCS = {
    "power": "pow",
    "arctan2": "atan2",
    "absolute": "abs",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "sqrt": "sqrt",
    "log": "log",
    # HACK
    "negative": "-",
    "exp": "exp",
    "tanh": "tanh",
    "sinh": "sinh",
    "cosh": "cosh",
}

ufunc_lookup = {
    "matmul": backend.np.matmul,
    "add": backend.np.add,
    "multiply": backend.np.cupy.multiply,
    "subtract": backend.np.cupy.subtract,
    "true_divide": backend.np.true_divide,
}


def calc_shape(left, right, op=None):
    if left == (0,):
        return right
    if right == (0,):
        return left
    if op.__name__ in OPS:
        return left
    if op.__name__ == "dot":
        # for now
        if len(left) > 1 and len(right) > 1:
            return (left[0], right[1])
        elif len(left) > 1:
            return (left[0],)
        else:
            return (0,)
    else:
        return left


class Memoiser(type):
    """Metaclass implementing caching"""

    def __new__(meta, *args, **kwargs):
        cls = super(Memoiser, meta).__new__(meta, *args, **kwargs)
        meta._cache = {}
        return cls

    def __call__(cls, *args):
        if type(args[0]).__name__ == "ndarray":
            key = id(args[0])
        else:
            key = hash(args)
        if key not in cls._cache:
            Memoiser._cache[key] = super(Memoiser, cls).__call__(*args)
        return cls._cache[key]


def reset():
    # hacks
    Memoiser._cache.clear()


class NumpyEx(DelayArray, metaclass=Memoiser):
    children: List["NumpyEx"]
    """Numpy expression"""

    def __init__(self, children: List["NumpyEx"] = []):
        super().__init__()
        self.dtype = None
        self.children = children

    def __hash__(self):
        """
        Should work because of the Memoizer
        """
        return id(self)


class Funcable:
    def to_op(self):
        return OPS[self.func.__name__]


class ReduceEx(NumpyEx, Funcable):
    def __init__(self, func, arg):
        super().__init__(children=[arg])
        self.func = func
        self.shape = (0,)

    # func: np.ufunc
    # arg: NumpyEx


class UnaryFuncEx(NumpyEx, Funcable):
    def __init__(self, func, arg):
        super().__init__(children=[arg])
        self.func = func
        self.shape = arg.shape
        self.dtype = arg.dtype

    def to_op(self):
        return FUNCS[self.func.__name__]


class BinaryFuncEx(NumpyEx):
    def __init__(self, func, left, right):
        super().__init__(children=[left, right])
        self.func = func
        self.shape = calc_shape(left.shape, right.shape, func)
        self.dtype = calc_type(left, right)

    def to_op(self):
        return FUNCS[self.func.__name__]


def pow_ex(func, left, right):
    if not isinstance(right.val, int):
        return BinaryFuncEx(func, left, right)
    ex = left
    for i in range(right.val - 1):
        # will give odd expression tree, but OK
        ex = BinaryNumpyEx(np.multiply, ex, left)

    return ex


def create_ex(func, args):
    if func.__name__ in OPS:
        return BinaryNumpyEx(func, *args)
    if func.__name__ == "square":
        return BinaryNumpyEx(np.multiply, args[0], args[0])
    if len(args) == 1:
        return UnaryFuncEx(func, *args)
    if func.__name__ == "power":
        return pow_ex(func, *args)
    return BinaryFuncEx(func, *args)


class BinaryNumpyEx(NumpyEx, Funcable):
    """Binary numpy expression"""

    def __init__(self, func, left, right):
        super().__init__(children=[left, right])
        self.func = func
        self.shape = calc_shape(left.shape, right.shape, func)
        self.dtype = calc_type(left, right)


class MMEx(NumpyEx, Funcable):
    # arg1: NumpyEx
    # arg2: NumpyEx
    def __init__(self, arg1, arg2):
        super().__init__()
        self.arg1 = arg1
        self.arg2 = arg2
        self.shape = calc_shape(arg1.shape, arg2.shape, np.dot)


class MVEx(NumpyEx, Funcable):
    # arg1: NumpyEx
    # arg2: NumpyEx
    def __init__(self, arg1, arg2):
        super().__init__()
        self.arg1 = arg1
        self.arg2 = arg2
        self.shape = calc_shape(arg1.shape, arg2.shape, np.dot)


class DotEx(NumpyEx, Funcable):
    def __init__(self, left, right):
        super().__init__()
        self.arg1 = left
        self.arg2 = right
        self.shape = calc_shape(left.shape, right.shape, np.dot)
        self._inshape = left.shape


class NPArray(NumpyEx):
    """ndarray"""

    def __init__(self, array):
        super().__init__()
        self.array = array
        self.shape = array.shape
        self.dtype = array.dtype

    def __hash__(self):
        return id(self.array)

    def __eq__(self, other):
        try:
            return self.array is other.array
        except AttributeError:
            return False

    def astype(self, *args, **kwargs):
        old = self.array
        cast_arr = self.array.astype(*args, **kwargs)
        del NPArray._cache[id(old)]
        NPArray._cache[id(cast_arr)] = self
        self.array = cast_arr
        self.dtype = cast_arr.dtype
        return self


class NPRef(NumpyEx):
    """Only for when breaking dependency chains for fusion"""

    def __init__(self, node: NumpyEx, shape: Shape):
        super().__init__()
        self.ref = node
        self.children = []
        self.shape = shape

    @property
    def array(self):
        return self.ref.array


class Scalar(NumpyEx):
    """a scalar"""

    # val: Number
    def __init__(self, val):
        super().__init__()
        self.val = val
        self.shape = (0,)

    def __hash__(self):
        return hash(self.val)


class Visitor:
    """Visitor ABC"""

    def visit(self, node) -> Any:
        """Visit a node."""
        if isinstance(node, list):
            visitor = self.list_visit
        else:
            method = "visit_" + node.__class__.__name__
            visitor = getattr(self, method, self.default_visit)
        return visitor(node)

    def list_visit(self, lst, **kwargs):
        return [self.visit(node) for node in lst]

    def default_visit(self, node):
        return node


class NumpyVisitor(Visitor):
    """Visits Numpy Expression"""

    def __init__(self):
        self.visits = 0

    def visit(self, node):
        """Visit a node."""
        self.visits += 1
        return super(NumpyVisitor, self).visit(node)

    def visit_BinaryExpression(self, node):
        return node

    def walk(self, tree):
        """ top-level walk of tree"""
        self.visits = 0
        return self.visit(tree)


def is_matrix_matrix(left, right):
    return len(left) > 1 and len(right) > 1


def is_matrix_vector(left, right):
    return len(left) > 1 and len(right) == 1


def calc_type(node1: NumpyEx, node2: NumpyEx) -> np.dtype:
    if node1.dtype is not None:
        node2.dtype = node1.dtype
        return node1.dtype
    node1.dtype = node2.dtype
    return node2.dtype


def arg_to_numpy_ex(arg: Any) -> NumpyEx:
    import cupy
    if isinstance(arg, DelayArray):
        return arg
    elif isinstance(arg, Number):
        return Scalar(arg)
    elif isinstance(arg, cupy.core.core.ndarray) or isinstance(arg, np.ndarray):
        return NPArray(arg)
    else:
        print(type(arg))
        raise NotImplementedError


class PrettyPrinter(Visitor):
    def visit(self, node):
        if isinstance(node, list):
            return self.list_visit(node)
        print(type(node).__name__)
        self.visit(node.children)


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
