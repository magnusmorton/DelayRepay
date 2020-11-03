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
from typing import Tuple, List, Any
import cupy  # type: ignore
import numpy as np  # type: ignore
import numpy.lib.mixins  # type: ignore
from .delayarray import DelayArray


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
    "matmul": cupy.matmul,
    "add": cupy.add,
    "multiply": cupy.multiply,
    "subtract": cupy.subtract,
    "true_divide": cupy.true_divide,
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
