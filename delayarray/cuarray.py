'''Delay array and related stuff'''

from typing import Any, List, Dict, Tuple, Optional
import cupy  # type: ignore
import numpy as np  # type: ignore
import numpy.lib.mixins  # type: ignore



OPS = {
    'matmul': '@',
    'add': '+',
    'multiply': '*',
    'subtract': '-',
    'true_divide': '/',
}

FUNCS = {
    'power': 'pow'
}


class DelayArray(numpy.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, *args, **kwargs):
        self._memo = None

    def __repr__(self):
        return str(self.__array__())

    def __array__(self):
        # return NumpyFunction(self.ex)()
        if isinstance(self, NPArray):
            return self.array
        return run_gpu(self)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc.__name__ == 'matmul':
            return self._dot(inputs, kwargs)
        # cls = func_to_numpy_ex(ufunc)
        args = [arg_to_numpy_ex(arg) for arg in inputs]
        return create_ex(ufunc, args)

    def _dot_mv(self, args, kwargs):
        return MVEx(args[0], args[1])

    def _dot_mm(self, args, kwargs):
        return MMEx(args[0], args[1])

    def _dot(self, args, kwargs):
        # scalar result dot
        args = [arg_to_numpy_ex(arg) for arg in args]
        if is_matrix_matrix(args[0].shape, args[1].shape):
            return self._dot_mm(args, kwargs)
        if is_matrix_vector(args[0].shape, args[1].shape):
            return self._dot_mv(args, kwargs)
        res = np.array(DelayArray(self.shape, ops=(np.dot, args, kwargs),
                                  ex=DotEx(args[0], args[1])))
        return np.sum(res)

    def __array_function__(self, func, types, args, kwargs):
        if func.__name__ == "dot":
            return self._dot(args, kwargs)
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def astype(self, *args, **kwargs):
        if isinstance(self, NPArray):
            self.array = self.array.astype(*args, **kwargs)
        else:
            raise Exception("Dont call astype here")
        return self

    def dot(self, other, out=None):
        return np.dot(self, other)

    def get(self):
        return self.__array__().get()


def calc_shape(left, right, op=None):
    if left == (0,):
        return right
    if right is (0,):
        return left
    if op.__name__ in OPS:
        return left
    if op.__name__ == 'dot':
        # for now
        if len(left) > 1 and len(right) > 1:
            return (left[0], right[1])
        elif len(left) > 1:
            return (left[0],)
        else:
            return (0,)
    else:
        return left


class NumpyEx(DelayArray):
    '''Numpy expression'''
    
    @property
    def name(self):
        return f"var{id(self)}"


class Funcable:
    def to_op(self):
        return OPS[self.func.__name__]


class ReduceEx(NumpyEx, Funcable):
    def __init__(self, func, arg):
        super().__init__()
        self.func = func
        self.arg = arg

    # func: np.ufunc
    # arg: NumpyEx


class UnaryFuncEx(NumpyEx, Funcable):

    def __init__(self, func, arg):
        super().__init__()
        self.arg = arg
        self.func = func
        self.shape = arg.shape


class BinaryFuncEx(NumpyEx):

    def __init__(self, func, left, right):
        super().__init__()
        self.left = left
        self.right = right
        self.func = func
        self.shape = calc_shape(left.shape, right.shape, func)

    def to_op(self):
        return FUNCS[self.func.__name__]


def create_ex(func, args):
    if func.__name__ in OPS:
        return BinaryNumpyEx(func, *args)
    if len(args) == 1:
        return UnaryFuncEx(func, *args)
    return BinaryFuncEx(func, *args)


class BinaryNumpyEx(NumpyEx, Funcable):
    '''Binary numpy expression'''
    # left: NumpyEx
    # right: NumpyEx
    # func: np.ufunc

    def __init__(self, func, left, right):
        super().__init__()
        self.left = left
        self.right = right
        self.func = func
        self.shape = calc_shape(left.shape, right.shape, func)


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


class MemoMeta(type):
    '''Metaclass implementing caching'''

    def __new__(meta, *args, **kwargs):
        cls = super(MemoMeta, meta).__new__(meta, *args, **kwargs)
        cls._cache = {}
        return cls

    def __call__(cls, array):
        if id(array) not in cls._cache:
            cls._cache[id(array)] = super(MemoMeta, cls).__call__(array)
        return cls._cache[id(array)]


class NPArray(NumpyEx, DelayArray, metaclass=MemoMeta):
    '''ndarray'''

    def __init__(self, array):
        super().__init__()
        self.array = array
        self.shape = array.shape

    def __hash__(self):
        return id(self.array)

    def __eq__(self, other):
        return self.array is other.array


class Scalar(NumpyEx):
    '''a scalar'''
    # val: Number
    def __init__(self, val):
        super().__init__()
        self.val = val
        self.shape = (0,)


class Visitor:
    '''Visitor ABC'''
    def visit(self, node, **kwargs):
        """Visit a node."""
        if isinstance(node, list):
            visitor = self.list_visit
        else:
            method = 'visit_' + node.__class__.__name__
            visitor = getattr(self, method, self.default_visit)
        return visitor(node, **kwargs)

    def list_visit(self, lst):
        return [self.visit(node) for node in lst]

    def default_visit(self, node):
        return node


class NumpyVisitor(Visitor):
    '''Visits Numpy Expression'''
    def __init__(self):
        self.visits = 0

    def visit(self, node, **kwargs):
        """Visit a node."""
        self.visits += 1
        return super(NumpyVisitor, self).visit(node, **kwargs)

    def visit_BinaryExpression(self, node):
        return node

    def walk(self, tree):
        ''' top-level walk of tree'''
        self.visits = 0
        return self.visit(tree)


def is_matrix_matrix(left, right):
    return len(left) > 1 and len(right) > 1


def is_matrix_vector(left, right):
    return len(left) > 1 and len(right) == 1


class ShapeAnnotator(NumpyVisitor):

    def calc_shape(left, right, op=None):
        if left == (0,):
            return right
        if right is (0,):
            return left
        if op.__name__ in OPS:
            return left
        if op.__name__ == 'dot':
            # for now
            if len(left) > 1 and len(right) > 1:
                return (left[0], right[1])
            elif len(left) > 1:
                return (left[0],)
            else:
                return (0,)

    def visit_BinaryNumpyEx(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        node.shape = ShapeAnnotator.calc_shape(left.shape,
                                               right.shape,
                                               node.func)
        return node

    def visit_NPArray(self, node):
        node.shape = node.array.shape
        return node

    def visit_Scalar(self, node):
        node.shape = (0,)
        return node

    def visit_DotEx(self, node):
        left = self.visit(node.arg1)
        right = self.visit(node.arg2)
        node.shape = ShapeAnnotator.calc_shape(left.shape, right.shape, np.dot)
        node._inshape = left.shape
        return node


class ReduceTransformer(NumpyVisitor):
    def visit_DotEx(self, node):
        # TODO This is just for vector x vector
        left = self.visit(node.arg1)
        right = self.visit(node.arg2)

        if is_matrix_vector(left.shape, right.shape):
            return MVEx(left, right, node.shape, node._inshape)
        else:
            muls = BinaryNumpyEx(np.multiply, left, right)
            muls.shape = node._inshape
            red = ReduceEx(np.add, muls)
            red.shape = node._inshape
            red._inshape = node._inshape
            return red


def cast(func):
    '''cast to Delay array decorator'''
    def wrapper(*args, **kwargs):
        arr = func(*args, **kwargs)
        if not isinstance(arr, DelayArray):
            arr = NPArray(arr)
        return arr
    return wrapper


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


def arg_to_numpy_ex(arg: Any) -> NumpyEx:
    from numbers import Number
    if isinstance(arg, DelayArray):
        return arg
    elif isinstance(arg, Number):
        return Scalar(arg)
    else:
        print(arg)
        print(type(arg))
        raise NotImplementedError


# def func_to_numpy_ex(func):
#     return {
#         'matmul': Matmul,
#         'add': Add,
#         'multiply': Multiply
#         }[func.__name__]


@implements(np.diag)
def diag(arr, k=0):
    if isinstance(arr.ex, NPArray):
        arr._ndarray = np.ascontiguousarray(np.diag(arr._ndarray, k))
        assert(arr._ndarray.flags['C_CONTIGUOUS'])
        arr.ex = NPArray(arr._ndarray)
        return arr
    else:
        return NotImplemented


@implements(np.diagflat)
@cast
def diagflat(arr, k=0):
    # keep it simple for now
    return np.diagflat(np.asarray(arr, order='C'))


add = cast(np.add)
dot = cast(np.dot)
cos = cast(np.cos)
sin = cast(np.sin)
tan = cast(np.tan)
subtract = cast(np.subtract)
exp = cast(np.exp)
power = cast(np.power)

# Ones and zeros
empty = cast(np.empty)
empty_like = cast(np.empty_like)
eye = cast(np.eye)
identity = cast(np.identity)
ones = cast(cupy.ones)
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

InputDict = Dict[str, 'BaseFragment']

class BaseFragment:
    def __init__(self):
        self.name = None
        self.stmts = []
        self._expr = None
        self._inputs = {}
        
    @property
    def inputs(self) -> InputDict:
        return self._inputs

    @property
    def kernel_args(self) -> InputDict:
        return self._inputs





class Fragment(BaseFragment):

    def __init__(self,
                 name: str,
                 expr: str,
                 inputs: InputDict):
        self.name = name
        self._expr = expr
        self._inputs = inputs
        self.dtype = np.float32

    def ref(self) -> str:
        return self.name

    def expr(self) -> str:
        return self._expr

    def to_input(self):
        return {self.name: self.node.array}

    def to_kern(self) -> cupy.ElementwiseKernel:
        inargs = [f"float32 {arg}" for arg in self.kernel_args]
        kern = cupy.ElementwiseKernel(
            ",".join(inargs),
            f"float32 {self.name}",
            f"{self.name} = {self._expr}"
        )
        return kern


class Kernel(Fragment):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    @property
    def inputs(self) -> InputDict:
        return {self.ref(): self}

    def expr(self) -> str:
        return self.name


class InputFragment(BaseFragment):

    def __init__(self, arr: NPArray):
        super().__init__()
        self.name = arr.name
        self._inputs = {self.name: arr.array}

    def ref(self) -> str:
        return f"{self.name}"

    def expr(self) -> str:
        return f"{self.name}"


class ScalarFragment(BaseFragment):
    def __init__(self, val: Scalar):
        super().__init__()
        self.val = val.val

    def ref(self) -> str:
        return str(self.val)

    def expr(self) -> str:
        suffix = ""
        if isinstance(self.val, float):
            suffix = 'f'
        return f"{self.val}{suffix}"


def combine_inputs(*args: InputDict) -> InputDict:
    ret = {}
    for arg in args:
        ret.update(arg)
    return ret


class CupyEmitter(Visitor):

    def __init__(self):
        super().__init__()
        self.ins = {}
        self.outs = []
        self.kernels = []

    def visit_BinaryNumpyEx(self,
                            node: BinaryNumpyEx,
                            callshape: Tuple[int, int] = None) -> BaseFragment:
        op = node.to_op()
        left = self.visit(node.left, callshape=node.shape)
        right = self.visit(node.right, callshape=node.shape)
        name = f"var{id(node)}"
        expr = f"{left.expr()} {op} {right.expr()}"
        
        # stmts = left.stmts + right.stmts + [stmt]
        if callshape is None or callshape != node.shape:
            kern: Fragment = Kernel(name,
                                    expr,
                                    combine_inputs(left.inputs, right.inputs))
            self.kernels.append(kern)
        else:
            kern = Fragment(name,
                            expr,
                            combine_inputs(left.inputs, right.inputs))
        return kern

    def visit_UnaryFuncEx(self,
                          node: UnaryFuncEx,
                          callshape: Tuple[int, int] = None) -> BaseFragment:
        inner = self.visit(node.arg)
        name = f"var{id(node)}"
        expr = f"{node.func.__name__}({inner.expr()})"
        if callshape is None or callshape != node.shape:
            kern: Fragment = Kernel(name,
                                    expr,
                                    inner.inputs)
            self.kernels.append(kern)
        else:
            kern = Fragment(name,
                            expr,
                            inner.inputs)
        return kern

    def visit_BinaryFuncEx(self,
                           node: BinaryFuncEx,
                           callshape: Tuple[int, int] = None) -> BaseFragment:
        op = node.to_op()
        left = self.visit(node.left)
        right = self.visit(node.right)
        name = f"var{id(node)}"
        # TODO: sort out the float literal thing
        expr = f"{op}({left.expr()}, {right.expr()})"

        if callshape is None or callshape != node.shape:
            kern: Fragment = Kernel(name,
                                    expr,
                                    combine_inputs(left.inputs, right.inputs))
            self.kernels.append(kern)
        else:
            kern = Fragment(name,
                            expr,
                            combine_inputs(left.inputs, right.inputs))
        return kern

    def visit_NPArray(self,
                      node: NPArray,
                      callshape: Tuple[int, int] = None) -> BaseFragment:
        return InputFragment(node)

    def visit_Scalar(self,
                     node: Scalar,
                     callshape: Tuple[int, int] = None) -> BaseFragment:
        return ScalarFragment(node)


def run_gpu(ex: NumpyEx) -> cupy.array:
    visitor = CupyEmitter()
    visitor.visit(ex)
    kerns = visitor.kernels
    assert(len(kerns))
    results: Dict[str, cupy.array] = {}
    for kern in kerns:
        compiled = kern.to_kern()
        inputs = [results[key] if isinstance(value, Kernel) else value for key, value in kern.kernel_args.items()]
                
        ret = compiled(*inputs)
        results[kern.ref()] = ret
        return ret
