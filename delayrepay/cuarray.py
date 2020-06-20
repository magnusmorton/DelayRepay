'''Delay array and related stuff'''

from typing import Any, List, Dict, Tuple, Optional, Union
import cupy  # type: ignore
import numpy as np  # type: ignore
import numpy.lib.mixins  # type: ignore

Shape = Tuple[int,int]

OPS = {
    'matmul': '@',
    'add': '+',
    'multiply': '*',
    'subtract': '-',
    'true_divide': '/',
}

FUNCS = {
    'power': 'pow',
    'arctan2': 'atan2',
    'absolute': 'abs',
    'sin':'sin',
    'cos':'cos',
    'tan':'tan',
    'sqrt':'sqrt'
}


def cast(func):
    '''cast to Delay array decorator'''
    def wrapper(*args, **kwargs):
        arr = func(*args, **kwargs)
        if not isinstance(arr, DelayArray):
            arr = NPArray(arr)
        return arr
    return wrapper

class DelayArray(numpy.lib.mixins.NDArrayOperatorsMixin):
    _id: int = 0
    
    def __init__(self, *args, **kwargs):
        self._memo = None
        self._id = DelayArray._id
        DelayArray._id += 1

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
        # if is_matrix_matrix(args[0].shape, args[1].shape):
        #     return self._dot_mm(args, kwargs)
        # if is_matrix_vector(args[0].shape, args[1].shape):
        #     return self._dot_mv(args, kwargs)

        left = args[0].__array__()
        right = args[1].__array__()
        return cupy.dot(left, right)

    def __array_function__(self, func, types, args, kwargs):
        if func.__name__ == "dot":
            return self._dot(args, kwargs)
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

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
        if isinstance(item, DelayArray):
            item = item.__array__()
        arr[key] = item
        return NPArray(arr)

    @cast
    def __getitem__(self, key):
        return self.__array__()[key]

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


class Memoiser(type):
    '''Metaclass implementing caching'''

    def __new__(meta, *args, **kwargs):
        cls = super(Memoiser, meta).__new__(meta, *args, **kwargs)
        cls._cache = {}
        return cls

    def __call__(cls, *args):
        if type(args[0]).__name__ == "ndarray":
            key = id(args[0])
        else:
            key = hash(args)
        if key not in cls._cache:
            cls._cache[key] = super(Memoiser, cls).__call__(*args)
        return cls._cache[key]


class NumpyEx(DelayArray, metaclass=Memoiser):
    children : List['NumpyEx']
    '''Numpy expression'''
    def __init__(self, children: List['NumpyEx']=[]):
        super().__init__()
        self.dtype = None
        self.children = children
    
    @property
    def name(self):
        return f"var{self._id}"

    def __hash__(self):
        '''
        Should work because of the Memoizer
        '''
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
        super().__init__(children=[left,right])
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
        ex = BinaryNumpyEx(multiply, ex, left)

    return ex



def create_ex(func, args):
    if func.__name__ in OPS:
        return BinaryNumpyEx(func, *args)
    if func.__name__ == 'square':
        return BinaryNumpyEx(multiply, args[0], args[0])
    if len(args) == 1:
        return UnaryFuncEx(func, *args)
    if func.__name__ == 'power':
        return pow_ex(func, *args)
    return BinaryFuncEx(func, *args)


class BinaryNumpyEx(NumpyEx, Funcable):
    '''Binary numpy expression'''

    def __init__(self, func, left, right):
        super().__init__(children=[left,right])
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




class NPArray(NumpyEx, DelayArray):
    '''ndarray'''

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
        del(NPArray._cache[id(old)])
        NPArray._cache[id(cast_arr)] = self
        self.array = cast_arr
        self.dtype = cast_arr.dtype
        return self

class NPRef(NumpyEx, DelayArray):
    '''Only for when breaking dependency chains for fusion'''
    
    name:str
    ref:NumpyEx

    def __init__(self, node:NumpyEx):
        name = node.name
        ref = node

class Scalar(NumpyEx):
    '''a scalar'''
    # val: Number
    def __init__(self, val):
        super().__init__()
        self.val = val
        self.shape = (0,)

    def __hash__(self):
        return hash(self.val)


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




# def calc_type(func, type1, type2):
#     if 'float64' in (type1, type2):
#         return 'float64'
#     elif 'float32' in (type1, type2):
#         return 'float32'
#     elif 'int64' in (type1, type2):
#         return 'int64'
#     else:
#         return type1

def calc_type(node1: NumpyEx, node2: NumpyEx) -> np.dtype:
    if node1.dtype is not None:
        node2.dtype = node1.dtype
        return node1.dtype
    node1.dtype = node2.dtype
    return node2.dtype


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


#@implements(np.sum)
#def sum(arr, axis=None, dtype=None, out=None, keepdims=None, initial=None, where=None):
#    print("BLAH")
#    return ReduceEx(np.add, arr)

@implements(np.var)
def var(arr, *args, **kwargs):
    return cupy.var(arr.__array__(), *args, **kwargs)

@implements(np.sum)
def sum(arr, *args, **kwargs):
    return cupy.sum(arr.__array__(), *args, **kwargs)

@implements(np.transpose)
@cast
def transpose(arr, *args, **kwargs):
    return cupy.transpose(arr.__array__(), *args, **kwargs)


@implements(np.roll)
@cast
def roll(arr, *args, **kwargs):
    return cupy.roll(arr.__array__(), *args, **kwargs)

#sum = cast(cupy.sum)
add = np.add
multiply = np.multiply
dot = np.dot
cos = np.cos
sin = np.sin
tan = np.tan
arctan2 = np.arctan2
subtract = np.subtract
exp = np.exp
power = np.power
sqrt = np.sqrt
square = np.square
abs = np.abs
newaxis = cupy.newaxis

#dtypes etc.
double = np.double
float32 = np.float32

# Ones and zeros
empty = cast(cupy.empty)
empty_like = cast(cupy.empty_like)
eye = cast(cupy.eye)
identity = cast(cupy.identity)
ones = cast(cupy.ones)
ones_like = cast(cupy.ones_like)
zeros = cast(cupy.zeros)
zeros_like = cast(cupy.zeros_like)
full = cast(cupy.full)
full_like = cast(cupy.full_like)


# From existing data

array = cast(cupy.array)
asarray = cast(cupy.asarray)
asanyarray = cast(cupy.asanyarray)
ascontiguousarray = cast(cupy.ascontiguousarray)
asmatrix = cast(np.asmatrix)
copy = cast(cupy.copy)
frombuffer = cast(np.frombuffer)
fromfile = cast(np.fromfile)
fromfunction = cast(np.fromfunction)
fromiter = cast(np.fromiter)
fromstring = cast(np.fromstring)
loadtxt = cast(np.loadtxt)

# Numerical ranges
arange = cast(cupy.arange)
linspace = cast(cupy.linspace)
logspace = cast(cupy.logspace)
geomspace = cast(np.geomspace)


# Building matrices
tri = cast(cupy.tri)
tril = cast(cupy.tril)
triu = cast(cupy.triu)
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


def dedup(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


class Fragment(BaseFragment):

    def __init__(self,
                 name: str,
                 stmts: List[str],
                 inputs: InputDict) -> None:
        self.name = name
        self.stmts = stmts
        self._inputs = inputs
        #self.dtype = np.float32
        

    def ref(self) -> str:
        return self.name

    # def expr(self) -> str:
    #     return self._expr

    def to_input(self):
        return {self.name: self.node.array}

    def to_kern(self) -> cupy.ElementwiseKernel:
        body = ";\n".join(dedup(self.stmts))
        inargs = [f"T {arg}" for arg in self.kernel_args]
        kern = cupy.ElementwiseKernel(
            ",".join(inargs),
            f"T out",
            f"{body};\nout = {self.name}"
        )
        return kern



class InputFragment(BaseFragment):

    def __init__(self, arr: Union[NPArray, NPRef]) -> None:
        super().__init__()
        self.name = arr.name
        self._inputs = {self.name: arr}

    def ref(self) -> str:
        return f"{self.name}"

    def expr(self) -> str:
        return f"{self.name}"


# dtype_map = {np.dtype("float32"): "float",
#              np.dtype("float64"): "double",
#              np.dtype("int32"): "int",
#              np.dtype("int64"): "long"}

# dtype_map = {np.dtype("float32"): "f",
#              np.dtype("float64"): "",
#              np.dtype("int32"): "",
#              np.dtype("int64"): ""}

class ScalarFragment(BaseFragment):
    def __init__(self, val: Scalar) -> None:
        super().__init__()
        self.val = val.val
        self.dtype = val.dtype

    def ref(self) -> str:
        return str(self.val)

    def expr(self) -> str:
        return str(self.val)


class ReductionKernel(Fragment):

    def to_kern(self):
        kern = cupy.ReductionKernel(','.join(inargs),
                                    'T out',
                                    self.expr,
                                    self.redex,
                                    'out = a',
                                    0,
                                    self.name)
        return kern





def combine_inputs(*args: InputDict) -> InputDict:
    ret = {}
    for arg in args:
        ret.update(arg)
    return ret

class Fuser(Visitor):
    def __init__(self):
        super().__init__()
        self.seen = {}
    
    def fuse(self, node):
        
        self.splits = [node]
        self.visit(node)
        return self.splits

    def visit(self, node) -> Shape:
        if isinstance(node, list):
            return self.list_visit(node)
        child_shapes = self.list_visit(node.children)
        new = []
        for child, shape in zip(node.children, child_shapes):
            if shape != node.shape and shape != (0,):
                new.append(NPRef(child))
                self.splits.append(child)

            else:
                new.append(child)

        node.children = new

        return node.shape

class CupyEmitter(Visitor):

    def __init__(self):
        super().__init__()
        self.ins = {}
        self.outs = []
        self.kernels = []
        self.seen = {}

    def visit(self, node):
        if node in self.seen:
            visited = self.seen[node]
        else:
            visited = super().visit(node)
            self.seen[node] = visited
        return visited

    def visit_BinaryNumpyEx(self,
                            node: BinaryNumpyEx) -> BaseFragment:
        op = node.to_op()
        left = self.visit(node.children[0])
        right = self.visit(node.children[1])
        name = node.name
        decl = f"T {name} = {left.ref()} {op} {right.ref()}"
        stmts = left.stmts + right.stmts + [decl]
        return Fragment(name, stmts, combine_inputs(left.inputs, right.inputs))

    def visit_UnaryFuncEx(self,
                          node: UnaryFuncEx) -> BaseFragment:
        inner = self.visit(node.children[0])
        name = node.name
        decls = inner.stmts + [f"T {name} = {node.to_op()}({inner.ref()})"]
        return Fragment(name, decls, inner.inputs)

    def visit_BinaryFuncEx(self,
                           node: BinaryFuncEx) -> BaseFragment:
        op = node.to_op()
        left = self.visit(node.children[0])
        right = self.visit(node.children[1])
        name = node.name
        decl = f"T {name} = {op}({left.ref()}, {right.ref()})"
        stmts = left.stmts + right.stmts + [decl]
        return Fragment(name, stmts, combine_inputs(left.inputs, right.inputs))
    
    def visit_NPArray(self,
                      node: NPArray) -> BaseFragment:
        return InputFragment(node)

    def visit_Scalar(self,
                     node: Scalar) -> BaseFragment:
        return ScalarFragment(node)

    def visit_ReduceEx(self,
                       node: ReduceEx) -> BaseFragment:
        inner = self.visit(node.children[0])
        name = node.name
        op = node.to_op()

        return NotImplemented


def run_gpu(ex: NumpyEx) -> cupy.array:
    fuser = Fuser()
    splits = fuser.fuse(ex)
    visitor = CupyEmitter()
    kerns = []
    for split in splits:
        res = visitor.visit(ex)
        kerns.append(res)
    assert(len(kerns))
    for kern in kerns:
        compiled = kern.to_kern()
        inputs = [value.array for key, value in kern.kernel_args.items()]
        ret = compiled(*inputs)
        kern.array = ret
    return ret
