'''Numpy abstractions'''

import logging
from dataclasses import dataclass
from numbers import Number
from typing import List, Tuple
import numpy as np

logger = logging.getLogger("delayRepay.num")

OPS = {
    'matmul': '@',
    'add': '+',
    'multiply': '*',
    'subtract': '-',
    'true_divide': '/'
}

@dataclass
class NumpyEx:
    '''Numpy expression'''

@dataclass
class DotEx(NumpyEx):
    arg1: NumpyEx
    arg2: NumpyEx

@dataclass
class MapEx(NumpyEx):
    func: np.ufunc
    arg: NumpyEx


class Funcable:
    def to_op(self):
        return OPS[self.func.__name__]


@dataclass
class ReduceEx(NumpyEx, Funcable):
    func: np.ufunc
    arg: NumpyEx


@dataclass
class BinaryNumpyEx(NumpyEx, Funcable):
    '''Binary numpy expression'''
    left: NumpyEx
    right: NumpyEx
    func: np.ufunc

@dataclass
class MMEx(NumpyEx, Funcable):
    arg1: NumpyEx
    arg2: NumpyEx

@dataclass
class MVEx(NumpyEx, Funcable):
    arg1: NumpyEx
    arg2: NumpyEx
    
#@dataclass
#class DotEx(NumpyEx, Funcable):
#    left: NumpyEx
#    right: NumpyEx
#    func: np.dot
    
# @dataclass
# class Dot(BinaryNumpyEx):
#     '''np.dot func'''
#     func: np.ufunc = np.dot

# @dataclass
# class Multiply(BinaryNumpyEx):
#     '''np.multiply'''
#     func = np.multiply

# @dataclass
# class Matmul(BinaryNumpyEx):
#     '''np.matmul'''
#     func = np.matmul

# @dataclass
# class Add(BinaryNumpyEx):
#     '''np.add'''
#     func  = np.add


@dataclass
class NPAtom(NumpyEx):
    '''an atom'''


@dataclass
class NPArray(NPAtom):
    '''ndarray'''
    array: np.ndarray

    def __hash__(self):
        return id(self.array)

    def __eq__(self, other):
        return self.array is other.array

@dataclass
class Var(NPAtom):
    '''a variable'''
    name: str

    def __repr__(self):
        return self.name

@dataclass
class Scalar(NPAtom):
    '''a scalar'''
    val: Number


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
        print("list_visit")
        return [self.visit(node) for node in lst]

    def default_visit(self, node):
        print("default_visit")
        return node


class NumpyVisitor(Visitor):
    '''Visits Numpy Expression'''
    def __init__(self):
        self.visits = 0
    '''visitor ABC'''
    # def visit(self, node):
    #     '''generic visit'''
    #     self.visits += 1
    #     logger.debug("visiting {}. visits: {}".format(node, self.visits))
    #     if isinstance(node, BinaryNumpyEx):
    #         ret =  self.visit_binary(node)
    #     elif isinstance(node, NPArray):
    #         ret = self.visit_array(node)
    #     elif isinstance(node, Var):
    #         ret =  self.visit_var(node)
    #     elif isinstance(node, Scalar):
    #         ret =  self.visit_scalar(node)
    #     else:
    #         logger.critical("!!!!!Not Implemented!!!!!")
    #         ret =  NotImplemented
    #     return ret

    def visit(self, node, **kwargs):
        """Visit a node."""
        self.visits += 1
        return super(NumpyVisitor, self).visit(node, **kwargs)

    def visit_BinaryExpression(self, node):
        return node

    def walk(self, tree):
        ''' top-level walk of tree'''
        self.visits = 0
        logger.debug("walking from top")
        return self.visit(tree)


class NumpyVarVisitor(NumpyVisitor):
    '''visits and returns new tree with arrays replaced with vars'''
    def __init__(self):
        self.arrays = {}
        super(NumpyVarVisitor, self).__init__()

    def visit_BinaryNumpyEx(self, node):
        '''visit BinaryNumpyEx'''
        return type(node)(
            self.visit(node.left),
            self.visit(node.right)
        )

    def visit_NPArray(self, node):
        '''visit NPArray'''
        if node in self.arrays:
            name = self.arrays[node]
        else:
            name = "npvar" + str(len(self.arrays))
            self.arrays[node] = name
        return Var(name)

    def visit_Scalar(self, node):
        '''Visit scalar'''
        return node

class StringVisitor(NumpyVisitor):
    '''returns string of numpy expression'''

    def visit_binary(self, node: BinaryNumpyEx):
        '''visit BinaryNumpyEx'''
        return "{}({}, {})".format("np." + node.func.__name__, self.visit(node.left), self.visit(node.right))

    def visit_var(self, node: Var):
        return node.name

    def visit_scalar(self, node: Scalar):
        return node.val

class NumpyFunction:
    '''complete numpy function'''
   
    def __init__(self, body: NumpyEx):
        visitor = NumpyVarVisitor()
        transformed = visitor.visit(body)
        self.body = transformed
        self.args = visitor.arrays

    def __call__(self):
        visitor = StringVisitor()
        exp = str(self)
        args = {}
        exec(exp, globals(), args )
        return args['jitfunc'](*[arr.array for arr in self.args.keys()])

    def _cpu(self):
        string = StringVisitor().visit(self.body)
        return "import numba\n@numba.jit\ndef jitfunc({}):\n    {}".format(','.join(self.args.values()), string)

    def _gpu(self):
        
        raise Error("GPU stuff not complete")

    def __str__(self):
        return self._cpu()


def is_matrix_vector(left, right):
    return len(left) > 1 and len(right) == 0

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
        node.shape = ShapeAnnotator.calc_shape(left.shape, right.shape, node.func)
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

    # def visit_MMEx(self, node):
    #     left = self.visit(node.arg1)
    #     right = self.visit(node.arg2)
    #     node.shape = ShapeAnnotator.calc_shape(left.shape, right.shape, np.dot)
    #     node._inshape = left.shape
    #     return node

    # def visit_MVEx(self, node):
    #     left = self.visit(node.arg1)
    #     right = self.visit(node.arg2)
    #     node.shape = ShapeAnnotator.calc_shape(left.shape, right.shape, np.dot)
    #     node._inshape = left.shape
    #     return node
    
class ReduceTransformer(NumpyVisitor):
    def visit_DotEx(self, node):
        # TODO This is just for vector x vector
        left = self.visit(node.arg1)
        right = self.visit(node.arg2)
        # if (len(left.array.shape) > 1 and len(right.array.shape) > 1):
        #     if (left.array.shape[0] > 1 and left.array.shape[1] > 1) and (right.array.shape[0] > 1 and right.array.shape[1] > 1):
        #         # print("matrix x matrix")
        #         muls = DotEx(left, right)
        #         muls.shape = (left.array.shape[0],right.array.shape[0])
        #         return muls
        #     elif is_matrix_vector(left.shape, right.shape):
        #         # matrix x vector
        #         print("matrix x vector")
        #         return MVEx(left, right, node.shape, node._inshape)
                
        #     elif (left.array.shape[0] > 1 or left.array.shape[1] > 1) and (right.array.shape[0] > 1 or right.array.shape[1] > 1):
        #         # vector x vector
        #         print("vector x vector")
        #         muls = BinaryNumpyEx(left, right, np.multiply)
        #         muls.shape = node._inshape
        #         red = ReduceEx(np.add, muls)
        #         red.shape = node.shape
        #         red._inshape = node._inshape
        #         return red
        # else:
        #     print("scalar?")
        if is_matrix_vector(left.shape, right.shape):
            print("matrix x vector")
            return MVEx(left, right, node.shape, node._inshape)
        else:
            print("vector x vector")
            muls = BinaryNumpyEx(left, right, np.multiply)
            muls.shape = node._inshape
            red = ReduceEx(np.add, muls)
            red.shape = node._inshape
            red._inshape = node._inshape
            return red
