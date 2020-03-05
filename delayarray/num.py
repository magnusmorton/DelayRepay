'''Numpy abstractions'''

import logging
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


def calc_shape(left, right, op=None):
    print(left)
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

class NumpyEx:
    '''Numpy expression'''

class Funcable:
    def to_op(self):
        return OPS[self.func.__name__]


class ReduceEx(NumpyEx, Funcable):
    def __init__(self, func, arg):
        self.func = func
        self.arg = arg

    # func: np.ufunc
    # arg: NumpyEx


class BinaryNumpyEx(NumpyEx, Funcable):
    '''Binary numpy expression'''
    # left: NumpyEx
    # right: NumpyEx
    # func: np.ufunc
    
    def __init__(self, left, right, func):
        self.left = left
        self.right = right
        self.func = func
        self.shape = calc_shape(left.shape, right.shape, func)

class MMEx(NumpyEx, Funcable):
    # arg1: NumpyEx
    # arg2: NumpyEx
    def __init__(self, arg1, arg2):
        self.arg1 = arg1
        self.arg2 = arg2
        self.shape = calc_shape(arg1.shape, arg2.shape, np.dot)

class MVEx(NumpyEx, Funcable):
    # arg1: NumpyEx
    # arg2: NumpyEx
    def __init__(self, arg1, arg2):
        self.arg1 = arg1
        self.arg2 = arg2
        self.shape = calc_shape(arg1.shape, arg2.shape, np.dot)
    
class DotEx(NumpyEx, Funcable):
   # left: NumpyEx
   # right: NumpyEx
   # func: np.dot
    def __init__(self, left, right):
        self.arg1 = left
        self.arg2 = right
        self.shape = calc_shape(left.shape, right.shape, np.dot)
        self._inshape  = left.shape


class NPArray(NumpyEx):
    '''ndarray'''

    def __init__(self, array):
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



def is_matrix_matrix(left, right):
    return len(left) > 1 and len(right) > 1


def is_matrix_vector(left, right):
    print(len(left))
    print(len(right))
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


    
class ReduceTransformer(NumpyVisitor):
    def visit_DotEx(self, node):
        # TODO This is just for vector x vector
        left = self.visit(node.arg1)
        right = self.visit(node.arg2)
        
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
