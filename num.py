'''Numpy abstractions'''

import logging
from dataclasses import dataclass
from numbers import Number
from typing import List
import numpy as np

logger = logging.getLogger("delayRepay.num")

@dataclass
class NumpyEx:
    '''Numpy expression'''

@dataclass
class BinaryNumpyEx(NumpyEx):
    '''Binary numpy expression'''
    left: NumpyEx
    right: NumpyEx
    
@dataclass
class Dot(BinaryNumpyEx):
    '''np.dot func'''
    func: np.ufunc = np.dot

@dataclass
class Multiply(BinaryNumpyEx):
    '''np.multiply'''
    func = np.multiply

@dataclass
class Matmul(BinaryNumpyEx):
    '''np.matmul'''
    func = np.matmul

@dataclass
class Add(BinaryNumpyEx):
    '''np.add'''
    func  = np.add

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

class NumpyVisitor:
    def __init__(self):
        self.visits = 0
    '''visitor ABC'''
    def visit(self, node):
        '''generic visit'''
        self.visits += 1
        logger.debug("visiting {}. visits: {}".format(node, self.visits))
        if isinstance(node, BinaryNumpyEx):
            ret =  self.visit_binary(node)
        elif isinstance(node, NPArray):
            ret = self.visit_array(node)
        elif isinstance(node, Var):
            ret =  self.visit_var(node)
        elif isinstance(node, Scalar):
            ret =  self.visit_scalar(node)
        else:
            logger.critical("!!!!!Not Implemented!!!!!")
            ret =  NotImplemented
        return ret

    def walk(self, tree):
        ''' top-level walk of tree'''
        self.visits = 0
        logger.debug("walking from top")
        return self.visit(tree)
    
class NumpyVarVisitor(NumpyVisitor):
    '''visits and returns new tree with arrays replaced with vars'''
    def __init__(self):
        self.arrays = {}

    def visit_binary(self, node):
        '''visit BinaryNumpyEx'''
        return type(node)(
            self.visit(node.left),
            self.visit(node.right)
        )

    def visit_array(self, node):
        '''visit NPArray'''
        if node in self.arrays:
            name = self.arrays[node]
        else:
            name = "npvar" + str(len(self.arrays))
            self.arrays[node] = name
        return Var(name)

    def visit_scalar(self, node):
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
        

