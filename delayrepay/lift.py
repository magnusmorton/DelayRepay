from dataclasses import dataclass
from typing import List, Any
import numpy
from .visitor import Visitor


np = numpy
fallback = numpy


@dataclass
class LiftNode:
    '''A lift IR node '''

    def compose(self, other):
        '''convenience method for composition'''
        return Compose(self, other)

    def apply(self, other):
        '''convenience method for application'''
        return Apply(self, other)


@dataclass
class LiftType(LiftNode):
    '''A lift type'''


@dataclass
class Atom(LiftNode):
    '''Atoms'''


@dataclass
class Seq(LiftType):
    '''Sequence type'''
    types: List[LiftType]


@dataclass
class Float(LiftType):
    '''A float'''


@dataclass
class Int(LiftType):
    '''An Int'''


@dataclass
class Var(Atom):
    '''Not functions/classes'''
    val: str

    def pp(self):
        return self.val

@dataclass
class Number(Atom):
    '''Numbers'''
    val: float

    def pp(self):
        return str(self.val)


@dataclass
class String(Atom):
    '''Strings'''
    val: str


@dataclass
class SizeVar(LiftNode):
    ''' An array sizevar'''
    name: String


@dataclass
class ArrayTypeWSWC(LiftType):
    '''Funciton input array type'''
    dtype: LiftType
    sizevar: SizeVar


@dataclass
class Unary(LiftNode):
    '''Unary function'''
    arg: LiftNode


@dataclass
class Binary(LiftNode):
    '''Binary Function'''
    func: str
    left: LiftNode
    right: LiftNode


@dataclass
class Compose(Binary):
    '''Composition function'''


@dataclass
class Apply(Binary):
    '''Function application ($)'''

@dataclass
class Lambda(LiftNode):
    '''A scala lambda'''
    args: List[Var]
    body: LiftNode


@dataclass
class Map(Unary):
    '''Map base class'''
    fun: Lambda
    # arr: Any

    def pp(self):
        print(f"Map({self.fun.pp()})")

@dataclass
class MapWrg(Map):
    '''Map function over workgroups?'''


@dataclass
class Join(LiftNode):
    '''Join. Not sure what it does. Barrier? '''


@dataclass
class MapLcl(Map):
    '''Map fuction over work items?'''


@dataclass
class MapSeq(Map):
    '''Map sequentially'''


@dataclass
class Split(Unary):
    '''Split data'''


@dataclass
class Zip(Binary):
    '''Zip'''


@dataclass
class Array(LiftNode):
    '''Lift Array type'''
    members: List[String]




@dataclass
class UserFun(LiftNode):
    '''User Functions for OpenCL'''
    name: String
    args: Array
    body: String
    arg_type: LiftType
    ret_type: LiftType


@dataclass
class LiftFunction:
    '''A lift fun...'''
    types: List[LiftType]
    body: LiftNode


def new_var():
    new_var.counter += 1
    return Var("x%d" % new_var.counter)


new_var.counter = 0


class LiftVisitor(Visitor):
    def visit_NPArray(self, node):
        return new_var()

    def visit_Scalar(self, node):
        return Number(node.val)

    def visit_BinaryNumpyEx(self, node):
        children = self.visit(node.children)
        print(children)
        body = Map(Lambda(new_var(),
                          Binary(node.func.__name__, *children)))
        return body


def run(ex):
    visitor = LiftVisitor()
    return visitor.visit(ex).pp()
