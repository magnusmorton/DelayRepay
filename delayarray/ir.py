from dataclasses import dataclass
from typing import List
from num import NumpyVisitor


@dataclass
class IRNode:
    '''Base IR node'''

@dataclass
class Var(IRNode):
    '''var or value'''
    value: str

@dataclass
class Exp(IRNode):
    outvar: Var
    left: Var
    right: Var
    op: str

@dataclass
class InfixExp(IRNode):
    '''infix op'''


class IRTransformer(NumpyVisitor):

    # def __init__(self):
    #     self.ins = {}
    #     self.outs = []
    #     super(IRTransformer, self).__init__()

    def visit_BinaryNumpyEx(self, node):
        left_res = self.visit(node.left)
        right_res = self.visit(node.right)
        ir = (node.to_op(), left_res, right_res)
        return ir

    def visit_Scalar(self, node):
        return node.val

    def visit_NPArray(self, node):
        self.ins["arr"] = node.array
        return "arr"



def to_kern(op, left, right):
    "out[i] = {} {} {}".format(left, op, right)
