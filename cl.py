import logging
from dataclasses import dataclass
from numbers import Number
from typing import List

import num


logger = logging.getLogger("delayRepay.cl")



@dataclass
class CLTree:
    '''CL tree node'''

@dataclass
class Expression(CLTree):
    '''Expression'''

@dataclass
class BinaryExpression(Expression):
    op: str
    left: Expression
    right: Expression

@dataclass
class Assignment(CLTree):
    '''Assignment statement'''
    left: Expression
    right: Expression

@dataclass
class Var(Expression):
    '''a variable'''
    name: str

@dataclass
class Scalar(Expression):
    val: Number
    
@dataclass
class Subscript(Expression):
    '''a subscript e.g. a[i]'''
    var: Var
    sub: Expression


@dataclass
class CLFunction:
    '''Complete CL function'''
    args: List[Var]
    name: str
    body: List[CLTree]


class Visitor:
    def visit(self, node):
        """Visit a node."""
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a node."""
        for field, value in iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, Expression):
                        self.visit(item)
            elif isinstance(value, Expression):
                self.visit(value)
    
class CLEmitter(Visitor):
    
    

    def visit_BinaryExpression(self, node):
        return "{} {} {}".format(self.visit(node.left), node.op, self.visit(node.right))


    def visit_Subscript(self, node):
        return "{}[{}]".format(self.visit(node.var), self.visit(node.sub))

    def visit_Scalar(self, node):
        return str(node.val)

    def visit_Var(self, node):
        return node.name

    def visit_Assignment(self, node):
        return "{} = {};".format(self.visit(node.left), self.visit(node.right))

class CLVarVisitor(Visitor):
    pass


class GPUTransformer(num.NumpyVisitor):

    def __init__(self):
        self.ins = []
        self.outs = []

    def visit_binary(self, node):
        cur_visits = self.visits
        print(self.visits)
        if isinstance(node, num.Add):
            op = "+"
        ex = BinaryExpression(op, self.visit(node.left), self.visit(node.right))
        if cur_visits == 1:
            ex = Assignment(Subscript(Var("foo"), Var("i")), ex)
        return ex

    def visit_array(self, node):
        return Subscript(Var('a'), Var('i'))

    def visit_scalar(self, node):
        return Scalar(node.val)


def function_coverter(tree: CLTree):
    pass
    
