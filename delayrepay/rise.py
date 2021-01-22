from dataclasses import dataclass

import numpy
from .visitor import Visitor


np = numpy
fallback = numpy


@dataclass
class Expression:
    """Expression"""


@dataclass
class Identifier(Expression):
    """an identifier"""

    name: str

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name is other.name

    def print(self):
        return self.name


def var(name):
    return Identifier(name)


def new_var():
    new_var.counter += 1
    return Identifier("x%d" % new_var.counter)


new_var.counter = 0


@dataclass
class TupleAccess(Expression):
    expr: Expression
    access: str

    def print(self):
        return "%s.%s" % (self.expr.print(), self.access)


def fst(x):
    return TupleAccess(x, "_1")


def snd(x):
    return TupleAccess(x, "_2")


@dataclass
class Lambda(Expression):
    x: Identifier
    body: Expression

    def print(self):
        return "fun(%s => %s)" % (self.x.print(), self.body.print())


@dataclass
class TypedLambda(Expression):
    x: Identifier
    body: Expression
    typ: str

    def print(self):
        return f"fun({self.typ})({self.x.print()} => {self.body.print()})"


@dataclass
class DepFun(Lambda):

    def print(self):
        return f"depFun(({self.x}) => {self.body.print()}"


def fun(x, body):
    return Lambda(x, body)


def typedfun(x, body, typ):
    pass

def depFun(typ, body):
    return NotImplemented


@dataclass
class Literal(Expression):
    val: str

    def print(self):
        return self.val


def l(s):
    return Literal(s)


@dataclass
class BinaryExpression(Expression):
    op: str
    lhs: Expression
    rhs: Expression

    def print(self):
        return "%s %s %s" % (self.lhs.print(), self.op, self.rhs.print())


def bin_op(op, lhs, rhs):
    return BinaryExpression(op, lhs, rhs)


@dataclass
class Zip(Expression):
    lhs: Expression
    rhs: Expression

    def print(self):
        return "zip(%s, %s)" % (self.lhs.print(), self.rhs.print())


def zip(lhs, rhs):
    return Zip(lhs, rhs)


@dataclass
class Map(Expression):
    f: Expression
    xs: Expression

    def print(self):
        return "map(%s)(%s)" % (self.f.print(), self.xs.print())


def map(f, xs):
    return Map(f, xs)


@dataclass
class Generate(Expression):
    f: Expression

    def print(self):
        return "generate(%s)" % self.f.print()


def generate(f):
    return Generate(f)


class RISETransformer(Visitor):
    def visit_BinaryNumpyEx(self, node):
        children = self.visit(node.children)
        from .delayarray import Scalar

        if isinstance(node.left, Scalar):
            lhs = generate(fun(new_var(), l(node.left.val)))
        else:
            lhs = self.visit(node.left)
        if isinstance(node.right, Scalar):
            rhs = generate(fun(new_var(), l(node.right.val)))
        else:
            rhs = self.visit(node.right)
        x = new_var()
        return map(fun(x, bin_op(node.to_op(), fst(x), snd(x))), zip(lhs, rhs))

    def visit_NPArray(self, node):
        return var("xs")

    def visit_Scalar(self, node):
        return l(node.val)


class ShineTransformer(Visitor):

    def walk(self, node):
        rise_ex = self.visit(node)
        funed = DepFun('n: Nat', TypedLambda(var('xs'), rise_ex, 'n`.`f32'))
        return funed

    def visit_BinaryNumpyEx(self, node):
        from .delayarray import Scalar

        # if both are scalar
        return fun(x, bin_op(node.to_op(), lhs, rhs))
        # if one is scalar
        return Map(fun(x, bin_op(node.to_op(), x, lhs)), rhs)

        # if both are tensors, zip
        return map(fun(x, bin_op(node.to_op(), fst(x), snd(x))), Zip(lhs, rhs))
        
        if isinstance(node.children[0], Scalar):
            lhs = generate(fun(new_var(), l(node.children[0].val)))
        else:
            lhs = self.visit(node.children[0])
        if isinstance(node.children[1], Scalar):
            rhs = generate(fun(new_var(), l(node.children[1].val)))
        else:
            rhs = self.visit(node.children[1])
        x = new_var()


    def visit_NPArray(self, node):
        return var("xs")

    def visit_Scalar(self, node):
        return l(node.val)


def to_rise(numpy_ex):
    transformer = ShineTransformer()
    rise_ex = transformer.walk(numpy_ex)
    return rise_ex.print()


run = to_rise