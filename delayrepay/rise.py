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
    pos: int

    def print(self):
        return "%s.%s" % (self.expr.print(), self.access)


def fst(x):
    return TupleAccess(x, 1)


def snd(x):
    return TupleAccess(x, 2)


@dataclass
class Lambda(Expression):
    x: Identifier
    body: Expression

    def print(self):
        return "fun(%s => %s)" % (self.x.print(), self.body.print())

    @property
    def children(self):
        return [self.x, self.body]


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


class DelayException(Exception):
    pass


class ShineTransformer(Visitor):

    def walk(self, node):
        rise_ex = self.visit(node)
        print(type(rise_ex))
        funed = DepFun('n: Nat', TypedLambda(var('xs'), rise_ex, 'n`.`f32'))
        return funed

    def visit_BinaryNumpyEx(self, node):
        from .delayarray import Scalar, NPArray
        left, right = self.visit(node.children)

        # TODO use BinaryNumpyEx over BinaryExpression. Make Var part of NumpyEx?
        if isinstance(left, Scalar):
            if isinstance(right, Scalar):
                # if both are scalar
                raise DelayException("should not happen")
            else:
                # if one is scalar
                x = new_var()
                return Map(Lambda(x, BinaryExpression(node.to_op(), x, left)),
                           right)
        else:
            x = new_var()
            if isinstance(right, Scalar):
                # if one is scalar
                return Map(Lambda(x, BinaryExpression(node.to_op(), x, right)),
                           left)
            else:
                # both are tensors
                return Map(Lambda(x, BinaryExpression(node.to_op(),
                                                      fst(x),
                                                      snd(x))),
                           Zip(left, right))


class ShineEmitter(Visitor):

    def visit_Scalar(self, node):
        return node.val

    def visit_BinaryExpression(self, node):
        lhs = self.visit(node.lhs)
        rhs = self.visit(node.rhs)
        return f"{lhs} {node.op} {rhs}"

    def visit_Lambda(self, node):
        x = self.visit(node.x)
        body = self.visit(node.body)
        return f"fun({x} => {body})"

    def visit_Map(self, node):
        f = self.visit(node.f)
        xs = self.visit(node.xs)
        return f"map({f})({xs})"

    def visit_Zip(self, node):
        lhs = self.visit(node.lhs)
        rhs = self.visit(node.rhs)
        return f"zip({lhs}, {rhs})"

    def visit_TupleAccess(self, node):
        expr = self.visit(node.expr)
        return f"{expr}._{node.pos}"

    def visit_TypedLambda(self, node):
        x = self.visit(node.x)
        body = self.visit(node.body)
        return f"fun({node.typ})({x} => {body})"

    def visit_DepFun(self, node):
        body = self.visit(node.body)
        return f"depFun(({node.x}) => {body})"

    def visit_Identifier(self, node):
        return node.name



def to_rise(numpy_ex):
    transformer = ShineTransformer()
    emitter = ShineEmitter()
    rise_ex = transformer.walk(numpy_ex)
    rise_str = emitter.visit(rise_ex)
    return rise_str


run = to_rise
