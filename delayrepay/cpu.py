import numpy
from .visitor import Visitor

np = numpy
fallback = numpy

class CpuVisitor(Visitor):
    def visit_NPArray(self, node):
        return node.array

    def visit_Scalar(self, node):
        return node.val

    def visit_UnaryFuncEx(self, node):
        return node.func(*self.visit(node.children))

    def visit_BinaryFuncEx(self, node):
        return node.func(*self.visit(node.children))

    def visit_BinaryNumpyEx(self, node):
        return node.func(*self.visit(node.children))


def run(ex):
    visitor = CpuVisitor()
    return visitor.visit(ex)
