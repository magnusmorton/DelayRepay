from typing import Any
from functools import lru_cache

class Visitor:
    """Visitor ABC"""

    def visit(self, node) -> Any:
        """Visit a node."""
        if isinstance(node, list):
            visitor = self.list_visit
        else:
            visitor = self.single_visit
        return visitor(node)

    def list_visit(self, lst, **kwargs):
        return [self.visit(node) for node in lst]

    @lru_cache(maxsize=None)
    def single_visit(self, node):
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, self.default_visit)
        return visitor(node)

    def default_visit(self, node):
        return node


class NumpyVisitor(Visitor):
    """Visits Numpy Expression"""

    def __init__(self):
        self.visits = 0

    def visit(self, node):
        """Visit a node."""
        self.visits += 1
        return super(NumpyVisitor, self).visit(node)

    def visit_BinaryExpression(self, node):
        return node

    def walk(self, tree):
        """ top-level walk of tree"""
        self.visits = 0
        return self.visit(tree)


class PrettyPrinter(Visitor):
    def visit(self, node):
        if isinstance(node, list):
            return self.list_visit(node)
        print(type(node).__name__)
        self.visit(node.children)
