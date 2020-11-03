""" CUDA numpy array with delayed-evaluation semantics

Needs split up

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""
# Copyright (C) 2020 by Univeristy of Edinburgh

from typing import Any, List, Dict, Tuple, Union
import cupy  # type: ignore

import delayrepay.ir as ir

Shape = Tuple[int, int]

OPS = {
    "matmul": "@",
    "add": "+",
    "multiply": "*",
    "subtract": "-",
    "true_divide": "/",
}

FUNCS = {
    "power": "pow",
    "arctan2": "atan2",
    "absolute": "abs",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "sqrt": "sqrt",
    "log": "log",
    # HACK
    "negative": "-",
    "exp": "exp",
    "tanh": "tanh",
    "sinh": "sinh",
    "cosh": "cosh",
}

ufunc_lookup = {
    "matmul": cupy.matmul,
    "add": cupy.add,
    "multiply": cupy.multiply,
    "subtract": cupy.subtract,
    "true_divide": cupy.true_divide,
}


class Visitor:
    """Visitor ABC"""

    def visit(self, node) -> Any:
        """Visit a node."""
        if isinstance(node, list):
            visitor = self.list_visit
        else:
            method = "visit_" + node.__class__.__name__
            visitor = getattr(self, method, self.default_visit)
        return visitor(node)

    def list_visit(self, lst, **kwargs):
        return [self.visit(node) for node in lst]

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


def is_matrix_matrix(left, right):
    return len(left) > 1 and len(right) > 1


def is_matrix_vector(left, right):
    return len(left) > 1 and len(right) == 1


InputDict = Dict[str, "BaseFragment"]


class BaseFragment:
    def __init__(self):
        self.name = None
        self.stmts = []
        self._expr = None
        self._inputs = {}

    @property
    def inputs(self) -> InputDict:
        return self._inputs

    @property
    def kernel_args(self) -> InputDict:
        return self._inputs


def dedup(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


class Fragment(BaseFragment):
    def __init__(self, name: str, stmts: List[str], inputs: InputDict) -> None:
        self.name = name
        self.stmts = stmts
        self._inputs = inputs
        # self.dtype = np.float32

    def ref(self) -> str:
        return self.name

    # def expr(self) -> str:
    #     return self._expr

    def to_input(self):
        return {self.name: self.node.array}

    def to_kern(self) -> cupy.ElementwiseKernel:
        body = ";\n".join(dedup(self.stmts))
        inargs = [f"T {arg}" for arg in self.kernel_args]
        kern = cupy.ElementwiseKernel(
            ",".join(inargs),
            "T out",
            f"{body};\nout = {self.name}",
            f"delay_repay_{self.name}",
        )
        return kern


class InputFragment(BaseFragment):
    def __init__(self, name: str, arr: Union[ir.NPArray, ir.NPRef]) -> None:
        super().__init__()
        self.name = name
        self._inputs = {self.name: arr}

    def ref(self) -> str:
        return f"{self.name}"

    def expr(self) -> str:
        return f"{self.name}"


# dtype_map = {np.dtype("float32"): "float",
#              np.dtype("float64"): "double",
#              np.dtype("int32"): "int",
#              np.dtype("int64"): "long"}

# dtype_map = {np.dtype("float32"): "f",
#              np.dtype("float64"): "",
#              np.dtype("int32"): "",
#              np.dtype("int64"): ""}


class ScalarFragment(BaseFragment):
    def __init__(self, val: ir.Scalar) -> None:
        super().__init__()
        self.val = val.val
        self.dtype = val.dtype

    def ref(self) -> str:
        return str(self.val)

    def expr(self) -> str:
        return str(self.val)


class ReductionKernel(Fragment):
    def to_kern(self):
        kern = cupy.ReductionKernel(
            ",".join(self.inargs),
            "T out",
            self.expr,
            self.redex,
            "out = a",
            0,
            self.name,
        )
        return kern


def combine_inputs(*args: InputDict) -> InputDict:
    ret = {}
    for arg in args:
        ret.update(arg)
    return ret


class PrettyPrinter(Visitor):
    def visit(self, node):
        if isinstance(node, list):
            return self.list_visit(node)
        print(type(node).__name__)
        self.visit(node.children)


class Fuser(Visitor):
    def __init__(self):
        super().__init__()
        self.seen = {}
        self.splits = []

    def fuse(self, node):

        self.visit(node)
        self.splits.append(node)
        return self.splits

    def visit(self, node) -> Shape:
        if isinstance(node, list):
            return self.list_visit(node)
        child_shapes = self.list_visit(node.children)
        new = []
        for child, shape in zip(node.children, child_shapes):
            if shape != node.shape and shape != (0,):
                new.append(ir.NPRef(child, node.shape))
                self.splits.append(child)
            else:
                new.append(child)

        node.children = new

        return node.shape


class CupyEmitter(Visitor):
    def __init__(self):
        super().__init__()
        self.ins = {}
        self.outs = []
        self.kernels = []
        self.seen = {}
        self.count = 0

    def visit(self, node):
        if node in self.seen:
            visited = self.seen[node]
        else:
            visited = super().visit(node)
            self.seen[node] = visited
            self.count += 1
        return visited

    def visit_BinaryNumpyEx(self, node: ir.BinaryNumpyEx) -> BaseFragment:
        op = node.to_op()
        left = self.visit(node.children[0])
        right = self.visit(node.children[1])
        name = f"binex{self.count}"
        stmt = f"T {name} = {left.ref()} {op} {right.ref()}"
        stmts = left.stmts + right.stmts + [stmt]
        return Fragment(name, stmts, combine_inputs(left.inputs, right.inputs))

    def visit_UnaryFuncEx(self, node: ir.UnaryFuncEx) -> BaseFragment:
        inner = self.visit(node.children[0])
        name = f"unfunc{self.count}"
        stmts = inner.stmts + [f"T {name} = {node.to_op()}({inner.ref()})"]
        return Fragment(name, stmts, inner.inputs)

    def visit_BinaryFuncEx(self, node: ir.BinaryFuncEx) -> BaseFragment:
        op = node.to_op()
        left = self.visit(node.children[0])
        right = self.visit(node.children[1])
        name = f"binfunc{self.count}"
        stmt = f"T {name} = {op}({left.ref()}, {right.ref()})"
        stmts = left.stmts + right.stmts + [stmt]
        return Fragment(name, stmts, combine_inputs(left.inputs, right.inputs))

    def visit_NPArray(self, node: ir.NPArray) -> BaseFragment:
        return InputFragment(f"arr{self.count}", node)

    def visit_NPRef(self, node: ir.NPRef) -> BaseFragment:
        return InputFragment(f"ref{self.count}", node)

    def visit_Scalar(self, node: ir.Scalar) -> BaseFragment:
        return ScalarFragment(node)

    def visit_ReduceEx(self, node: ir.ReduceEx) -> BaseFragment:
        inner = self.visit(node.children[0])
        name = node.name
        op = node.to_op()

        return NotImplemented


def run_gpu(ex: ir.NumpyEx) -> cupy.array:
    visitor = CupyEmitter()
    kerns = [visitor.visit(ex)]
    for kern in kerns:
        compiled = kern.to_kern()
        inputs = [value.array for key, value in kern.kernel_args.items()]
        ret = compiled(*inputs)
        kern.array = ret
    return ret
