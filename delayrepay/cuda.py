""" CUDA numpy array with delayed-evaluation semantics

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

import cupy  # type: ignore

from .visitor import Visitor

np = cupy
fallback = cupy


def is_ndarray(arr):
    return isinstance(arr, cupy.core.core.ndarray)


def dedup(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def make_kernel(inputs, body, name):
    body = ";\n".join(dedup(body))
    inargs = [f"T {arg}" for arg in inputs]
    return cupy.ElementwiseKernel(
        ",".join(inargs),
        "T out",
        f"{body};\nout = {name}",
        f"delay_repay_{name}",
    )


class CupyEmitter(Visitor):
    def __init__(self):
        super().__init__()
        self.ins = {}
        self.outs = []
        self.kernels = []
        self.seen = {}
        self.count = 0

    def visit_BinaryNumpyEx(self, node):
        op = node.to_op()
        (lname, lbody), (rname, rbody) = self.visit(node.children)
        name = node.name
        stmt = f"T {name} = {lname} {op} {rname}"
        body = lbody + rbody + [stmt]
        return (name, body)

    def visit_UnaryFuncEx(self, node):
        inname, inbody = self.visit(node.children[0])
        name = node.name
        body = inbody + [f"T {name} = {node.to_op()}({inname})"]
        return (name, body)

    def visit_BinaryFuncEx(self, node):
        op = node.to_op()
        lname, lbody = self.visit(node.children[0])
        rname, rbody = self.visit(node.children[1])
        name = node.name
        stmt = f"T {name} = {op}({lname}, {rname})"
        body = lbody + rbody + [stmt]
        return (name, body)

    def visit_NPArray(self, node):
        return (node.name, [])

    def visit_NPRef(self, node):
        return NotImplemented

    def visit_Scalar(self, node):
        return (node.name, [])

    def visit_ReduceEx(self, node):
        return NotImplemented


def run(ex) -> cupy.array:
    visitor = CupyEmitter()
    body = visitor.visit(ex)[1]
    kern = make_kernel(ex.inputs, body, ex.name)
    inputs = [value.array for value in ex.inputs.values()]
    return kern(*inputs)
