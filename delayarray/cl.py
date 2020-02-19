import logging
from dataclasses import dataclass
from numbers import Number
from typing import List, Dict

import delayarray.num as num
import delayarray.kernels as kernels
import pyopencl as cl
import numpy as np

logger = logging.getLogger("delayRepay.cl")
preamble = "int i = get_global_id(0);"

@dataclass
class CLTree:
    '''CL tree node'''


@dataclass
class Expression(CLTree):
    '''Expression'''


@dataclass
class DotExpression(Expression):
    arg1: Expression
    arg2: Expression


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

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name is other.name


@dataclass
class Scalar(Expression):
    val: Number

    def to_input(self, _=0):
        return self.val


@dataclass
class Subscript(Expression):
    '''a subscript e.g. a[i]'''
    var: Var
    sub: Expression


@dataclass
class CLArgs(CLTree):
    params: List[Var]
    types: List[str]


@dataclass
class CLFunction(CLTree):
    '''Complete CL function'''
    args: CLArgs
    name: str
    body: List[CLTree]


@dataclass
class CLKernel:
    name: str
    body: str
    inputs: Dict
    reducing: bool = False

    def to_kern(self):
        out = "__kernel void {} ({}, __global float *output){{\n{}\n{}\n}}"
        inargs = ["__global const float* {}".format(name)
                  for name in self.inputs.keys()]
        return out.format("foo", ", ".join(inargs), preamble, self.body)


class CLEmitter(num.Visitor):

    def visit_BinaryExpression(self, node):
        return "{} {} {}".format(
            self.visit(node.left), node.op, self.visit(node.right)
        )

    def visit_DotExpression(self, node):
        return "dot({}, {})"

    def visit_Subscript(self, node):
        return "{}[{}]".format(self.visit(node.var), self.visit(node.sub))

    def visit_Scalar(self, node):
        return str(node.val)

    def visit_Var(self, node):
        return node.name

    def visit_Assignment(self, node):
        return "{} = {};".format(self.visit(node.left), self.visit(node.right))

    def visit_CLArgs(self, node):
        args = ["__global {} {}".format(typ, var)
                for typ, var in zip(node.types, self.visit(node.params))]
        return ", ".join(args)

    def visit_CLFunction(self, node):
        return "__kernel void {} ({}) {{\n{}\n{}\n}}".format(
            node.name, self.visit(node.args),
            preamble,
            "\n".join(self.visit(node.body))
        )


class GPUTransformer(num.NumpyVisitor):

    def __init__(self):
        self.ins = {}
        self.outs = []
        super(GPUTransformer, self).__init__()

    def visit_BinaryNumpyEx(self, node):
        cur_visits = self.visits
        ex = BinaryExpression(node.to_op(),
                              self.visit(node.left),
                              self.visit(node.right))
        if cur_visits == 1:
            ex = Assignment(Subscript(Var("foo"), Var("i")), ex)
            self.outs.append(Var("foo"))
        return ex

    def visit_NPArray(self, node):
        var = Var('delayvar{}'.format(len(self.ins)))
        self.ins[var] = node.array
        return Subscript(var, Var('i'))

    def visit_Scalar(self, node):
        return Scalar(node.val)

    def visit_DotEx(self, node):
        ex = DotExpression(self.visit(node.arg1), self.visit(node.arg2))
        return ex


class GPUEmitter(num.NumpyVisitor):

    def __init__(self):
        self.ins = {}
        self.outs = []
        self.kernels = []
        super(GPUEmitter, self).__init__()

    def visit_BinaryNumpyEx(self, node, callshape=None):
        op = node.to_op()
        curr_visit = self.visits
        left, lin, lstmts, llocals = self.visit(node.left, callshape=node.shape)
        right, rin, rstmts, rlocals = self.visit(node.right, callshape=node.shape)
        name = "var{}".format(curr_visit)
        stmts = []
        for local in llocals + rlocals:
            stmts.append("float {};".format(local))

        outvar = name
        if callshape is None or callshape != node.shape:
            outvar = "output[i]"
        stmt = "{} = {} {} {};".format(outvar, left, op, right)
        # I've made this too complicated for myself
        kernel = CLKernel(name, "\n".join(stmts + lstmts + rstmts + [stmt]), {**lin, **rin})
        if callshape is None or callshape != node.shape:
            self.kernels.append(kernel)
            return (name+"[i]", {name: kernel}, [], [])
        else:
            return (name, {**lin, **rin}, [stmt], [name] + llocals + rlocals)

    def visit_NPArray(self, node, callshape=None):
        name = "var{}".format(self.visits)
        self.ins[name] = node.array
        return (name+"[i]", {name: node.array}, [], [])

    def visit_Scalar(self, node, callshape=None):
        return (node.val, {}, [], [])

    def visit_DotEx(self, node, callshape=None):
        ex = DotExpression(self.visit(node.arg1), self.visit(node.arg2))
        return ex

    def visit_ReduceEx(self, node):
        curr_visit = self.visits
        arg, input_arg, stmts, mlocals = self.visit(node.arg, callshape=node._inshape)
        decls = ["float {};".format(local) for local in mlocals]
        stmt = kernels.mag_sum.format(arg)
        name = "input{}".format(curr_visit)
        kernel = CLKernel(name, "\n".join(decls + stmts + [stmt]), input_arg, reducing=True)
        self.kernels.append(kernel)
        return (name+"[i]", {name: kernel})


def executor(kernel, in_arrs, out_arr):
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    in_bufs = [cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=in_arr)
               for in_arr in in_arrs]
    prog = cl.Program(ctx, kernel).build()
    res_g = cl.Buffer(ctx, mf.WRITE_ONLY, in_arrs[0].nbytes)
    prog.gfunc(queue, in_arrs[0].shape, None, *in_bufs, res_g)
    res_np = np.empty_like(in_arrs[0])
    cl.enqueue_copy(queue, res_np, res_g)
    return res_np


def run_gpu(numpy_ex):
    trans = GPUEmitter()
    trans.walk(num.ReduceTransformer().visit(num.ShapeAnnotator().visit(numpy_ex)))
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    bufs = {}
    for kernel in trans.kernels:
        for ref, source in kernel.inputs.items():
            if isinstance(source, np.ndarray):
                # TODO: fix sizing;get rid of first_arr
                first_arr = source
                bufs[ref] = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                      hostbuf=source)
            else:
                bufs[ref] = cl.Buffer(ctx, mf.READ_WRITE, first_arr.nbytes)
            kernel.prog = cl.Program(ctx, kernel.to_kern()).build()
    last_kern = trans.kernels[-1]
    resshape = first_arr.shape
    shape = first_arr.shape
    if len(shape) > 1:
        shape = (shape[0] * shape[1],)
    if last_kern.reducing:
        resshape = (resshape[0] // 64,)
        bufs[last_kern.name] = cl.Buffer(ctx, mf.READ_WRITE, first_arr.nbytes // 64)
    else:
        bufs[last_kern.name] = cl.Buffer(ctx, mf.READ_WRITE, first_arr.nbytes)
    events = []
    for kernel in trans.kernels:
        group_shape = (64,)
        inputs = [bufs[key] for key in kernel.inputs.keys()]
        events.append(kernel.prog.foo(queue,
                                      shape,
                                      group_shape,
                                      *inputs,
                                      bufs[kernel.name]))
    res_np = np.empty(resshape).astype(np.float32)
    cl.enqueue_copy(queue, res_np, bufs[last_kern.name], wait_for=events)
    return res_np
