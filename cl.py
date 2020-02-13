import logging
from dataclasses import dataclass
from numbers import Number
from typing import List, Dict, Callable

import num
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

    def visit_BinaryNumpyEx(self, node):
        op = node.to_op()
        curr_visit = self.visits
        left, lin = self.visit(node.left)
        right, rin = self.visit(node.right)
        stmt = "output[i] = {} {} {};".format(left, op, right)
        # I've made this too complicated for myself
        name = "input{}".format(curr_visit)
        kernel = CLKernel(name, stmt, {**lin, **rin})
        self.kernels.append(kernel)
        return (name+"[i]", {name: kernel})

    def visit_NPArray(self, node):
        name = "input{}".format(self.visits)
        self.ins[name] = node.array
        return (name+"[i]", {name: node.array})

    def visit_Scalar(self, node):
        return (node.val, {})

    def visit_DotEx(self, node):
        ex = DotExpression(self.visit(node.arg1), self.visit(node.arg2))
        return ex

    def visit_ReduceEx(self, node):
        curr_visit = self.visits
        arg, input_arg = self.visit(node.arg)
        op = node.to_op()
        stmt = """
         int local_id = get_local_id(0);
    int group_size = get_local_size(0);

   // get me stuff in local mem plsthnx
    __local float localSums[64];
    localSums[local_id] = {};
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int offset = 1; offset < group_size; offset <<= 1) {{
        int mask = (offset << 1) - 1;
        if ((local_id & mask) == 0) {{
            localSums[local_id] += localSums[offset];
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
    }}
    if (local_id == 0) {{
        output[get_group_id(0)] = localSums[0];

    }}
        """.format(arg)
        name = "input{}".format(curr_visit)
        kernel = CLKernel(name, stmt, input_arg, reducing=True)
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

    if last_kern.reducing:
        bufs[last_kern.name] = cl.Buffer(ctx, mf.READ_WRITE, first_arr.nbytes // 64)
    else:
        bufs[last_kern.name] = cl.Buffer(ctx, mf.READ_WRITE, first_arr.nbytes)
    events = []
    for kernel in trans.kernels:
        inputs = [bufs[key] for key in kernel.inputs.keys()]

        events.append(kernel.prog.foo(queue,
                                      first_arr.shape,
                                      (64,),
                                      *inputs,
                                      bufs[kernel.name]))
    res_np = np.empty((160,)).astype(np.float32)
    cl.enqueue_copy(queue, res_np, bufs[last_kern.name], wait_for=events)
    return res_np
