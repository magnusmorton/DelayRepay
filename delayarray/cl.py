import logging
from numbers import Number
from typing import List, Dict, Tuple

import delayarray.num as num
import delayarray.kernels as kernels
import pyopencl as cl
import numpy as np

logger = logging.getLogger("delayRepay.cl")
PREAMBLE = "int i = get_global_id(0);"
PREAMBLE2D = """
#define TILEX 4
#define TILEX_SHIFT 2
#define TILEY 4
#define TILEY_SHIFT 2
int2 shape = (int2)(get_global_size(0), get_global_size(1));
int2 pos = (int2)(get_global_id(0), get_global_id(1));
int i = pos.y * shape.x + pos.x;
"""


class CLKernel:

    def __init__(self, name, body, inputs, shape):
        self.name = name
        self.body = body
        self.inputs = inputs
        self.shape = shape
        self.reducing = False
        self.preamble = PREAMBLE
        self.out_type = "float4"

    def to_kern(self):
        inargs = []
        for name in self.inputs.keys():
            if "var" in name:
                inargs.append("__global float4* {}".format(name))
            else:
                inargs.append("const uint {}".format(name))

        return f'__kernel void foo ({", ".join(inargs)}, __global {self.out_type} *output){{\n{self.preamble}\n{self.body}\n}}'

    def global_shape(self):
        return tuple(dim // 4 for dim in self.shape)

    def outshape(self):
        return self.shape

    @staticmethod
    def factory(name, body, inputs, shape, reducing=False):
        print(shape)
        if len(shape) > 1:
            return Kernel2D(name, body, inputs, shape)
        if reducing:
            return ReducingKernel(name, body, inputs, shape)
        return CLKernel(name, body, inputs, shape)


class ReducingKernel(CLKernel):

    def __init__(self, *args):
        "initialiser"
        super(ReducingKernel, self).__init__(*args)
        self.reducing = True
        self.out_type = "float"

    def outshape(self):
        return self.global_shape()


class Kernel2D(CLKernel):
    '''
    for 2D kernels
    '''
    def __init__(self, *args, **kwargs):

        super(Kernel2D, self).__init__(*args, **kwargs)
        self.preamble = PREAMBLE2D

    def global_shape(self):
        return tuple(dim // 2 for dim in self.shape)



class GPUEmitter(num.NumpyVisitor):

    def __init__(self):
        self.ins = {}
        self.outs = []
        self.kernels = []
        super(GPUEmitter, self).__init__()

    def visit_BinaryNumpyEx(self, node, callshape=None):
        op = node.to_op()
        curr_visit = self.visits
        left, lin, lstmts, llocals = self.visit(node.left,
                                                callshape=node.shape)
        right, rin, rstmts, rlocals = self.visit(node.right,
                                                 callshape=node.shape)
        name = "var{}".format(curr_visit)
        stmts = []
        for local in llocals + rlocals:
            stmts.append("float4 {};".format(local))

        outvar = name
        if callshape is None or callshape != node.shape:
            outvar = "output[i]"
        stmt = "{} = {} {} {};".format(outvar, left, op, right)
        # I've made this too complicated for myself
        kernel = CLKernel.factory(name, "\n".join(stmts + lstmts + rstmts + [stmt]),
                          {**lin, **rin}, node.shape)
        if callshape is None or callshape != node.shape:
            self.kernels.append(kernel)
            return (name+"[i]", {name: kernel}, [], [])
        else:
            return (name, {**lin, **rin},lstmts + rstmts + [stmt],
                    [name] + llocals + rlocals)

    def visit_NPArray(self, node, callshape=None):
        name = "var{}".format(id(node))
        self.ins[name] = node.array
        return (name+"[i]", {name: node.array}, [], [])

    def visit_Scalar(self, node, callshape=None):
        return (node.val, {}, [], [])

    def visit_MVEx(self, node, callshape=None):
        curr_visit = self.visits
        left, lin, lstmts, llocals = self.visit(node.arg1,
                                                callshape=node.arg1.array.shape)
        right, rin, rstmts, rlocals = self.visit(node.arg2,
                                                 callshape=node.arg2.array.shape)

        if lstmts == []:
            if left.endswith('[i]'):
                left = left[:-3]

        if rstmts == []:
            if right.endswith('[i]'):
                right = right[:-3]

        name = "var{}".format(curr_visit)
        stmts = []
        for local in llocals + rlocals:
            stmts.append("float4 {};".format(local))

        outvar = name
        if callshape is None or callshape != node.shape:
            outvar = "output"
            
        stmt = kernels.gemv.format(outvar, outvar, left, right)

        d = {**lin, **rin}
        d["num_rows_A"] = node.arg1.array.shape[0]
        d["num_cols_A"] = node.arg1.array.shape[1]
        
        kernel = CLKernel.factory(name, "\n".join(stmts + lstmts + rstmts + [stmt]),
                          d,
                          node.shape)
        
        if callshape is None or callshape != node.shape:
            self.kernels.append(kernel)
            return (name, {name: kernel}, [], [])
        else:
            return (name, {**lin, **rin}, [stmt], [name] + llocals + rlocals)
                
    def visit_MMEx(self, node, callshape=None):
        curr_visit = self.visits
        left, lin, lstmts, llocals = self.visit(node.arg1, callshape=node.arg1.array.shape)
        right, rin, rstmts, rlocals = self.visit(node.arg2, callshape=node.arg2.array.shape)

        if lstmts == []:
            if left.endswith('[i]'):
                left = left[:-3]

        if rstmts == []:
            if right.endswith('[i]'):
                right = right[:-3]
                    
        name = "var{}".format(curr_visit)
        stmts = []
        for local in llocals + rlocals:
            stmts.append("float {};".format(local))

        outvar = name
        if callshape is None or callshape != node.shape:
            outvar = "output"
        stmt = kernels.gemm_new.format(matrixA=left, matrixB=right,
                                       matrixC=outvar)

        d = {**lin, **rin}
        d["num_cols_A"] = node.arg1.array.shape[1]
        d["num_cols_B"] = node.arg2.array.shape[1]
        kernel = CLKernel.factory(name, "\n".join(stmts + lstmts + rstmts + [stmt]),
                          d,
                          node.shape)
    
        if callshape is None or callshape != node.shape:
            self.kernels.append(kernel)
            return (name, {name: kernel}, [], [])
        else:
            return (name, {**lin, **rin}, [stmt], [name] + llocals + rlocals)
    
    def visit_ReduceEx(self, node):
        curr_visit = self.visits
        arg, input_arg, stmts, mlocals = self.visit(node.arg,
                                                    callshape=node._inshape)
        decls = ["float4 {};".format(local) for local in mlocals]
        stmt = kernels.mag_sum.format(arg)
        name = "input{}".format(curr_visit)
        kernel = CLKernel.factory(name, "\n".join(decls + stmts + [stmt]), input_arg,
                          node.shape, reducing=True)
        self.kernels.append(kernel)
        return (name+"[i]", {name: kernel})

    def visit_DotEx(self, node):
        curr_visit = self.visits
        left, lin, lstmts, llocals = self.visit(node.arg1, callshape=node._inshape)
        right, rin, rstmts, rlocals = self.visit(node.arg2, callshape=node._inshape)
        d = {**lin, **rin}
        stmts = []
        for local in llocals + rlocals:
            stmts.append("float4 {};".format(local))
        stmt = kernels.dot.format(left, right)
        name = "input{}".format(curr_visit)
        kernel = CLKernel.factory(name, "\n".join(stmts + [stmt]), d,
                                node._inshape, reducing=True)
        self.kernels.append(kernel)
        return (name+"[i]", {name: kernel})


def run_gpu(numpy_ex):
    trans = GPUEmitter()
    trans.walk(numpy_ex)
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    bufs = {}

    if trans.kernels == []:

        raise Exception("No kernels...")
    # allocating memory
    for kernel in trans.kernels:
        # print(kernel.to_kern())     
        for ref, source in kernel.inputs.items():
            if isinstance(source, np.ndarray) and ref not in bufs:
                first_arr = source
                bufs[ref] = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                      hostbuf=source)
            elif isinstance(source, int):
                bufs[ref] = np.uint32(source)
            else:
                bufs[ref] = cl.Buffer(ctx, mf.READ_WRITE, first_arr.nbytes)
        kernel.prog = cl.Program(ctx, kernel.to_kern()).build()

    last_kern = trans.kernels[-1]
        
    resshape = last_kern.outshape()
    print(resshape)
    shape = first_arr.shape
    if len(shape) > 1:
        shape = (shape[0] * shape[1],)

    # todo fixme
    # if last_kern.reducing:
    #     resshape = (resshape[0] // 64,)
    #     res_np = np.empty(resshape,dtype=np.float32)
    #     bufs[last_kern.name] = cl.Buffer(ctx, mf.READ_WRITE, first_arr.nbytes // 64)
    # else:
    res_np = np.empty(resshape,dtype=np.float32)
    bufs[last_kern.name] = cl.Buffer(ctx, mf.READ_WRITE, res_np.nbytes)

    # scheduling
    events = []
    for kernel in trans.kernels:
        group_shape = (64,)
        print(kernel.global_shape())
        print(kernel.to_kern())
        inputs = [bufs[key] for key in kernel.inputs.keys()]
        events.append(kernel.prog.foo(queue,
                                      kernel.global_shape(),
                                      None,
                                      *inputs,
                                      bufs[kernel.name]))

    cl.enqueue_copy(queue, res_np, bufs[last_kern.name], wait_for=events)
    return res_np
