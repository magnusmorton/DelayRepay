import logging
import delayarray.num as num
import delayarray.kernels as kernels
import pyopencl as cl
import numpy as np

logger = logging.getLogger("delayRepay.cl")
PREAMBLE = "int i = get_global_id(0);"
PREAMBLE2D = """
int2 shape = (int2)(get_global_size(0), get_global_size(1));
int2 pos = (int2)(get_global_id(0), get_global_id(1));
int i = pos.y * shape.x + pos.x;
"""

REDUCTION_FACTOR = 32


class CLKernel:

    def __init__(self, name, body, inputs, shape):
        self.name = name
        self.body = body
        self.inputs = inputs
        self.shape = shape
        self.reducing = False
        self.preamble = PREAMBLE
        self.out_type = "float"
        self.dtype = np.float32

    def to_kern(self):
        inargs = []
        for name in self.inputs.keys():
            # TODO something else
            if "var" in name:
                inargs.append("__global float* {}".format(name))
            elif "local" in name:
                inargs.append(f"__local float* {name}")
            else:
                inargs.append("const uint {}".format(name))

        return f'__kernel void foo ({", ".join(inargs)}, __global {self.out_type} *output){{\n{self.preamble}\n{self.body}\n}}'

    def global_shape(self):
        return self.shape

    def local_shape(self):
        return None

    def outshape(self):
        return self.shape

    def outbytes(self):
        return np.prod(self.outshape()) * self.dtype().itemsize

    @staticmethod
    def factory(name, body, inputs, shape, reducing=False):
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

    def local_shape(self):
        return (REDUCTION_FACTOR,)

    def outshape(self):
        return tuple(dim // REDUCTION_FACTOR for dim in self.shape)


class Kernel2D(CLKernel):
    '''
    for 2D kernels
    '''
    def __init__(self, *args, **kwargs):

        super(Kernel2D, self).__init__(*args, **kwargs)
        self.preamble = PREAMBLE2D

    def local_shape(self):
        return (self.shape[0], 1, 1)

    def global_shape(self):
        return (self.shape[0] * self.shape[0], 1, 1)

class VisitResult:

    def __init__(self, name, input_args, statements, local_defs):
        "docstring"
        self.name = name
        self.input_args = input_args
        self.statements = statements
        self.local_defs = local_defs

    def __add__(self, other):
        new = VisitResult(self.name, {**self.local_defs, **other.local_defs}, self.local_defs + other.local_defs)
        
        


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
            stmts.append("float {};".format(local))

        outvar = name
        if callshape is None or callshape != node.shape:
            outvar = "output[i]"
        stmt = "{} = {} {} {};".format(outvar, left, op, right)
        # I've made this too complicated for myself
        kernel = CLKernel.factory(name, "\n".join(stmts + lstmts + rstmts +
                                                  [stmt]), {**lin, **rin},
                                  node.shape)
        if callshape is None or callshape != node.shape:
            self.kernels.append(kernel)
            return (name+"[i]", {name: kernel}, [], [])
        else:
            return (name, {**lin, **rin}, lstmts + rstmts + [stmt],
                    [name] + llocals + rlocals)


    def visit_UnaryFuncEx(self, node, callshape=None):
        stmt = '{} = {}({});'
        curr_visit = self.visits
        return None

    
        

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
            stmts.append("float {};".format(local))

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
        stmt = kernels.gemm.format(matrixA=left, matrixB=right, matrixC=outvar)

        d = {**lin, **rin}
        d["num_cols_A"] = node.arg1.array.shape[1]
        d["num_rows_A"] = node.arg1.array.shape[0]
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
        arg, input_args, stmts, mlocals = self.visit(node.arg,
                                                    callshape=node._inshape)
        decls = ["float {};".format(local) for local in mlocals]
        stmt = kernels.dot.format(arg)
        name = "input{}".format(curr_visit)
        input_args["localSums"] = cl.LocalMemory(
            node.shape[0] // REDUCTION_FACTOR
        )
        
        kernel = CLKernel.factory(name, "\n".join(decls + stmts + [stmt]), input_args,
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
            stmts.append("float {};".format(local))
        stmt = kernels.dot.format(left, right)
        name = "input{}".format(curr_visit)
        kernel = CLKernel.factory(name, "\n".join(stmts + [stmt]), d,
                                node._inshape, reducing=True)
        self.kernels.append(kernel)
        return (name+"[i]", {name: kernel})


def run_gpu(numpy_ex):
    trans = GPUEmitter()
    trans.walk(num.ReduceTransformer().walk(numpy_ex))
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    bufs = {}

    if trans.kernels == []:
        raise Exception("No kernels...")

    events = []
    for i, kernel in enumerate(trans.kernels):
        inputs = []
        for ref, source in kernel.inputs.items():
            if isinstance(source, np.ndarray) and ref not in bufs:
                bufs[ref] = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                      hostbuf=source)
            elif isinstance(source, int):
                bufs[ref] = np.uint32(source)
            elif isinstance(source, cl.LocalMemory):
                bufs[ref] = source
            # else:
            #     bufs[ref] = cl.Buffer(ctx, mf.READ_WRITE, first_arr.nbytes)
            inputs.append(bufs[ref])

        bufs[kernel.name] = cl.Buffer(ctx, mf.READ_WRITE, kernel.outbytes())
        last_kern = kernel
        kernel.prog = cl.Program(ctx, kernel.to_kern()).build()
        events.append(kernel.prog.foo(queue,
                                      kernel.global_shape(),
                                      kernel.local_shape(),
                                      *inputs,
                                      bufs[kernel.name]))

    res_np = np.empty(last_kern.outshape(), dtype=last_kern.dtype)
    cl.enqueue_copy(queue, res_np, bufs[last_kern.name], wait_for=events)
    return res_np
