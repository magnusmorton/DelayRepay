import logging
from dataclasses import dataclass
from numbers import Number
from typing import List

import num
import pyopencl as cl
import numpy as np

logger = logging.getLogger("delayRepay.cl")



@dataclass
class CLTree:
    '''CL tree node'''

@dataclass
class Expression(CLTree):
    '''Expression'''

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



    
class CLEmitter(num.Visitor):
    
    preamble = "int i = get_global_id(0);"

    def visit_BinaryExpression(self, node):
        return "{} {} {}".format(self.visit(node.left), node.op, self.visit(node.right))


    def visit_Subscript(self, node):
        return "{}[{}]".format(self.visit(node.var), self.visit(node.sub))

    def visit_Scalar(self, node):
        return str(node.val)

    def visit_Var(self, node):
        return node.name

    def visit_Assignment(self, node):
        return "{} = {};".format(self.visit(node.left), self.visit(node.right))

    def visit_CLArgs(self, node):
        args = ["__global {} {}".format(typ, var) for typ, var in zip(node.types, self.visit(node.params)) ]
        return ", ".join(args)

    def visit_CLFunction(self, node):
        return "__kernel void {} ({}) {{\n{}\n{}\n}}".format(node.name, self.visit(node.args), self.preamble,  "\n".join(self.visit(node.body)))

class CLVarVisitor(num.Visitor):
    pass


class GPUTransformer(num.NumpyVisitor):

    def __init__(self):
        self.ins = {}
        self.outs = []

    def visit_BinaryNumpyEx(self, node):
        cur_visits = self.visits
        ex = BinaryExpression(node.to_op(), self.visit(node.left), self.visit(node.right))
        if cur_visits == 1:
            ex = Assignment(Subscript(Var("foo"), Var("i")), ex)
            self.outs.append(Var("foo"))
        return ex

    def visit_NPArray(self, node):
        var = Var('a')
        self.ins[var] = node.array
        return Subscript(var, Var('i'))

    def visit_Scalar(self, node):
        return Scalar(node.val)


def function_coverter(tree: CLTree):
    pass
    


def executor(kernel, in_arr, out_arr):
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=in_arr)
    prog = cl.Program(ctx,kernel).build()
    res_g = cl.Buffer(ctx, mf.WRITE_ONLY, in_arr.nbytes)
    prog.gfunc(queue, in_arr.shape, None, a_g, res_g)
    res_np = np.empty_like(in_arr)
    cl.enqueue_copy(queue, res_np, res_g)
    return res_np
    

def run_gpu(numpy_ex):
    transformer =  GPUTransformer()
    gpu_ex = transformer.walk(numpy_ex)
    args = CLArgs(list(transformer.ins.keys()) + transformer.outs, ["float*", "float*"])
    func = CLFunction(args, "gfunc", [gpu_ex])
    kernel = CLEmitter().visit(func)
    return executor(kernel, list(transformer.ins.values())[0], None)
