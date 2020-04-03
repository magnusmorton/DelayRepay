import num
import cupy


class CupyEmitter(num.NumpyVisitor):

    def __init__(self):
        self.ins = {}
        self.outs = []
        self.kernels = []

    def visit_BinaryFuncEx(self, node):
        pass


    def visit_NPArray(self, node):
        pass
