import unittest
import logging

from array import *

# class TestScalarAdd(unittest.TestCase):
#     def setUp(self):
#         self.arr = array([1,2,3])
#         self.res = self.arr + 1
#         self.func = NumpyFunction(self.res.ex)

#     def test_e2e(self):
#         print(self.func)
#         exec(str(self.func), globals())
#         print(jitfunc)
#         print(self.func())
#         print(self.res)


#     def test_numpy_function(self):
#         print(self.res.ex)
#         print(self.func)

#     def test_gpu_function_def(self):
#         print(self.func._gpu)

arr = array([1.0,2.0,3.0])
arr = ones(50000)
res = arr + 1.0
import cl
import num

trans = cl.GPUTransformer()
res = trans.walk(res.ex)
print(res)
print(trans.ins)
print(trans.outs)

args = cl.CLArgs(list(trans.ins.keys()) + trans.outs, ["float*", "float*"])

fun = cl.CLFunction(args, "gfunc", [res])
print(fun)
emit = cl.CLEmitter()
kern = emit.visit(fun)
print(kern)

print(cl.executor(kern, arr._ndarray, None))
# if __name__ == '__main__':
#     # logging.basicConfig( level=logging.DEBUG)
#     # unittest.main()
#     pass
