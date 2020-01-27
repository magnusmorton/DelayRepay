import unittest
import logging
import numpy as np
import numpy.testing as npt
from array import *

class TestElWise(unittest.TestCase):
    # def setUp(self):
    #     self.arr = array([1,2,3])
    #     self.res = self.arr + 1
    #     self.func = NumpyFunction(self.res.ex)

    def test_scalar_add(self):
        arr = ones(5000).astype(np.float32)
        np_arr = np.ones(5000).astype(np.float32)
        res = arr + 1
        npt.assert_array_almost_equal(res, np_arr + 1)

    def test_scalar_mul(self):
        arr = ones(5000).astype(np.float32)
        np_arr = np.ones(5000).astype(np.float32)
        res = arr * 3
        npt.assert_array_almost_equal(res, np_arr * 3)

# arr = ones(5000000).astype(np.float32)
# res = arr + 1
# import cl
# import num

# # trans = cl.GPUTransformer()
# # res = trans.walk(res.ex)
# # # print(res)
# # # print(trans.ins)
# # # print(trans.outs)

# # args = cl.CLArgs(list(trans.ins.keys()) + trans.outs, ["float*", "float*"])

# # fun = cl.CLFunction(args, "gfunc", [res])
# # # print(fun)
# # emit = cl.CLEmitter()
# # kern = emit.visit(fun)
# # # print(kern)

# # print(cl.run_gpu(res.ex))
# print(res)
if __name__ == '__main__':
    # logging.basicConfig( level=logging.DEBUG)
    unittest.main()
    
