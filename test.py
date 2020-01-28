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

    def test_var_add(self):
        arr = ones(5000).astype(np.float32)
        np_arr = np.ones(5000).astype(np.float32)
        a = 7
        res = a * arr
        npt.assert_array_almost_equal(res, np_arr * 7)
                


if __name__ == '__main__':
    # logging.basicConfig( level=logging.DEBUG)
    unittest.main()
    
