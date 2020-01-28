import unittest
import logging
import numpy as np
import numpy.testing as npt
from array import *

class TestElWise(unittest.TestCase):
    def setUp(self):
        self.arr = ones(5000).astype(np.float32)
        self.np_arr = np.ones(5000).astype(np.float32)

    def test_scalar_add(self):
        res = self.arr + 1
        npt.assert_array_almost_equal(res, self.np_arr + 1)

    def test_scalar_mul(self):
        res = self.arr * 3
        npt.assert_array_almost_equal(res, self.np_arr * 3)

    def test_var_add(self):
        a = 7
        res = a * self.arr
        npt.assert_array_almost_equal(res, self.np_arr * 7)

if __name__ == '__main__':
    # logging.basicConfig( level=logging.DEBUG)
    unittest.main()
    
