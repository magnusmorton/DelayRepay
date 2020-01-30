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

    def test_axpy(self):
        def axpy(a,x,y):
            return a*x + y
        res = axpy(8, self.arr, 9)
        npt.assert_array_almost_equal(res, axpy(8, self.np_arr, 9))


class TestVector(unittest.TestCase):
    def test_dot(self):
        arr = array([1,2,3])
        np_arr = np.array([1,2,3])
        res = arr.dot(arr)
        npt.assert_array_almost_equal(res, np_arr.dot(np_arr))

if __name__ == '__main__':
    logging.basicConfig( level=logging.DEBUG)
    unittest.main()
    
