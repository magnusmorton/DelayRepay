'''Test simple cl examples'''
import unittest
import logging
from array import ones, array, full,  ReduceTransformer
import numpy as np
import numpy.testing as npt

class TestElWise(unittest.TestCase):
    # pylint: disable=C
    
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
        def axpy(a, x, y):
            return a*x + y
        res = axpy(8, self.arr, 9)
        npt.assert_array_almost_equal(res, axpy(8, self.np_arr, 9))

    def test_ir(self):
        res = self.arr + 3
        assert(res)


class TestVector(unittest.TestCase):
    # pylint: disable=C
    SIZE = 10240

    def setUp(self):
        self.arr = full((self.SIZE,), 7).astype(np.float32)
        self.arr2 = full((self.SIZE,), 3).astype(np.float32)
        self.np_arr = np.full((self.SIZE,), 7).astype(np.float32)
        self.np_arr2 = np.full((self.SIZE,), 3).astype(np.float32)


    def test_vecadd(self):
        res = self.arr + self.arr2
        npt.assert_array_almost_equal(res, self.np_arr + self.np_arr2)

    def test_vecmul(self):
        res = self.arr * self.arr2
        npt.assert_array_almost_equal(res, self.np_arr * self.np_arr2)

    def test_dot(self):
        res = self.arr.dot(self.arr2)
        test = self.np_arr.dot(self.np_arr2)
        self.assertEqual(res, test)


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    unittest.main()
