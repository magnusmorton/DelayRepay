'''Test simple cl examples'''
import unittest
import logging
from array import ones, array, full,  ReduceTransformer
import numpy as np
import numpy.testing as npt

SIZE = 10240
class TestElWise(unittest.TestCase):
    # pylint: disable=C
    
    def setUp(self):
        self.arr = ones(SIZE).astype(np.float32)
        self.np_arr = np.ones(SIZE).astype(np.float32)

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

    def setUp(self):
        self.arr = full((SIZE,), 7).astype(np.float32)
        self.arr2 = full((SIZE,), 3).astype(np.float32)
        self.np_arr = np.full((SIZE,), 7).astype(np.float32)
        self.np_arr2 = np.full((SIZE,), 3).astype(np.float32)


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


    def test_matmul(self):
        res = self.arr @ self.arr2
        test = self.np_arr @ self.np_arr2
        self.assertEqual(res, test)



class TestMatrix(unittest.TestCase):

    def setUp(self):
        self.mat = full((SIZE,SIZE), 7).astype(np.float32)
        self.vec = full((SIZE,), 3).astype(np.float32)
        self.np_mat = np.full((SIZE,SIZE), 7).astype(np.float32)
        self.np_vec = np.full((SIZE,), 3).astype(np.float32)

    def test_scalar_mul(self):
        res = self.mat * 3
        npt.assert_array_almost_equal(res, self.np_mat * 3)

    def test_matvec(self):
        res = self.mat @ self.vec
        test =  self.np_mat @ self.np_vec
        print(res)
        print(test)
        
if __name__ == '__main__':
    unittest.main()
