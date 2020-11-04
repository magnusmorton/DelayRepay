'''Test simple cl examples'''
import unittest
from delayrepay import ones, full, NPArray, array, sum
import numpy as np
import numpy.testing as npt

SIZE = 64

def assertEqF(one, two):
    return abs(one-two) < 0.001

class TestElwise(unittest.TestCase):
    # pylint: disable=C

    def setUp(self):
        self.arr = ones(SIZE)
        self.np_arr = np.ones(SIZE)

    def test_scalar_add(self):
        res = self.arr + 1
        npt.assert_array_almost_equal(res.get(), self.np_arr + 1)

    def test_scalar_mul(self):
        res = self.arr * 3
        npt.assert_array_almost_equal(res.get(), self.np_arr * 3)

    def test_var_add(self):
        a = 7
        res = a * self.arr
        npt.assert_array_almost_equal(res.get(), self.np_arr * 7)

    def test_axpy(self):
        def axpy(a, x, y):
            return a*x + y
        res = axpy(8, self.arr, 9)
        npt.assert_array_almost_equal(res.get(), axpy(8, self.np_arr, 9))

    def test_regression(self):
        def fun(mat):
            return mat + mat * 3 + 9

        res = fun(self.arr)
        npt.assert_array_almost_equal(res.get(), fun(self.np_arr))

    def test_ir(self):
        res = self.arr + 3
        assert(res)

    def test_cos(self):
        res = np.cos(self.arr)
        npt.assert_array_almost_equal(res.get(), np.cos(self.np_arr))

    def test_exp(self):
        res = self.arr ** 2
        npt.assert_array_almost_equal(res.get(), self.np_arr ** 2)

    def test_exp32(self):
        arr = self.arr.astype(np.float32)
        res = arr ** 2
        npt.assert_array_almost_equal(res.get(), self.np_arr.astype(np.float32) ** 2)

    def test_fuse_bench(self):
        res = np.sin(self.arr) ** 2 + np.cos(self.arr) ** 2
        resn = np.sin(self.np_arr) ** 2 + np.cos(self.np_arr) ** 2
        npt.assert_array_almost_equal(res.get(), resn)



class TestVector(unittest.TestCase):
    # pylint: disable=C

    def setUp(self):
        self.arr = full((SIZE,), 7).astype(np.float32)
        self.arr2 = full((SIZE,), 3).astype(np.float32)
        self.np_arr = np.full((SIZE,), 7).astype(np.float32)
        self.np_arr2 = np.full((SIZE,), 3).astype(np.float32)

    def test_vecadd(self):
        res = self.arr + self.arr2
        npt.assert_array_almost_equal(res.get(), self.np_arr + self.np_arr2)

    def test_vecmul(self):
        res = self.arr * self.arr2
        npt.assert_array_almost_equal(res.get(), self.np_arr * self.np_arr2)

    def test_dot_method(self):
        res = self.arr.dot(self.arr2)
        test = self.np_arr.dot(self.np_arr2)
        assertEqF(res, test)

    def test_dot_func(self):
        res = np.dot(self.arr, self.arr2)
        test = self.np_arr.dot(self.np_arr2)
        assertEqF(res, test)

    def test_matmul(self):
        res = self.arr @ self.arr2
        test = self.np_arr @ self.np_arr2
        assertEqF(res, test)

    def test_sum(self):
        res = sum(self.arr)
        test = np.sum(self.np_arr)
        print(res)
        self.assertEqual(res, test)

    def test_atan2(self):
        res = np.arctan2(self.arr, self.arr2)
        test = np.arctan2(self.np_arr, self.np_arr2)
        npt.assert_array_almost_equal(res.get(), test)


class TestMatrix(unittest.TestCase):

    def setUp(self):
        self.mat = full((SIZE, SIZE), 7).astype(np.float32)
        self.vec = full((SIZE,), 3).astype(np.float32)
        self.np_mat = np.full((SIZE, SIZE), 7).astype(np.float32)
        self.np_vec = np.full((SIZE,), 3).astype(np.float32)

    def test_scalar_mul(self):
        res = self.mat * 3
        npt.assert_array_almost_equal(res.get(), self.np_mat * 3)

    def test_matvec(self):
        res = self.mat @ self.vec
        print(res)
        test = self.np_mat @ self.np_vec
        npt.assert_array_almost_equal(res.get(), test)

    def test_kuba(self):
        a = full((64, 64), 10.0, dtype=np.float32)
        b = full((64,), 2.0, dtype=np.float32)
        an = np.full((64, 64), 10.0, dtype=np.float32)
        bn = np.full((64,), 2.0, dtype=np.float32)
        npt.assert_array_almost_equal((a @ b).get(), an @ bn)

    def test_gemm(self):
        res = self.mat @ self.mat
        npt.assert_array_almost_equal(res.get(), self.np_mat @ self.np_mat)


class TestMeta(unittest.TestCase):

    def test_memoise(self):
        arr = np.array([1, 2, 3])
        ar1 = NPArray(arr)
        ar2 = NPArray(arr)
        self.assertIs(ar1, ar2)

    def test_no_memoize(self):
        arr = full((3,), 5).astype(np.float32)
        arr2 = full((3,), 3).astype(np.float32)
        self.assertIsNot(arr, arr2)

    def test_memo_ex(self):
       arr = array([1, 2, 3])
       ex1 = np.sin(arr)
       print(id(ex1))
       ex2 = np.sin(arr)
       print(id(ex2))
       self.assertIs(ex1, ex2)

    
if __name__ == '__main__':
    unittest.main()
