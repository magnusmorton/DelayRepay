import unittest
import logging

from array import *

class TestScalarAdd(unittest.TestCase):
    def setUp(self):
        self.arr = array([1,2,3])
        self.res = self.arr + 1
        self.func = NumpyFunction(self.res.ex)

    def test_e2e(self):
        print(self.func)
        exec(str(self.func), globals())
        print(jitfunc)
        print(self.func())
        print(self.res)


    def test_numpy_function(self):
        print(self.res.ex)
        print(self.func)


if __name__ == '__main__':
    # logging.basicConfig( level=logging.DEBUG)
    unittest.main()
