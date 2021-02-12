import unittest
import os

os.environ['DELAY_LIFT'] = '1'

import delayrepay as dr
import numpy as np


class TestLift(unittest.TestCase):
    def setUp(self):
        pass

    def test_pp(self):
        arr = dr.ones(32)
        print(arr * 3)

    def test_axpy_pp(self):
        arr = dr.ones(32)
        print(arr*3 + 2)

    def test_two_arrs(self):
        arr1 = dr.ones(32)
        arr2 = dr.full((32,),3)
        print(arr1 + arr2)


if __name__ == '__main__':
    unittest.main()
