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


if __name__ == '__main__':
    unittest.main()