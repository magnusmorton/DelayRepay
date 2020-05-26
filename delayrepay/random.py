import cupy.random
#from numpy.random import *
from .cuarray import cast

rand = cast(cupy.random.rand)
randn = cast(cupy.random.randn)
random = cast(cupy.random.random)
