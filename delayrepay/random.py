import cupy.random
#from numpy.random import *
from .cuarray import cast

rand = cast(cupy.random.rand)
randn = cast(cupy.random.randn)
random = cast(cupy.random.random)
seed = cupy.random.seed
randint = cast(cupy.random.randint)
choice = cast(cupy.random.choice)
