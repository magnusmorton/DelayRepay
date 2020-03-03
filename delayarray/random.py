import numpy.random
#from numpy.random import *
from .array import cast

rand = cast(numpy.random.rand)
randn = cast(numpy.random.randn)
random = cast(numpy.random.random)
