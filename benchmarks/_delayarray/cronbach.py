# NEEDS:
# var (numpy variance)
# sum

#from: http://stackoverflow.com/questions/20799403/improving-performance-of-cronbach-alpha-code-python-numpy
#pythran export cronbach(float [][])
#setup: import numpy as np ; np.random.seed(0); N = 600 ; items = np.random.rand(N,N)
#run: cronbach(items)

from delayarray import zeros, diag, diagflat
from delayarray.random import rand
import numpy as np

def cronbach(itemscores):
    itemvars = itemscores.var(axis=1, ddof=1)
    tscores = itemscores.sum(axis=0)
    nitems = len(itemscores)
    return nitems / (nitems-1) * (1 - itemvars.sum() / tscores.var(ddof=1))

#np.random.seed(0)
N = 600
items = rand(N,N)
cronbach(items)
