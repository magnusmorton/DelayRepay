# NEEDS:
# exp, sqrt, log, sum

#setup: import numpy as np ; N = 100000 ; a = np.random.random(N); b = 0.1; c =1.1
#run: log_likelihood(a, b, c)
#from: http://arogozhnikov.github.io/2015/09/08/SpeedBenchmarks.html
from delayarray import zeros, diag, diagflat
from delayarray.random import random
import numpy as np

#pythran export log_likelihood(float64[], float64, float64)
def log_likelihood(data, mean, sigma):
    s = (data - mean) ** 2 / (2 * (sigma ** 2))
    pdfs = exp(- s)
    pdfs /= sqrt(2 * np.pi) * sigma
    return log(pdfs).sum()

N = 100000
a = np.random.random(N)
b = 0.1
c = 1.1
log_likelihood(a,b,c)
