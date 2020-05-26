# NEEDS:
# average
# sum

#setup: import numpy as np ; N = 500000 ; X, Y = np.random.rand(N), np.random.rand(N)
#run: lstsqr(X, Y)
#from: http://nbviewer.ipython.org/github/rasbt/One-Python-benchmark-per-day/blob/master/ipython_nbs/day10_fortran_lstsqr.ipynb

#pythran export lstsqr(float[], float[])

from delayrepay import zeros, diag, diagflat
from delayrepay.random import rand
import numpy as np

def lstsqr(x, y):
    """ Computes the least-squares solution to a linear matrix equation. """
    x_avg = average(x)
    y_avg = average(y)
    dx = x - x_avg
    dy = y - y_avg
    var_x = sum(dx**2)
    cov_xy = sum(dx * (y - y_avg))
    slope = cov_xy / var_x
    y_interc = y_avg - slope*x_avg
    return (slope, y_interc)

N = 500000
X, Y = rand(N), rand(N)
lstsqr(X, Y)
