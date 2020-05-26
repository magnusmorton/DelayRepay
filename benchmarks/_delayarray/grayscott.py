# DelayArray is not subscriptable: u, v = U[1:-1,1:-1], V[1:-1,1:-1]

#from http://stackoverflow.com/questions/26823312/numba-or-cython-acceleration-in-reaction-diffusion-algorithm
#setup: pass
#run: grayscott(40, 0.16, 0.08, 0.04, 0.06)

#pythran export grayscott(int, float, float, float, float)
from delayrepay import zeros, diag, diagflat
from delayrepay.random import random
import numpy as np
def grayscott(counts, Du, Dv, F, k):
    n = 300
    U = zeros((n+2,n+2), dtype=np.float32)
    V = zeros((n+2,n+2), dtype=np.float32)
    u, v = U[1:-1,1:-1], V[1:-1,1:-1]

    r = 20
    u[:] = 1.0
    U[n//2-r:n//2+r,n//2-r:n//2+r] = 0.50
    V[n//2-r:n//2+r,n//2-r:n//2+r] = 0.25
    u += 0.15*random((n,n))
    v += 0.15*random((n,n))

    for i in range(counts):
        Lu = (                 U[0:-2,1:-1] +
              U[1:-1,0:-2] - 4*U[1:-1,1:-1] + U[1:-1,2:] +
                               U[2:  ,1:-1] )
        Lv = (                 V[0:-2,1:-1] +
              V[1:-1,0:-2] - 4*V[1:-1,1:-1] + V[1:-1,2:] +
                               V[2:  ,1:-1] )
        uvv = u*v*v
        u += Du*Lu - uvv + F*(1 - u)
        v += Dv*Lv + uvv - (F + k)*v

    return V

grayscott(40, 0.16, 0.08, 0.04, 0.06)
