from numpy import array, zeros, diag, diagflat, dot
import numpy.random as npr
from benchmarks.common import JACOBI_SIZE
def jacobi(A,b,N=25,x=None):
    """Solves the equation Ax=b via the Jacobi iterative method."""
    # Create an initial guess if needed                                                                                                                                                            
    if x is None:
        x = zeros(len(A[0]))

    # Create a vector of the diagonal elements of A                                                                                                                                                
    # and subtract them from A                                                                                                                                                                     
    D = diag(A)
    print(D.shape)
    print(diagflat(D).flags['C_CONTIGUOUS'])
    R = A - diagflat(D)

    # Iterate for N times                                                                                                                                                                          
    for i in range(N):
        x = (b - dot(R,x)) / D
    return x

def time_jacobi():
    A = npr.rand(JACOBI_SIZE, JACOBI_SIZE)
    b = npr.rand(JACOBI_SIZE, JACOBI_SIZE)
    guess = npr.rand(JACOBI_SIZE)

    sol = jacobi(A,b,N=25,x=guess)

    print("A:")
    print(A)

    print("b:")
    print(b)

    print("x:")
    print(sol)

if __name__=='__main__':
    time_jacobi()
