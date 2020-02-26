from delayarray import zeros, diag, diagflat
from numpy import dot
import delayarray.random as npr
from benchmarks.common import JACOBI_SIZE
def jacobi(A,b,N=25,x=None):
    """Solves the equation Ax=b via the Jacobi iterative method."""
    # Create an initial guess if needed                                                                                                                                                            
    if x is None:
        x = zeros(len(A[0]))

    # Create a vector of the diagonal elements of A                                                                                                                                                
    # and subtract them from A                                                                                                                                                                     
    D = diag(A)
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
