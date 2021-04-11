import numpy as np
from scipy.linalg import lu

'''Generating random band matrices from uniform distribution'''
def RandomBand(rownum, colnum, band=1, low=-1.0, high=1.0):
    A = np.zeros((rownum, colnum))
    B = np.random.uniform(low, high, size=(colnum, 2*band +1))
    
    for m in range(rownum):
        for n in range(colnum):
            if (n-m)>=0 and (n-m)<=band:
                A[m,n] = B[n, band + n-m]
            if (m-n)>0 and (m-n)<=band:
                A[m,n] = B[n, band + n-m]
    return(A)

'''
Testing the function:
RandomBand(10, 5)
RandomBand(10, 5, 2)
'''

'''Generating random toeplitz band matrices from uniform distribution'''
def RandomToeplitzBand (rownum, colnum, band=1, low = -1.0, high = 1.0):
    A = np.zeros((rownum, colnum))
    B1 = np.random.uniform(low, high, size=2*band + 1)
    B = np.ones((colnum, 2*band+1))
    
    for i in range(2*band + 1):
        B[:,i] = B1[i]
        
    for m in range(rownum):
        for n in range(colnum):
            if (n-m)>=0 and (n-m)<=band:
                A[m,n] = B[n, band + n-m]
            if (m-n)>0 and (m-n)<=band:
                A[m,n] = B[n, band + n-m]
    return(A)

'''
Testing the function:
GenRandBandMat(10, 5)
GenRandBandMatToep(10, 5, 2)
'''

'''Generate random matrix from uniform distribution'''
def FullUniformMatrix(rownum, colnum, low = -1.0, high = 1.0):
    B = np.random.uniform(low, high, size=(rownum, colnum))
    return B

# FullUniformMatrix(5,4).shape

# Finds the 'PLU' based sparse generalized inverse of matrix
def PLUginv(A):
    m, n = A.shape # dimensions
    r = np.linalg.matrix_rank(A) # rank
    
    # if m>n, p= (m x m), l=(m x n), u=(n x n) with entries in (r x r) submatrix
    p, l, u = lu(A)
    
    # Transform l to (m x m) matrix
    L = np.identity(m)
    for i in range(m):
        for j in range(n):
            L[i,j] = l[i,j]
    
    # Transform u to (m x n) matrix
    U = np.zeros((m,n))
    for i in range(n):
        for j in range(n):
            U[i,j] = u[i,j]
    
    # P remains the same.
    P = p
    
    E = np.linalg.inv(P@L)
    u_inv = np.linalg.inv(u[:r,:r])
    B = np.zeros((n,m))
    for i in range(r):
        for j in range(r):
            B[i,j] = u_inv[i,j]
    
    return B@E

eps = np.finfo(float).eps

def regularize(matrix, atol=eps):
    matrix[np.abs(matrix)<atol]=0
    return matrix

def maxdiag(matrix):
    rows, cols = matrix.shape

    max_array = np.full(rows+cols-1, -np.inf)
    
    for i in range(rows):
        for j in range(cols):
            if matrix[i,j]>= max_array[-i+j+(rows-1)]:
                max_array[-i+j+(rows-1)] = matrix[i,j]
    
    return max_array