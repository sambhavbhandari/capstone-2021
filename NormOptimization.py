import numpy as np
#import Mat as mat

import os
import sys

import cvxpy as cp
import mosek

'''
L_p optimizing code.
'''
def get_Lp(A, lp=1, condns = [1], solver = cp.MOSEK):
    m = A.shape[0]
    n = A.shape[1]

    I = np.identity(n)

    # variable
    x = cp.Variable((n,m))

    # objective function
    objective = cp.Minimize(cp.pnorm(x,p=lp))

    # constraints = Left inverse
    PenroseConditions = {1:[A@x@A==A], 2:[x@A@x ==x], 3:[(A@x).T==A@x],4:[(x@A).T==x@A]}
    constraints = []
    for i in condns:
        constraints+=PenroseConditions[i]

    # problem
    prob = cp.Problem(objective, constraints)

    # Optimize
    prob.solve(solver=solver)

    return x, prob

'''
Weight functions
'''
# Polynomial weighted function
def poly_weighted(m,n,pw=2):
    A = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            p = i+1
            q = j+1
            A[i,j] = (1+abs(p-q))**pw
    return A

# Exponential weighted function
def exp_weighted(m,n):
    A = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            p = i+1
            q = j+1
            A[i,j] = np.exp(abs(p-q))
    return A

# Big number weighted
def large_weighted(m,n, bandwidth, penalty=100):
    A = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            p = i+1
            q = j+1
            if abs(p-q)<= bandwidth:
                A[i,j] = 1
            else:
                A[i,j] = penalty
    return A


'''
Weighted norms
'''
def get_weighted_norm(A, weight_type='poly', pw=2, bandwidth=1, penalty=100, lp=2, solver=cp.MOSEK):
    m = A.shape[0]
    n = A.shape[1]

    I = np.identity(n)

    x = cp.Variable((n,m))

    if weight_type=="poly":
        weight = poly_weighted(n,m,pw)
    elif weight_type=="exp":
        weight = exp_weighted(n,m)
    else:
        weight = large_weighted(n,m, bandwidth, penalty)

    # objective function
    objective = cp.Minimize(cp.pnorm(cp.multiply(weight,x),p=lp))

    # constraints
    constraints = [A@x@A==A] # generalized inverse constraint

    prob = cp.Problem(objective, constraints)

    prob.solve(solver=solver)

    return x, prob
