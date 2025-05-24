#==============================================================
# Problem 3.8: Test Our Algorithms on Other Relevant Problems
#==============================================================
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
import random
from prob3_5 import LPippd

# print_n options: n = 5, 10, 100, 1000 
# Errors can sometimes occur: if this occurs, just rerun
print_n = 10
#==============================================================
# 3.8.1: Using LPippd on random problems of different sizes
#==============================================================
# First we make a function to produce random problems of the form LPippd accepts:
#
#       o--------------------o
#       | min   g.T @ x      |  
#       |  x                 |  
#       | s.t.  A.T @ x  = b | 
#       |             x >= 0 | 
#       o--------------------o
#
#==============================================================
# The following function makes random feasible LP systems by building them from the KKT conditions.
# The function itself is based on the one we used for 1.4. It accepts:
#
#       n: the number of x variables
#    beta: a parameter which gives m by m = beta * n
# density: a parameter giving the proportion of A which is nonzero
#
# The first thing we do is build A of shape (n,m) such that A.T has full row rank. This means we do not have repeat equality constraints.
#
# The next thing we do is define x0 to be a random position vector (x >= 0)
# We do the same with the Lagrange multipliers for the equality constraint y0, and for the bound constraints s0 (dual feasibility)
#
# We then define b = A.T @ x0 (primal feasibility) and g = A @ y0 + s0 (stationarity)
#
# This almost always generates a feasible LP system.
def random_lp_strict(n, beta=0.5,density=0.15): 
        m = int(np.round(beta * n))
        A = sp.sparse.random(n, m, density=density, format='csr', data_rvs=np.random.randn).toarray()

        it_count=0
        while np.linalg.matrix_rank(A.T) != A.T.shape[0]:
            A = sp.sparse.random(n, m, density=density, format='csr', data_rvs=np.random.randn).toarray()
            it_count+=1
            if it_count>10000: raise ValueError("###---Too Many Iterations---###")

        x0 = np.abs(np.random.randn(n,1)) # x >= 0
        y0 = np.abs(np.random.randn(m,1)) 
        s0 = np.abs(np.random.randn(n,1)) # dual feasibility

        b = A.T @ x0                      # primal feasibility
        g = A@y0 + s0                     # stationarity
        
        return g, A, b, x0, y0, s0
#==============================================================
# First we test our solver on a single generated problem
g, A, b, x0, y0, s0 = random_lp_strict(5,0.5,0.15)
x, mu, la, ResidualsArray, it_count, cpu_time = LPippd(g,A,b)
sol = sp.optimize.linprog(c=g.flatten(),A_eq=A.T,b_eq=b.flatten(),method='highs')
sp_x = np.array(sol.x) 
if print_n == 5:
    print("\n---------- Testing on 1 Problem of Size n = 5 ----------")
    print("Real x:",sp_x.flatten())
    print("Our x:",x.flatten())
    print("Euclidean Distance Between Solutions:",np.linalg.norm(x.flatten()-sp_x.flatten(),ord=2))
    print("Real Cost:", (g.T@sp_x)[0])
    print("Our Cost:", (g.T@x)[0,0])

#==============================================================
# Here we test on many test problems with 10 x variables
# This sometimes errors for reasons I am not certain about.
# This behavior occasionally happens even after setting the random seed.
# If it errors, just run it again and it should work.
random.seed(1)
if print_n == 10:
    num=1000
    size = 10
    # Initialize vectors
    EucDistVec = np.ones(num)
    CostDifVec = np.ones(num)
    ItVec      = np.ones(num)
    CPUTimeVec = np.ones(num)
    # For loop
    total_t = time.process_time_ns()
    for i in range(num):
        g, A, b, x0, y0, s0 = random_lp_strict(size,0.5,0.15)
        sol = sp.optimize.linprog(c=g.flatten(),A_eq=A.T,b_eq=b.flatten(),method='highs')
        sp_x = np.array(sol.x).T 
        x, mu, la, ResidualsArray, it_count, cpu_time = LPippd(g,A,b)
        EucDistVec[i] = np.linalg.norm(x.flatten()-sp_x.flatten(),ord=2)
        CostDifVec[i] = np.abs((g.T@sp_x)[0]-(g.T@x)[0,0])
        ItVec[i]      = it_count
        CPUTimeVec[i] = cpu_time
    total_t = time.process_time_ns() - total_t  

    print("\n---------- Testing on",len(EucDistVec),"Problems of Size n =",size,"----------")
    print("Mean Euclidean Distance:",np.mean(EucDistVec))
    print("Mean Cost Difference:", np.mean(CostDifVec))
    print("Mean Number of LPippd Iterations:",np.mean(ItVec))
    print("Mean LPippd CPU Time (Seconds):",np.mean(CPUTimeVec)/1000000000)
    print("Total CPU Time (Seconds):",total_t/1000000000)
#==============================================================
# Here we test on many test problems with 100 x variables
# This sometimes errors for reasons I am not certain about.
# This behavior occasionally happens even after setting the random seed.
# If it errors, just run it again and it should work.
random.seed(1)
if print_n == 100:
    num=500
    size = 100
    # Initialize vectors
    EucDistVec = np.ones(num)
    CostDifVec = np.ones(num)
    ItVec      = np.ones(num)
    CPUTimeVec = np.ones(num)
    # For loop
    total_t = time.process_time_ns()
    for i in range(num):
        g, A, b, x0, y0, s0 = random_lp_strict(size,0.5,0.15)
        sol = sp.optimize.linprog(c=g.flatten(),A_eq=A.T,b_eq=b.flatten(),method='highs')
        sp_x = np.array(sol.x).T 
        x, mu, la, ResidualsArray, it_count, cpu_time = LPippd(g,A,b)
        EucDistVec[i] = np.linalg.norm(x.flatten()-sp_x.flatten(),ord=2)
        CostDifVec[i] = np.abs((g.T@sp_x)[0]-(g.T@x)[0,0])
        ItVec[i]      = it_count
        CPUTimeVec[i] = cpu_time
    total_t = time.process_time_ns() - total_t  

    print("\n---------- Testing on",len(EucDistVec),"Problems of Size n =",size,"----------")
    print("Mean Euclidean Distance:",np.mean(EucDistVec))
    print("Mean Cost Difference:", np.mean(CostDifVec))
    print("Mean Number of LPippd Iterations:",np.mean(ItVec))
    print("Mean LPippd CPU Time (Seconds):",np.mean(CPUTimeVec)/1000000000)
    print("Total CPU Time (Seconds):",total_t/1000000000)

#==============================================================
# Here we test on a couple test problems with 1000 x variables
# This sometimes errors for reasons I am not certain about.
# This behavior occasionally happens even after setting the random seed.
# If it errors, just run it again and it should work.
random.seed(1)
if print_n == 1000:
    num=5
    size = 1000
    # Initialize vectors
    EucDistVec = np.ones(num)
    CostDifVec = np.ones(num)
    ItVec      = np.ones(num)
    CPUTimeVec = np.ones(num)
    # For loop
    total_t = time.process_time_ns()
    for i in range(num):
        g, A, b, x0, y0, s0 = random_lp_strict(size,0.5,0.15)
        sol = sp.optimize.linprog(c=g.flatten(),A_eq=A.T,b_eq=b.flatten(),method='highs')
        sp_x = np.array(sol.x).T 
        x, mu, la, ResidualsArray, it_count, cpu_time = LPippd(g,A,b)
        EucDistVec[i] = np.linalg.norm(x.flatten()-sp_x.flatten(),ord=2)
        CostDifVec[i] = np.abs((g.T@sp_x)[0]-(g.T@x)[0,0])
        ItVec[i]      = it_count
        CPUTimeVec[i] = cpu_time
    total_t = time.process_time_ns() - total_t  

    print("\n---------- Testing on",len(EucDistVec),"Problems of Size n =",size,"----------")
    print("Mean Euclidean Distance:",np.mean(EucDistVec))
    print("Mean Cost Difference:", np.mean(CostDifVec))
    print("Mean Number of LPippd Iterations:",np.mean(ItVec))
    print("Mean LPippd CPU Time (Seconds):",np.mean(CPUTimeVec)/1000000000)
    print("Total CPU Time (Seconds):",total_t/1000000000)

