#==============================================================
# Problem 2.9: 
#==============================================================
#Imports
import numpy as np
import scipy as sp
import casadi as ca
from casadi import *
import matplotlib.pyplot as plt
import time
from prob2_8 import InequalityQPSolver, standard_NoA_IP
from prob1 import run_Casadi

#The run options are "single" for one problem, or "many" for testing many different n values.
# Please note that running the script on "many" mode will take several minutes to run.
run = "many"
#==============================================================
# Problem 2.9.1: Using InequalityQPSolver on random problems of different sizes
#==============================================================
# First we make a function to make random inequality constrained quadratic programs of the form:
# 
#      o-------------------------------------o
#      | min (0.5 * x.T @ H @ x) + (g.T @ x) |
#      |  x                                  |
#      | s.t.     dl <= C.T @ x <= du        |
#      |              l <= x <= u            |
#      o-------------------------------------o
#
#==============================================================
# The following function is a modification of "random_qp_strict" from 1.4 which produces inequality constrained QPs.
# 
# Just like "random_qp_strict," our function accepts 4 parameters:
#
#       n: the number of x variables (positive integer)
#   alpha: a regularization parameter ensuring H is positive definite (positive real number)
#    beta: a parameter which gives m by m = beta * n (real number between 0 and 1)
# density: a parameter giving the proportion of A which is nonzero (real number between 0 and 1)
#
# The function produces H, g, C, dl, du, l, and u
def random_inequality_qp_strict(n, alpha=1, beta=0.5, density=0.15):
        m = int(np.round(beta * n))
        C = sp.sparse.random(n, m, density=density, format='csr', data_rvs=np.random.randn).toarray()

        it_count=0
        while np.linalg.matrix_rank(C.T) != C.T.shape[0]:
            C = sp.sparse.random(n, m, density=density, format='csr', data_rvs=np.random.randn).toarray()
            it_count+=1
            if it_count>10000: raise ValueError("###---Too Many Iterations---###")

        x0 = np.random.randn(n,1)
        ya0 = np.abs(np.random.randn(m,1))            #dual feasibility: 
        yb0 = np.abs(np.random.randn(m,1))
        yc0 = np.abs(np.random.randn(n,1))
        yd0 = np.abs(np.random.randn(n,1))

        dl = C.T @ x0 - np.abs(np.random.randn(m,1))  #primal feasibility
        du = C.T @ x0 + np.abs(np.random.randn(m,1))  
        l = x0 - np.abs(np.random.randn(n,1))
        u = x0 + np.abs(np.random.randn(n,1))

        M = sp.sparse.random(n, n, density=density, format='csr', data_rvs=np.random.randn)
        H = (M @ M.T).toarray() + (alpha * np.eye(n))

        g = -(H@x0 + C@(ya0-yb0)+yc0-yd0)             #stationarity

        return H, g, C, dl, du, l, u
#==============================================================
# We then use the function to make a random QP
if run == "single":
    n=10
    H, g, Cin, dlin, duin, lin, uin = random_inequality_qp_strict(n)
    # We can then use the function "standard_NoA_ip" from 2.8 to convert the inequality constrained problem to standard form.
    C, d = standard_NoA_IP(Cin, dlin, duin, lin, uin)
    # We then solve the problem with both "InequalityQPSolver" and CasADi
    xout, zout, sout, iter, res, cputime = InequalityQPSolver(H,g,C,d,eps = 0.001)
    xtrue = np.array(run_Casadi(H, g, Cin, dlin, duin, lin, uin))
    # Some statistics on the performance of the solver:
    print("\n---------- Testing on 1 Problem of Size n =",n,"----------")
    print("Real x:",xtrue.flatten())
    print("Our x:",xout.flatten())
    print("Euclidean Distance Between Solutions:",np.linalg.norm(xout.flatten()-xtrue.flatten(),ord=2))
    print("Real Cost:", (g.T@xtrue)[0,0])
    print("Our Cost:", (g.T@xout)[0,0])
    print("Cost Difference:",(g.T@xtrue)[0,0]-(g.T@xout)[0,0])
#==============================================================
# We next decided to examine the performance of the solver as a function of n
# Please note that this will take several minutes to run:
if run == "many":
    #setting parameters:
    eps = 0.001 # we fix epsilon
    n_list = np.arange(5,201)
    #preparing the arrays and labels for the graphs
    Err_labels = ["Euclidean Distance","Cost Difference"]
    Dist_and_Cost_array = np.zeros((len(n_list),2))

    Res_labels =["rL","rC","mu"]
    Res_array = np.zeros((len(n_list),3))

    Iter_labels = ["Iterations","CPU Time","CPU Time Per Iteration"]
    Iter_array = np.zeros((len(n_list),3))

    for i in range(len(n_list)):
        # Making the problem
        H, g, Cin, dlin, duin, lin, uin = random_inequality_qp_strict(n_list[i])
        C, d = standard_NoA_IP(Cin, dlin, duin, lin, uin)
        # Solving the problem
        xout, zout, sout, iter, res, cputime = InequalityQPSolver(H,g,C,d,eps = eps)
        xtrue = np.array(run_Casadi(H, g, Cin, dlin, duin, lin, uin))  
        # Calculating the statistics
        Dist = np.linalg.norm(xout.flatten()-xtrue.flatten(),ord=2)
        Cost = np.abs((g.T@xtrue)[0,0]-(g.T@xout)[0,0])  
        # Saving statistics 
        Dist_and_Cost_array[i,0] = Dist
        Dist_and_Cost_array[i,1] = Cost
        res_row = res[-1] #taking the final residuals values (which were accepted as an optimized solution)
        Res_array[i,0] = res_row[0]
        Res_array[i,1] = res_row[1]
        Res_array[i,2] = res_row[2]
        Iter_array[i,0] = iter
        Iter_array[i,1] = cputime / 1000000000 #convert from ns to seconds
        Iter_array[i,2] = (cputime / 1000000000) / iter # seconds per iteration


    #Plotting the error statistics
    plt.figure(figsize=(10, 6))
    for j in range(Dist_and_Cost_array.shape[1]):
        plt.plot(n_list, Dist_and_Cost_array[:, j], label=Err_labels[j])
    plt.xlabel('n Values')
    plt.ylabel('Distance and Cost Values')
    plt.title('Euclidean Distance from Solution and Cost Difference as a Function of n')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/2_9_Dist_and_Cost.png")
    #plt.show() #If you want the figure to pop up

    # Plotting the L2 norm of the final accepted residuals values as a function of n
    plt.figure(figsize=(10, 6))
    for j in range(Res_array.shape[1]):
        plt.plot(n_list, Res_array[:, j], label=Res_labels[j])
    plt.xlabel('n Values')
    plt.ylabel('Residuals')
    plt.title('L2 Norm of Final Residuals Values as a Function of n')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/2_9_Fin_Res.png")
    #plt.show() #If you want the figure to pop up

    # Plotting the timing statistics
    fig, ax1 = plt.subplots()
    ax1.plot(n_list, Iter_array[:,0], 'g', label=Iter_labels[0])  
    ax1.set_xlabel('n Values')
    ax1.set_ylabel('Iterations', color='k')
    ax1.tick_params(axis='y', labelcolor='k')
    ax2 = ax1.twinx()
    ax2.plot(n_list, Iter_array[:,1], 'r', label=Iter_labels[1])
    ax2.plot(n_list, Iter_array[:,2], 'b', label=Iter_labels[2])  
    ax2.set_ylabel('CPU Time in Seconds', color='k')
    ax2.tick_params(axis='y', labelcolor='k')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.title('Iteration Count and CPU Time as Function of n')
    plt.tight_layout()
    plt.savefig("figures/2_9_Iter_and_CPU.png")
    #plt.show()
    
#==============================================================
# End 2.9
#==============================================================