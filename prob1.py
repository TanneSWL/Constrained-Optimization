#==============================================================
# Problems 1.3, 1.4, 1.5, and 1.6
#==============================================================
# Imports:
import numpy as np
import scipy as sp
import pandas as pd
from scipy.sparse.linalg import splu, spsolve
from scipy.sparse import csc_matrix, csr_matrix, dia_matrix
import casadi as ca
from casadi import *
import matplotlib.pyplot as plt
import time

#==============================================================
# Problem definition:
#==============================================================
# The goal here is to solve a problem of the form:
#
#      o-------------------------------------o
#      | min (0.5 * x.T @ H @ x) + (g.T @ x) |
#      |  x                                  |
#      | s.t.            A.T @ x = b         |   
#      o-------------------------------------o
#
# We note from our write-up for 1.2 and 1.3 that this is the same as finding x and y s.t.
#
#      o--------------------------------o
#      |                                |
#      |  |  H   -A  | @ | x | = |-g |  |
#      |  |-A.T   0  |   | y |   |-b |  |
#      |                                |
#      o--------------------------------o
#
#==============================================================
# 1.3: Solvers
#==============================================================
# Dense LU Solver:
def EqualityQPSolverLUdense(H,g,A,b):
    _, Am = A.shape #note that An = Hn = Hm, so we use Am for the dimensions of the 0 matrix
    #Factorization
    KKT = np.block([[H, -A],[-A.T,np.zeros([Am,Am])]])
    L, U = sp.linalg.lu(KKT, permute_l = True)
    #Solving for x and y
    rhs = -np.concatenate([g, b], axis=0) 
    midd = np.linalg.solve(L, rhs)
    sol = np.linalg.solve(U, midd)
    #Returning
    xout = sol[:len(g)]
    yout = sol[len(g):]
    return xout, yout

# Sparse LU Solver:
def EqualityQPSolverLUsparse(H,g,A,b):
    _, Am = A.shape #note that An = Hn = Hm, so we use Am for the dimensions of the 0 matrix
    #Factorization
    KKT = np.block([[H, -A],[-A.T,np.zeros([Am,Am])]])
    L, U = sp.linalg.lu(KKT, permute_l = True)
    L = csc_matrix(L) #converting to sparse matrices
    U = csc_matrix(U)
    #Solving for x and y
    rhs = csc_matrix(-np.concatenate([g, b], axis=0) )
    midd = sp.sparse.linalg.spsolve(L, rhs)
    sol = np.array(sp.sparse.linalg.spsolve(U, midd))
    #Returning
    xout = sol[:len(g)]
    yout = sol[len(g):]
    return np.array([xout]).T, np.array([yout]).T

# Smart Sparse LU Solver:
def EqualityQPSolverLUsparse_smart(H,g,A,b):
    _, Am = A.shape #note that An = Hn = Hm, so we use Am for the dimensions of the 0 matrix
    #Factorization
    KKT = sp.sparse.csc_matrix(np.block([[H, -A],[-A.T,np.zeros([Am,Am])]])) #define the KKT matrix as a csc matrix
    LUsparse= splu(KKT) # the sparse lu function from scipy uses csc matrices. This produces an object with a .solve function
    # Solving for x and y
    rhs = -np.concatenate([g, b], axis=0)
    sol = LUsparse.solve(rhs) #using the .solve functionality to get the solution
    #Returning x and y
    xout = sol[:len(g)]
    yout = sol[len(g):]
    return xout, yout

# Dense LDL.T Solver:
def EqualityQPSolverLDLdense(H,g,A,b):
    _, Am = A.shape #note that An = Hn = Hm, so we use Am for the dimensions of the 0 matrix
    #Factorization 
    KKT = np.block([[H, -A],[-A.T,np.zeros([Am,Am])]])
    L, D, perm = sp.linalg.ldl(KKT) #scipy's linalg.ldl performs a dense LDL.T decomposition
    #Solving for x and y
    rhs = -np.concatenate([g, b], axis=0) 
    midd1 = np.linalg.solve(L, rhs) #here we solve the system using the LDL.T factorization
    midd2 = np.linalg.solve(D, midd1)
    sol = np.linalg.solve(L.T, midd2)
    #Returning x and y 
    xout = sol[:len(g)]
    yout = sol[len(g):]
    return xout, yout

# Sparse LDL.T Solver:
def EqualityQPSolverLDLsparse(H,g,A,b):
    _, Am = A.shape #note that An = Hn = Hm, so we use Am for the dimensions of the 0 matrix
    # Factorization
    KKT = np.block([[H, -A],[-A.T,np.zeros([Am,Am])]])
    L, D, perm = sp.linalg.ldl(KKT) #scipy's linalg.ldl performs a dense LDL.T decomposition
    L = csc_matrix(L) #converting to sparse matrices
    D = csc_matrix(D)
    # Solving for x and y
    rhs = csc_matrix(-np.concatenate([g, b], axis=0)) 
    midd1 = sp.sparse.linalg.spsolve(L, rhs) #here we solve the system using the LDL.T factorization
    midd2 = sp.sparse.linalg.spsolve(D, midd1)
    sol = sp.sparse.linalg.spsolve(L.T, midd2) 
    # Returning x and y
    xout = sol[:len(g)]
    yout = sol[len(g):]
    return np.array([xout]).T, np.array([yout]).T

# Range Space Solver:
def EqualityQPSolverRange(H,g,A,b):
    Hinv = sp.linalg.inv(H) #most costly line
    A_T = A.T
    v = np.linalg.solve(H,g)
    y = np.linalg.solve((A_T@Hinv@A),b+(A_T@v))
    x = np.linalg.solve(H,(A@y-g))
    return x, y

# Null Space Solver:
def EqualityQPSolverNull(H,g,A,b):
    #This one is built from: Lecture 5 QuadraticOptimization page 22/23: https://learn.inside.dtu.dk/d2l/le/lessons/242131/topics/969401
    _, Am = A.shape 
    A_T = A.T
    Y,R = sp.linalg.qr(A,mode='economic') #goal: A^T@Y is nonsingular.
    Z = sp.linalg.null_space(A_T) #Note that we want A^T@Z=0, so Z is the null space of A^T
    Y_T = Y.T
    Z_T = Z.T
    x_Y = np.linalg.solve((A_T@Y),b)
    x_Z = np.linalg.solve((Z_T@H@Z),-Z_T@((H@Y@x_Y)+g))
    x = (Y@x_Y)+(Z@x_Z)
    y = np.linalg.solve((A_T@Y).T,Y_T@((H@x)+g))
    return x, y

# Defining the control center function:
def EqualityQPSolver(H,g,A,b,solver="LDLdense"):
    if solver == "LUdense":
        return EqualityQPSolverLUdense(H,g,A,b)
    if solver == "LUsparse":
        return EqualityQPSolverLUsparse(H,g,A,b)
    if solver == "LDLdense":
        return EqualityQPSolverLDLdense(H,g,A,b)
    if solver == "LDLsparse":
        return EqualityQPSolverLDLsparse(H,g,A,b)
    if solver == "Range":
        return EqualityQPSolverRange(H,g,A,b)
    if solver == "Null":
        return EqualityQPSolverNull(H,g,A,b)
    #specialty solvers:
    if solver == "LUsparse_smart":
        return EqualityQPSolverLUsparse_smart(H,g,A,b)
    else:
        raise ValueError("###---Unknown Solver---###")
    
#==============================================================
# 1.4: Random QP Generator
#==============================================================
# The goal is to make a function that can construct random inequality constrained QPs with guaranteed solutions.
# Here, the number of constraints m are a fraction (beta) of n. We then generate a random sparse
# constraint matrix A with gaussian entries, a vector x0 to serve as the optimal solution, vector y0 as the known
# dual variables. Then we set the lower and upper constraint bounds bl, lu to AT x0, making all constraints
# equalities, and ensuring x0 is feasible. Then we generate a random sparse matrix, a positive definite Hessian
# and set the gradient g so that the KKT conditions are satisfied at (x0, y0).
def random_qp(n, alpha, beta, density):
        m = int(np.round(beta * n))
        A = sp.sparse.random(n, m, density=density, format='csr', data_rvs=np.random.randn).toarray()
        x0 = np.random.randn(n,1)
        y0 = np.random.randn(m,1)
        bl = A.T @ x0
        bu = bl
        M = sp.sparse.random(n, n, density=density, format='csr', data_rvs=np.random.randn)
        H = (M @ M.T).toarray() + (alpha * np.eye(n))
        g = -(H@x0 - A@y0)
        l = np.full(n, -np.inf)
        u = np.full(n,  np.inf)
        return H, g, A, bl, bu, l, u

# We also define a version of the function which redraws A until it finds one for which A.T has full row rank.
def random_qp_strict(n, alpha, beta, density):
        m = int(np.round(beta * n))
        A = sp.sparse.random(n, m, density=density, format='csr', data_rvs=np.random.randn).toarray()

        it_count=0
        while np.linalg.matrix_rank(A.T) != A.T.shape[0]:
            A = sp.sparse.random(n, m, density=density, format='csr', data_rvs=np.random.randn).toarray()
            it_count+=1
            if it_count>10000: raise ValueError("###---Too Many Iterations---###")

        x0 = np.random.randn(n,1)
        y0 = np.random.randn(m,1)
        bl = A.T @ x0
        bu = bl
        M = sp.sparse.random(n, n, density=density, format='csr', data_rvs=np.random.randn)
        H = (M @ M.T).toarray() + (alpha * np.eye(n))
        g = -(H@x0 - A@y0)
        l = np.full(n, -np.inf)
        u = np.full(n,  np.inf)
        return H, g, A, bl, bu, l, u

#==============================================================
# 1.5 Part 1: CasADi Basline Check
#==============================================================
# We first need to confirm that our solution methods work by comparing to the CasADi solver.

def run_Casadi(H,g,A,bl,bu,l,u):
    nh, _ = H.shape
    #here we convert all of the entries in CasADi's formatting
    H = ca.SX(H) 
    g = ca.SX(g)
    A = ca.SX(A.T) # switching back to A @ x = b
    lbg = ca.DM(bl)
    ubg = ca.DM(bu)
    lbx = ca.DM(l)
    ubx = ca.DM(u)
    # Set up x as a symbolic variable to be solved for
    x = ca.SX.sym('x', nh) 
    # Minimize (0.5 * x.T @ H @ x) + (g.T @ x)
    objective = 0.5 * ca.dot(x, ca.mtimes(H, x)) + ca.dot(g, x)
    # Constraints
    constraint = ca.mtimes(A, x)
    # Define the QP in symbolic form
    qp = {
        'x': x,       
        'f': objective,
        'g': constraint
    }
    opts = {
    "printLevel": "none",   # Print level options: none, low, medium, high, debug
    }                     
    # Create solver
    solver = ca.qpsol('S', 'qpoases', qp, opts)
    # Call solver with bounds
    sol = solver(
        lbx=lbx, 
        ubx=ubx,  
        lbg=lbg,  
        ubg=ubg 
    )
    # Returning x
    return sol['x']

#==============================================================
# 1.5 Part 2: Checking Accuracy of Solvers
#==============================================================
# Note: This code will likely take 1 or 2 minutes to run. Do not be surprised if nothing immidiately pops out.
# Note: This will also fill the terminal with many CasADi calls since CasADi is used many dozens of times.
#
# Next we define a function which will return the accuracy for all of the solvers given n, alpha, beta, and the density.
def return_accuracy(solvers= ["LUdense","LUsparse","LDLdense","LDLsparse","Range","Null","LUsparse_smart"], n=100,alpha=1,beta=0.5,density=0.15):
    # First make the random QP
    H, g, A, bl, bu, l, u = random_qp_strict(n,alpha,beta,density)
    # Then get the CasADi solution
    Casadi_x = np.array(run_Casadi(H,g,A,bl,bu,l,u))
    # Next calculate the Euclidean distance between each solver and the CasADi solution and return it as a row.
    error_row = np.zeros(len(solvers))
    for i in range(len(solvers)):
        xout, _ = EqualityQPSolver(H,g,A,bl,solvers[i])
        error_row[i]=np.linalg.norm(Casadi_x-xout,ord=2) 
    return(error_row)

#Here we define the different n values we want, and which solveres we want to use
n_list = np.arange(5,201) #Even the strict QP generator struggles to make valid problems with n < 5
solvers= ["LUdense","LUsparse","LDLdense","LDLsparse","Range","Null","LUsparse_smart"]
# initialize the error matrix 
x_err_mat = np.zeros((len(n_list),len(solvers)))
#for every n, calculate the error with every solver.
for i in range(len(n_list)):
    x_err_mat[i,:] = return_accuracy(solvers,n_list[i])
#plotting it all:
plt.figure(figsize=(10, 6))
for i in range(x_err_mat.shape[1]):
    plt.plot(n_list, x_err_mat[:, i], label=solvers[i])
plt.xlabel('n values')
plt.ylabel('error values')
plt.title('Euclidean Distance from Solution as a Function of n')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/1-5_Error.png")
#plt.show() #If you want the figure to pop up

#==============================================================
# 1.5 Part 3: Checking CPU Time of Solvers
#==============================================================
# Note: This code will likely take 1 or 2 minutes to run. Do not be surprised if nothing immidiately pops out.
# Note: This will also fill the terminal with many CasADi calls since CasADi is used many dozens of times.
#
# This is just like return_accuracy, but instead of Euclidean distance from solution, it returns CPU time for each solver given n, alpha, beta, and density.
def return_cputime(solvers= ["LUdense","LUsparse","LDLdense","LDLsparse","Range","Null","LUsparse_smart"], n=100,alpha=1,beta=0.5,density=0.15):
    # Make a random QP
    H, g, A, bl, bu, l, u = random_qp_strict(n,alpha,beta,density)
    # Calculate the CPU time for each solver averaged over num runs
    time_row = np.zeros(len(solvers))
    for i in range(len(solvers)):
        average_time = 0
        num = 1 # for the assignment, we used 25 runs.
        for j in range(num):
            t = time.process_time_ns()
            xout, _ = EqualityQPSolver(H,g,A,bl,solvers[i])
            average_time+=time.process_time_ns()-t
        time_row[i]=average_time / num
    return(time_row)

# Define which ns we want, and which solvers to use
n_list = np.arange(5,251)
solvers= ["LUdense","LUsparse","LDLdense","LDLsparse","Range","Null","LUsparse_smart"]
# initialize the time matrix
cputime_mat = np.zeros((len(n_list),len(solvers)))
# for ever n, calculate the CPU time for each solver
for i in range(len(n_list)):
    cputime_mat[i,:] = return_cputime(solvers,n_list[i])
#plotting it all:
plt.figure(figsize=(10, 6))
for i in range(cputime_mat.shape[1]):
    plt.plot(n_list, cputime_mat[:, i], label=solvers[i])
plt.xlabel('n values')
plt.ylabel('cpu time ns')
plt.title('Average CPU Time per Solver as Function of n')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/1-5_Time-CPU.png")
#plt.show() #If you want the figure to pop up

#==============================================================
# 1.6 Part 1: CasADi Baseline Check
#==============================================================
# Just like in 1.5, we need a baseline to compare to. We again use CasADi.
# The first step is make a function which will convert from the standard form (H,g,A,b) to CasADi's form (H,g,A,lbx,ubx,lbg,ubg
#We need a converter to get to Casadi formatting
def QP_Convert_To_Casadi(Hin, gin, Ain, bin): #This function accepts H, g, A, and b, and outputs Casadi's inputs
    Hn, _ = Hin.shape
    # Convert all arrays to CASADI types first
    Hout = ca.SX(Hin)
    gout = ca.SX(gin)
    # Build constraint matrix. Again, note how we shift to the form A @ x = b
    Aout = ca.SX(Ain.T)
    # Build bounds
    lbg = ca.DM(bin)
    ubg = ca.DM(bin)
    lbx = ca.DM(np.full((1, Hn), -np.inf))
    ubx = ca.DM(np.full((1, Hn),  np.inf))
    # Return it all    
    return Hout, gout, Aout, lbx, ubx, lbg, ubg

#This function accepts H, g, A, and b, and returns the Casadi solution
def Casadi_EC_QP(Hin,gin,Ain,bin): 
    nh, _ = Hin.shape
    # Here we use our previous function to compute to the CasADi input form    
    H, g, A, lbx_, ubx_, lbg_, ubg_ = QP_Convert_To_Casadi(Hin,gin,Ain,bin)
    # Define x as a symbolic vector
    x = ca.SX.sym('x', nh) 
    # Minimize (0.5 * x.T @ H @ x) + (g.T @ x)
    objective = 0.5 * ca.dot(x, ca.mtimes(H, x)) + ca.dot(g, x)
    # Constraints
    constraint = ca.mtimes(A, x)
    # Define the QP in symbolic form
    qp = {
        'x': x,       
        'f': objective,
        'g': constraint
    }
    opts = {
    "printLevel": "none",   # Print level options: none, low, medium, high, debug
    }                     
    # Create solver
    solver = ca.qpsol('S', 'qpoases', qp, opts)
    # Call solver with bounds 
    sol = solver(
        lbx=lbx_, 
        ubx=ubx_,  
        lbg=lbg_,   
        ubg=ubg_   
    )
    # Return x
    return sol['x']

#==============================================================
# 1.6 Part 2: Problem Definiton and Solution
#==============================================================
#We are asked to solve:

H = np.array([
    [6.0000, 1.8600, 1.2400, 1.4800, -0.4600],
    [1.8600, 4.0000, 0.4400, 1.1200, 0.5200],
    [1.2400, 0.4400, 3.8000, 1.5600, -0.5400],
    [1.4800, 1.1200, 1.5600, 7.2000, -1.1200],
    [-0.4600, 0.5200, -0.5400, -1.1200, 7.8000]
])

g = np.array([
    [-16.1000],
    [-8.5000],
    [-15.7000],
    [-10.0200],
    [-18.6800]
])

A = np.array([
    [16.1000, 1.0000],
    [8.5000, 1.0000],
    [15.7000, 1.0000],
    [10.0200, 1.0000],
    [18.6800, 1.0000]
])


b = np.array([
    [15],
    [1]
])

# First we solve the test problem with one of our solvers:
x, y = EqualityQPSolver(H,g,A,b,"LDLdense")
Our_x = x.flatten()
# Then we get CasADi's solution
Cas_x = np.array(Casadi_EC_QP(H, g, A, b)).flatten()
# Calcute the absolute difference between the elements of the two solutions
AbsDif = np.abs(Our_x-Cas_x)
# And the Euclidean distance between the two solutions
EucDist = np.linalg.norm(Our_x - Cas_x, ord=2)
EucDistVec = np.full(len(Our_x), np.nan)
EucDistVec[0]= EucDist
# Create a dataframe to store all of this in
df = pd.DataFrame({
    'Our Solution': Our_x,           # Our solution
    'CasADi Solution': Cas_x,        # CasADi's solution
    "Absolute_Difference": AbsDif,   # The absolute difference between each component of Our_x and Cas_x
    'Euclidean_Distance': EucDistVec # This is the Euclidean distance between CasADi's solution and ours
})
# Save to CSV
df.to_csv('figures/1-6_Solution-Comparison.csv')

#==============================================================
# 1.6 Part 3: x_i as a Function of b_1
#==============================================================
# We next want to see how the elements of the solution to x change asa function of b_1:
# How does |x_1 x_2 x_3 x_4 x_5|.T change as a function of b_1?
#
# Redefine the QP for consistency's sake:
H = np.array([
    [6.0000, 1.8600, 1.2400, 1.4800, -0.4600],
    [1.8600, 4.0000, 0.4400, 1.1200, 0.5200],
    [1.2400, 0.4400, 3.8000, 1.5600, -0.5400],
    [1.4800, 1.1200, 1.5600, 7.2000, -1.1200],
    [-0.4600, 0.5200, -0.5400, -1.1200, 7.8000]
])

g = np.array([
    [-16.1000],
    [-8.5000],
    [-15.7000],
    [-10.0200],
    [-18.6800]
])

A = np.array([
    [16.1000, 1.0000],
    [8.5000, 1.0000],
    [15.7000, 1.0000],
    [10.0200, 1.0000],
    [18.6800, 1.0000]
])


b = np.array([
    [15],      #b_1 will be changed
    [1]
])

# Define different b_1 values we want
bn = 1000
b1_space = np.linspace(8.50,18.68,bn)
# Initialize a matrix to fit all of the solutions for x into
xsol_matrix = np.zeros((bn,5))
# Calculate x for each different b_1
for b_id in range(len(b1_space)):
    b[0,0]=b1_space[b_id]
    xsol, _ = EqualityQPSolver(H,g,A,b,"LDLdense")
    xsol_matrix[b_id,:] = xsol.flatten()
# Plot it all up:
plt.figure(figsize=(10, 6))
for i in range(xsol_matrix.shape[1]):
    plt.plot(b1_space, xsol_matrix[:, i], label=f'x_{i + 1}')
plt.xlabel('b_1 value')
plt.ylabel('x values')
plt.title('x Solution Values as a Function of b_1')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/1-6_xi-by-b1.png")
#plt.show()

#==============================================================
# End Problem 1
#==============================================================



