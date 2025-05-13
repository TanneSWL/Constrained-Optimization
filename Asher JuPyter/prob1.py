#==============================================================================================
#Exam Problem 1:
#==============================================================================================
#Imports
#----------------------------------------------------------------------------------------------
import numpy as np
import scipy as sp
from scipy.sparse.linalg import splu, spsolve
from scipy.sparse import csc_matrix, csr_matrix, dia_matrix
import casadi as ca
from casadi import *
import matplotlib.pyplot as plt
#==============================================================================================
#Problem 1.3:
#==============================================================================================
#Defining the Solver Functions
#----------------------------------------------------------------------------------------------
def EqualityQPSolverLUdense(H,g,A,b):
    _, Am = A.shape #note that An = Hn = Hm, so we use Am for the dimensions of the 0 matrix
    KKT = np.block([[H, -A],[-A.T,np.zeros([Am,Am])]])
    L, U = sp.linalg.lu(KKT, permute_l = True)
    vec = -np.concatenate([g, b], axis=0) 
    midd = np.linalg.solve(L, vec)
    out_vec = np.linalg.solve(U, midd)
    xout = out_vec[:len(g)]
    yout = out_vec[len(g):]
    return xout, yout

def EqualityQPSolverLUsparse(H,g,A,b):
    _, Am = A.shape #note that An = Hn = Hm, so we use Am for the dimensions of the 0 matrix
    KKT = np.block([[H, -A],[-A.T,np.zeros([Am,Am])]])
    L, U = sp.linalg.lu(KKT, permute_l = True)
    L = csc_matrix(L) 
    U = csc_matrix(U)
    vec = csc_matrix(-np.concatenate([g, b], axis=0) )
    midd = sp.sparse.linalg.spsolve(L, vec)
    out_vec = np.array(sp.sparse.linalg.spsolve(U, midd))
    xout = out_vec[:len(g)]
    yout = out_vec[len(g):]
    return np.array([xout]).T, np.array([yout]).T

def EqualityQPSolverLDLdense(H,g,A,b):
    _, Am = A.shape 
    KKT = np.block([[H, -A],[-A.T,np.zeros([Am,Am])]]) #note that An = Hn = Hm, so we use Am for the dimensions of the 0 matrix
    L, D, perm = sp.linalg.ldl(KKT) #scipy's linalg.ldl performs a dense LDL.T decomposition
    vec = -np.concatenate([g, b], axis=0) 
    midd1 = np.linalg.solve(L, vec) #here we solve the system using the LDL.T factorization
    midd2 = np.linalg.solve(D, midd1)
    out_vec = np.linalg.solve(L.T, midd2) 
    xout = out_vec[:len(g)]
    yout = out_vec[len(g):]
    return xout, yout

def EqualityQPSolverLDLsparse(H,g,A,b):
    _, Am = A.shape 
    KKT = np.block([[H, -A],[-A.T,np.zeros([Am,Am])]]) #note that An = Hn = Hm, so we use Am for the dimensions of the 0 matrix
    L, D, perm = sp.linalg.ldl(KKT) #scipy's linalg.ldl performs a dense LDL.T decomposition
    L = csc_matrix(L)
    D = csc_matrix(D)
    vec = csc_matrix(-np.concatenate([g, b], axis=0)) 
    midd1 = sp.sparse.linalg.spsolve(L, vec) #here we solve the system using the LDL.T factorization
    midd2 = sp.sparse.linalg.spsolve(D, midd1)
    out_vec = sp.sparse.linalg.spsolve(L.T, midd2) 
    xout = out_vec[:len(g)]
    yout = out_vec[len(g):]
    return np.array([xout]).T, np.array([yout]).T #Note that the np.array and transpose commands are to get the output shape correct.

def EqualityQPSolverRange(H,g,A,b):
    Hinv = sp.linalg.inv(H) #most costly line
    A_T = A.T
    v = np.linalg.solve(H,g)
    y = np.linalg.solve((A_T@Hinv@A),b+(A_T@v))
    x = np.linalg.solve(H,(A@y-g))
    return x, y

def EqualityQPSolverNull(H,g,A,b):
    #This one is built from: Lecture 5 QuadraticOptimization page 22/23: https://learn.inside.dtu.dk/d2l/le/lessons/242131/topics/969401
    _, Am = A.shape 
    A_T = A.T
    Y,R = sp.linalg.qr(A,mode='economic') #This works but must be justified: goal: A^T@Y is nonsingular.
    Z = sp.linalg.null_space(A_T) #Note that we want A^T@Z=0, so Z is the null space of A^T
    Y_T = Y.T
    Z_T = Z.T
    x_Y = np.linalg.solve((A_T@Y),b)
    x_Z = np.linalg.solve((Z_T@H@Z),-Z_T@((H@Y@x_Y)+g))
    x = (Y@x_Y)+(Z@x_Z)
    y = np.linalg.solve((A_T@Y).T,Y_T@((H@x)+g))
    return x, y

#Smart more efficient LU solver
def EqualityQPSolverLUsparse_smart(H,g,A,b):
    _, Am = A.shape #note that An = Hn = Hm, so we use Am for the dimensions of the 0 matrix
    KKT = sp.sparse.csc_matrix(np.block([[H, -A],[-A.T,np.zeros([Am,Am])]])) #define the KKT matrix as a csc matrix
    LUsparse= splu(KKT) # the sparse lu function from scipy uses csc matrices. This produces an object with a .solve function
    vec = -np.concatenate([g, b], axis=0)
    out_vec = LUsparse.solve(vec) #using the .solve functionality to get the solution
    xout = out_vec[:len(g)]
    yout = out_vec[len(g):]
    return xout, yout

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
#----------------------------------------------------------------------------------------------
#The following are tests 
#----------------------------------------------------------------------------------------------        
    #Defining the test problem for 1.6
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

solvers = ["LUdense","LUsparse","LDLdense","LDLsparse","Range","Null","LUsparse_smart"]
for s in solvers: #testing each solver
    print("-----",s,"-----")
    x, y = EqualityQPSolver(H,g,A,b,s)
    print("    x.T:",x.T)
    #print("x.shape:",x.shape)