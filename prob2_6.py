import numpy as np
import scipy as sp
from scipy.linalg import solve
from scipy.sparse import random as sparse_random
from scipy.sparse import csr_matrix
import casadi as ca
import time

#returns the solution of the Inequality constrained QP problem
# min 1/2 x^T H x + g^T x
# s.t. A x + b >= 0
# where H is a positive definite matrix, g is the gradient vector,
# A is the constraint matrix and b is the constraint vector.

def equality_qp_subproblem(H, g, A, b):
    n = H.shape[0]
    m = A.shape[1]

    KKT = np.block([
        [H, -A],
        [-A.T, np.zeros((m, m))]
    ])
    rhs = np.concatenate([-g, b])

    z = np.linalg.solve(KKT, rhs)
    x = z[:n]
    lamb = z[n:]
    return x, lamb


def qpsolver_active_set(H, g, A, b, x0):
    """
    Solve a convex QP using a primal active set algorithm.
    """
    tol = 1.0e-8
    tolLx = tol
    tolc = tol
    tollambda = tol
    tolp = tol

    n, m = A.shape
    Wset = []
    IWset = list(range(m))
    lambda_full = np.zeros(m)
    x = x0.copy()

    # QP data
    gk = H @ x + g
    nablaxL = gk - A @ lambda_full
    #c = A.T @ x + b  # c(x) = A' x + b >= 0
    c = np.matmul(A.T,x) + b
    # Check if the initial point is optimal
    KKT_stationarity = np.linalg.norm(nablaxL, np.inf) < tolLx
    KKT_conditions = KKT_stationarity  # Other conditions are satisfied

    # Main loop
    maxit = 100 * (n + m)
    it = 0
    while not KKT_conditions and it < maxit:
        it += 1

        # Solve equality constrained QP
        A_w = A[:, Wset] if Wset else np.zeros((n, 0))
        b_w = np.zeros(len(Wset))
        p, lambdaWset = equality_qp_subproblem(H, gk, A_w, b_w)

        if np.linalg.norm(p, np.inf) > tolp:  # p is non-zero
            # Find binding constraint (if any)
            alpha = 1.0
            idc = -1
            for i, idx in enumerate(IWset):
                pA = A[:, idx].T @ p
                if pA < 0.0:    
                    alpha_pA = -c[idx] / pA
                    if alpha_pA < alpha:
                        alpha = alpha_pA
                        idc = i

            #take step, update data and working set
            x += alpha * p
            gk = H @ x + g
            c = A.T @ x + b
            if idc >= 0:
                Wset.append(IWset[idc])
                IWset.pop(idc)
        else:  #p is zero
            #Find minimum lambda
            idlambdaWset = -1
            minlambdaWset = 0.0
            for i, lam in enumerate(lambdaWset):
                if lam < minlambdaWset:
                    idlambdaWset = i
                    minlambdaWset = lam

            if idlambdaWset >= 0:  #Update the working set, x = x
                #If minimum lambda < 0, remove constraint from working set
                IWset.append(Wset[idlambdaWset])
                Wset.pop(idlambdaWset)
            else:  #Optimal solution found
                KKT_conditions = True
                xopt = x
                lambdaopt = np.zeros(m)
                for i, idx in enumerate(Wset):
                    lambdaopt[idx] = lambdaWset[i]
                return xopt, lambdaopt, Wset, it

    #If max iterations exceeded
    return None, None, None, it

# Convert the problem to the desired form
def convert_form(A, bl, bu, l, u):
    n = A.shape[0]
    # move upper and lower bounds to single inequality
    Abar = np.hstack([
        A,
        -A,
        np.eye(n),
        -np.eye(n)
    ])
    bbar = np.concatenate([
        -bl,
        bu,
        -l,
        u
    ])
    return Abar, bbar

# compare to casadi
def solve_with_casadi(H, g, A, bl, bu, l, u):
    n = H.shape[0]
    m = A.shape[1]

    x = ca.MX.sym("x", n)

    H_sym = ca.DM(H)
    g_sym = ca.DM(g)
    A_sym = ca.DM(A)

    obj = 0.5 * ca.mtimes([x.T, H_sym, x]) + ca.mtimes(g_sym.T, x)
    constraints = ca.mtimes(A_sym.T, x)  # shape: (m,)

    qp = {
        "x": x,
        "f": obj,
        "g": constraints
    }

    solver = ca.qpsol("qp_solver", "qpoases", qp)

    sol = solver(
        lbg=bl,
        ubg=bu,
        lbx=l,
        ubx=u
    )

    x_sol = np.array(sol['x']).flatten()
    x_l_sol = np.array(sol['lam_x']).flatten()
    return x_sol,x_l_sol

if __name__ == "__main__":
    QP_test = sp.io.loadmat('sources/QP_Test.mat')
    #print(QP_test.keys())

    C = QP_test["C"]
    H = QP_test["H"]
    dl = QP_test["dl"].flatten()
    du = QP_test["du"].flatten()
    l = QP_test["l"].flatten()
    u = QP_test["u"].flatten()
    g = QP_test["g"].flatten()
    
    # print("C:",C.dtype)
    # print("H:",H.dtype)
    # print("dl:",dl.dtype)
    # print("du:",du.dtype)
    # print("l:",l.dtype)
    # print("u:",u.dtype)
    u = u.astype(np.int32)
    # print("g:",g.dtype)


    n = C.shape[0]

    x0 = np.zeros(n)

    Abar, bbar = convert_form(C, dl, du, l, u)
    
    if np.all(np.transpose(Abar)@x0+bbar>=0):
        print("Infeasible initial point. Finding feasible initial point")
        x0 = np.loadtxt('prob2_6_x0.out')
        
    t0 = time.process_time()

    xopt, lambdaopt, Wset, it = qpsolver_active_set(H, g, Abar, bbar, x0)

    t1 = time.process_time()

    x_casadi, lambda_casadi = solve_with_casadi(H, g, C, dl, du, l, u)
    
    print("########################## Our solution:  #########################")
    print("completed in ",it," itterations")
    print("CPU time:", t1-t0)
    diff = np.linalg.norm(xopt - x_casadi)
    print("Solution Difference Norm:", diff)
    print("Solution saved to 'matlab_x_solution.mat")
    # save to matlab data file:
    m_sol = {"x":xopt,"lambda":lambdaopt,"Wset": Wset}
    sp.io.savemat("matlab_x_solution.mat", m_sol)
