#==============================================================
# Prob 2.8: Inequality Constrained Quadratic Program Interior Point Solver
#==============================================================
#Imports
import numpy as np
import scipy as sp
import casadi as ca
from casadi import *
import matplotlib.pyplot as plt
import time

#==============================================================
# Part 1: Preparing Our Data
#==============================================================
# Import from the file:
if __name__ == "__main__":
    QP_test = sp.io.loadmat('./sources/QP_Test.mat')
    #we import with the float datatype to prevent numerical problems further down the line
    QPt_C = (QP_test['C']).astype(float)
    QPt_H = (QP_test['H']).astype(float)
    QPt_dl = (QP_test['dl']).astype(float)
    QPt_du = (QP_test['du']).astype(float)
    QPt_l = (QP_test['l']).astype(float)
    QPt_u = (QP_test['u']).astype(float)
    QPt_g =(QP_test['g']).astype(float)
    QPt_n, _ = QPt_H.shape

#==============================================================
# The problem we were given was in the form:
# 
#      o-------------------------------------o
#      | min (0.5 * x.T @ H @ x) + (g.T @ x) |
#      |  x                                  |
#      | s.t.     dl <= C.T @ x <= du        |
#      |              l <= x <= u            |
#      o-------------------------------------o
#
# However, we need to convert this to standard form:
#
#      o-------------------------------------o
#      | min (0.5 * x.T @ H @ x) + (g.T @ x) |
#      |  x                                  |
#      | s.t.       C_bar.T @ x >= d_bar     |   
#      o-------------------------------------o
#
# We do this refomrulating our dl, du, l, and u bounds into d_bar.
# Note that:
#                dl <= C.T @ x <= du
# Is the same as:
#                     C.T @ x  >=  dl
#                   -(C.T @ x) >= -du
# Similarly:
#                    l <= x <= u  
# Is the same as:
#                       I @ x  >=  l
#                     -(I @ x) >= -u 
# This means we want:
# 
#         C_bar = | C |   d_bar = | dl|
#                 |-C |           |-du|
#                 | I |           | l |
#                 |-I |           |-u |
#            
# We the define the following function to convert to standard form:
def standard_NoA_IP(C, bl, bu, l, u): #converted to the form expected by interior point
    n = C.shape[0]
    Cbar = np.hstack([
        C,
        -C,
        np.eye(n),
        -np.eye(n)
    ])
    dbar = np.concatenate([
        bl,
        -bu,
        l,
        -u
    ])
    return Cbar, dbar

#==============================================================
# Part 2: The Solver
#==============================================================
# We make an inequality constrained quadratic program interior point solver for problems of the form:
#
#      o-------------------------------------o
#      | min (0.5 * x.T @ H @ x) + (g.T @ x) |
#      |  x                                  |
#      | s.t.           C.T @ x >= d         |   
#      o-------------------------------------o
#
def InequalityQPSolver(H,g,C,d,x_init= "NoEntry",z_init="NoEntry",s_init="NoEntry",eps = 0.1): #in QP form using C^T * x >= d
    t = time.process_time_ns()
    # First we define some functions which will come in handly later:
    def pos_test(z): #checks if all elements in an array are strictly positive
        if np.all(z > 0):
            return True
        else:
            return False
        
    def NoA_Build_r_val(H,g,C,d,x,z,s,C_T= "NoEntry",print_test = False): #calculate residuals
        if isinstance(C_T,str):
            C_T = C.T
        rL = ( H @ x ) + g  - (C @ z)
        rC = s + d - (C_T @ x)
        rsz = np.multiply(z,s)
        mu = np.mean(rsz)   
        return rL, rC, rsz, mu 
    
    def NoA_Opt_Check(H,g,C,d,x,z,s,eps,mu0,C_T="NoEntry",rL="NoEntry",rC="NoEntry",rsz="NoEntry",mu="NoEntry"):
        #Taking transposes if they weren't provided
        if isinstance(C_T,str):
            C_T = C.T
        #Calculated r values and mu if they weren't provided
        if isinstance(rL,str) or isinstance(rC,str) or isinstance(rsz,str) or isinstance(mu,str):
            rL, rC, rsz, mu = NoA_Build_r_val(H,g,C,d,x,z,s,C_T)
        #Actual calcuation
        goodness = 0
        if np.linalg.norm(rL,ord=np.inf) <= eps * max(1, np.linalg.norm(np.hstack((H,g,C)),ord=np.inf)):
            goodness += 1
        dn, dm = d.shape
        if np.linalg.norm(rC,ord=np.inf) <= eps * max(1, np.linalg.norm(np.hstack((np.eye(dn),d, C_T)),ord=np.inf)):
            goodness += 1
        if mu <= eps * (1/100) * mu0:
            goodness += 1
        if goodness >=3: return True
        else: return False
        
    def find_a_alt(z,dz,s,ds,x,mult=1): #finds alpha
        zs_stack = -np.concatenate([z, s], axis=0)
        dzds_stack = np.concatenate([dz,ds],axis=0)
        dmask = dzds_stack < 0
        if len(dzds_stack[dmask]) == 0:
            a = 1
        else:
            a_vec = zs_stack[dmask]/dzds_stack[dmask]
            a = min(a_vec)
            #if a > 1:
            #    a = 1
            if a <= 0:
                raise ValueError("###---a nonpositive---###")
        a *= mult
        return a
    #grabbing dimensions
    Hn, Hm = H.shape
    Cn, Cm = C.shape
    #initialize x, z, and s
    if isinstance(x_init,str):
        # x_init = np.ones((Hn,1))*10
        x_init = np.zeros((Hn,1))
    if isinstance(z_init,str):
        z_init = np.ones((Cm,1))
    if isinstance(s_init,str):
        s_init = np.ones((Cm,1))    
    #setting variables datatype to float. In tessting we experienced datatype conflicts without this for providing initial x, z, and s values
    x = x_init.astype(float)
    z = z_init.astype(float)
    s = s_init.astype(float)

    #Check for consistent dimensions
    if not np.all(z > 0):
        raise ValueError("###---z must be positive---###")
    if not np.all(s > 0):
        raise  ValueError("###---s must be positive---###")
    if Hn != len(x):
        raise ValueError("###---H and x dimensions do not match---###")
    if Hm != len(g):
        raise ValueError("###---H and g dimensions do not match---###")
    if Cm != len(z):
        raise ValueError("###---C and z dimensions do not match---###")
    #Collect Transposes, this means they don't need to be repeatedly calculated. Note that C is never reassigned, so these should be good forever
    C_T = C.T
    #Get r and mu values
    rL, rC, rsz, mu = NoA_Build_r_val(H,g,C,d,x,z,s,C_T) #this function does what it says on the tin: it makes the rL, rA, rC, rsz (and mu) values following the documentation
    mu0=mu #this grabs the first mu for use in the optimality conditions
    # Prepare the residuals array
    ResArray = []
    ResArray.append([np.linalg.norm(rL,ord=2),np.linalg.norm(rC,ord=2),mu])
    #start the iteration counter
    itcount=0
    #Check for optimality
    while NoA_Opt_Check(H,g,C,d,x,z,s,eps,mu0,C_T,rL,rC,rsz,mu) != True: #this loop runs so long as the optimality conditions are not met
        itcount+=1
        #guardrails
        if any(np.clip(z,1e-18,1e18) != z):
            print("z clipped\n",min(z),max(z))
        z_div = np.clip(z,1e-18,1e18)
        if any(np.clip(s,1e-18,1e18) != s):
            print("s clipped\n",min(s),max(s))
        s_div = np.clip(s,1e-18,1e18) 
         #this is not strictly in the instructions: I was getting division by 0 errors since z and s were getting really small 
        if not np.all(z >= 0):
            raise ValueError("###---while loop error: z must be positive---###")
        if not np.all(s >= 0):
            raise ValueError("###---while loop error: s must be positive---###")
        SinvZ = np.diag((z / s_div).flatten()) #this calculates S^-1 * Z, which is a diagonal matrix.
        ZinvS = np.diag((s / z_div).flatten()) #calculating Z^-1 * S
        Zinv  = np.diag((1 / z_div).flatten()) #Z^-1
        #==============================================================
        # KKT Matrix and Factorization:
        #Note that since z and s are technically 2D, I need to flatten z/s before I can plug them into np.diag
        H_bar = H + (C @ SinvZ @ C_T)
        #Next I make the KKT matrix
        KKT = H_bar 
        L, D, perm = sp.linalg.ldl(KKT) #this is the LDL' decomposition of the KKT matrix
        L_T = L.T
        #==============================================================
        # Affine Step:
        #Next step is to calculate the affine direction
        #this next thing is a little weird: (SinvZ @ (rC - Zinv@rsz)) appears several times in the equations, so I just make it a variable
        #old:               (SinvZ @ (rC - (Zinv@rsz)))
        #(rC - (Zinv@rsz)) = s + d - C_T@x - Z_invSZe = s + d - C_T@x - Se = s - s + d - C_T@x = d - (C_T@x)
        reference_block_1 = (SinvZ @ (d - (C_T@x)))
        rL_bar = rL - ( C @ reference_block_1 ) #calculating rL_bar
        #here we solve the system using the LDL factorization
        midd1 = np.linalg.solve(L, -rL_bar)
        midd2 = np.linalg.solve(D,midd1)
        xaff = np.linalg.solve(L_T, midd2) 
        zaff = -(SinvZ @ C_T @ xaff) + reference_block_1
        saff = - s - d + (C_T @ (xaff + x))
        #need to get alpha affine
        aaff = find_a_alt(z,zaff,s,saff,x) #Actually used this step
        #print("A affine used:",aaff) #testing
        #next is the duality gap and centering parameter
        ztest = z + (aaff * zaff)
        stest = s + (aaff * saff)
        muaff = np.mean(np.multiply(ztest,stest)) # note
        #guardrails for numerical stability
        #note that original definitial is sigma = (muaff / mu)**3
        sigma = np.clip(min(1, (muaff / mu)**3),0.001,0.999)
        #==============================================================
        # Center-Correction step:
        #affine-centering-correction direction
        rsz_bar = rsz + np.multiply(saff,zaff) - (sigma * muaff * np.ones((len(rsz),1)))
        rL_bar = rL - ( C @ SinvZ @ ( rC - ( Zinv @ rsz_bar ) ) )
        midd1 = np.linalg.solve(L, -rL_bar)
        midd2 = np.linalg.solve(D,midd1)
        dx = np.linalg.solve(L_T, midd2) 
        dz = -(SinvZ @ C_T @ dx) + (SinvZ @ (rC - (Zinv@rsz_bar)))
        ds = - (Zinv @ rsz_bar) - (-(C_T @ dx) + (rC - (Zinv@rsz_bar)))
        #calculate a
        a = find_a_alt(z,dz,s,ds,x)
        nu = 0.995
        a_bar = nu * a
        x += a_bar * dx
        z += a_bar * dz
        s += a_bar * ds
        rL, rC, rsz, mu = NoA_Build_r_val(H,g,C,d,x,z,s,C_T)
        ResArray.append([np.linalg.norm(rL,ord=2),np.linalg.norm(rC,ord=2),mu]) 
    cpu_time = time.process_time_ns() - t  
    return x, z, s, itcount, ResArray, cpu_time

#==============================================================
# Part 3: Baseline CasADi Solution
#==============================================================
# Preparing the variables
if __name__ == "__main__":
    H = QPt_H
    g = QPt_g
    A = QPt_C
    lbx_ = QPt_l
    ubx_ = QPt_u
    lbg_ = QPt_dl
    ubg_ = QPt_du
    nh = QPt_n
    # Define x as a symbolic variable
    x = ca.SX.sym('x', nh) 
    # min (0.5 * x.T @ H @ x) + (g.T @ x)
    objective = 0.5 * ca.dot(x, ca.mtimes(H, x)) + ca.dot(g, x)
    # Constraint: 
    constraint = ca.mtimes(A.T, x)
    # Define the QP in symbolic form
    qp = {
        'x': x,      
        'f': objective,
        'g': constraint
    }
    # Create solver
    solver = ca.qpsol('S', 'qpoases', qp)
    # Call solver with bounds
    sol = solver(
        lbx=lbx_, 
        ubx=ubx_,  
        lbg=lbg_,  
        ubg=ubg_  
    )
    # getting the actual solution object
    Cas_X_Opt = sol['x']

    print(sol['x'])

#==============================================================
# Part 4: Using Our Solver and Statistics
#==============================================================
# preparing our variables
if __name__ == "__main__":
    H = QPt_H
    g = QPt_g
    C, d = standard_NoA_IP(QPt_C,QPt_dl,QPt_du,QPt_l,QPt_u) #using our converer
    #using our function
    xout, zout, sout, iter, res, cpu_time = InequalityQPSolver(H,g,C,d,eps=10e-10)
    print("x:\n",xout.T)
    # Compare the Euclidean distance between CasADi's solution and ours:
    print("Euclidean Distance from Solution:",np.linalg.norm(np.array(Cas_X_Opt).flatten()-xout.flatten(),ord=2))

    res = np.array(res)
    res_names = ["rL", "rC", "mu"]
    plt.figure(figsize=(10, 6))

    for i in range(res.shape[1]):
        plt.plot(res[:, i], label=res_names[i])

    plt.xlabel('Iteration Number')
    plt.ylabel('Residual L2 Norm')
    plt.title('InequalityQPSolver: L2 Norm of Residuals By Iteration')
    plt.yscale('log') 
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.savefig("figures/2_8_Residuals.png")
    plt.show()

    print("Number of Iterations:", iter)
    print("CPU Time in seconds:", cpu_time / 1000000000)

#==============================================================
# End 2.8
#==============================================================