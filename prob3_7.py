import numpy as np
import scipy as sp
import time


# defoíne Simplex algorithm
def LP_simplex(g,A,b,Bs,NBs):
    #initialize 
    cycles = 0
    Stop = False #loop condition
    B = A[:,Bs]
    N = A[:,NBs]

    xB = np.linalg.solve(B,b)
    
    xN = np.zeros((len(NBs)))

    #test for looping
    i_j_prev = 0
    i_s_prev = 0


    if xB.min()<0: #test for negative values in xB
        print("xB has a negative value. check your initial basis",xB)
        Stop = True
    
    
    while not(Stop):
        cycles+=1 #track cycles
        # get B and N matrices
        B = A[:,Bs]
        N = A[:,NBs]
        
        µ = np.linalg.solve(np.transpose(B),g[Bs])
        
        lambdaN = g[NBs] - np.dot(np.transpose(N),µ)
        
        test=np.array(lambdaN)>=0 #test if all lambda are non-negative
        if test.all():
            print("Optimal solution found after %d itterations" % (cycles))
            Stop = True

        else:
            #find a s (first s chosen)
            s = np.arange(0,lambdaN.size)[lambdaN < 0][0]
            i_s = NBs[s]
            
            h = np.linalg.solve(B,A[:,i_s])

            temp=np.ones(h.size)*xB.max()#keep shape of the divitions to keep index correct
            #xB.max() value chosen so it would never interfere.
            temp[h>0]=np.divide(xB[h>0],h[h>0])#actial divition to determine J and alpha
            
            J = np.argmin(temp) #gives first argmin result

            if h[h>0].size==0: #equivalent to J being empty
                print("unbound solution found after %d itterations (J)" % (cycles))
                Stop = True
            else:
                j = J
                alpha = temp[j] #get relevant alpha
                xB = xB - alpha*h #x step
                
                #xB[j] = 0
                #xN[s] = alpha
                #We update the values at there destinations after the the basis change:
                xB[j] = alpha.copy()
                xN[s] = 0 #technically superfluous

                #Update basis 
                i_j=Bs[j]
                Bs[j] = i_s.copy()
                NBs[s] = i_j.copy()

                
                # in case of looping
                if i_j_prev == i_s and i_s_prev == i_j:
                    print("end due to loop")
                    print("min LambdaN: ",np.min(lambdaN))
                    return(Bs,NBs,xB,xN,µ,lambdaN)
                i_j_prev = i_j
                i_s_prev = i_s

    return(Bs,NBs,xB,xN,µ,lambdaN)

# convert to standard form function
def standard_form_E3(g, A, b, u = "empty",l = "empty"): 
    # from c'x, A'x=b, l<=x<=u

    # to c'x, Ax=b, 0<=x

    A_new = A.T #A is transposed when arriving
    (m,n)=A_new.shape
    g_new = g
    b_new = b

    if not(isinstance(u,str)):
        A_new = np.hstack((A_new,np.zeros((m,n)))) # space for new variables
        g_new = np.concatenate((g_new,np.zeros(n)))

        # upper boundary A matrix:
        Ub = np.hstack((np.eye(n),np.eye(n)))
        # add under current A and b
        A_new = np.vstack((A_new,Ub))
        b_new = np.concatenate((b_new,u))

    if not(isinstance(l,str)) and max(abs(l))>0: #test if upper bound is needed
        # should be handeld differently if negative x is needed
        # not nessesary for this problem
        n_l=len(l[l>0]) #needed lower bounds
        (m_2,n_2)=A_new.shape

        A_new = np.hstack((A_new,np.zeros((m_2,n_l)))) # space for new variables
        g_new = np.concatenate((g_new,np.zeros(n_l)))

        temp_I = np.eye(n)
        Lb = np.hstack((temp_I[l>0,:],np.zeros((n_l,n_2-n)),np.eye(n_l)))

        A_new = np.vstack((A_new,Lb))
        b_new = np.concatenate((b_new,l))

    return g_new, A_new, b_new

# finding a feasible initial point
def Find_initial_point(A,b):
    # FINDING A FEASIBLE POINT

    (m,n)=A.shape

    # new g vector (is 1 at index of t)
    g_In = np.zeros(n+1+m+m)
    g_In[n] = 1
    
    # generate filler matrices:
    e_In = np.ones((m,1))
    I_In=np.identity(m)
    zero_In = np.zeros((m,m))

    # make the new "A" constraint matrix
    temp_upper=np.concatenate((A.copy(),e_In,-I_In,zero_In),axis=1)
    temp_lower=np.concatenate((-A.copy(),e_In,zero_In,-I_In),axis=1)
    A_In = np.concatenate((temp_upper,temp_lower),axis=0)
    # print(A_In)

    # make the new "b" constraint matrix
    b_In = np.concatenate((b.copy(),-b.copy()),axis=0)
    # print(b_In)

    # make initial sets
    Bs_In = np.arange(n+1,n+1+m+m)
    NBs_In = np.arange(0,n+1)

    max_b_i = np.argmax((b_In)) #find index of max abs(b) (In this case just max of b_In)

    # print("i: ",max_b_i,n+1+max_b_i)

    NBs_In[-1] = n+1+max_b_i # add this s1 to non-basic set

    Bs_In [max_b_i] = n # add t to the Basic set

    # use simplex to find feasible initial point
    [Bs_0,NBs_0,_,_,_,_]=LP_simplex(g_In,A_In,b_In,Bs_In,NBs_In)

    Bs_0.sort() #sort and take the first m values to be the basic set
    Bs_0=Bs_0[0:m]
    NBs_0.sort() #repeat for non-basic set
    NBs_0=NBs_0[0:(n-m)]
    return Bs_0, NBs_0


# remember to change loadmat
if __name__ == '__main__':
    LP_test = sp.io.loadmat('./sources/LP_test.mat')
    # print(LP_test.keys())
    
    LP_U = LP_test["U"]
    # print("LP_U: ",LP_U.dtype)
    LP_C = LP_test["C"]
    # print("LP_C: ",LP_C.dtype)
    LP_Pd = LP_test["Pd_max"]
    # print("LP_Pd: ",LP_Pd.dtype)
    LP_Pg = LP_test["Pg_max"]
    # print("LP_Pg: ",LP_Pg.dtype)
    # change data types
    LP_U = LP_U.astype(np.int32)
    LP_C = LP_C.astype(np.int32)
    LP_Pd = LP_Pd.astype(np.int32)
    LP_Pg = LP_Pg.astype(np.int32)

    # converting data to standard form:
    LP_g = np.concatenate((-LP_U,LP_C)).flatten()
    n = len(LP_g)
    LP_A = np.vstack((np.ones((len(LP_U),1)),-np.ones((len(LP_C),1))))
    LP_b = np.zeros(1)
    l = np.zeros(n)

    u = np.concatenate((LP_Pd,LP_Pg)).flatten()
    
    [g, A, b] = standard_form_E3(LP_g,LP_A,LP_b, u ,l )

    # define initial basic and non-basic sets:
    Bs_0 = np.arange(g.shape[0]-b.shape[0],g.shape[0])
    NBs_0 = np.arange(0,g.shape[0]-b.shape[0])
    
    t_start = time.process_time() #CPU timing start

    # test if initial point is feasible. If not use simpex to find one.
    if not(np.all(np.linalg.solve(A[:,Bs_0],b)>=0)): 
        print("Infeasible initial point. Finding feasible initial point")
        [Bs_0,NBs_0]=Find_initial_point(A,b)
    [Bs,NBs,xB,xN,µ,lambdaN]=LP_simplex(g,A,b,Bs_0,NBs_0)

    t_end = time.process_time() #CPU timing end

    x_full = np.zeros(g.size)
    x_full[Bs]=xB
    # x_full[NBs]=xN #superfluous

    x = x_full[0:n]

    print("CPU time: ",t_end-t_start,"seconds")
    print("Objective value (test problem) = ",-LP_g@x)
    print("market clearing price: ",-µ[0])
    print("x saved in 'x_simplex_solution.txt'")
    np.savetxt('figures/3_7_x_simplex_solution.txt', x, delimiter=',',header="optimal x form the simplex algorithm")  




    
    
