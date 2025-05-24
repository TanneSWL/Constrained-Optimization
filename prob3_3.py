import scipy.io
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linprog
import time

# Problem 3.3

# Load data from LP_Test.mat
data = scipy.io.loadmat('sources/LP_Test.mat')

U_d = data['U'].flatten().astype('float64')   # Bid prices
C_g = data['C'].flatten()   # Offer proces

P_d_max = data['Pd_max'].flatten()  # Maximum loads
P_g_max = data['Pg_max'].flatten()  # Capacities

n_d = len(U_d)  # Number of demands
n_g = len(C_g)  # Number of generators

g = np.concatenate((-U_d, C_g))  
bounds = [(0, P_d_max[i]) for i in range(n_d)] + [(0, P_g_max[i]) for i in range(n_g)]
A = np.concatenate([np.ones(n_d), -np.ones(n_g)]).reshape(1, n_d + n_g) # Stored as row vector --> no need for transpose

b = np.array([0])

# Solve using linprog
start_time = time.time()
result = linprog(c=g, A_eq=A, b_eq=b, bounds=bounds, method='highs')
stop_time = time.time()

# Results)
x_opt = result.x
p_d_opt = x_opt[:n_d]
p_g_opt = x_opt[n_d:]
market_clearing_price = -result.eqlin.marginals[0]  # Dual variable (Lagrange multiplier). Minus because we flipped sign of g

print("Optimal demand:", p_d_opt)
print("Optimal generation:", p_g_opt)
print("Solution:", x_opt)
print("Market clearing price:", market_clearing_price)
print("Objective value:", -result.fun)  # flip sign back to maximize welfare

# solution statistics
print("Number of iterations:", result.nit)
print("CPU time (seconds):", stop_time - start_time)


# Plot supply-demand curve
# Demand sorted decreasing (higher utility first)
sorted_Ud_idx = np.argsort(-U_d)
sorted_Ud = U_d[sorted_Ud_idx]
sorted_Pd_max = P_d_max[sorted_Ud_idx]

# Supply sorted increasing (lower cost first)
sorted_Cg_idx = np.argsort(C_g)
sorted_Cg = C_g[sorted_Cg_idx]
sorted_Pg_max = P_g_max[sorted_Cg_idx]

# Build cumulative quantities
demand_cum = np.cumsum(sorted_Pd_max)
supply_cum = np.cumsum(sorted_Pg_max)

plt.figure(figsize=(8, 5))
plt.step(demand_cum, sorted_Ud, where='post', label='Demand', linestyle='--')
plt.step(supply_cum, sorted_Cg, where='post', label='Supply', linestyle='-')
plt.axhline(y=market_clearing_price, color='red', linestyle=':', label='Market Clearing Price')
plt.xlabel('Energy Quantity')
plt.ylabel('Price')
plt.title('Supply and Demand Curve')
plt.legend()
plt.grid(True)
plt.show()