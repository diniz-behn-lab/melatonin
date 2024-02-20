# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 18:18:02 2024

@author: sstowe

Optimize the melatonin forcing parameters (B1, B2, theta_M1, theta_M2, epsilon)
"""


#------- Set Up ----------------

import numpy as np
from scipy.optimize import differential_evolution
from run_HannayBreslow_Burgess2008_PRC import objective_func


#---------- Create an array of the data values -----------------

Burgess_2008_PRC = [
0.782608696,#13
-0.556521739,#17
-1.782608696,#21
#-2.695652174,#22
-0.330434783,#0
1.095652174,#4
#2.182608696,#5
1.634782609,#9
]

# Eyeballed it
Burgess_2008_PRC_times = [2.5,6.5,10.5,14.5,18.5,22.5]

data_vals = Burgess_2008_PRC







#params = np.array([0.74545016,-0.05671999,0.76024892,-0.05994563,-0.18366069])
#anything = objective_func(data_vals)


#------------- Run the differential evolution algorithm ----------
# To be optimized: [(B_1), (B_2), (theta_M1), (theta_M2), (epsilon)]

# Set bounds for the five parameters to be optimized
bounds = [(-1, 1), (-1, 1), (0, np.pi/2), (0, np.pi/2), (-0.5, 0.5)]
#params = np.array([0,0,0,0,0])

optimized = differential_evolution(objective_func, bounds)#, args=(data_vals,), popsize=15, maxiter=4, disp=True)

# Print the best solution found
print(optimized.x)
# Print the function value at the best solution
print(optimized.fun)
# Print the number of iterations
print(optimized.nit)
# Print the success flag
print(optimized.success)




'''

# B_1, B_2, theta_M1, theta_M2, epsilon
opt_lower_bound = np.array([-1,-1,0,0,-0.5])
opt_upper_bound = np.array([1,1,np.pi/2,np.pi/2,0.5])
bounds = [(l,u) for l, u in zip(opt_lower_bound,opt_upper_bound)]

#x0 = np.array([0,-.3,0,5*np.pi/12,0])
#x0 = np.array([0.74545016,-0.05671999,0.76024892,-0.05994563,-0.18366069])

x = [0.74545016,-0.05671999,0.76024892,-0.05994563,-0.18366069]

optimized = differential_evolution(run_HannayBreslow(x,data_vals), bounds, args=(data_vals,), popsize=15, maxiter=4, disp=True)

print(optimized)

'''

