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
import time


#---------- Create an array of the data values -----------------

Burgess_2008_PRC = [
0.782608696,#13
-0.556521739,#17
-1.782608696,#21
#-2.695652174,#22
-0.330434783,#0
#1.095652174,#4
2.182608696,#5
1.634782609,#9
]

# Eyeballed it
Burgess_2008_PRC_times = [2.5,6.5,10.5,14.5,18.5,22.5]

data_vals = Burgess_2008_PRC


#------------- Run the differential evolution algorithm ----------
now = time.time()
# To be optimized: [(B_1), (B_2), (theta_M1), (theta_M2), (epsilon)]

# Set bounds for the five parameters to be optimized
bounds = [(-1, 1), (-1, 1), (0, np.pi/2), (0, np.pi/2), (-0.5, 0.5)]
#bounds = [(-0.1, 0.1), (-0.1, 0.1), (0, 0.1), (0, 0.1), (-0.1, 0.1)]

optimized = differential_evolution(objective_func, bounds, args=(data_vals,), popsize=15, maxiter=20, disp=True)

# Print how long (mins) the run took
print((time.time() - now)/60) 

# Print the best solution found
print(optimized.x)
# Print the function value at the best solution
print(optimized.fun)
# Print the number of iterations
print(optimized.nit)
# Print the success flag
print(optimized.success)
# Print the success flag
print(optimized.message)




