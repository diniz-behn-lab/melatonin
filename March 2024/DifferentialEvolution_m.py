# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 13:23:47 2024

@author: sstowe

Optimize the constant m

"""

#------- Set Up ----------------

import numpy as np
from scipy.optimize import differential_evolution
from run_HannayBreslow_Zeitzer2000_MelSupp import objective_func
import time


#---------- Create an array of the data values -----------------

Zeitzer_2000 = [
0,
0,
0,
0.1,
0.2,
0.25,
0.8,
0.9,
1.0,
1.0
]

# Eyeballed it
Zeitzer_2000_lux = [3,10,50,60,80,100,200,500,1000,5000]

data_vals = Zeitzer_2000


#------------- Run the differential evolution algorithm ----------
now = time.time()
# To be optimized: [(B_1), (B_2), (theta_M1), (theta_M2), (epsilon)]

# Set bounds for the one parameter to be optimized
bounds = [(-10, 10)]

optimized = differential_evolution(objective_func, bounds, args=(data_vals,), popsize=15, maxiter=4, disp=True)

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

