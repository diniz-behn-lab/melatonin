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

# ----- Fit to data (determined from WebPlotDigitizer) -----
# n = 21
Zeitzer_2000 = [0.11,
0.08,
-0.08,
-0.01,
-0.12,
0.22,
0.88,
0.26,
0.80,
0.75,
0.69,
0.85,
0.96,
0.99,
0.92,
0.96,
0.94,
0.97,
0.98,
0.95,
0.98
]

data_vals = Zeitzer_2000


#------------- Run the differential evolution algorithm ----------
now = time.time()
# To be optimized: [(m)]

# Set bounds for the one parameter to be optimized
bounds = [(4, 5)]

optimized = differential_evolution(objective_func, bounds, args=(data_vals,), popsize=50, maxiter=4, tol=0.01, disp=True)

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

