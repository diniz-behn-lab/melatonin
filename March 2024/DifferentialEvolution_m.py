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

''' 
# ------- Fit to curve -----
Zeitzer_2000 = [
0, # 3
0, # 10
0, # 20
0, # 30
0, # 50
0.1, # 60
0.2, # 70
0.3, # 80
0.5, # 100
0.7, # 150
0.8, # 180
0.9, # 300
0.9, # 500
0.95, # 1000
0.95 # 5000
]

# Eyeballed it
Zeitzer_2000_lux = [3,10,20,30,50,60,70,80,100,150,180,300,500,1000,5000]
'''

# ----- Fit to data (determined from WebPlotDigitizer) -----
Zeitzer_2000 = [0.11,
0.08,
-0.08,
-0.01,
-0.12,
0.23,
0.88,
0.26,
0.75,
0.80,
0.69,
0.86,
0.96,
0.99,
0.92,
0.96,
0.93,
0.97,
0.98,
0.95,
0.98]

data_vals = Zeitzer_2000


#------------- Run the differential evolution algorithm ----------
now = time.time()
# To be optimized: [(B_1), (B_2), (theta_M1), (theta_M2), (epsilon)]

# Set bounds for the one parameter to be optimized
bounds = [(4, 6)]

optimized = differential_evolution(objective_func, bounds, args=(data_vals,), popsize=75, maxiter=4, disp=True)

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

