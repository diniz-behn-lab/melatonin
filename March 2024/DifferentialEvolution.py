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
#from run_HannayBreslow_Burgess2010_PRC import objective_func
import time


#---------- Create an array of the data values -----------------

# From WebPlotDigitizer (n = 27)
Burgess_2008_PRC = [-0.43,
                    -0.08,
                    0.45,
                    0.78,
                    -0.17,
                    0.08,
                    -0.37,
                    -0.55,
                    -1.17,
                    -2.69,
                    -1.02,
                    -1.78,
                    -1.37,
                    -1.05,
                    -0.43,
                    0.35,
                    -0.62,
                    1.00,
                    -0.32,
                    1.58,
                    2.60,
                    1.09,
                    2.18,
                    1.66,
                    0.60,
                    1.68,
                    1.63
                    ]

# From WebPlotDigitizer (n = 34)
Burgess_2010_PRC = [1.2,
                    -0.5,
                    -1.5,
                    -1.2,
                    0.7,
                    -0.1,
                    -1.0,
                    -0.6,
                    0.9,
                    1.2,
                    -1.2,
                    -1.1,
                    0.9,
                    -1.8,
                    -1.8,
                    -0.3,
                    -1.2,
                    -0.9,
                    -0.7,
                    -1.3,
                    -0.9,
                    -0.3,
                    0.7,
                    1.4,
                    1.7,
                    0.7,
                    2.7,
                    2.9,
                    -0.2,
                    0.8,
                    2.5,
                    1.0,
                    0.9,
                    1.0
                    ]


data_vals = Burgess_2008_PRC
#data_vals = Burgess_2010_PRC


#------------- Run the differential evolution algorithm ----------
now = time.time()
# To be optimized: [(B_1), (theta_M1), (B_2), (theta_M2), (epsilon)]

# Set bounds for the five parameters to be optimized
#bounds = [(-1, 1), (0, np.pi/2), (-1, 1), (0, np.pi/2), (-0.5, 0.5)]
#bounds = [(-3, 3), (-np.pi/2, np.pi/2), (-3, 3), (-np.pi/2, np.pi/2), (-0.5, 0.5)]
bounds = [(-1.5, 1.5), (-np.pi/2, np.pi/2), (-1.5, 1.5), (-np.pi/2, np.pi/2), (-0.5, 0.5)]

optimized = differential_evolution(objective_func, bounds, args=(data_vals,), popsize=20, maxiter=50, disp=True)

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




