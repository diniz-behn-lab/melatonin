# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 20:49:12 2024

@author: sstowe

Small example of differential evolution 

"""

# Import the differential_evolution function
from scipy.optimize import differential_evolution

# Define the objective function (Rosenbrock function)
def rosenbrock(x):
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2 for i in range(len(x) - 1))

# Define the bounds of the input variables (2 dimensions)
bounds = [(-5, 5), (-5, 5)]

# Call the differential_evolution function with the objective function and the bounds
result = differential_evolution(rosenbrock, bounds)

# Print the best solution found
print(result.x)
# Print the function value at the best solution
print(result.fun)
# Print the number of iterations
print(result.nit)
# Print the success flag
print(result.success)