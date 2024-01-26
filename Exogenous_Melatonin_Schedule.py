# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 12:40:23 2024

@author: sstowe

Code to generate plot of exogenous melatonin administration
"""

import numpy as np
import matplotlib.pyplot as plt

j = 1 # Number of days to be plotted

n = j*24

ex_melatonin = list()

for t in range(0, n):
    dosage = 10.0
    timing = 10

    if np.mod(t,24) == timing:
        melatonin = dosage
    else:
        melatonin = 0

    ex_melatonin.append(melatonin)

    #return light_level
        
plt.plot(range(0,n),ex_melatonin)