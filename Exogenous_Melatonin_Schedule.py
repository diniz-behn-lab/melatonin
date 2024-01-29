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
    dosage = 0.2
    timing = 18

    mel = np.mod(t, 24) == timing

    ex_melatonin.append(mel*(dosage/1e-9))
        
plt.plot(range(0,n),ex_melatonin)