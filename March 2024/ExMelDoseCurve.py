# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 16:01:57 2024

@author: sstowe

Plot the curve for melatonin dosing 

"""


#-------- Set Up -------------------------------

# Imports 

import numpy as np
import matplotlib.pyplot as plt

#-------- Plot the curve and the data -----

x = [0.1, 0.3, 1, 5, 10]
y = [45, 150, 400, 1900, 6500] # pg/mL (plasma)

x_line = np.linspace(0,10,100);
y_line = 6.413*pow(x_line,3) - 41.94*pow(x_line,2) + 426.9*(x_line) + 12.36;

plt.plot(x_line,y_line,lw=5,color='mediumvioletred')
plt.plot(x,y,'o',color='black',markersize=8)
plt.yscale('log')
plt.ylim([10,10000])
plt.title('Exogenous Melatonin Dosing Curve')
plt.xlabel('Exogenous Melatonin Dose (mg)')
plt.ylabel('Plasma Melatonin (pg/mL)')
