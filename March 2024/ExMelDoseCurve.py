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
y = [45, 150, 400, 2000, 6500] # pg/mL (plasma)

x_line = np.linspace(0,10,100);
y_line = 5.3861*pow(x_line,3) - 30.502*pow(x_line,2) + 415.01*(x_line) + 14.06;

plt.plot(x_line,y_line,lw=3,color='hotpink')
plt.plot(x,y,'o',color='black')
plt.yscale('log')
plt.ylim([10,10000])
plt.title('Exogenous Melatonin Dosing Curve')
plt.xlabel('Exogenous Melatonin Dose (mg)')
plt.ylabel('Peak Plasma Melatonin Concentration (pg/mL)')
