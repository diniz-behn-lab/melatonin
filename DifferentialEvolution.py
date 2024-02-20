# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 18:18:02 2024

@author: sstowe

Optimize the melatonin forcing parameters (B1, B2, theta_M1, theta_M2, epsilon)
"""


#------- Set Up ----------------

import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
import time


#------ Set bounds & ICs for the five parameters to be optimized ------------

opt_lower_bound = np.array([-0.5,-1,-1,0,0])
opt_upper_bound = np.array([0.5,1,1,np.pi/2,np.pi/2])
bounds = [(l,u) for l, u in zip(opt_lower_bound,opt_upper_bound)]

x0 =np.array([0,-.3,0,5*np.pi/12,0])

#---------- Create an array of the data values -----------------

Burgess_2008_PRC = [
-0.330434783,#0
1.095652174,#4
#2.182608696,#5
1.634782609,#9
0.782608696,#13
-0.556521739,#17
-1.782608696,#21
#-2.695652174,#22
]

# Eyeballed it
Burgess_2008_PRC_times = [14.5,18.5,22.5,2.5,6.5,10.5]


