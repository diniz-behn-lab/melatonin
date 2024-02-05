# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 17:30:20 2024

@author: sstowe

Code to generate plot of light level
"""

import numpy as np
import matplotlib.pyplot as plt

j = 2 # Number of days to be plotted

n = j*24

light_level = list()

for t in range(0, n):
    full_light = 10#1000
    dim_light = 10#300 # reduced light
    wake_time = 7
    sleep_time = 23
    sun_up = 8
    sun_down = 19

    is_awake = np.mod(t - wake_time,24) <= np.mod(sleep_time - wake_time,24)
    sun_is_up = np.mod(t - sun_up,24) <= np.mod(sun_down - sun_up,24)

    light_level.append(is_awake*(full_light*sun_is_up + dim_light*(1 - sun_is_up)))

    #return light_level
        
plt.plot(range(0,n),light_level)
