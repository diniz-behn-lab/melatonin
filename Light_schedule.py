# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 17:30:20 2024

@author: sstowe

Code to generate plot of light level
"""

import numpy as np
import matplotlib.pyplot as plt


#------------- Standard Light Schedule (16:8) --------------

j = 2 # Number of days to be plotted

n = j*24

standard_lightschedule = list()

for t in range(0, n):
    full_light = 1000
    dim_light = 300 # reduced light
    wake_time = 7
    sleep_time = 23
    sun_up = 8
    sun_down = 19

    is_awake = np.mod(t - wake_time,24) <= np.mod(sleep_time - wake_time,24)
    sun_is_up = np.mod(t - sun_up,24) <= np.mod(sun_down - sun_up,24)

    standard_lightschedule.append(is_awake*(full_light*sun_is_up + dim_light*(1 - sun_is_up)))

        
plt.plot(range(0,n),standard_lightschedule)
plt.title("Standard Light:Dark Schedule (16:8)")
plt.show()


#------------------- Wyatt 2006 Baseline Light Schedule ------------

j = 2 # Number of days to be plotted

n = j*24

WyattBaseline_lightschedule = list()

for t in range(0, n):
    full_light = 15
    dim_light = 15 # reduced light
    wake_time = 6
    sleep_time = 22
    sun_up = 6
    sun_down = 22

    is_awake = np.mod(t - wake_time,24) <= np.mod(sleep_time - wake_time,24)
    sun_is_up = np.mod(t - sun_up,24) <= np.mod(sun_down - sun_up,24)

    WyattBaseline_lightschedule.append(is_awake*(full_light*sun_is_up + dim_light*(1 - sun_is_up)))

        
plt.plot(range(0,n),WyattBaseline_lightschedule)
plt.title("Wyatt 2006 Light:Dark Schedule")
plt.show()


#------------------- Burgess 2008 Ultradian Light Schedule ------------

j = 1 # Number of days to be plotted

n = j*24

BurgessUltradian_lightschedule = list()
time = np.linspace(0,n,97)

for t in time:
    full_light = 150
    
    if 1 <= np.mod(t,4) <= 2.5 :
        BurgessUltradian_lightschedule.append(0)
    else:
        BurgessUltradian_lightschedule.append(full_light)

        
plt.plot(time[0:40],BurgessUltradian_lightschedule[0:40])
plt.title("Burgess 2008 Ultradian Light:Dark Schedule (2.5:1.5)")
#plt.axvline(1)
#plt.axvline(2.5)
#plt.axvline(5)
#plt.axvline(6.5)
plt.show()