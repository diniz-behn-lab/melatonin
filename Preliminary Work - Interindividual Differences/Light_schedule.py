# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 17:30:20 2024

@author: sstowe

Code to generate plot of light level
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 15})


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

        
plt.plot(range(0,n),standard_lightschedule,color='black',lw=3)
plt.xticks(np.arange(0, 52, 6)) 
plt.title("Standard Light:Dark Schedule")
plt.xlabel('Time (hours)')
plt.ylabel('Light Intensity (lux)')
#plt.axvline(7)
plt.show()




# ---------- Burke 2013 5 Day Protocol ----------
# Intervention 1: Bright morning light 
j = 5

n = j*24

BurkeProtocol_lightschedule = list()

for t in range(0,n): 
    wake = 1000
    dim = 2
    dark = 0
    bright = 3000
    wake_time = 6
    sleep_time = 22
    
    if 0 <= t < 24*1: # Enter the lab 4hrs before habitual sleep
        if t < wake_time:
            lux = dark
        elif wake_time <= t < sleep_time-4: 
            lux = wake
        elif sleep_time-4 <= t < sleep_time:
            lux = dim
        else:
            lux = dark
            
        BurkeProtocol_lightschedule.append(lux)

    elif 24*1 <= t < 24*2: # Start of CR1 (28 h)
        t = np.mod(t,24)
        if t < wake_time:
            lux = dark
        else:
            lux = dim
            
        BurkeProtocol_lightschedule.append(lux)
        
    elif 24*2 <= t <  24*3: 
        t = np.mod(t,24)
        if t < wake_time+4: # Need 4hrs more of dim to complete the 28h CR
            lux = dim
        elif wake_time+4 <= t < wake_time+9:
            lux = dark
        elif wake_time+9 <= t < sleep_time:
            lux = dim
        else:
            lux = dark
            
        BurkeProtocol_lightschedule.append(lux)
        
    elif 24*3 <= t <  24*4: # Light intervention and start of CR2 (19h)
        t = np.mod(t,24)
        if t < wake_time-1:
            lux = dark
        elif wake_time-1 <= t < wake_time+2:
            lux = bright 
        else: 
            lux = dim
            
        BurkeProtocol_lightschedule.append(lux)
    
    elif 24*4 <= t <  24*5: # Day 2
        t = np.mod(t,24)
        if t < wake_time-3:
            lux = dim
        elif wake_time-3 <= t < wake_time+3:
            lux = dark 
        else: 
            lux = wake
            
        BurkeProtocol_lightschedule.append(lux)
    
    

plt.plot(range(0,n),BurkeProtocol_lightschedule,color='black',lw=3)
plt.xticks(np.arange(0, 122, 24))
plt.xlabel('Time (hours)')
plt.ylabel('Light Intensity (lux)')
plt.title("Burke 2013 Protocol")
#plt.axvline(24,linestyle='dashed',color='grey')
#plt.axvline(24*2,linestyle='dashed',color='grey')
#plt.axvline(24*3,linestyle='dashed',color='grey')
#plt.axvline(24*4,linestyle='dashed',color='grey')
plt.axvline((sleep_time-5.75)+(24*2),color='mediumvioletred',lw=3)
plt.axvspan(78, 81, facecolor='gold', alpha=0.4)
#plt.axvspan(21.25+(24*2), 3.75+(24*3), facecolor='grey', alpha=0.4)
#plt.axvline(7+24,color='orange')
#plt.axvline(23,color='orange')
plt.legend(['5.0 mg dose'],loc='upper left')
plt.show()     
      
 

