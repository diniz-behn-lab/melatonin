# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 17:30:20 2024

@author: sstowe

Code to generate plot of light level
"""

import numpy as np
import matplotlib.pyplot as plt

'''
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

        
plt.plot(range(0,n),standard_lightschedule,color='goldenrod')
plt.title("Standard Light:Dark Schedule (16:8)")
#plt.axvline(7)
plt.show()


    
#------------- Light at Night Light Schedule (16:8) --------------

j = 2 # Number of days to be plotted

n = j*24

standard_lightschedule = list()

for t in range(0, n):
    full_light = 1000
    dim_light = 300 # reduced light
    wake_time = 10
    sleep_time = 2
    sun_up = 8
    sun_down = 19

    is_awake = np.mod(t - wake_time,24) <= np.mod(sleep_time - wake_time,24)
    sun_is_up = np.mod(t - sun_up,24) <= np.mod(sun_down - sun_up,24)

    standard_lightschedule.append(is_awake*(full_light*sun_is_up + dim_light*(1 - sun_is_up)))

        
plt.plot(range(0,n),standard_lightschedule,color='goldenrod')
plt.title("Night Owl Light:Dark Schedule (16:8)")
#plt.axvline(8)
plt.show()    


#------------- Light in the Morning Light Schedule (16:8) --------------

j = 2 # Number of days to be plotted

n = j*24

standard_lightschedule = list()

for t in range(0, n):
    full_light = 1000
    dim_light = 300 # reduced light
    wake_time = 4
    sleep_time = 20
    sun_up = 8
    sun_down = 19

    is_awake = np.mod(t - wake_time,24) <= np.mod(sleep_time - wake_time,24)
    sun_is_up = np.mod(t - sun_up,24) <= np.mod(sun_down - sun_up,24)

    standard_lightschedule.append(is_awake*(full_light*sun_is_up + dim_light*(1 - sun_is_up)))

        
plt.plot(range(0,n),standard_lightschedule,color='goldenrod')
#plt.axvline(5)
plt.title("Morning Lark Light:Dark Schedule (16:8)")
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

        
#plt.plot(time[0:40],BurgessUltradian_lightschedule[0:40])
plt.plot(time,BurgessUltradian_lightschedule)
plt.title("Burgess 2008 Ultradian Light:Dark Schedule (2.5:1.5)")
#plt.axvline(1)
#plt.axvline(2.5)
#plt.axvline(5)
#plt.axvline(6.5)
plt.show()


#------------------- Burgess 2008 5 Day Protocol ------------

j = 5 # Number of days to be plotted

n = j*24

BurgessProtocol_lightschedule = list()
time = np.linspace(0,n,100)

for t in time:
    full_light = 150
    
    if 24 <= t < 48:
        if 1 <= np.mod(t,4) <= 2.5 :
            BurgessProtocol_lightschedule.append(0)
        else:
            BurgessProtocol_lightschedule.append(full_light)
    if 48 <= t < 72:
        if 1 <= np.mod(t,4) <= 2.5 :
            BurgessProtocol_lightschedule.append(0)
        else:
            BurgessProtocol_lightschedule.append(full_light)
    if 72 <= t < 96:
        if 1 <= np.mod(t,4) <= 2.5 :
            BurgessProtocol_lightschedule.append(0)
        else:
            BurgessProtocol_lightschedule.append(full_light)
    else:
        BurgessProtocol_lightschedule.append(5)
        
plt.plot(time,BurgessProtocol_lightschedule)
plt.title("Burgess 2008 Protocol Light:Dark Schedule")
#plt.axvline(1)
#plt.axvline(2.5)
#plt.axvline(5)
#plt.axvline(6.5)
plt.show()



# ---------- Zeitzer 9 Day Protocol ----------

j = 9 

n = j*24

ZeitzerProtocol_lightschedule = list()
CBTmin = 4

for t in range(0,n): 
    bright_light = 1000
    dim_light = 150
    dark = 0
    
    if 78 <= t < 96:
        if np.mod(t,24) > np.mod(CBTmin - 6.75,24):
            ZeitzerProtocol_lightschedule.append(bright_light)
        else: 
            ZeitzerProtocol_lightschedule.append(dim_light)    
    if 96 <= t < 120:
        if np.mod(t,24) < np.mod(CBTmin - 0.25,24):
            ZeitzerProtocol_lightschedule.append(bright_light)
        else:
            ZeitzerProtocol_lightschedule.append(dim_light)
    else:
        ZeitzerProtocol_lightschedule.append(dim_light)
    

plt.plot(range(0,n),ZeitzerProtocol_lightschedule)
plt.title("Zeitzer 2000 Protocol Light:Dark Schedule")
plt.axvline(72)
plt.axvline(100)
#plt.axvline(5)
#plt.axvline(6.5)
plt.show()




# ---------- Zeitzer 2 Days of Protocol ----------

j = 2

n = j*24

ZeitzerProtocol_lightschedule = list()
CBTmin = 4

for t in range(0,n): 
    bright_light = 300
    dim_light = 10
    dark = 0
    
    if 0 <= t < 24:
        if np.mod(t,24) > np.mod(CBTmin - 6.75,24):
            ZeitzerProtocol_lightschedule.append(bright_light)
        else: 
            ZeitzerProtocol_lightschedule.append(dim_light)    
    if 24 <= t < 48:
        if np.mod(t,24) < np.mod(CBTmin - 0.25,24):
            ZeitzerProtocol_lightschedule.append(bright_light)
        else:
            ZeitzerProtocol_lightschedule.append(dim_light)
    

plt.plot(range(0,n),ZeitzerProtocol_lightschedule)
plt.title("Zeitzer 2000 Protocol Light:Dark Schedule")
plt.axvline(28,color='black')
plt.axvline(24,linestyle='dashed')
#plt.axvline(5)
#plt.axvline(6.5)
plt.show()



# ---------- Zeitzer 5 Days of Protocol ----------

j = 5

n = j*24

ZeitzerProtocol_lightschedule = list()
CBTmin = 4

for t in range(0,n): 
    bright_light = 550
    dim_light = 10
    dark = 0
    
    if 0 <= t < 24*3:
        full_light = 150
        dim_light = 150
        wake_time = 7
        sleep_time = 23
        sun_up = 8
        sun_down = 19
        
        is_awake = np.mod(t - wake_time,24) <= np.mod(sleep_time - wake_time,24)
        sun_is_up = np.mod(t - sun_up,24) <= np.mod(sun_down - sun_up,24)

        light = is_awake*(full_light*sun_is_up + dim_light*(1 - sun_is_up))
        
        ZeitzerProtocol_lightschedule.append(light)
        
    if 24*3 <= t < 24*4:
        if np.mod(t,24) > np.mod(CBTmin - 6.75,24):
            ZeitzerProtocol_lightschedule.append(bright_light)
        else: 
            ZeitzerProtocol_lightschedule.append(dim_light)    
    if 24*4 <= t < 24*5:
        if np.mod(t,24) < np.mod(CBTmin - 0.25,24):
            ZeitzerProtocol_lightschedule.append(bright_light)
        else:
            ZeitzerProtocol_lightschedule.append(dim_light)
    
plt.axvline(4+(24*4),color='black',linestyle='dashdot')
plt.plot(range(0,n),ZeitzerProtocol_lightschedule,color='goldenrod')
plt.xlabel('Time (hours)')
plt.ylabel('Light Intensity (lux)')
plt.title("Zeitzer 2000 Protocol Light:Dark Schedule")
plt.axvline(24,linestyle='dashed',color='grey')
plt.axvline(24*2,linestyle='dashed',color='grey')
plt.axvline(24*3,linestyle='dashed',color='grey')
plt.axvline(24*4,linestyle='dashed',color='grey')
plt.axvspan(21.25+(24*3), 3.75+(24*4), facecolor='gold', alpha=0.4)
plt.axvspan(21.25+(24*2), 3.75+(24*3), facecolor='grey', alpha=0.4)
#plt.axvline(7+24,color='orange')
#plt.axvline(23,color='orange')
plt.legend(['CBTmin'])
plt.show()




# ---------- Burgess 2008 & 2010 Protocol ---------

j = 5

n = j*24

BurgessProtocol_lightschedule = list()
DLMO = 21
mel_timing = 0 # Timing after DLMO 

for t in range(0,n): 
    bright_light = 40
    dim_light = 5
    dark = 0
    
    if 24 <= t < 96: # Days 2,3,4 (ultradian light schedule)
        t = np.mod(t,24)
        dose_time = np.mod(DLMO+mel_timing,24)
        
        # Find the lights on time
        if dose_time >= 4:
            lights_on = np.mod(dose_time,4)
        else: 
            lights_on = dose_time

        # Ultradian light schedule (2.5:1.5 light:dark)
        if 0 <= np.mod(t-lights_on, 4) < 2.5:
            light = bright_light
            BurgessProtocol_lightschedule.append(light)
        else: 
            light = dark
            BurgessProtocol_lightschedule.append(light)
            
    else:
        light = dim_light
        BurgessProtocol_lightschedule.append(light)
       
plt.axvline(21,color='black',linestyle='dashed')        
plt.axvline(dose_time+24,color='hotpink')       
plt.plot(range(0,n),BurgessProtocol_lightschedule,color='goldenrod')
plt.title("Burgess 2008/2010 Protocol Light:Dark Schedule")
plt.axvline(dose_time+24+24,color='hotpink')
plt.axvline(dose_time+24+24+24,color='hotpink')
plt.axvline(24,linestyle='dashed',color='grey')
plt.axvline(24*2,linestyle='dashed',color='grey')
plt.axvline(24*3,linestyle='dashed',color='grey')
plt.axvline(24*4,linestyle='dashed',color='grey')
plt.ylim([-1, 70])
plt.xlabel('Time (hours)')
plt.ylabel('Light Intensity (lux)')
plt.legend(['DLMO','Exogenous Melatonin Dose'])
#plt.axvline(6.5)
plt.show()        

 

# ---------- Dawson 1995 7 Days of Protocol ----------
# Intervention 1: Bright Light
j = 7

n = j*24

DawsonProtocol_lightschedule = list()

for t in range(0,n): 
    wake = 1000
    dim = 50
    dark = 0
    free_time = 500 # Not a controlled level in the study
    bright_light = 4000 # Intervention, set to 50 for placebo 
    
    if 0 <= t < 24*1: # Enter the lab
    
        if t < 7:
            lux = dark
        else: 
            lux = wake
            
        DawsonProtocol_lightschedule.append(lux)

    elif 24*1 <= t < 24*2: # Adaption Day, first baseline DLMO assessement 
        t = np.mod(t,24)
        if t < 8:
            lux = dark
        elif 8 <= t < 16:
            lux = wake
        elif 16<= t < 23: 
            lux = dim
        else:
            lux = dark
            
        DawsonProtocol_lightschedule.append(lux)
        
    elif 24*2 <= t <  24*3: # Baseline Day, second baseline DLMO assessment
        t = np.mod(t,24)
        if t < 9: 
            lux = dark
        elif 9 <= t < 16:
            lux = wake
        elif 16 <= t < 23:
            lux = dim
        else:
            lux = wake
            
        DawsonProtocol_lightschedule.append(lux)
        
    elif 24*3 <= t <  24*4: # Day 1
        t = np.mod(t,24)
        if t < 4:
            lux = bright_light
        elif 4 <= t < 7:
            lux = wake
        elif 7 <= t < 9:
            lux = dim 
        elif 9 <= t < 17:
            lux = dark
        elif 17 <= t < 23:
            lux = free_time
        else: 
            lux = wake
            
        DawsonProtocol_lightschedule.append(lux)
    
    elif 24*4 <= t <  24*5: # Day 2
        t = np.mod(t,24)
        if t < 4:
            lux = bright_light
        elif 4 <= t < 7:
            lux = wake
        elif 7 <= t < 9:
            lux = dim 
        elif 9 <= t < 17:
            lux = dark
        elif 17 <= t < 23:
            lux = free_time
        else: 
            lux = wake
            
        DawsonProtocol_lightschedule.append(lux)
    
    elif 24*5 <= t <  24*6: # Day 3
        t = np.mod(t,24)
        if t < 4:
            lux = bright_light
        elif 4 <= t < 7:
            lux = wake
        elif 7 <= t < 9:
            lux = dim 
        elif 9 <= t < 17:
            lux = dark
        else: 
            lux = dim
            
        DawsonProtocol_lightschedule.append(lux)
    
    elif 24*6 <= t <  24*7: # Day 4
        lux = dim
        
        DawsonProtocol_lightschedule.append(lux)
    

plt.plot(range(0,n),DawsonProtocol_lightschedule,color='goldenrod')
plt.xlabel('Time (hours)')
plt.ylabel('Light Intensity (lux)')
plt.title("Dawson 1995 Protocol")
plt.axvline(24,linestyle='dashed',color='grey')
plt.axvline(24*2,linestyle='dashed',color='grey')
plt.axvline(24*3,linestyle='dashed',color='grey')
plt.axvline(24*4,linestyle='dashed',color='grey')
plt.axvline(24*5,linestyle='dashed',color='grey')
plt.axvline(24*6,linestyle='dashed',color='grey')
#plt.axvspan(21.25+(24*3), 3.75+(24*4), facecolor='gold', alpha=0.4)
#plt.axvspan(21.25+(24*2), 3.75+(24*3), facecolor='grey', alpha=0.4)
#plt.axvline(7+24,color='orange')
#plt.axvline(23,color='orange')
#plt.legend(['CBTmin'])
plt.show()       
    

# ---------- Dawson 1995 7 Days of Protocol ----------
# Intervention 2: Exogenous Melatonin
j = 7

n = j*24

DawsonProtocol_lightschedule = list()

for t in range(0,n): 
    wake = 1000
    dim = 50
    dark = 0
    free_time = 500 # Not a controlled level in the study
    
    if 0 <= t < 24*1: # Enter the lab
    
        if t < 7:
            lux = dark
        else: 
            lux = wake
            
        DawsonProtocol_lightschedule.append(lux)

    elif 24*1 <= t < 24*2: # Adaption Day, first baseline DLMO assessement 
        t = np.mod(t,24)
        if t < 8:
            lux = dark
        elif 8 <= t < 16:
            lux = wake
        elif 16<= t < 23: 
            lux = dim
        else:
            lux = dark
            
        DawsonProtocol_lightschedule.append(lux)
        
    elif 24*2 <= t <  24*3: # Baseline Day, second baseline DLMO assessment
        t = np.mod(t,24)
        if t < 9: 
            lux = dark
        elif 9 <= t < 16:
            lux = wake
        elif 16 <= t < 23:
            lux = dim
        else:
            lux = wake
            
        DawsonProtocol_lightschedule.append(lux)
        
    elif 24*3 <= t <  24*4: # Day 1
        t = np.mod(t,24)
        if t < 7:
            lux = wake
        elif 7 <= t < 9:
            lux = dim 
        elif 9 <= t < 17:
            lux = dark
        elif 17 <= t < 23:
            lux = free_time
        else: 
            lux = wake
            
        DawsonProtocol_lightschedule.append(lux)
    
    elif 24*4 <= t <  24*5: # Day 2
        t = np.mod(t,24)
        if t < 7:
            lux = wake
        elif 7 <= t < 9:
            lux = dim 
        elif 9 <= t < 17:
            lux = dark
        elif 17 <= t < 23:
            lux = free_time
        else: 
            lux = wake
            
        DawsonProtocol_lightschedule.append(lux)
    
    elif 24*5 <= t <  24*6: # Day 3
        t = np.mod(t,24)
        if t < 7:
            lux = wake
        elif 7 <= t < 9:
            lux = dim 
        elif 9 <= t < 17:
            lux = dark
        else: 
            lux = dim
            
        DawsonProtocol_lightschedule.append(lux)
    
    elif 24*6 <= t <  24*7: # Day 4
        lux = dim
        
        DawsonProtocol_lightschedule.append(lux)
    

plt.plot(range(0,n),DawsonProtocol_lightschedule,color='goldenrod')
plt.xlabel('Time (hours)')
plt.ylabel('Light Intensity (lux)')
plt.title("Dawson 1995 Protocol")
plt.axvline(24,linestyle='dashed',color='grey')
plt.axvline(24*2,linestyle='dashed',color='grey')
plt.axvline(24*3,linestyle='dashed',color='grey')
plt.axvline(24*4,linestyle='dashed',color='grey')
plt.axvline(24*5,linestyle='dashed',color='grey')
plt.axvline(24*6,linestyle='dashed',color='grey')
plt.axvline(8+24*3,color='hotpink')
plt.axvline(8+24*4,color='hotpink')
plt.axvline(8+24*5,color='hotpink')
plt.axvline(11+24*3,color='hotpink')
plt.axvline(11+24*4,color='hotpink')
plt.axvline(11+24*5,color='hotpink')
plt.axvline(14+24*3,color='hotpink')
plt.axvline(14+24*4,color='hotpink')
plt.axvline(14+24*5,color='hotpink')
#plt.axvspan(21.25+(24*3), 3.75+(24*4), facecolor='gold', alpha=0.4)
#plt.axvspan(21.25+(24*2), 3.75+(24*3), facecolor='grey', alpha=0.4)
#plt.axvline(7+24,color='orange')
#plt.axvline(23,color='orange')
#plt.legend(['CBTmin'])
plt.show()     


'''

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
    
    if 0 <= t < 24*1: # Enter the lab
        if t < 7:
            lux = dark
        elif 7 <= t < 19: 
            lux = wake
        elif 19 <= t < 23:
            lux = dim
        else:
            lux = dark
            
        BurkeProtocol_lightschedule.append(lux)

    elif 24*1 <= t < 24*2: # Start of CR1 (28 h)
        t = np.mod(t,24)
        if t < 7:
            lux = dark
        else:
            lux = dim
            
        BurkeProtocol_lightschedule.append(lux)
        
    elif 24*2 <= t <  24*3: 
        t = np.mod(t,24)
        if t < 11: 
            lux = dim
        elif 11 <= t < 16:
            lux = dark
        elif 16 <= t < 23:
            lux = dim
        else:
            lux = dark
            
        BurkeProtocol_lightschedule.append(lux)
        
    elif 24*3 <= t <  24*4: # Light intervention and start of CR2 (19h)
        t = np.mod(t,24)
        if t < 6:
            lux = dark
        elif 6 <= t < 9:
            lux = bright 
        else: 
            lux = dim
            
        BurkeProtocol_lightschedule.append(lux)
    
    elif 24*4 <= t <  24*5: # Day 2
        t = np.mod(t,24)
        if t < 4:
            lux = dim
        elif 4 <= t < 10:
            lux = dark 
        else: 
            lux = wake
            
        BurkeProtocol_lightschedule.append(lux)
    
    

plt.plot(range(0,n),BurkeProtocol_lightschedule,color='goldenrod')
plt.xlabel('Time (hours)')
plt.ylabel('Light Intensity (lux)')
plt.title("Burke 2013 Protocol")
plt.axvline(24,linestyle='dashed',color='grey')
plt.axvline(24*2,linestyle='dashed',color='grey')
plt.axvline(24*3,linestyle='dashed',color='grey')
plt.axvline(24*4,linestyle='dashed',color='grey')
plt.axvline(17.25+(24*2),color='hotpink')
plt.axvspan(78, 81, facecolor='gold', alpha=0.4)
#plt.axvspan(21.25+(24*2), 3.75+(24*3), facecolor='grey', alpha=0.4)
#plt.axvline(7+24,color='orange')
#plt.axvline(23,color='orange')
#plt.legend(['CBTmin'])
plt.show()     
      
 

