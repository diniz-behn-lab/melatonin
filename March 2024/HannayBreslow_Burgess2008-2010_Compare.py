# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 12:09:03 2024

@author: sstowe

Plotting Burgess 2008 and Burgess 2010 together 
"""


#-------- Set Up -------------------------------

# Imports 

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


#--------- Burgess 2008 ---------

ExMel_times_2008 = [0, 2, 3, 4, 7, 10, 11, 12, 13, 15, 16, 17, 18, 19, 21, 22, 23]

phase_shifts_corrected_2008 = [0.728035,
                               0.388097,
                               0.258366,
                               0.0844391,
                               -0.545787,
                               -1.17357,
                               -1.20764,
                               -1.12698,
                               -0.899798,
                               0.0856513,
                               0.67831,
                               1.40988,
                               1.52414,
                               1.50699,
                               1.28684,
                               1.10435,
                               0.908155
                               ]



plt.title("3 Pulse PRC To 3mg Exogenous Melatonin")
plt.xlabel("Time After DLMO (hours)")
plt.ylabel("Phase Shift (hours)")
plt.axhline(y=0,color='black',linestyle='dashed',lw=3)
#plt.axvline(x=19.4,color='grey',linestyle='dashed',lw=3) # Burgess 2008 fitted peak (max phase advance)
#plt.axhline(y=1.8,color='grey',linestyle='dashed',lw=3) # Burgess 2008 fitted max advance 
#plt.axhline(y=-1.3,color='grey',linestyle='dashed',lw=3) # Burgess 2008 fitted max delay
#plt.axvline(x=11.1,color='grey',linestyle='dashed',lw=3) # Burgess 2008 fitted trough (max phase delay)
#plt.plot(ExMel_times,phase_shifts_corrected, lw=5,color='mediumturquoise')
plt.plot(ExMel_times_2008,phase_shifts_corrected_2008, '-o',color='mediumturquoise')
#plt.legend(["Burgess 2008 Data", "Hannay-Breslow 2024 Model"])
plt.show()

#--------- Burgess 2010 ---------

ExMel_times_2010 = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 19, 20, 21, 22, 23]

phase_shifts_corrected_2010 = [0.707141,
                               0.557256,
                               0.418006,
                               0.323161,
                               0.0996589,
                               -0.00262554,
                               -0.106339,
                               -0.271563,
                               -0.625701,
                               -0.732232,
                               -0.807022,
                               -0.800361,
                               -0.0616216,
                               0.264914,
                               0.930055,
                               1.26036,
                               1.16951,
                               1.02335,
                               0.858506
                               ]


plt.title("Simulated 3 Pulse PRC to 0.5 mg Melatonin")
plt.xlabel("Time After DLMO (hours)")
plt.ylabel("Phase Shift (hours)")
plt.axhline(y=0,color='black',linestyle='dashed',lw=3)
#plt.axvline(x=20.7,color='grey',linestyle='dashed',lw=3) # Burgess 2010 fitted peak (max phase advance)
#plt.axhline(y=1.5,color='grey',linestyle='dashed',lw=3) # Burgess 2010 fitted max advance 
#plt.axhline(y=-1.3,color='grey',linestyle='dashed',lw=3) # Burgess 2010 fitted max delay
#plt.axvline(x=13.6,color='grey',linestyle='dashed',lw=3) # Burgess 2010 fitted trough (max phase delay)
plt.plot(ExMel_times_2010,phase_shifts_corrected_2010, 'o-',color='forestgreen')
#plt.legend(["Burgess 2010 Data", "Model"])
plt.show()



#-------- Plot Together -----------

plt.rcParams.update({'font.size': 15})

plt.title("Simulated 3 Pulse PRCs to Melatonin")
plt.xlabel("Time After DLMO (hours)")
plt.ylabel("Phase Shift (hours)")
plt.plot(ExMel_times_2008,phase_shifts_corrected_2008, '-o',color='mediumturquoise')
plt.plot(ExMel_times_2010,phase_shifts_corrected_2010, 'o-',color='forestgreen')
plt.axhline(y=0,color='black',linestyle='dashed',lw=3)
plt.legend(["3 mg", "0.5 mg"])
plt.ylim([-2.9,2.9])
plt.show()