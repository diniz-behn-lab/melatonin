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

phase_shifts_corrected_2008 = [7.999999999999971578e-01,
5.000000000000000000e-01,
3.999999999999914735e-01,
1.999999999999886313e-01,
-5.000000000000000000e-01,
-1.200000000000002842e+00,
-1.300000000000011369e+00,
-1.200000000000002842e+00,
-9.000000000000056843e-01,
1.999999999999886313e-01,
7.999999999999971578e-01,
1.399999999999991473e+00,
1.500000000000000000e+00,
1.500000000000000000e+00,
1.299999999999997158e+00,
1.099999999999994316e+00,
1.000000000000000000e+00,
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

phase_shifts_corrected_2010 = [7.999999999999971578e-01,
5.999999999999943157e-01,
5.000000000000000000e-01,
5.000000000000000000e-01,
2.999999999999971578e-01,
9.999999999999431566e-02,
0.000000000000000000e+00,
-2.000000000000028422e-01,
-6.000000000000085265e-01,
-8.000000000000113687e-01,
-8.000000000000113687e-01,
-8.000000000000113687e-01,
0.000000000000000000e+00,
3.999999999999914735e-01,
1.000000000000000000e+00,
1.199999999999988631e+00,
1.099999999999994316e+00,
1.000000000000000000e+00,
8.999999999999914735e-01,
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