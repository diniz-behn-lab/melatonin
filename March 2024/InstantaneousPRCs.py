# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 20:46:48 2024

@author: sstowe

Plotting the instantaneous phase response curves to light and melatonin 

"""


#-------- Set Up -------------------------------

# Imports 

import numpy as np
import matplotlib.pyplot as plt


psi = np.linspace(0,2*np.pi,100)

# ---------------- Instantaneous PRC to Light -----------

# Light forcing parameters (from Hannay 2019)

A_1 = 0.3855 
beta_L1 = -0.0026
A_2 = 0.1977 
beta_L2 = -0.957756
sigma = 0.0400692

Q_L_list = []

for i in range(0,len(psi)):
    Q_L = sigma - A_1*np.sin(psi[i] + beta_L1) - A_2*np.sin(2*psi[i] + beta_L2)
    Q_L_list.append(Q_L)
    
    
plt.plot(psi,Q_L_list) 
plt.title('Instantaneous PRC to Light')
plt.show()


# ---------------- Instantaneous PRC to Melatonin -----------

# x = [B1, theta_M1, B2, theta_M2, epsilon]

# Fitting to melatonin_dose = 70000
#x = [-1.91498106,  0.49976944, -1.26086337,  1.05335774,  0.29083548]

# Fitting to melatonin_dose = 179893 (fit to Wyatt 2006)
#x = [-1.27270047,  0.38065833, -0.86828183,  0.67052757,  0.14395031] # Error = 3.5785611633727887 #  Optimization terminated successfully!!
# Fitting to the cubic dose curve 
x = [-1.42992587,  0.43158586, -0.89095487,  0.82059878,  0.11236468] # Error = 3.6102769976831413  # Optimization terminated successfully!!

B_1 = x[0]
theta_M1 = x[1]
B_2 = x[2]
theta_M2 = x[3]
epsilon = x[4]

Q_M_list = []

for i in range(0,len(psi)):
    Q_M = epsilon - B_1*np.sin(psi[i] + theta_M1) - B_2*np.sin(2*psi[i] + theta_M2)
    Q_M_list.append(Q_M)
    
    
plt.plot(psi,Q_M_list) 
plt.title('Instantaneous PRC to Melatonin')   
plt.show()


# --------- Plot PRCs together ---------

plt.plot(psi,Q_M_list)
plt.plot(psi,Q_L_list) 
plt.axhline(0, linestyle='dashed')
plt.axvline(1.26,color='black') # DLMO 
plt.title('Instantaneous Phase Response Curves')
plt.legend(["Melatonin PRC","Light PRC"])
plt.xlabel('Psi')
plt.ylabel('Instantaneous Phase Response')
plt.show()
