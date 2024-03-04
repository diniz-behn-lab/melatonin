# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 10:40:18 2024

@author: sstowe

A of psi 

Plotting A of psi 

"""

import numpy as np
import matplotlib.pyplot as plt

# ---------- Breslow 2017 ------
# A(phi)

phi = np.linspace(0, 2*np.pi)

A_function_Breslow = list()

phi_on = 6.113
phi_off = 4.352

a = 1*60
delta = 600/60
r = 200/60

for i in range(0,len(phi)):
    
    if (np.mod(phi[i],2*np.pi) > phi_off) and (np.mod(phi[i],2*np.pi) < phi_on): 
        A = a*(np.exp(-r*np.mod(phi[i] - phi_off,2*np.pi)))
        A_function_Breslow.append(A)
    else: 
        A = a*((1 - np.exp(-delta*np.mod(phi[i] - phi_on,2*np.pi))/1 - np.exp(-delta*np.mod(phi_off - phi_on,2*np.pi))));
        A_function_Breslow.append(A)
        
#plt.plot(phi,A_function_Breslow)
#plt.show()

# ------ Hannay-Breslow 2024 -----------
# A(psi)

psi = np.linspace(0, 2*np.pi)

A_function = list()

psi_on = 1.0472
psi_off = 3.92699

a = 0.01*60
delta_M = 600/60
r = 200/60

for i in range(0,len(psi)):
    if (np.mod(psi[i],2*np.pi) > psi_on) and (np.mod(psi[i],2*np.pi) < psi_off): 
        A = a*((1 - np.exp(-delta_M*np.mod(psi[i] - psi_on,2*np.pi))/1 - np.exp(-delta_M*np.mod(psi_off - psi_on,2*np.pi))));
        A_function.append(A)
    else: 
        A = a*(np.exp(-r*np.mod(psi[i] - psi_off,2*np.pi)))
        A_function.append(A)
        
plt.plot(psi,A_function)
plt.show()
