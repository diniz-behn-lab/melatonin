# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:55:48 2024

@author: sstowe

Plot A(phi), the circadian modulation of the pineal gland's melatonin production

"""

import numpy as np
import matplotlib.pyplot as plt

a = 6*60#4*60 #1.0442e-3 # CHANGED, the tiny value is from Breslow
delta_M = 600/3600 # CHANGED, converting secs to hrs
r = 15.36/3600 # CHANGED, converting secs to hrs

psi_on = 1.0472 #6.113 CHANGED 
psi_off = 3.92699 #4.352 CHANGED

psi = np.linspace(4.8,11,100)


dlmo_phase = 5*np.pi/12
psi = np.mod(psi - dlmo_phase,2*np.pi)


'''
if psi > self.psi_off and psi <= self.psi_on:
    print("Pineal on")
    self.a * (1 - np.exp(-self.delta_M*np.mod(self.psi_on - psi,2*np.pi))) / (1 - np.exp(-self.delta_M*np.mod(self.psi_on - self.psi_off,2*np.pi)))
else:
    print("Pineal off")
    self.a*np.exp(-self.r*np.mod(self.psi_on - self.psi_off,2*np.pi))
'''