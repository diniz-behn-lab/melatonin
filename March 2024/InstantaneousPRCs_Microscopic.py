# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:35:29 2024

@author: sstowe

Microscopic PRC analysis 

"""


#-------- Set Up -------------------------------

# Imports 

import numpy as np
import matplotlib.pyplot as plt

# ----------- Modified from Will's code: prcs_galore.py ------

from itertools import repeat

# Corrected Hsat and sigma_M 
#x = [-0.98204363, -0.07764001, -0.7152688,   0.8511226,   0.07833321] # Error = 3.655967724146368 # Optimization terminated successfully!!

# Updated for linear interpolation (May 2024)
x = [-1.00356684, -0.01949989, -0.56044428,  0.60146017,  0.07781353] # Error = 3.7152403300234877 # Optimization terminated successfully!! 

B1 = x[0]
theta1 = x[1] 
B2 = x[2]
theta2 = x[3] 
epsilon = x[4]

# ----------- Microscopic PRC to melatonin --------------
micro_prc = lambda phi: epsilon - B1*np.sin(phi + theta1) - B2*np.sin(2*phi + theta2)
phi = np.linspace(0,2*np.pi,361,endpoint=True)
response = micro_prc(phi)
DLMO = 5*np.pi/12
phi_rel_to_DLMO = np.mod(phi - DLMO,2*np.pi)
phi_rel_to_DLMO[phi_rel_to_DLMO > np.pi] = phi_rel_to_DLMO[phi_rel_to_DLMO > np.pi] - 2*np.pi
phi_argsort = np.argsort(phi_rel_to_DLMO)
max_advance_idx = np.argmax(response)
max_delay_idx = np.argmin(response)

plt.plot(12*phi_rel_to_DLMO[phi_argsort]/np.pi,response[phi_argsort],lw=2)
plt.plot(12*phi_rel_to_DLMO[phi_argsort]/np.pi, np.zeros_like(response),'k--')
plt.plot(list(repeat(12*phi_rel_to_DLMO[max_advance_idx]/np.pi,2)), [0,response[max_advance_idx]],'bo--')
plt.plot(list(repeat(12*phi_rel_to_DLMO[max_delay_idx]/np.pi,2)), [0,response[max_delay_idx]],'o--')
plt.grid()
plt.xlabel("Time Relative to DLMO (hours)")
plt.ylabel("Phase Response")
plt.title("Microscopic Melatonin Phase Response Curve")
plt.xlim((-12,12))
max_advance_time = abs(12*phi_rel_to_DLMO[max_advance_idx]/np.pi)
max_delay_time = abs(12*phi_rel_to_DLMO[max_delay_idx]/np.pi)
text_str = "Max advance at {:.2} hours \n before DLMO".format(max_advance_time)
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
plt.text(2,0.9,text_str,bbox=props)
text_str = "Max delay at {:.2} hours \n before DLMO".format(max_delay_time)
plt.text(-5,-1,text_str,bbox=props)
plt.show()



# -------------- Micrscopic PRC to light ------------------
A1 = 0.3855 
beta1 = -0.0026
A2 = 0.1977 
beta2 = -0.957756
sigma = 0.0400692

micro_prc = lambda phi: sigma - A1*np.sin(phi + beta1) - A2*np.sin(2*phi + beta2)
phi = np.linspace(0,2*np.pi,361,endpoint=True)
light_response = micro_prc(phi)
DLMO = 5*np.pi/12
phi_rel_to_DLMO = np.mod(phi - DLMO,2*np.pi)
phi_rel_to_DLMO[phi_rel_to_DLMO > np.pi] = phi_rel_to_DLMO[phi_rel_to_DLMO > np.pi] - 2*np.pi
phi_argsort = np.argsort(phi_rel_to_DLMO)
max_advance_idx = np.argmax(light_response)
max_delay_idx = np.argmin(light_response)
plt.plot(12*phi_rel_to_DLMO[phi_argsort]/np.pi,light_response[phi_argsort],lw=2)
plt.plot(12*phi_rel_to_DLMO[phi_argsort]/np.pi, np.zeros_like(light_response),'k--')
plt.plot(list(repeat(12*phi_rel_to_DLMO[max_advance_idx]/np.pi,2)), [0,light_response[max_advance_idx]],'bo--')
plt.plot(list(repeat(12*phi_rel_to_DLMO[max_delay_idx]/np.pi,2)), [0,light_response[max_delay_idx]],'o--')
plt.grid()
plt.xlabel("Time Relative to DLMO (hours)")
plt.ylabel("Phase Response")
plt.title("Microscopic Light Phase Response Curve")
plt.xlim((-12,12))
max_advance_time = abs(12*phi_rel_to_DLMO[max_advance_idx]/np.pi)
max_delay_time = abs(12*phi_rel_to_DLMO[max_delay_idx]/np.pi)
text_str = "Max advance at {:.2} hours \n before DLMO".format(max_advance_time)
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
plt.text(-5,0.3,text_str,bbox=props)
text_str = "Max delay at {:.2} hours \n after DLMO".format(max_delay_time)
plt.text(3,-0.43,text_str,bbox=props)
plt.show()



# Determine phase difference between melatonin and light PRCs 
mel_fourier = np.fft.fft(response)
light_fourier = np.fft.fft(np.flip(light_response))
cross_cor_four = mel_fourier*light_fourier
cross_cor = np.real(np.fft.ifft(cross_cor_four))
cross_cor = np.fft.fftshift(cross_cor)/(np.linalg.norm(response)*np.linalg.norm(light_response))
phi[phi > np.pi] = phi[phi > np.pi] - 2*np.pi
phi = np.fft.fftshift(phi)
max_alignment_idx = np.argmax(cross_cor)
plt.plot(12*phi/np.pi,cross_cor)
plt.plot(12*phi/np.pi,np.zeros_like(phi),'k--')
plt.plot(list(repeat(12*phi[max_alignment_idx]/np.pi,2)), [0,cross_cor[max_alignment_idx]],'bo--')
hours_out = -12*phi[max_alignment_idx]/np.pi
text_str = "PRC curves are about {:2} hours \n out of phase".format(hours_out)
plt.text(-7,0.6,text_str,bbox=props)
plt.xlabel('Phase Delay (hours)')
plt.xlim((-12,12))
plt.ylabel('Cross Correlation')
plt.title("Melatonin-Light PRC Correlation")
plt.grid()
plt.show()


# ------------- Shelby Plotting ----------------------

# Microscopic PRC to light
plt.rcParams.update({'font.size': 15})
plt.plot(12*phi_rel_to_DLMO[phi_argsort]/np.pi,light_response[phi_argsort],lw=5,color='goldenrod') # PRC 
plt.plot(12*phi_rel_to_DLMO[phi_argsort]/np.pi, np.zeros_like(light_response),'k--') # Line at 0
plt.plot(list(repeat(12*phi_rel_to_DLMO[max_advance_idx]/np.pi,2)), [0,light_response[max_advance_idx]],'o--',color='black') # Line of max advance 
plt.plot(list(repeat(12*phi_rel_to_DLMO[max_delay_idx]/np.pi,2)), [0,light_response[max_delay_idx]],'o--',color='black') # Line of max delay
plt.xticks(np.arange(-12, 14, 4))
plt.xlabel("Time Relative to DLMO (hours)")
plt.ylabel("Phase Response")
plt.title("Microscopic Phase Response Curve to Light")
plt.xlim((-12,12))
plt.ylim([-1.75,1.4])
max_advance_time_mel = abs(12*phi_rel_to_DLMO[max_advance_idx]/np.pi)
max_delay_time_mel = abs(12*phi_rel_to_DLMO[max_delay_idx]/np.pi)
plt.show()



# Microscopic PRC to Melatonin 

micro_prc = lambda phi: epsilon - B1*np.sin(phi + theta1) - B2*np.sin(2*phi + theta2)
phi = np.linspace(0,2*np.pi,361,endpoint=True)
response = micro_prc(phi)
DLMO = 5*np.pi/12
phi_rel_to_DLMO = np.mod(phi - DLMO,2*np.pi)
phi_rel_to_DLMO[phi_rel_to_DLMO > np.pi] = phi_rel_to_DLMO[phi_rel_to_DLMO > np.pi] - 2*np.pi
phi_argsort = np.argsort(phi_rel_to_DLMO)
max_advance_idx = np.argmax(response)
max_delay_idx = np.argmin(response)

plt.plot(12*phi_rel_to_DLMO[phi_argsort]/np.pi,response[phi_argsort],lw=5,color='mediumpurple')
plt.plot(12*phi_rel_to_DLMO[phi_argsort]/np.pi, np.zeros_like(response),'k--')
plt.plot(list(repeat(12*phi_rel_to_DLMO[max_advance_idx]/np.pi,2)), [0,response[max_advance_idx]],'ko--')
plt.plot(list(repeat(12*phi_rel_to_DLMO[max_delay_idx]/np.pi,2)), [0,response[max_delay_idx]],'ko--')
plt.xticks(np.arange(-12, 14, 4))
plt.xlabel("Time Relative to DLMO (hours)")
plt.ylabel("Phase Response")
plt.title("Microscopic Phase Response Curve to Melatonin")
plt.xlim((-12,12))
plt.ylim([-1.75,1.4])
max_advance_time_mel = abs(12*phi_rel_to_DLMO[max_advance_idx]/np.pi)
max_delay_time_mel = abs(12*phi_rel_to_DLMO[max_delay_idx]/np.pi)
plt.show()



# Cross Correlation
mel_fourier = np.fft.fft(response)
light_fourier = np.fft.fft(np.flip(light_response))
cross_cor_four = mel_fourier*light_fourier
cross_cor = np.real(np.fft.ifft(cross_cor_four))
cross_cor = np.fft.fftshift(cross_cor)/(np.linalg.norm(response)*np.linalg.norm(light_response))
phi[phi > np.pi] = phi[phi > np.pi] - 2*np.pi
phi = np.fft.fftshift(phi)
max_alignment_idx = np.argmax(cross_cor)
plt.plot(12*phi/np.pi,cross_cor,lw=5,color='forestgreen')
plt.plot(12*phi/np.pi,np.zeros_like(phi),'k--')
plt.plot(list(repeat(12*phi[max_alignment_idx]/np.pi,2)), [0,cross_cor[max_alignment_idx]],'ko--')
hours_out = -12*phi[max_alignment_idx]/np.pi
plt.xticks(np.arange(-12, 14, 4))
plt.xlabel('Phase Delay (hours)')
plt.xlim((-12,12))
plt.ylabel('Cross-correlation')
plt.title("Melatonin-Light PRC Correlation")
plt.show()
