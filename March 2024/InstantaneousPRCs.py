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

'''
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
#x = [-1.42992587,  0.43158586, -0.89095487,  0.82059878,  0.11236468] # Error = 3.6102769976831413  # Optimization terminated successfully!!

# Corrected Hsat and sigma_M 
x = [-0.98204363, -0.07764001, -0.7152688,   0.8511226,   0.07833321] # Error = 3.655967724146368 # Optimization terminated successfully!! 

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
#DLMO = 5*np.pi/12
#psi_rel_to_DLMO = np.mod(psi - DLMO,2*np.pi)
#psi_rel_to_DLMO[psi_rel_to_DLMO > np.pi] = psi_rel_to_DLMO[psi_rel_to_DLMO > np.pi] - 2*np.pi
#plt.plot(psi_rel_to_DLMO,Q_M_list)
#plt.plot(psi_rel_to_DLMO,Q_L_list)

plt.plot(psi,Q_M_list)
plt.plot(psi,Q_L_list) 
#plt.axvline(1.26,color='black') # DLMO 
plt.axvline(1.31,color='grey',linestyle='dashed') # DLMO 
plt.axvline(3.14,color='grey') # CBTmin 
plt.axhline(0, linestyle='dashed',color='black')
plt.title('Instantaneous Phase Response Curves')
plt.legend(["Melatonin PRC","Light PRC","DLMO","CBTmin"])
plt.xlabel('Psi')
plt.ylabel('Instantaneous Phase Response')
plt.show()
'''

# ----------- Modified from Will's code: prcs_galore.py ------

from itertools import repeat

# Corrected Hsat and sigma_M 
x = [-0.98204363, -0.07764001, -0.7152688,   0.8511226,   0.07833321] # Error = 3.655967724146368 # Optimization terminated successfully!! 

B1 = x[0]
theta1 = x[1] 
B2 = x[2]
theta2 = x[3] 
epsilon = x[4]

# Microscopic PRC to melatonin
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



# Macroscopic PRC to melatonin
Mhat = 0.019513/(1 + np.exp((301 - 4.10668)/(17.5))) 
R = 0.75969 # 0.83 # THIS CHOICE MATTERS !!!
macro_prc = lambda phi: epsilon*Mhat-B1*Mhat*0.5*(pow(R,3.0)+1.0/R)*np.sin(phi+theta1)-B2*Mhat*0.5*(1.0+pow(R,8.0))*np.sin(2.0*phi+theta2)
macro_response = macro_prc(phi)

max_advance_idx = np.argmax(macro_response)
max_delay_idx = np.argmin(macro_response)
plt.plot(12*phi_rel_to_DLMO[phi_argsort]/np.pi,macro_response[phi_argsort],lw=2)
plt.plot(12*phi_rel_to_DLMO[phi_argsort]/np.pi, np.zeros_like(macro_response),'k--')
plt.plot(list(repeat(12*phi_rel_to_DLMO[max_advance_idx]/np.pi,2)), [0,macro_response[max_advance_idx]],'bo--')
plt.plot(list(repeat(12*phi_rel_to_DLMO[max_delay_idx]/np.pi,2)), [0,macro_response[max_delay_idx]],'o--')
plt.grid()
plt.xlabel("Time Relative to DLMO (hours)")
plt.ylabel("Phase Response (radians/hour)")
plt.title("Macroscopic Melatonin Phase Response Curve")
plt.xlim((-12,12))
max_advance_time = abs(12*phi_rel_to_DLMO[max_advance_idx]/np.pi)
max_delay_time = abs(12*phi_rel_to_DLMO[max_delay_idx]/np.pi)
text_str = "Max advance at {:.2} hours \n before DLMO".format(max_advance_time)
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
plt.text(-10,0.5e-9,text_str,bbox=props)
text_str = "Max delay at {:.2} hours \n before DLMO".format(max_delay_time)
plt.text(0,-0.5e-9,text_str,bbox=props)
plt.show()


# Micrscopic PRC to light
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


# Macroscopic PRC to light
G = 33.75
alpha_0 = 0.05
p = 1.5
I_0 = 9325
light = 1000#5#12 # THIS CHOICE DOES NOT SEEM TO MATTER FOR THE CROSS CORRELATION

Bhat = G*(alpha_0*pow(light, p)/(pow(light, p) + I_0)) # Corresponds to a dim light response

macro_prc = lambda phi: sigma*Bhat-A1*Bhat*0.5*(pow(R,3.0)+1.0/R)*np.sin(phi+beta1)-A2*Bhat*0.5*(1.0+pow(R,8.0))*np.sin(2.0*phi+beta2)
macro_light_response = macro_prc(phi)

max_advance_idx = np.argmax(macro_light_response)
max_delay_idx = np.argmin(macro_light_response)
plt.plot(12*phi_rel_to_DLMO[phi_argsort]/np.pi,macro_light_response[phi_argsort],lw=2)
plt.plot(12*phi_rel_to_DLMO[phi_argsort]/np.pi, np.zeros_like(macro_light_response),'k--')
plt.plot(list(repeat(12*phi_rel_to_DLMO[max_advance_idx]/np.pi,2)), [0,macro_light_response[max_advance_idx]],'bo--')
plt.plot(list(repeat(12*phi_rel_to_DLMO[max_delay_idx]/np.pi,2)), [0,macro_light_response[max_delay_idx]],'o--')
plt.grid()
plt.xlabel("Time Relative to DLMO (hours)")
plt.ylabel("Phase Response (radians/hour)")
plt.title("Macroscopic Light Phase Response Curve")
plt.xlim((-12,12))
max_advance_time = abs(12*phi_rel_to_DLMO[max_advance_idx]/np.pi)
max_delay_time = abs(12*phi_rel_to_DLMO[max_delay_idx]/np.pi)
text_str = "Max advance at {:.2} hours \n before DLMO".format(max_advance_time)
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
plt.text(-5.2,0.35,text_str,bbox=props)
text_str = "Max delay at {:.2} hours \n after DLMO".format(max_delay_time)
plt.text(2.75,-0.55,text_str,bbox=props)
plt.show()


# Determine phase difference between melatonin and light PRCs 
mel_fourier = np.fft.fft(macro_response)
light_fourier = np.fft.fft(np.flip(macro_light_response))
cross_cor_four = mel_fourier*light_fourier
cross_cor = np.real(np.fft.ifft(cross_cor_four))
cross_cor = np.fft.fftshift(cross_cor)/(np.linalg.norm(macro_response)*np.linalg.norm(macro_light_response))
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

# Macroscopic Light PRC 

max_advance_idx = np.argmax(macro_light_response)
max_delay_idx = np.argmin(macro_light_response)
plt.plot(12*phi_rel_to_DLMO[phi_argsort]/np.pi,macro_light_response[phi_argsort],lw=5,color='goldenrod') # PRC 
plt.plot(12*phi_rel_to_DLMO[phi_argsort]/np.pi, np.zeros_like(macro_light_response),'k--') # Line at 0
plt.plot(list(repeat(12*phi_rel_to_DLMO[max_advance_idx]/np.pi,2)), [0,macro_light_response[max_advance_idx]],'o--',color='black') # Line of max advance 
plt.plot(list(repeat(12*phi_rel_to_DLMO[max_delay_idx]/np.pi,2)), [0,macro_light_response[max_delay_idx]],'o--',color='black') # Line of max delay
plt.xlabel("Time Relative to DLMO (hours)")
plt.ylabel("Phase Response (radians/hour)")
plt.title("Macroscopic Light Phase Response Curve")
plt.xlim((-12,12))
max_advance_time_light = abs(12*phi_rel_to_DLMO[max_advance_idx]/np.pi)
max_delay_time_light = abs(12*phi_rel_to_DLMO[max_delay_idx]/np.pi)
plt.show()


# Macroscopic Melatonin PRC

max_advance_idx = np.argmax(macro_response)
max_delay_idx = np.argmin(macro_response)
plt.plot(12*phi_rel_to_DLMO[phi_argsort]/np.pi,macro_response[phi_argsort],lw=5,color='mediumpurple')
plt.plot(12*phi_rel_to_DLMO[phi_argsort]/np.pi, np.zeros_like(macro_response),'k--')
plt.plot(list(repeat(12*phi_rel_to_DLMO[max_advance_idx]/np.pi,2)), [0,macro_response[max_advance_idx]],'ko--')
plt.plot(list(repeat(12*phi_rel_to_DLMO[max_delay_idx]/np.pi,2)), [0,macro_response[max_delay_idx]],'ko--')
plt.xlabel("Time Relative to DLMO (hours)")
plt.ylabel("Phase Response (radians/hour)")
plt.title("Macroscopic Melatonin Phase Response Curve")
plt.xlim((-12,12))
max_advance_time_mel = abs(12*phi_rel_to_DLMO[max_advance_idx]/np.pi)
max_delay_time_mel = abs(12*phi_rel_to_DLMO[max_delay_idx]/np.pi)
plt.show()



# Cross Correlation
plt.plot(12*phi/np.pi,cross_cor,lw=5,color='forestgreen')
plt.plot(12*phi/np.pi,np.zeros_like(phi),'k--')
plt.plot(list(repeat(12*phi[max_alignment_idx]/np.pi,2)), [0,cross_cor[max_alignment_idx]],'ko--')
hours_out = -12*phi[max_alignment_idx]/np.pi
plt.xlabel('Phase Delay (hours)')
plt.xlim((-12,12))
plt.ylabel('Cross Correlation')
plt.title("Melatonin-Light PRC Correlation")
plt.show()