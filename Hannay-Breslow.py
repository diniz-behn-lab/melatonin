# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:57:02 2023

@author: sstowe
"""

# Code to run the Hannay-Breslow Model 

#Armelle has been here 

#-------- Set Up -------------------------------

# Imports 




# Parameter values

## Breslow Model
beta_IP = 7.83e-4*60*60
beta_CP = 3.35e-4*60*60
beta_AP = 1.62-4*60*60

a = 600
r = 15.36

psi_on = 6.113
psi_off = 4.352

M_max = 0.019513
H_sat = 861
sigma_M = 50
m = 7

## Hannay Model
D = 1
gamma = 0.024
K = 0.06358 
beta = -0.09318
omega_0 = 0.263524

A_1 = 0.3855 
beta_L1 = -0.0026
A_2 = 0.1977 
beta_L2 = -0.957756
sigma = 0.0400692

delta = 0.0075
alpha_0 = 0.05
p = 1.5
I_0 = 9325
G = 33.75

B_1 = 1
theta_M1 = 1
B_2 = 1
theta_M2 = 1
epsilon = 1



# Initial Conditions 




# Time




# Light Schedule 



#--------- Create Model Function ---------------





#--------- Plot Model Output -------------------




