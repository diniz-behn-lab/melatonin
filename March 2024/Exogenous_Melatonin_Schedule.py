# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 12:40:23 2024

@author: sstowe

Code to generate plot of exogenous melatonin administration
"""

import numpy as np
import matplotlib.pyplot as plt


j = 3 # Number of days to be plotted

n = j*24

ex_melatonin_1 = list()

for t in range(0, n):
    dosage = 0.2
    timing = 18

    mel = np.mod(t, 24) == timing

    ex_melatonin_1.append(mel*(dosage/1e-9))
        
plt.plot(range(0,n),ex_melatonin_1)
plt.show()


#------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

def dirac_approximation(x, sig):
    if -(1 / (2 * sig)) <= x <= (1 / (2 * sig)):
        return sig
    else:
        return 0

X = np.linspace(-5, 5, 1000)
dirac_values = [dirac_approximation(x, sig=1) for x in X]

plt.plot(X, dirac_values)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Approximation of Dirac Delta Function")
plt.grid(True)
plt.show()

#----------------------------------

from scipy import signal
#impulse = signal.unit_impulse(100)  # Creates an impulse at the 0th element
impulse = signal.unit_impulse(100,2)  # Creates an impulse at the 3rd element
print(impulse)

T = np.linspace(0,10,100)

plt.plot(T, impulse)
plt.show()


#---------------------------

#array1 = [1,2,3,4,5,6]
array1 = [10]
array2 = [2,6,3,4,1,5,10]

Y = np.intersect1d(array1, array2, return_indices=True)[2]

print(Y)

#---------------------

j = 3 # Number of days to be plotted

n = j*24

T = np.linspace(0,n,720)
T_round = np.round(T,decimals=1)

mel_timing = 10.0+24*np.arange(j)

Times = np.intersect1d(mel_timing, T_round, return_indices=True)[2]

ex_melatonin = signal.unit_impulse(720,Times)

plt.plot(T,ex_melatonin)
plt.title("Unit Pulse")
plt.show()


#---------------------------------

def smooth_pulse(time, start, stop, cycle=1, steep=1000):
    #xi = 3/4 - (stop - start) / (2 * cycle) * (cycle / (stop - start))
    #return 1 / (1 + np.exp(steep * (np.sin(2 * np.pi * ((time - start) / cycle + xi)) - np.sin(2 * np.pi * xi))))
    
    xi = 3/4 - (stop - start) / (2 * cycle)
    return (cycle/(stop - start)) / (1 + np.exp(steep * (np.sin(2 * np.pi * ((time - start) / cycle + xi)) - np.sin(2 * np.pi * xi))))

x = np.arange(0, 5, 0.001)
y_smooth = smooth_pulse(x, start=0.2, stop=0.3, cycle=0.5)


plt.plot(x, y_smooth, label='Smooth Pulse')
plt.xlabel('Time')
plt.ylabel('f(t)')
plt.title('Smooth Differentiable Pulse Function')
plt.xlim(0, 2)
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.show()

#--------------------------------------

def smooth_pulse(time, start, stop, cycle=1, steep=1000):
    #xi = 3/4 - (stop - start) / (2 * cycle) * (cycle / (stop - start))
    #return 1 / (1 + np.exp(steep * (np.sin(2 * np.pi * ((time - start) / cycle + xi)) - np.sin(2 * np.pi * xi))))
    
    xi = 3/4 - (stop - start) / (2 * cycle)
    return (cycle/(stop - start)) / (1 + np.exp(steep * (np.sin(2 * np.pi * ((time - start) / cycle + xi)) - np.sin(2 * np.pi * xi))))

x = np.arange(0, 48, 0.001)
y_smooth = smooth_pulse(x, start=12.0, stop=12.1, cycle=24)


plt.plot(x, y_smooth, label='Smooth Pulse')
plt.xlabel('Time')
plt.ylabel('f(t)')
plt.title('Smooth Differentiable Pulse Function')
plt.xlim(0, 24)
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.show()

#--------------------------------------

def ex_melatonin(time, start, stop, cycle=24, steep=1000, dosage=80):
    
    xi = 3/4 - (stop - start) / (2 * cycle)
    return (cycle/(stop - start)) / (1 + np.exp(steep * (np.sin(2 * np.pi * ((time - start) / cycle + xi)) - np.sin(2 * np.pi * xi))))

x = np.arange(0, 48, 0.1)
y_smooth = ex_melatonin(x, start=10.0, stop=10.01, cycle=24, dosage=800)

max_value = max(y_smooth)

normalize_y_smooth = (1/max_value)*y_smooth

dose_y_smooth = 2*normalize_y_smooth


plt.plot(x, dose_y_smooth, label='Smooth Pulse')
plt.xlabel('Time')
plt.ylabel('ex_melatonin(t)')
plt.title('Exogenous Melatonin Dosing')
plt.xlim(0, 24)
plt.ylim(0, 2)
plt.legend()
plt.show()


#-----------------------

# Hill Function 
T = np.linspace(0,250,100)
V_max = 100
K = 50
n = 2

V = (V_max*pow(T,n))/(pow(K,n) + pow(T,n)) 

plt.plot(T,V)
plt.title("Hill Equation")
plt.show()

#----------------

# Steady-state firing rate response function
T = np.linspace(-5,5,100)
X_max = 6
s_x = 1
beta_x = -0.34
alpha_x = 0.5

X_infty = (X_max/2)*(s_x + np.tanh((T-beta_x)/alpha_x))

plt.plot(T,X_infty)
plt.show()

#-------------------

# Guassian function 
T = np.linspace(0,1,100)
sigma = np.sqrt(0.0002)#np.sqrt(0.002)
mu = 1/2

Guassian = (1/sigma*np.sqrt(2*np.pi))*np.exp((-pow(T-mu,2))/(2*pow(sigma,2)))

max_val = max(Guassian)
print(max_val)

normalize_Guassian = (1/max_val)*Guassian

plt.plot(T,normalize_Guassian)
plt.show()

#----------------------------

# Generalized Logistic Function 
T = np.linspace(0,4,100)
A = 0 
K = 1
C = 1
Q = -0.5
B = 3
nu = 0.5

G = A + (K - A)/pow((C + Q*np.exp(-B*T)),(1/nu)) 

plt.plot(T,G)
plt.title("Generalized Logistic Function")
plt.show()

#---------------------------------

# Logistic Growth Function
T = np.linspace(0,5,100)
L = 100
k = 3
T0 = 2

Logistic = (L/(1 + np.exp(-k*(T - T0))))

plt.plot(T,Logistic)
plt.title("Logistic Growth Function")
plt.show()