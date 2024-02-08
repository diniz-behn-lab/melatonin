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
T_round = np.round(T)

mel_timing = 10.0+24*np.arange(j)

Times = np.intersect1d(mel_timing, T_round, return_indices=True)[2]

ex_melatonin = signal.unit_impulse(720,Times)

plt.plot(T,ex_melatonin)
plt.show()


#---------------------------------

def smooth_pulse(time, start, stop, cycle=1, steep=1000):
    xi = 3/4 - (stop - start) / (2 * cycle) * (cycle / (stop - start))
    return 1 / (1 + np.exp(steep * (np.sin(2 * np.pi * ((time - start) / cycle + xi)) - np.sin(2 * np.pi * xi))))
    #return (cycle/(stop - start)) / (1 + np.exp(steep * (np.sin(2 * np.pi * ((time - start) / cycle + xi)) - np.sin(2 * np.pi * xi))))

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