# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:27:57 2024

@author: sstowe

Plotting bar graph of Burke2013 results 
"""


'''
# creating the dataset
data = {'DLP':-0.2, 'DLM':0.2, 'BLP':0.8, 
        'BLM':1.3}
protocol = list(data.keys())
phase_shift = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot


plt.bar(protocol, phase_shift, color ='maroon', 
        width = 0.4)
plt.axhline(0,linestyle='dashed',color='black')
plt.xlabel("Experimental Condition")
plt.ylabel("Phase Advance (h)")
plt.show()

'''

import numpy as np 
import matplotlib.pyplot as plt 

font = {'weight' : 'normal',
        'size'   : 12}

plt.rc('font', **font)
   
model = [-0.2, 0.2, 0.8, 1.3] 
Burke2013 = [-0.43, 0.19, 0.27, 0.7] 
  
n=4
r = np.arange(n) 
width = 0.25
  
  
plt.bar(r, model, color = 'mediumturquoise', 
        width = width, edgecolor = 'black', 
        label='Model') 
plt.bar(r + width, Burke2013, color = 'grey', 
        width = width, edgecolor = 'black', 
        label='Data') 
  
plt.xlabel("Experimental Condition") 
plt.ylabel("Phase Shift (h)") 
  
# plt.grid(linestyle='--') 
plt.xticks(r + width/2,['DLP','DLM','BLP','BLM']) 
plt.axhline(0,linestyle='dashed',color='black')
plt.legend() 
plt.show()

