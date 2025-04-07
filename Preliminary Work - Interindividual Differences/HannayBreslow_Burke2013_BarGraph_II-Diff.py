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
   
model = [-0.24, 0.16, 0.8, 1.23] 
Burke2013 = [-0.43, 0.19, 0.27, 0.7] 

# SD
error_SD = [0.16, 0.16, 0.15, 0.19]

# Interindividual Variability 
II_max = [0.44, 0.54, 0.63, 0.81]
II_min = [0.63, 1.21, 1.08, 0.79]

  
n=4
r = np.arange(n) 
width = 0.25
  

plt.bar(r, Burke2013, color = 'grey', 
        width = width, edgecolor = 'black', 
        label='Burke 2013 Data')   
plt.errorbar(r, Burke2013, yerr=error_SD,color='black',fmt='none',capsize=6,lw=3)
eb2 = plt.errorbar(r, Burke2013, yerr=[II_min, II_max],fmt='none',color='black',capsize=5)
eb2[-1][0].set_linestyle('-.')
plt.bar(r + width, model, color = 'mediumturquoise', 
        width = width, edgecolor = 'black', 
        label='Model') 

  
plt.xlabel("Experimental Condition") 
plt.ylabel("Phase Shift (h)") 
  
# plt.grid(linestyle='--') 
plt.xticks(r + width/2,['DLP','DLM','BLP','BLM']) 
plt.axhline(0,linestyle='dashed',color='black')
#plt.plot(r + width, [-0.2336755581568042, 0.23274257763995365, 0.6858073254417718, 1.188526950066418],'o',color = 'blue',label = 'Delayed Light History')
#plt.plot(r + width, [-0.23716934215681107, 0.10075535361085741, 0.8948723974875961, 1.2614399567569663],'o',color = 'red',label = 'Advanced Light History')
#plt.plot(r + width, [-0.23660555518726056, 0.18660832498951407, 0.7606946917866324, 1.2173625756642181],'o',color = 'purple',label = 'Sun Up =7, Sun down = 23')
#plt.plot(r + width, [-0.2294450911619066, 0.16380992318019594, 0.8090005638228916, 1.2395028425216879],'o',color = 'yellow',label = 'Low production')
#plt.plot(r + width, [-0.18316323309520754, 0.1807228813350079, 0.8679435614973343, 1.2585500980756947],'o',color = 'black',label = 'High production')
plt.plot(r + width, [-0.1709497169616938, 0.15551370937236442, 0.9327722241789687, 1.28894405393374],'o',color = 'orange',label = 'Early Chronotype')
plt.plot(r + width, [-0.32191905787230723, 0.1403791183858658, 0.6575592545435711, 1.1548843790439776],'o',color = 'midnightblue',label = 'Late Chronotype')
#plt.plot(r + width, [-0.31877011448162307, 0.1395872067574544, 0.6628040486751416, 1.1566682699919824],'o',color = 'fuchsia',label = 'Late Chronotype + Low Production')
plt.legend(loc ='center right', bbox_to_anchor=(1.7, 0.5)) 
plt.show()

