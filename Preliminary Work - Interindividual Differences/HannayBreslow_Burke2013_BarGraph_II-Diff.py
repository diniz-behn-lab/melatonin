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
#[-0.23644603353316285, 0.16099583689388197, 0.8025134255383435, 1.2310223985522342]

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
plt.bar(r + width, model, color = 'royalblue', 
        width = width, edgecolor = 'black', 
        label='Model') 

  
plt.xlabel("Experimental Condition") 
plt.ylabel("Phase Shift (h)") 
  
# plt.grid(linestyle='--') 
plt.xticks(r + width/2,['DLP','DLM','BLP','BLM']) 
plt.plot(r + width, [-0.21897837755552274, -0.21897837755552274, 0.7634377932709313, 0.7634377932709313],'o',color = 'darkorange', label = 'Delayed Light History')
plt.plot(r + width, [-0.23676848429042252, 0.17465555156651646, 0.7786504182689384, 1.2250925127694217],'o',color = 'darkorange',label = 'Advanced Light History')
plt.plot(r + width, [-0.2294450911619066, 0.16380992318019594, 0.8090005638228916, 1.2395028425216879],'o',color = 'darkorange',label = 'Low production')
plt.plot(r + width, [-0.18316323309520754, 0.1807228813350079, 0.8679435614973343, 1.2585500980756947],'o',color = 'darkorange',label = 'High production')
plt.plot(r + width, [-0.17115303825973882, -0.17115695697379607, 0.8756813789396247, 0.8765825262056737],'o',color = 'darkorange',label = 'Early Chronotype')
plt.plot(r + width, [-0.3234762446624444, -0.3234762446624444, 0.7132851442735699, 0.7132851442735699],'o',color = 'darkorange',label = 'Late Chronotype')
plt.plot(r + width, [-0.23653804417646285, 0.18080009999820135, 0.7690794651279518, 1.221421273841564],'o',color = 'darkorange',label = 'Increased Light Intensity')
plt.plot(r + width, [-0.2363937376018157, 0.08931684235497528, 0.9157121172801084, 1.2677500901559142],'o',color = 'darkorange',label = 'Decreased Light Intensity')
#plt.plot(r + width, [-0.29966864361298207, -0.29966864361298207, 0.7390668527878788, 0.7390668527878788],'o',color = 'green',label = 'High Production + Late Chronotype')
plt.axhline(0,linestyle='dashed',color='black')
#plt.legend(["Model", "Data"], loc ='center right', bbox_to_anchor=(1.7, 0.5)) 
plt.show()

