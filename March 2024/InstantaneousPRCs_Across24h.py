# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:12:51 2024

@author: sstowe

Determing the cross correlation between the macro PRCs across 24 h with the 
model entrained and the light changing dynamically

"""

#-------- Set Up -------------------------------

# Imports 

import numpy as np
import matplotlib.pyplot as plt

#from itertools import repeat
'''

# Create array of light levels 

light_list = [0, # 0:00
         0, # 1:00
         0, # 2:00
         0, # 3:00
         0, # 4:00
         0, # 5:00
         0, # 6:00
         300, # 7:00
         1000, # 8:00
         1000, # 9:00
         1000, # 10:00
         1000, # 11:00
         1000, # 12:00
         1000, # 13:00
         1000, # 14:00
         1000, # 15:00
         1000, # 16:00
         1000, # 17:00
         1000, # 18:00
         300, # 19:00
         300, # 20:00
         300, # 21:00
         300, # 22:00
         0, # 23:00 
         ]


plt.plot(np.linspace(0,23,24),light_list)
plt.show()

# Create array of R values on the hour 

R_list = [0.831465, # 0:00
     0.825455, # 1:00
     0.819796, # 2:00
     0.814591, # 3:00
     0.81024, # 4:00
     0.807036, # 5:00
     0.804898, # 6:00
     0.800842, # 7:00
     0.785607, # 8:00
     0.770174, # 9:00
     0.75969, # 10:00
     0.754191, # 11:00
     0.757813, # 12:00
     0.772668, # 13:00
     0.796714, # 14:00
     0.824486, # 15:00
     0.849996, # 16:00
     0.869321, # 17:00
     0.881161, # 18:00
     0.885841, # 19:00
     0.880708, # 20:00
     0.87176, # 21:00
     0.857716, # 22:00
     0.838658, # 23:00 
     ]

plt.plot(np.linspace(0,23,24),R_list)
plt.show()


# Create array of H2 levels 

H2 = [164.593, # 0:00
      254.839, # 1:00
      285.237, # 2:00
      294.537, # 3:00
      297.304, # 4:00
      298.15, # 5:00
      287.531, # 6:00
      137.01, # 7:00
      44.8895, # 8:00
      13.6705, # 9:00
      4.10668, # 10:00
      1.23035, # 11:00
      0.368426, # 12:00
      0.110297, # 13:00
      0.0330187, # 14:00
      0.00988949, # 15:00
      0.00295905, # 16:00
      0.000886705, # 17:00
      0.000265289, # 18:00
      7.9469e-05, # 19:00
      6.48027, # 20:00
      39.5963, # 21:00
      46.5628, # 22:00
      46.407, # 23:00 
         ]

plt.plot(np.linspace(0,23,24),H2)
plt.show()


# Corrected Hsat and sigma_M 
x = [-0.98204363, -0.07764001, -0.7152688,   0.8511226,   0.07833321] # Error = 3.655967724146368 # Optimization terminated successfully!! 

B1 = x[0]
theta1 = x[1]
B2 = x[2]
theta2 = x[3]
epsilon = x[4]

M_max = 0.019513 # Breslow 2013
H_sat = 301 # Scaled for our peak concentration #861 # Breslow 2013
sigma_M = 17.5 # Scaled for our peak concentration #50 # Breslow 2013
    

# Light forcing function parameters
A1 = 0.3855 
beta1 = -0.0026
A2 = 0.1977 
beta2 = -0.957756
sigma = 0.0400692




# ---------------- Calculate Cross Correlation at Each Hour ------

Hours_Out = list()

for t in range(0,24): 

    # Macroscopic PRC to melatonin
    Mhat = M_max/(1 + np.exp((H_sat - H2[t])/(sigma_M)))
    R = R_list[t]
    macro_prc = lambda phi: epsilon*Mhat-B1*Mhat*0.5*(pow(R,3.0)+1.0/R)*np.sin(phi+theta1)-B2*Mhat*0.5*(1.0+pow(R,8.0))*np.sin(2.0*phi+theta2)
    phi = np.linspace(0,2*np.pi,361,endpoint=True)
    macro_response = macro_prc(phi)



    # Macroscopic PRC to light
    G = 33.75
    alpha_0 = 0.05
    p = 1.5
    I_0 = 9325
    light = light_list[t]

    Bhat = G*(alpha_0*pow(light, p)/(pow(light, p) + I_0)) # Corresponds to a dim light response
    
    macro_prc = lambda phi: sigma*Bhat-A1*Bhat*0.5*(pow(R,3.0)+1.0/R)*np.sin(phi+beta1)-A2*Bhat*0.5*(1.0+pow(R,8.0))*np.sin(2.0*phi+beta2)
    macro_light_response = macro_prc(phi)



    # Determine phase difference between melatonin and light PRCs 
    mel_fourier = np.fft.fft(macro_response)
    light_fourier = np.fft.fft(np.flip(macro_light_response))
    cross_cor_four = mel_fourier*light_fourier
    cross_cor = np.real(np.fft.ifft(cross_cor_four))
    cross_cor = np.fft.fftshift(cross_cor)/(np.linalg.norm(macro_response)*np.linalg.norm(macro_light_response))
    phi[phi > np.pi] = phi[phi > np.pi] - 2*np.pi
    phi = np.fft.fftshift(phi)
    max_alignment_idx = np.argmax(cross_cor)

    hours_out = -12*phi[max_alignment_idx]/np.pi # plot this value for t = 0:24

    Hours_Out.append(hours_out)
    

plt.plot(np.linspace(0,23,24),Hours_Out)
plt.show()

'''

# ---- Do it again, but with finer resolution -----------




H2 = [164.593,
177.84,
190.059,
201.248,
211.45,
220.708,
229.061,
236.584,
243.349,
249.414,
254.839,
259.682,
264.002,
267.83,
271.255,
274.322,
277.057,
279.489,
281.645,
283.552,
285.237,
286.728,
288.053,
289.221,
290.261,
291.196,
292.032,
292.777,
293.438,
294.023,
294.537,
294.99,
295.387,
295.736,
296.044,
296.319,
296.562,
296.777,
296.971,
297.146,
297.304,
297.444,
297.569,
297.679,
297.777,
297.862,
297.936,
298.001,
298.058,
298.107,
298.15,
298.189,
298.223,
298.255,
298.284,
298.309,
298.332,
298.352,
297.686,
294.334,
287.531,
277.383,
264.49,
249.59,
233.386,
216.546,
199.576,
182.87,
166.751,
151.419,
137.01,
123.574,
111.145,
99.729,
89.3138,
79.8569,
71.3048,
63.5965,
56.6675,
50.4529,
44.8895,
39.9167,
35.4778,
31.5192,
27.9932,
24.8538,
22.0612,
19.5784,
17.3715,
15.4114,
13.6705,
12.1249,
10.7533,
9.53576,
8.45567,
7.49754,
6.64748,
5.89372,
5.22516,
4.63229,
4.10668,
3.6405,
3.22726,
2.86092,
2.53599,
2.24804,
1.99281,
1.76643,
1.5658,
1.38803,
1.23035,
1.09052,
0.966654,
0.856923,
0.759611,
0.673227,
0.596694,
0.528938,
0.468909,
0.415723,
0.368426,
0.326483,
0.289358,
0.256515,
0.22742,
0.201635,
0.178711,
0.158365,
0.140347,
0.124408,
0.110297,
0.0977911,
0.0866838,
0.0768166,
0.0680733,
0.0603373,
0.0534921,
0.0474244,
0.0420446,
0.0372608,
0.0330187,
0.0292641,
0.0259426,
0.0229999,
0.0203922,
0.0180736,
0.0160158,
0.0141937,
0.0125817,
0.0111547,
0.00988949,
0.00876586,
0.00776838,
0.00688481,
0.0061029,
0.0054104,
0.00479736,
0.00425219,
0.00376783,
0.00333866,
0.00295905,
0.00262338,
0.00232601,
0.00206249,
0.0018281,
0.00161986,
0.00143536,
0.00127216,
0.00112784,
0.001,
0.000886705,
0.000785937,
0.000696412,
0.000617088,
0.000546926,
0.000484884,
0.000429921,
0.000381305,
0.000338016,
0.000299481,
0.000265289,
0.000235035,
0.00020831,
0.000184706,
0.000163815,
0.000145237,
0.000128736,
0.000114111,
0.000101146,
8.96548e-05,
7.9469e-05,
7.04411e-05,
6.24371e-05,
5.53439e-05,
4.90562e-05,
4.34828e-05,
3.85428e-05,
3.41638e-05,
0.514527,
2.90047,
6.48027,
10.6457,
14.9903,
19.2464,
23.2562,
26.9279,
30.2124,
33.1083,
35.6209,
37.7728,
39.5963,
41.1245,
42.3877,
43.4257,
44.2689,
44.942,
45.4692,
45.8752,
46.1768,
46.4022,
46.5628,
46.6675,
46.7255,
46.7454,
46.7365,
46.7074,
46.6642,
46.6093,
46.5464,
46.4782,
46.407,
50.1069,
59.4991,
72.4719,
87.4282,
103.337,
119.422,
135.156,
150.256,
164.483,
]



R_list = [0.831465,
0.830845,
0.83023,
0.829619,
0.829012,
0.828409,
0.827811,
0.827216,
0.826625,
0.826038,
0.825455,
0.824875,
0.824298,
0.823725,
0.823155,
0.822587,
0.822022,
0.82146,
0.820901,
0.820346,
0.819796,
0.819249,
0.818707,
0.81817,
0.817638,
0.817112,
0.816593,
0.816081,
0.815577,
0.81508,
0.814591,
0.814111,
0.81364,
0.813179,
0.812727,
0.812286,
0.811854,
0.811433,
0.811023,
0.810625,
0.81024,
0.809866,
0.809504,
0.809154,
0.808816,
0.80849,
0.808175,
0.807873,
0.807582,
0.807303,
0.807036,
0.806781,
0.806537,
0.806305,
0.806085,
0.805875,
0.805676,
0.805487,
0.805307,
0.80512,
0.804898,
0.804616,
0.80427,
0.803875,
0.803451,
0.803017,
0.80258,
0.802143,
0.801707,
0.801274,
0.800842,
0.79891,
0.797114,
0.795431,
0.793844,
0.792336,
0.790895,
0.789512,
0.788177,
0.786883,
0.785607,
0.783459,
0.781539,
0.779795,
0.778189,
0.776692,
0.775279,
0.773933,
0.77264,
0.771388,
0.770174,
0.768991,
0.767835,
0.766708,
0.765607,
0.764535,
0.763494,
0.762485,
0.761512,
0.760579,
0.75969,
0.758849,
0.75806,
0.757328,
0.756657,
0.756053,
0.75552,
0.755063,
0.754686,
0.754394,
0.754191,
0.754081,
0.754068,
0.754155,
0.754347,
0.754645,
0.755053,
0.755572,
0.756204,
0.75695,
0.757813,
0.758792,
0.759887,
0.761098,
0.762423,
0.76386,
0.765409,
0.767068,
0.768832,
0.7707,
0.772668,
0.77473,
0.77688,
0.779116,
0.781435,
0.78383,
0.786296,
0.788829,
0.791406,
0.794036,
0.796714,
0.799433,
0.80219,
0.804979,
0.807764,
0.810555,
0.813353,
0.816152,
0.818945,
0.821724,
0.824486,
0.827225,
0.829933,
0.832608,
0.835243,
0.837834,
0.840378,
0.842869,
0.845304,
0.847681,
0.849996,
0.852247,
0.854432,
0.856549,
0.858594,
0.860567,
0.862467,
0.864293,
0.866045,
0.867721,
0.869321,
0.870844,
0.872291,
0.873662,
0.874957,
0.876177,
0.877322,
0.878392,
0.879388,
0.880311,
0.881161,
0.88194,
0.882647,
0.883284,
0.883852,
0.884352,
0.884783,
0.885148,
0.885447,
0.88568,
0.885841,
0.885363,
0.884901,
0.884444,
0.883985,
0.883512,
0.883017,
0.882493,
0.881937,
0.881343,
0.880708,
0.880031,
0.879308,
0.878538,
0.87772,
0.876853,
0.875936,
0.874969,
0.873951,
0.872881,
0.87176,
0.870588,
0.869363,
0.868088,
0.86676,
0.865381,
0.86395,
0.862468,
0.860935,
0.859351,
0.857716,
0.85603,
0.854294,
0.852509,
0.850674,
0.848791,
0.846859,
0.844878,
0.842851,
0.840777,
0.838658,
0.837988,
0.837324,
0.836664,
0.836009,
0.835359,
0.834712,
0.834071,
0.833434,
0.832801,
]



# Corrected Hsat and sigma_M 
x = [-0.98204363, -0.07764001, -0.7152688,   0.8511226,   0.07833321] # Error = 3.655967724146368 # Optimization terminated successfully!! 

B1 = x[0]
theta1 = x[1]
B2 = x[2]
theta2 = x[3]
epsilon = x[4]

M_max = 0.019513 # Breslow 2013
H_sat = 301 # Scaled for our peak concentration #861 # Breslow 2013
sigma_M = 17.5 # Scaled for our peak concentration #50 # Breslow 2013
    

# Light forcing function parameters
A1 = 0.3855 
beta1 = -0.0026
A2 = 0.1977 
beta2 = -0.957756
sigma = 0.0400692




# ---------------- Calculate Cross Correlation at Each Hour ------

Hours_Out = list()

for t in range(0,240): 

    # Macroscopic PRC to melatonin
    Mhat = M_max/(1 + np.exp((H_sat - H2[t])/(sigma_M)))
    R = R_list[t]
    macro_prc = lambda phi: epsilon*Mhat-B1*Mhat*0.5*(pow(R,3.0)+1.0/R)*np.sin(phi+theta1)-B2*Mhat*0.5*(1.0+pow(R,8.0))*np.sin(2.0*phi+theta2)
    phi = np.linspace(0,2*np.pi,361,endpoint=True)
    macro_response = macro_prc(phi)



    # Macroscopic PRC to light
    G = 33.75
    alpha_0 = 0.05
    p = 1.5
    I_0 = 9325
    
    if 0 <= t < 70:
        light = 0
    elif 70<= t < 80:
        light = 300
    elif 80 <= t < 190:
        light = 1000
    elif 190 <= t < 230:
        light = 300
    else:
        light = 0

    Bhat = G*(alpha_0*pow(light, p)/(pow(light, p) + I_0)) # Corresponds to a dim light response
    
    macro_prc = lambda phi: sigma*Bhat-A1*Bhat*0.5*(pow(R,3.0)+1.0/R)*np.sin(phi+beta1)-A2*Bhat*0.5*(1.0+pow(R,8.0))*np.sin(2.0*phi+beta2)
    macro_light_response = macro_prc(phi)



    # Determine phase difference between melatonin and light PRCs 
    mel_fourier = np.fft.fft(macro_response)
    light_fourier = np.fft.fft(np.flip(macro_light_response))
    cross_cor_four = mel_fourier*light_fourier
    cross_cor = np.real(np.fft.ifft(cross_cor_four))
    cross_cor = np.fft.fftshift(cross_cor)/(np.linalg.norm(macro_response)*np.linalg.norm(macro_light_response))
    phi[phi > np.pi] = phi[phi > np.pi] - 2*np.pi
    phi = np.fft.fftshift(phi)
    max_alignment_idx = np.argmax(cross_cor)

    hours_out = -12*phi[max_alignment_idx]/np.pi # plot this value for t = 0:24

    Hours_Out.append(hours_out)
    
plt.rcParams.update({'font.size': 15})
plt.plot(np.linspace(0,239,240)/10,Hours_Out,lw=3,color='mediumturquoise')
plt.xlabel("Clock Time (hours)")
plt.ylabel("Hours Out of Phase (h)")
plt.title("Correlation Across the Day")
plt.ylim([10.7,11.2])
plt.xlim([7.5,22.5])

maximum = max(Hours_Out)
minimum = min(Hours_Out)


