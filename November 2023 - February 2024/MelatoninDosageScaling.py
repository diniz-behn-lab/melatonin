# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 11:47:39 2024

@author: sstowe

Figuring out how to scale the mg dosage of melatonin

"""

import numpy as np
import matplotlib.pyplot as plt

x = np.array([0.3, 2.0, 5.0])
y = np.array([24500, 145000, 295000])

plt.plot(x,y,'o')
plt.plot(x,y)
plt.show()