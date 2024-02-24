# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 16:45:40 2024

@author: sstowe

Two compartment model of melatonin

St. Hilaire JPR 2007 

"""

#-------- Set Up -------------------------------

# Imports 

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


# ---------- Create Model Class -------------------

class TwoCompartmentMelatonin(object):
    """Driver of ForwardModel simulation"""

# Initialize the class
    def __init__(self):
        self.set_params() # setting parameters every time an object of the class is created

# Set parameter values 
    def set_params(self):
        ## Breslow Model
        self.beta_IP = 7.83e-4*60*60 # converting 1/sec to 1/hr
        self.beta_CP = 3.35e-4*60*60 # converting 1/sec to 1/hr

        self.a = 6*60 #4*60 #1.0442e-3 # CHANGED, the tiny value is from Breslow
        self.lamb = 600/3600 # CHANGED, converting secs to hrs
        
        self.t_on = 1.04719755 #6.113 # CHANGED from Breslow 
        self.t_off = 3.92699 #4.352 # CHANGED from Breslow
        
        
        

# Defining the system of ODEs (2-dimensional system)
    def ODESystem(self,t,y):
        """
        This defines the ode system for the single population model and 
        returns dydt numpy array.
        """
        H1 = y[0]
        H2 = y[1]

        dydt=np.zeros(2)

        dydt[0] =
        dydt[1] = 

        return(dydt)



    def integrateModel(self, tend, tstart=0.0, initial=[1.0, 0.0]):
        """ 
        Integrate the model forward in time.
            tend: float giving the final time to integrate to
            initial=[H1, H2]: initial dynamical state (initial 
            conditions)
        Writes the integration results into the scipy array self.results.
        Returns the circadian phase (in hours) at the ending time for the system.
        """

        dt = 0.1
        self.ts = np.arange(tstart,tend,dt)
        initial[1] = np.mod(initial[1], 2*sp.pi) #start the initial phase between 0 and 2pi
    
        # Sanitize input: sometimes np.arange witll output values outside of the range of tstart -> tend
        self.ts = self.ts[self.ts <= tend]
        self.ts = self.ts[self.ts >= tstart]
        
        r_variable = sp.integrate.solve_ivp(self.ODESystem,(tstart,tend), initial, t_eval=self.ts, method='Radau')
        self.results = np.transpose(r_variable.y)

        return
#-------- end of HannayBreslowModel class ---------




#--------- Run the Model ---------------

model = TwoCompartmentMelatonin() # defining model as a new object built with the HannayBreslowModel class 
model.integrateModel(24*30) # use the integrateModel method with the object model
IC = model.results[-1,:] # get initial conditions from entrained model

#Uncomment this one to run it without exogenous melatonin
model.integrateModel(24*1,tstart=0.0,initial=IC) # run the model from entrained ICs

#Uncomment this one to run it with exogenous melatonin 
#model.integrateModel(24*2,tstart=0.0,initial=IC, melatonin_timing=12.0, melatonin_dosage=2500) # with pulse function
#model.integrateModel(24*1,tstart=0.0,initial=IC, melatonin_timing=20.0, melatonin_dosage=7500,schedule=2) # with Guassian
#model.integrateModel(24*2,tstart=0.0,initial=IC, melatonin_timing=12.0, melatonin_dosage=0.2)



#--------- Plot Model Output -------------------

# Plotting H1, H2, and H3 (melatonin concentrations, pmol/L)
plt.plot(model.ts,model.results[:,0],lw=2)
plt.plot(model.ts,model.results[:,1],lw=2)
#plt.axvline(x=23)
#plt.axvline(x=12)
plt.xlabel("Time (hours)")
plt.ylabel("Melatonin Concentration (pmol/L)")
plt.title("Melatonin Concentrations (pmol/L)")
plt.legend(["Pineal","Plasma"])
plt.show()
