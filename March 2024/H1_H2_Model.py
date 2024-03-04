# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:11:51 2024

@author: sstowe

Coding two compartment model of melatonin 

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
        self.beta_IP = 7.83e-4*60*60 # converting 1/sec to 1/hr
        self.beta_CP = 3.35e-4*60*60 # converting 1/sec to 1/hr

        self.a = 4*60 # pmol/L/sec
        self.delta_M = 600 # sec
        self.r = 500 # sec
        
        self.psi_on = 1.0472 # radians 
        self.psi_off = 3.92699 # radians 
        
        

# Defining the system of ODEs (2-dimensional system)
    def ODESystem(self,t,y):
        """
        This defines the ode system for the single population model and 
        returns dydt numpy array.
        """
        H1 = y[0]
        H2 = y[1]

        dydt=np.zeros(2)
        
        psi = t*(np.pi/12) + (8*np.pi/12) # Convert hours to radians, shift to align with Hannay's convention 
        
        if (np.mod(psi,2*np.pi) > self.psi_on) and (np.mod(psi,2*np.pi) < self.psi_off): 
            A = self.a*((1 - np.exp(-self.delta_M*np.mod(psi - self.psi_on,2*np.pi))/1 - np.exp(-self.delta_M*np.mod(self.psi_off - self.psi_on,2*np.pi))));
        else: 
            A = self.a*(np.exp(-self.r*np.mod(psi - self.psi_off,2*np.pi)))
    
    
        dydt[0] = -self.beta_IP*H1 + A
        dydt[1] = self.beta_IP*H1 - self.beta_CP*H2

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
model.integrateModel(10*24) # use the integrateModel method with the object model
IC = model.results[-1,:] # get initial conditions from entrained model

#Uncomment this one to run it without exogenous melatonin
model.integrateModel(1*24,tstart=0.0,initial=IC) # run the model from entrained ICs



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
