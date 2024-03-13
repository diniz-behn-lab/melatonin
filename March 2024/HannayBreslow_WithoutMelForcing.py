# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 12:45:54 2024

@author: sstowe

Hannay-Breslow model without melatonin forcing equations included (parameters have not been fit yet)

"""

#-------- Set Up -------------------------------

# Imports 

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


# ---------- Create Model Class -------------------

class HannayBreslowModel(object):
    """Driver of ForwardModel simulation"""


# Initialize the class
    def __init__(self):
        self.set_params() # setting parameters every time an object of the class is created


# Set parameter values 
    def set_params(self):
        ## Breslow Model
        self.beta_IP = (7.83e-4)*60*60 # converting 1/sec to 1/hr, Breslow 2013
        self.beta_CP = (3.35e-4)*60*60 # converting 1/sec to 1/hr, Breslow 2013
        self.beta_AP = (1.62e-4)*60*60 # converting 1/sec to 1/hr, Breslow 2013

        self.a = (0.1)*60*60 # pmol/L/sec converted to hours, I determined
        self.delta_M = 600 # sec, Breslow 2013
        self.r = 15.36 # sec, Breslow 2013
        
        self.psi_on = 1.2217 #1.13446401 # radians, I determined 
        self.psi_off = 3.5779  # radians, I determined
        
        self.m = 4.7565 # I determined by fitting to Zeitzer using differential evolution
        
        ## Hannay Model
        self.D = 0
        self.gamma = 0.024
        self.K = 0.06358 
        self.beta = -0.09318
        self.omega_0 = 0.263524

        self.A_1 = 0.3855 
        self.beta_L1 = -0.0026
        self.A_2 = 0.1977 
        self.beta_L2 = -0.957756
        self.sigma = 0.0400692

        self.delta = 0.0075
        self.alpha_0 = 0.05
        self.p = 1.5
        self.I_0 = 9325
        self.G = 33.75
        
        
# Set the light schedule (timings and intensities)
    def light(self,t,schedule):
        
        if schedule == 1: # standard 16:8 schedule 
            full_light = 1000
            dim_light = 300
            wake_time = 7
            sleep_time = 23
            sun_up = 8
            sun_down = 19
            
            is_awake = np.mod(t - wake_time,24) <= np.mod(sleep_time - wake_time,24)
            sun_is_up = np.mod(t - sun_up,24) <= np.mod(sun_down - sun_up,24)

            return is_awake*(full_light*sun_is_up + dim_light*(1 - sun_is_up))
        else: # Constant light environment
            return 0


# Define the alpha(L) function 
    def alpha0(self,t,schedule):
        """A helper function for modeling the light input processing"""
        
        return(self.alpha_0*pow(self.light(t,schedule), self.p)/(pow(self.light(t,schedule), self.p)+self.I_0));
    

# Set the exogenous melatonin administration schedule
    def ex_melatonin(self,t,melatonin_timing,melatonin_dosage):
        
        if melatonin_timing == None:
            return 0 # set exogenous melatonin to zero
        else: 
            t = np.mod(t,24)
            if melatonin_timing-0.1 <= t <= melatonin_timing+0.3:
                sigma = np.sqrt(0.002)
                mu = melatonin_timing+0.1

                ex_mel = (1/sigma*np.sqrt(2*np.pi))*np.exp((-pow(t-mu,2))/(2*pow(sigma,2))) # Guassian function

                x = np.arange(0, 1, 0.01)
                melatonin_values = self.max_value(x, sigma)
                max_value = max(melatonin_values)
                #print(max_value)
                
                converted_dose = self.mg_conversion(melatonin_dosage)
                #print(converted_dose)
            
                normalize_ex_mel = (1/max_value)*ex_mel # normalize the values so the max is 1
                dose_ex_mel = (converted_dose)*normalize_ex_mel # multiply by the dosage so the max = dosage
            
                return dose_ex_mel        
            else: 
                return 0
    
# Generate the curve for a 24h melatonin schedule so that the max value can be determined      
    def max_value(self, time, sigma):
        mu = 1/2
        Guassian = (1/sigma*np.sqrt(2*np.pi))*np.exp((-pow(time-mu,2))/(2*pow(sigma,2)))
        return Guassian    

# Convert mg dose to value to be used in the Guassian dosing curve
    def mg_conversion(self, melatonin_dosage):
        x_line = melatonin_dosage
        #y_line = (56686*x_line) + 16897 # with 2.0mg dose (3pts fit)
        #y_line = (57553*x_line) + 7234 # without 2.0mg dose (2pts fit)
        y_line = 832.37*pow(x_line,3) - 4840.4*pow(x_line,2) + 64949*x_line + 2189.4 # Cubic fit to 5 points 
        return y_line

        

# Defining the system of ODEs (2-dimensional system)
    def ODESystem(self,t,y,melatonin_timing,melatonin_dosage,schedule):
        """
        This defines the ode system for the single population model and 
        returns dydt numpy array.
        """
        
        R = y[0]
        Psi = y[1]
        n = y[2]
        H1 = y[3]
        H2 = y[4]
        H3 = y[5]

        
        # Pineal activation/deactivation 
        psi = Psi #t*(np.pi/12) + (8*np.pi/12) # Convert hours to radians, shift to align with Hannay's convention 
        if (np.mod(psi,2*np.pi) > self.psi_on) and (np.mod(psi,2*np.pi) < self.psi_off): 
            A = self.a*((1 - np.exp(-self.delta_M*np.mod(psi - self.psi_on,2*np.pi))/1 - np.exp(-self.delta_M*np.mod(self.psi_off - self.psi_on,2*np.pi))));
        else: 
            A = self.a*(np.exp(-self.r*np.mod(psi - self.psi_off,2*np.pi)))
    
    
        # Light interaction with pacemaker
        Bhat = self.G*(1.0-n)*self.alpha0(t,schedule)
    
        # Light forcing equations
        LightAmp = (self.A_1/2.0)*Bhat*(1.0 - pow(R,4.0))*np.cos(Psi + self.beta_L1) + (self.A_2/2.0)*Bhat*R*(1.0 - pow(R,8.0))*np.cos(2.0*Psi + self.beta_L2) # L_R
        LightPhase = self.sigma*Bhat - (self.A_1/2.0)*Bhat*(pow(R,3.0) + 1.0/R)*np.sin(Psi + self.beta_L1) - (self.A_2/2.0)*Bhat*(1.0 + pow(R,8.0))*np.sin(2.0*Psi + self.beta_L2) # L_psi
    
    
        # Switch pineal on and off in the presence of light 
        tmp = 1 - self.m*Bhat
        S = np.piecewise(tmp, [tmp >= 0, tmp < 0 and H1 < 0.001], [1, 0])
    
    
        dydt=np.zeros(6)
    
        # ODE System  
        dydt[0] = -(self.D + self.gamma)*R + (self.K/2)*np.cos(self.beta)*R*(1 - pow(R,4.0)) + LightAmp # dR/dt
        dydt[1] = self.omega_0 + (self.K/2)*np.sin(self.beta)*(1 + pow(R,4.0)) + LightPhase # dpsi/dt 
        dydt[2] = 60.0*(self.alpha0(t,schedule)*(1.0-n)-(self.delta*n)) # dn/dt
        
        dydt[3] = -self.beta_IP*H1 + A*(1 - self.m*Bhat)*S # dH1/dt
        dydt[4] = self.beta_IP*H1 - self.beta_CP*H2 + self.beta_AP*H3 # dH2/dt
        dydt[5] = -self.beta_AP*H3 + self.ex_melatonin(t,melatonin_timing,melatonin_dosage) # dH3/dt

        return(dydt)



    def integrateModel(self, tend, tstart=0.0, initial=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], melatonin_timing = None, melatonin_dosage=None, schedule = 1):
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
        
        r_variable = sp.integrate.solve_ivp(self.ODESystem,(tstart,tend), initial, t_eval=self.ts, method='Radau', args=(melatonin_timing,melatonin_dosage,schedule,))
        self.results = np.transpose(r_variable.y)

        return
#-------- end of HannayBreslowModel class ---------




#--------- Run the Model ---------------

model = HannayBreslowModel() # defining model as a new object built with the HannayBreslowModel class 
model.integrateModel(10*24,schedule=1) # use the integrateModel method with the object model
IC = model.results[-1,:] # get initial conditions from entrained model

#Uncomment this one to run it without exogenous melatonin
model.integrateModel(1*24,tstart=0.0,initial=IC,schedule=2) # run the model from entrained ICs

#Uncomment this one to run it with exogenous melatonin
#model.integrateModel(24*1,tstart=0.0,initial=IC, melatonin_timing=13.0, melatonin_dosage=3,schedule=2)

# Finding DLMO and DLMOff
plasma_mel = model.results[:,4]/4.3 # converting output to pg/mL
times = model.ts[:] # defining times from first 24hrs 
melatonin, = np.where(plasma_mel<=10) # finding all the indices where concentration is below 10pg/mL
DLMO = times[melatonin[-1]] # finding the time corresponding to the last index below threshold, DLMO
DLMOff = times[melatonin[0]] # finding the time corresponding to the first index below threshold, DLMOff


#--------- Plot Model Output -------------------

'''
# Plotting H1, H2, and H3 (melatonin concentrations, pmol/L)
plt.plot(model.ts,model.results[:,3],lw=2)
plt.plot(model.ts,model.results[:,4],lw=2)
plt.plot(model.ts,model.results[:,5],lw=2)
#plt.axvline(x=6)
#plt.axvline(x=20.3)
#plt.axvline(x=12)
#plt.axhline(200)
plt.xlabel("Time (hours)")
plt.ylabel("Melatonin Concentration (pmol/L)")
plt.title("Melatonin Concentrations (pmol/L)")
plt.legend(["Pineal","Plasma", "Exogenous"])
plt.show()
'''


# Plotting H1, H2, and H3 (melatonin concentrations, pg/mL)
plt.plot(model.ts,model.results[:,3]/4.3,lw=2)
plt.plot(model.ts,model.results[:,4]/4.3,lw=2)
plt.plot(model.ts,model.results[:,5]/4.3,lw=2)
#plt.axvline(x=6) # Set psi_off
#plt.axvline(x=20.3) # Set psi_on
plt.axvline(x=DLMOff)
plt.axvline(x=DLMO)
plt.axvline(x=21.4,linestyle='dashed',color='grey') # Burgess 2008 reported average DLMO
plt.axvline(x=8.3,linestyle='dashed',color='grey') # Burgess 2008 reported average DLMOff
plt.axhline(10,linestyle='dashed',color='black') # DLMO threshold 
#plt.axhline(1000)
plt.xlabel("Time (hours)")
plt.ylabel("Melatonin Concentration (pg/mL)")
plt.title("Melatonin Concentrations (pg/mL)")
plt.legend(["Pineal","Plasma", "Exogenous"])
plt.show()



# Plotting R
plt.plot(model.ts,model.results[:,0],lw=2)
plt.xlabel("Time (hours)")
plt.ylabel("R, Collective Amplitude")
plt.title("Time Trace of R, Collective Amplitude")
plt.show()


# Plotting psi
plt.plot(model.ts,model.results[:,1],lw=2)
plt.xlabel("Time (hours)")
plt.ylabel("Psi, Mean Phase (radians)")
plt.title("Time Trace of Psi, Mean Phase")
plt.show()

# Plotting psi mod 2pi
plt.plot(model.ts,np.mod(model.results[:,1],2*np.pi),'o')#lw=2)
plt.axhline(1.04719755)
plt.axhline(3.92699)
plt.axvline(x=7)
plt.axvline(x=23)
plt.xlabel("Time (hours)")
plt.ylabel("Psi, Mean Phase (radians)")
plt.title("Time Trace of Psi, Mean Phase")
plt.show()

# Plotting n
plt.plot(model.ts,model.results[:,2],lw=2)
#plt.axvline(x=7)
#plt.axvline(x=23)
plt.xlabel("Time (hours)")
plt.ylabel("Proportion of Activated Photoreceptors")
plt.title("Time Trace of Photoreceptor Activation")
plt.show()
