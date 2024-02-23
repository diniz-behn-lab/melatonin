# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:57:02 2023

@author: sstowe
"""

# Code to run the Hannay-Breslow Model 

#Armelle has been here 

# --------- Learning Python --------------------

# Class vs. Function
    # Functions are blocks of reusable code that perform specific tasks
    # Classes are blueprints for objects with attributes and methods
    # By defining functions within classes, you can create methods that are 
        # specific to that class
    
    # You can think of a class as a piece of code that specifies the data and 
        #behavior that represent and model a particular type of object.
    # A common analogy is that a class is like the blueprint for a house. You 
        # can use the blueprint to create several houses and even a complete 
        # neighborhood. Each concrete house is an object or instance that’s 
        # derived from the blueprint.

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
        self.beta_IP = 7.83e-4*60*60 # converting 1/sec to 1/hr
        self.beta_CP = 3.35e-4*60*60 # converting 1/sec to 1/hr
        self.beta_AP = 1.62e-4*60*60 #converting 1/sec to 1/hr

        self.a = 6*60 #4*60 #1.0442e-3 # CHANGED, the tiny value is from Breslow
        self.delta_M = 600/3600 # CHANGED, converting secs to hrs
        self.r = 15.36/3600 # CHANGED, converting secs to hrs
        
        self.psi_on = 1.04719755 #6.113 # CHANGED from Breslow 
        self.psi_off = 3.92699 #4.352 # CHANGED from Breslow

        self.M_max = 0.019513
        self.H_sat = 861
        self.sigma_M = 50
        self.m = 7*60 # CHANGED, converting 1/sec to 1/min 

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

        ## Melatonin Forcing Parameters
        self.B_1 = -0.74545016
        self.theta_M1 = 0.05671999
        self.B_2 = -0.76024892
        self.theta_M2 = 0.05994563
        self.epsilon = 0.18366069
        
        
        
# Set the exogenous melatonin administration schedule VERSION 5
    def ex_melatonin(self,t,melatonin_timing,melatonin_dosage):
        if melatonin_timing == None:
            return 0 #set exogenous melatonin to zero
        else: 
            if melatonin_timing-0.2 <= t <= melatonin_timing+0.3:
                sigma = np.sqrt(0.002)
                mu = melatonin_timing+0.1

                ex_mel = (1/sigma*np.sqrt(2*np.pi))*np.exp((-pow(t-mu,2))/(2*pow(sigma,2))) # Guassian function

                x = np.arange(0, 1, 0.01)
                melatonin_values = self.max_value(x, sigma)
                max_value = max(melatonin_values)
                print(max_value)
            
                normalize_ex_mel = (1/max_value)*ex_mel # normalize the values so the max is 1
                dose_ex_mel = (melatonin_dosage)*normalize_ex_mel # multiply by the dosage so the max = dosage
            
                return dose_ex_mel        
            else: 
                return 0
        
    
# Generate the curve for a 24h melatonin schedule so that the max value can be determined      
    def max_value(self, time, sigma):
        mu = 1/2
        Guassian = (1/sigma*np.sqrt(2*np.pi))*np.exp((-pow(time-mu,2))/(2*pow(sigma,2)))
        return Guassian

    

# Set the light schedule (timings and intensities)
    def light(self,t,schedule):
        
        if schedule == 1: # standard 16:8 schedule 
            full_light = 1000
            dim_light = 300
            wake_time = 7
            sleep_time = 15#23
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
    
    '''
# Melatonin dynamics from Breslow model
    def m_process(self,u):
        H2 = max(u[4],0)
        H2_conc = H2*861/200 # Convert from pg/L to pmol/L
        try:
            if H2_conc > 861*10:
                output = self.M_max
            else:
                output = self.M_max/(1 + np.exp((self.H_sat - H2_conc)/self.sigma_M))
        except:
            print("Slow the heck down there")
            output = self.M_max/(1 + np.exp((self.H_sat - H2_conc)/self.sigma_M))
        return output
    '''


# Timing of melatonin on and off
    def circ_response(self,psi):
        dlmo_phase = 5*np.pi/12
        psi = np.mod(psi - dlmo_phase,2*np.pi)
        
        if self.psi_on < psi < self.psi_off:
            #print("Pineal on")
            return self.a*np.exp(-self.r*np.mod(self.psi_on - self.psi_off,2*np.pi))
        else:
            #print("Pineal off") 
            return (1/360)*self.a * (1 - np.exp(-self.delta_M*np.mod(self.psi_on - psi,2*np.pi))) / (1 - np.exp(-self.delta_M*np.mod(self.psi_on - self.psi_off,2*np.pi)))

    '''
        if psi > self.psi_off and psi <= self.psi_on:
            print("Pineal off")
            return self.a * (1 - np.exp(-self.delta_M*np.mod(self.psi_on - psi,2*np.pi))) / (1 - np.exp(-self.delta_M*np.mod(self.psi_on - self.psi_off,2*np.pi)))
        else:
            print("Pineal on")
            return self.a*np.exp(-self.r*np.mod(self.psi_on - self.psi_off,2*np.pi))
    '''
    
    

# Defining the system of ODEs (6-dimensional system)
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

        # Light interaction with pacemaker
        Bhat = self.G*(1.0-n)*self.alpha0(t,schedule)
        LightAmp = (self.A_1/2.0)*Bhat*(1.0 - pow(R,4.0))*np.cos(Psi + self.beta_L1) + (self.A_2/2.0)*Bhat*R*(1.0 - pow(R,8.0))*np.cos(2.0*Psi + self.beta_L2) # L_R
        LightPhase = self.sigma*Bhat - (self.A_1/2.0)*Bhat*(pow(R,3.0) + 1.0/R)*np.sin(Psi + self.beta_L1) - (self.A_2/2.0)*Bhat*(1.0 + pow(R,8.0))*np.sin(2.0*Psi + self.beta_L2) # L_psi
        
        # Melatonin interaction with pacemaker
        #Mhat = self.m_process(y)
        Mhat = self.M_max/(1 + np.exp((self.H_sat - H2)/self.sigma_M))
        MelAmp = (self.B_1/2)*Mhat*(1.0 - pow(R,4.0))*np.cos(Psi + self.theta_M1) + (self.B_2/2.0)*Mhat*R*(1.0 - pow(R,8.0))*np.cos(2.0*Psi + self.theta_M2) # M_R
        MelPhase = self.epsilon*Mhat - (self.B_1/2.0)*Mhat*(pow(R,3.0)+1.0/R)*np.sin(Psi + self.theta_M1) - (self.B_2/2.0)*Mhat*(1.0 + pow(R,8.0))*np.sin(2.0*Psi + self.theta_M2) # M_psi

        tmp = 1 - self.m*Bhat # This m might need to be altered
        #S = not(H1 < 0.001 and tmp < 0)
        S = np.piecewise(tmp, [tmp >= 0, tmp < 0 and H1 < 0.001], [1, 0])
        #S = np.piecewise(tmp, [tmp >= 0, tmp < 0], [1, 0])
        
        
        dydt=np.zeros(6)

        dydt[0] = -1.0*(self.D + self.gamma)*R + (self.K/2.0)*np.cos(self.beta)*R*(1.0-pow(R,4.0)) + LightAmp #+ MelAmp # dR/dt
        dydt[1] = self.omega_0 + (self.K/2.0)*np.sin(self.beta)*(1 + pow(R,4.0)) + LightPhase #+ MelPhase # dpsi/dt
        dydt[2] = 60.0*(self.alpha0(t,schedule)*(1.0-n)-(self.delta*n)) # dn/dt

        dydt[3] = -self.beta_IP*H1 + self.circ_response(y[1])*tmp*S # dH1/dt
        dydt[4] = self.beta_IP*H1 - self.beta_CP*H2 + self.beta_AP*H3 # dH2/dt
        dydt[5] = -self.beta_AP*H3 + self.ex_melatonin(t,melatonin_timing,melatonin_dosage) # dH3/dt

        return(dydt)



    def integrateModel(self, tend, tstart=0.0, initial=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],melatonin_timing = None,melatonin_dosage=None, schedule = 1):
        """ 
        Integrate the model forward in time.
            tend: float giving the final time to integrate to
            initial=[R, Psi, n, H1, H2, H3]: initial dynamical state (initial 
            conditions)
            melatonin_timing is set to "None" as default
            melatonin_dosage is set to "None" as default
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
model.integrateModel(24*30,schedule=1) # use the integrateModel method with the object model
IC = model.results[-1,:] # get initial conditions from entrained model

#Uncomment this one to run it without exogenous melatonin
model.integrateModel(24*2,tstart=0.0,initial=IC, melatonin_timing=None, melatonin_dosage=None,schedule=2) # run the model from entrained ICs

#Uncomment this one to run it with exogenous melatonin 
#model.integrateModel(24*2,tstart=0.0,initial=IC, melatonin_timing=12.0, melatonin_dosage=2500) # with pulse function
#model.integrateModel(24*1,tstart=0.0,initial=IC, melatonin_timing=20.0, melatonin_dosage=7500,schedule=2) # with Guassian
#model.integrateModel(24*2,tstart=0.0,initial=IC, melatonin_timing=12.0, melatonin_dosage=0.2)



#--------- Plot Model Output -------------------

# Plotting H1, H2, and H3 (melatonin concentrations, pmol/L)
plt.plot(model.ts,model.results[:,3],lw=2)
plt.plot(model.ts,model.results[:,4],lw=2)
plt.plot(model.ts,model.results[:,5],lw=2)
plt.axvline(x=21.4)
plt.axvline(x=7)
plt.axvline(x=8.3)
plt.axvline(x=12)
plt.xlabel("Time (hours)")
plt.ylabel("Melatonin Concentration (pmol/L)")
plt.title("Melatonin Concentrations (pmol/L)")
plt.legend(["Pineal","Plasma", "Exogenous"])
plt.show()


'''
# Plotting H1, H2, and H3 (melatonin concentrations, pmol/L, zoomed)
plt.plot(model.ts[120:180],model.results[120:180,3],lw=2)
plt.plot(model.ts[120:180],model.results[120:180,4],lw=2)
plt.plot(model.ts[120:180],model.results[120:180,5],lw=2)
plt.axhline(200)
plt.axvline(12)
plt.xlabel("Time (hours)")
plt.ylabel("Melatonin Concentration (pmol/L)")
plt.title("Melatonin Concentrations")
plt.legend(["Pineal","Plasma", "Exogenous"])
plt.show()

'''


# Plotting H1, H2, and H3 (melatonin concentrations, pg/mL)
plt.plot(model.ts,model.results[:,3]/4.3,lw=2)
plt.plot(model.ts,model.results[:,4]/4.3,lw=2)
plt.plot(model.ts,model.results[:,5]/4.3,lw=2)
plt.axvline(x=21.4)
#plt.axvline(x=7)
plt.axvline(x=8.3)
plt.axhline(4*10)
plt.xlabel("Time (hours)")
plt.ylabel("Melatonin Concentration (pg/mL)")
plt.title("Melatonin Concentrations (pg/mL)")
plt.legend(["Pineal","Plasma", "Exogenous"])
plt.show()

'''

# Plotting H1, H2, and H3 (melatonin concentrations, pg/mL, zoomed)
plt.plot(model.ts[220:250],model.results[220:250,3]/4.3,lw=2)
plt.plot(model.ts[220:250],model.results[220:250,4]/4.3,lw=2)
plt.plot(model.ts[220:250],model.results[220:250,5]/4.3,lw=2)
plt.axhline(4)
plt.xlabel("Time (hours)")
plt.ylabel("Melatonin Concentration (pg/mL)")
plt.title("Melatonin Concentrations at DLMO (4pg/mL) Before")
plt.legend(["Pineal","Plasma", "Exogenous"])
plt.show()

# Plotting H1, H2, and H3 (melatonin concentrations, pg/mL, zoomed)
plt.plot(model.ts[450:479],model.results[450:479,3]/4.3,lw=2)
plt.plot(model.ts[450:479],model.results[450:479,4]/4.3,lw=2)
plt.plot(model.ts[450:479],model.results[450:479,5]/4.3,lw=2)
plt.axhline(4)
plt.xlabel("Time (hours)")
plt.ylabel("Melatonin Concentration (pg/mL)")
plt.title("Melatonin Concentrations at DLMO (4pg/mL) After")
plt.legend(["Pineal","Plasma", "Exogenous"])
plt.show()

'''

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
plt.plot(model.ts,np.mod(model.results[:,1],2*np.pi),lw=2)
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


