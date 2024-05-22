# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 13:30:41 2024

@author: sstowe

Comparison plots with Hannay 2019 and Hannay-Breslow 2024

"""

#-------- Set Up -------------------------------

# Imports 

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


# ---------- Create Hanny 2019 Model Class -------------------

class HannayModel(object):
    """Driver of ForwardModel simulation"""


# Initialize the class
    def __init__(self):
        self.set_params() # setting parameters every time an object of the class is created


# Set parameter values 
    def set_params(self):
        
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
        else: 
            return 0


# Define the alpha(L) function 
    def alpha0(self,t,schedule):
        """A helper function for modeling the light input processing"""
        
        return(self.alpha_0*pow(self.light(t,schedule), self.p)/(pow(self.light(t,schedule), self.p)+self.I_0));

        

# Defining the system of ODEs (3-dimensional system)
    def ODESystem(self,t,y,schedule):
        """
        This defines the ode system for the single population model and 
        returns dydt numpy array.
        """
        
        R = y[0]
        Psi = y[1]
        n = y[2]

    
        # Light interaction with pacemaker
        Bhat = self.G*(1.0-n)*self.alpha0(t,schedule)
        #print(Bhat)
        # Light forcing equations
        LightAmp = (self.A_1/2.0)*Bhat*(1.0 - pow(R,4.0))*np.cos(Psi + self.beta_L1) + (self.A_2/2.0)*Bhat*R*(1.0 - pow(R,8.0))*np.cos(2.0*Psi + self.beta_L2) # L_R
        LightPhase = self.sigma*Bhat - (self.A_1/2.0)*Bhat*(pow(R,3.0) + 1.0/R)*np.sin(Psi + self.beta_L1) - (self.A_2/2.0)*Bhat*(1.0 + pow(R,8.0))*np.sin(2.0*Psi + self.beta_L2) # L_psi
    
        dydt=np.zeros(3)
    
        # ODE System  
        dydt[0] = -(self.D + self.gamma)*R + (self.K/2)*np.cos(self.beta)*R*(1 - pow(R,4.0)) + LightAmp # dR/dt
        dydt[1] = self.omega_0 + (self.K/2)*np.sin(self.beta)*(1 + pow(R,4.0)) + LightPhase # dpsi/dt 
        dydt[2] = 60.0*(self.alpha0(t,schedule)*(1.0-n)-(self.delta*n)) # dn/dt
    

        return(dydt)



    def integrateModel(self, tend, tstart=0.0, initial=[1.0, 0.0, 0.0], schedule = 1):
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
        
        r_variable = sp.integrate.solve_ivp(self.ODESystem,(tstart,tend), initial, t_eval=self.ts, method='Radau', args=(schedule,))
        self.results = np.transpose(r_variable.y)

        return
#-------- end of HannayModel class ---------







# ---------- Create Hannay-Breslow Model Class -------------------

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

        self.a = (0.101)*60*60 # pmol/L/sec converted to hours, I determined
        self.delta_M = 600 # sec, Breslow 2013
        self.r = 15.36 # sec, Breslow 2013
        
        self.psi_on = 1.2566 # radians, I determined 
        self.psi_off = 3.6128 # radians, I determined
        
        self.M_max = 0.019513 # Breslow 2013
        self.H_sat = 301 # Scaled from Breslow 2013
        self.sigma_M = 17.5 # Scaled from Breslow 2013
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
        
        ## Melatonin Forcing Parameters   
        # Fitting to the cubic dose curve 
        # Corrected Hsat and sigma_M 
        x = [-0.98204363, -0.07764001, -0.7152688,   0.8511226,   0.07833321] # Error = 3.655967724146368 # Optimization terminated successfully!!
        self.B_1 = x[0] 
        self.theta_M1 = x[1]
        self.B_2 = x[2]
        self.theta_M2 = x[3]
        self.epsilon = x[4]
        
        
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
        else: 
            return 0


# Define the alpha(L) function 
    def alpha0(self,t,schedule):
        """A helper function for modeling the light input processing"""
        
        return(self.alpha_0*pow(self.light(t,schedule), self.p)/(pow(self.light(t,schedule), self.p)+self.I_0));
    

# Set the exogenous melatonin administration schedule VERSION 5
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
        #y_line = (56383*x_line) + 3085.1 # 2pts fit (Wyatt 2006)
        #y_line = 7500
        y_line = 832.37*pow(x_line,3) - 4840.4*pow(x_line,2) + 64949*x_line + 2189.4 # Cubic fit to 5 points
        return y_line

        

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

        
        # Pineal activation/deactivation 
        psi = Psi #t*(np.pi/12) + (8*np.pi/12) # Convert hours to radians, shift to align with Hannay's convention 
        if (np.mod(psi,2*np.pi) > self.psi_on) and (np.mod(psi,2*np.pi) < self.psi_off): 
            A = self.a*((1 - np.exp(-self.delta_M*np.mod(psi - self.psi_on,2*np.pi))/1 - np.exp(-self.delta_M*np.mod(self.psi_off - self.psi_on,2*np.pi))));
        else: 
            A = self.a*(np.exp(-self.r*np.mod(psi - self.psi_off,2*np.pi)))
    
    
        # Light interaction with pacemaker
        Bhat = self.G*(1.0-n)*self.alpha0(t,schedule)
        #print(Bhat)
        # Light forcing equations
        LightAmp = (self.A_1/2.0)*Bhat*(1.0 - pow(R,4.0))*np.cos(Psi + self.beta_L1) + (self.A_2/2.0)*Bhat*R*(1.0 - pow(R,8.0))*np.cos(2.0*Psi + self.beta_L2) # L_R
        LightPhase = self.sigma*Bhat - (self.A_1/2.0)*Bhat*(pow(R,3.0) + 1.0/R)*np.sin(Psi + self.beta_L1) - (self.A_2/2.0)*Bhat*(1.0 + pow(R,8.0))*np.sin(2.0*Psi + self.beta_L2) # L_psi
    
        # Melatonin interaction with pacemaker
        Mhat = self.M_max/(1 + np.exp((self.H_sat - H2)/self.sigma_M))
        #print(Mhat)
        # Melatonin forcing equations 
        MelAmp = (self.B_1/2)*Mhat*(1.0 - pow(R,4.0))*np.cos(Psi + self.theta_M1) + (self.B_2/2.0)*Mhat*R*(1.0 - pow(R,8.0))*np.cos(2.0*Psi + self.theta_M2) # M_R
        MelPhase = self.epsilon*Mhat - (self.B_1/2.0)*Mhat*(pow(R,3.0)+1.0/R)*np.sin(Psi + self.theta_M1) - (self.B_2/2.0)*Mhat*(1.0 + pow(R,8.0))*np.sin(2.0*Psi + self.theta_M2) # M_psi
    
        #print(MelAmp)
        #print(MelPhase)
    
        # Switch pineal on and off in the presence of light 
        tmp = 1 - self.m*Bhat
        S = np.piecewise(tmp, [tmp >= 0, tmp < 0 and H1 < 0.001], [1, 0])
    
    
        dydt=np.zeros(6)
    
        # ODE System  
        dydt[0] = -(self.D + self.gamma)*R + (self.K/2)*np.cos(self.beta)*R*(1 - pow(R,4.0)) + LightAmp + MelAmp # dR/dt
        dydt[1] = self.omega_0 + (self.K/2)*np.sin(self.beta)*(1 + pow(R,4.0)) + LightPhase + MelPhase # dpsi/dt 
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





#--------- Run the Hannay 2019 model to find initial conditions ---------------

model_IC = HannayModel() # defining model as a new object built with the HannayBreslowModel class 
model_IC.integrateModel(24*50,schedule=1) # use the integrateModel method with the object model
IC = model_IC.results[-1,:] # get initial conditions from entrained model



#--------- Run the Hannay 2019 model ---------------

model_Hannay = HannayModel()
model_Hannay.integrateModel(24*2,tstart=0.0,initial=IC,schedule=1) 






#--------- Run the Hannay-Breslow 2024 model to find initial conditions ---------------

model_IC = HannayBreslowModel() # defining model as a new object built with the HannayBreslowModel class 
model_IC.integrateModel(24*50,schedule=1) # use the integrateModel method with the object model
IC = model_IC.results[-1,:] # get initial conditions from entrained model



#--------- Run the Hannay-Breslow 2024 model without exogenous melatonin ---------------

model_HannayBreslow = HannayBreslowModel()
model_HannayBreslow.integrateModel(24*2,tstart=0.0,initial=IC, melatonin_timing=None, melatonin_dosage=None,schedule=1) 




#--------- Find DLMO and CBTmin for Hannay 2019 -----------

# By Hannay model definition 
# Updated (May 2024) to perform a linear interpolation 
psi_mod2pi_Hannay = np.mod(model_Hannay.results[0:240,1],2*np.pi)

DLMO_index = min(range(len(psi_mod2pi_Hannay)), key=lambda i: abs(psi_mod2pi_Hannay[i]-1.30899))
CBTmin_index = min(range(len(psi_mod2pi_Hannay)), key=lambda i: abs(psi_mod2pi_Hannay[i]-3.14159))

DLMO_psi_Hannay_1 = model_Hannay.ts[DLMO_index] # closest to psi = 1.30899
psi_value_1 = psi_mod2pi_Hannay[DLMO_index]
if psi_value_1 < 1.30899:
    psi_value_2 = psi_mod2pi_Hannay[DLMO_index + 1]
    DLMO_psi_Hannay_2 = model_Hannay.ts[DLMO_index + 1]
    DLMO_psi_Hannay = np.interp(1.30899, [psi_value_1, psi_value_2], [DLMO_psi_Hannay_1, DLMO_psi_Hannay_2])   
elif psi_value_1 > 1.30899:
    psi_value_2 = psi_mod2pi_Hannay[DLMO_index - 1]
    DLMO_psi_Hannay_2 = model_Hannay.ts[DLMO_index - 1]
    DLMO_psi_Hannay = np.interp(1.30899, [psi_value_2, psi_value_1], [DLMO_psi_Hannay_2, DLMO_psi_Hannay_1])
 

CBTmin_psi_Hannay_1 = model_Hannay.ts[CBTmin_index] # closest to psi = 3.14159
psi_value_1 = psi_mod2pi_Hannay[CBTmin_index]
if psi_value_1 < 3.14159:
    psi_value_2 = psi_mod2pi_Hannay[CBTmin_index + 1]
    CBTmin_psi_Hannay_2 = model_Hannay.ts[CBTmin_index + 1]
    CBTmin_Hannay = np.interp(3.14159, [psi_value_1, psi_value_2], [CBTmin_psi_Hannay_1, CBTmin_psi_Hannay_2])   
elif psi_value_1 > 3.14159:
    psi_value_2 = psi_mod2pi_Hannay[CBTmin_index - 1]
    CBTmin_psi_Hannay_2 = model_Hannay.ts[CBTmin_index - 1]
    CBTmin_Hannay = np.interp(3.14159, [psi_value_2, psi_value_1], [CBTmin_psi_Hannay_2, CBTmin_psi_Hannay_1])
 



#--------- Find DLMO and CBTmin for Hannay-Breslow 2024 -----------

# By Hannay model definition 
psi_mod2pi_HannayBreslow = np.mod(model_HannayBreslow.results[0:240,1],2*np.pi)

DLMO_index = min(range(len(psi_mod2pi_HannayBreslow)), key=lambda i: abs(psi_mod2pi_HannayBreslow[i]-1.30899))
CBTmin_index = min(range(len(psi_mod2pi_HannayBreslow)), key=lambda i: abs(psi_mod2pi_HannayBreslow[i]-3.14159))

DLMO_psi_HannayBreslow_1 = model_HannayBreslow.ts[DLMO_index] # closest to psi = 1.30899
psi_value_1 = psi_mod2pi_HannayBreslow[DLMO_index]
if psi_value_1 < 1.30899:
    psi_value_2 = psi_mod2pi_HannayBreslow[DLMO_index + 1]
    DLMO_psi_HannayBreslow_2 = model_HannayBreslow.ts[DLMO_index + 1]
    DLMO_psi_HannayBreslow = np.interp(1.30899, [psi_value_1, psi_value_2], [DLMO_psi_HannayBreslow_1, DLMO_psi_HannayBreslow_2])   
elif psi_value_1 > 1.30899:
    psi_value_2 = psi_mod2pi_HannayBreslow[DLMO_index - 1]
    DLMO_psi_HannayBreslow_2 = model_HannayBreslow.ts[DLMO_index - 1]
    DLMO_psi_HannayBreslow = np.interp(1.30899, [psi_value_2, psi_value_1], [DLMO_psi_HannayBreslow_2, DLMO_psi_HannayBreslow_1])
 

CBTmin_psi_HannayBreslow_1 = model_HannayBreslow.ts[CBTmin_index] # closest to psi = 3.14159
psi_value_1 = psi_mod2pi_HannayBreslow[CBTmin_index]
if psi_value_1 < 3.14159:
    psi_value_2 = psi_mod2pi_HannayBreslow[CBTmin_index + 1]
    CBTmin_psi_HannayBreslow_2 = model_HannayBreslow.ts[CBTmin_index + 1]
    CBTmin_HannayBreslow = np.interp(3.14159, [psi_value_1, psi_value_2], [CBTmin_psi_HannayBreslow_1, CBTmin_psi_HannayBreslow_2])   
elif psi_value_1 > 3.14159:
    psi_value_2 = psi_mod2pi_HannayBreslow[CBTmin_index - 1]
    CBTmin_psi_HannayBreslow_2 = model_HannayBreslow.ts[CBTmin_index - 1]
    CBTmin_HannayBreslow = np.interp(3.14159, [psi_value_2, psi_value_1], [CBTmin_psi_HannayBreslow_2, CBTmin_psi_HannayBreslow_1])
 




## DLMO 
# By threshold definition (10 pg/mL in plasma)
# Updated (May 2024) to perform a linear interpolation 
DLMO_threshold = 10

plasma_mel_concentrations = model_HannayBreslow.results[60:240,4]/4.3 # converting output to pg/mL
times = model_HannayBreslow.ts[60:240] # defining times from first 24hrs 
plasma_mel, = np.where(plasma_mel_concentrations<=DLMO_threshold) # finding all the indices where concentration is below 10pg/mL
DLMO_H2_HannayBreslow_below10 = times[plasma_mel[-1]] # finding the time corresponding to the last index below threshold, DLMO
DLMO_H2_HannayBreslow_above10 = DLMO_H2_HannayBreslow_below10+0.1

DLMO_H2_HannayBreslow = np.interp(DLMO_threshold, [plasma_mel_concentrations[plasma_mel[-1]],plasma_mel_concentrations[plasma_mel[-1]+1]], [DLMO_H2_HannayBreslow_below10, DLMO_H2_HannayBreslow_above10])


## DLMOff
# By threshold definition (10 pg/mL in plasma)
# Updated (May 2024) to perform a linear interpolation 
plasma_mel_concentrations = model_HannayBreslow.results[240:400,4]/4.3 # converting output to pg/mL
times = model_HannayBreslow.ts[240:400] # defining times from second 24hrs 
plasma_mel, = np.where(plasma_mel_concentrations>=DLMO_threshold) # finding all the indices where concentration is above 10pg/mL
DLMOff_H2_HannayBreslow_above10 = times[plasma_mel[-1]] # finding the time corresponding to the last index above threshold, DLMO
DLMOff_H2_HannayBreslow_below10 = DLMOff_H2_HannayBreslow_above10+0.1

DLMOff_HannayBreslow = np.interp(DLMO_threshold, [plasma_mel_concentrations[plasma_mel[-1]],plasma_mel_concentrations[plasma_mel[-1]+1]], [DLMOff_H2_HannayBreslow_above10, DLMOff_H2_HannayBreslow_below10])
DLMOff_mod = np.mod(DLMOff_HannayBreslow,24)



#--------- Plot Model Output -------------------
plt.rcParams.update({'font.size': 15})


# Plotting R
plt.plot(model_HannayBreslow.ts,model_HannayBreslow.results[:,0],lw=5,color='forestgreen')
plt.plot(model_Hannay.ts,model_Hannay.results[:,0],lw=3,color='black',linestyle='dashed')
plt.xlabel("Time (hours)")
plt.ylabel("$R$, Collective Amplitude")
plt.legend(['Light + Melatonin','Light (Hannay 2019)'],loc='upper center', bbox_to_anchor=(0.5, 1.3))
#plt.title("Time Trace of R, Collective Amplitude")
plt.show()

'''
# Plotting psi
plt.plot(model_HannayBreslow.ts,model_HannayBreslow.results[:,1],lw=3,color='mediumturquoise')
plt.plot(model_Hannay.ts,model_Hannay.results[:,1],lw=2,linestyle='dashed')
plt.xlabel("Clock Time (hours)")
plt.ylabel("Psi, Mean Phase (radians)")
plt.legend(['Hannay-Breslow 2024','Hannay 2019'])
plt.title("Time Trace of Psi, Mean Phase")
plt.show()
'''

# Plotting psi mod 2pi
plt.plot(model_HannayBreslow.ts[0:141],np.mod(model_HannayBreslow.results[0:141,1],2*np.pi),color='mediumturquoise',lw=5) #'.',markersize=10
plt.plot(model_Hannay.ts[0:143],np.mod(model_Hannay.results[0:143,1],2*np.pi),color='black',lw=3,linestyle='dashed') # '.',markersize=3,
plt.plot(model_HannayBreslow.ts[142:381],np.mod(model_HannayBreslow.results[142:381,1],2*np.pi),color='mediumturquoise',lw=5) #'.',markersize=10
plt.plot(model_HannayBreslow.ts[382:479],np.mod(model_HannayBreslow.results[382:479,1],2*np.pi),color='mediumturquoise',lw=5) #'.',markersize=10
plt.plot(model_Hannay.ts[144:382],np.mod(model_Hannay.results[144:382,1],2*np.pi),color='black',lw=3,linestyle='dashed') # '.',markersize=3,
plt.plot(model_Hannay.ts[383:479],np.mod(model_Hannay.results[383:479,1],2*np.pi),color='black',lw=3,linestyle='dashed') # '.',markersize=3,
#plt.axvline(DLMO_psi_HannayBreslow,color='grey')
#plt.axvline(DLMO_psi_Hannay,color='black',linestyle='dashed')
plt.axhline(5*np.pi/12, color='black',lw=2)
#plt.axhline(5*np.pi/12)
#plt.axhline(np.pi)
#plt.axvline(DLMO)
#plt.axvline(CBTmin)
#plt.axvline(x=20.6) # Checking pineal on 
#plt.axvline(x=5.7) # Checking pineal off
#plt.axvline(21)
plt.xlabel("Time (hours)")
plt.ylabel("$\Psi$, Mean Phase (radians)")
plt.legend(['Light + Melatonin','Light (Hannay 2019)'],loc='upper center', bbox_to_anchor=(0.5, 1.3))
#plt.title("Time Trace of Psi, Mean Phase")
plt.show()

# Plotting n
plt.plot(model_HannayBreslow.ts,model_HannayBreslow.results[:,2],lw=5,color='goldenrod')
plt.plot(model_Hannay.ts,model_Hannay.results[:,2],lw=3,linestyle='dashed',color='black')
#plt.axvline(x=7)
#plt.axvline(x=23)
plt.xlabel("Time (hours)")
plt.ylabel("$n$, Activated Photoreceptors")
plt.legend(['Light + Melatonin','Light (Hannay 2019)'],loc='upper center', bbox_to_anchor=(0.5, 1.3))
#plt.title("Time Trace of Photoreceptor Activation")
plt.show()


# Plotting H1, H2, and H3 (melatonin concentrations, pg/mL)
plt.plot(model_HannayBreslow.ts,model_HannayBreslow.results[:,3]/4.3,lw=3,color='mediumblue')
plt.plot(model_HannayBreslow.ts,model_HannayBreslow.results[:,4]/4.3,lw=3,color='darkorchid')
plt.plot(model_HannayBreslow.ts,model_HannayBreslow.results[:,5]/4.3,lw=3,color='hotpink')
plt.axvline(x=DLMOff_HannayBreslow,color='grey',linestyle='dashed') # Checking DLMOff
plt.axvline(x=DLMO_H2_HannayBreslow,color='grey') # Checking DLMO
#plt.axvline(x=20) # Checking pineal on 
#plt.axvline(x=5.7+24) # Checking pineal off
plt.axhline(10, color='black',lw=1)
plt.xlabel("Time (hours)")
plt.ylabel("Melatonin Concentration (pg/mL)")
#plt.title("Melatonin Concentrations (pg/mL)")
plt.legend(["Pineal","Plasma","Exogenous","DLMOff","DLMO"])
plt.show()