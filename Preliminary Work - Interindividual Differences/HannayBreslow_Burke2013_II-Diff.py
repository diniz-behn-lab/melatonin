# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 01:33:49 2024

@author: sstowe

April 2025

Simulating Burke 2013

Do differences in light history change the predicted phase shifts? 

Do differences in endogenous melatonin change the predicted phase shifts?

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

        self.a = (0.101)*60*60 # pmol/L/sec converted to hours, I determined
        self.delta_M = 600 # sec, Breslow 2013
        self.r = 15.36 # sec, Breslow 2013
        
        self.psi_on = 1.6406095 #1.2566 # radians, I determined 
        self.psi_off = 3.99680399 #3.6128 # radians, I determined
        
        self.M_max = 0.019513 # Breslow 2013
        self.H_sat = 301 # Scaled for our peak concentration #861 # Breslow 2013
        self.sigma_M = 17.5 # Scaled for our peak concentration #50 # Breslow 2013
        self.m = 4.7887 # I determined by fitting to Zeitzer using differential evolution
        
        
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
        # Updated for linear interpolation (May 2024)
        x = [-1.00356684, -0.01949989, -0.56044428,  0.60146017,  0.07781353] # Error = 3.7152403300234877 # Optimization terminated successfully!!
        self.B_1 = x[0] 
        self.theta_M1 = x[1]
        self.B_2 = x[2]
        self.theta_M2 = x[3]
        self.epsilon = x[4]
        
        # Light schedule times & intensities
        self.wake_time = 8
        self.sleep_time = 24
        self.wake = 1000 
        self.dim = 2
        self.dark = 0
        self.bright = 3000
        
        
# Set the light schedule (timings and intensities)
    def light(self,t,schedule):
        
        if schedule == 1: # standard 16:8 schedule 
            low_light = 300
            sun_up =  9 #8
            sun_down = 20 #19
            
            is_awake = np.mod(t - self.wake_time,24) <= np.mod(self.sleep_time - self.wake_time,24)
            sun_is_up = np.mod(t - sun_up,24) <= np.mod(sun_down - sun_up,24)

            return is_awake*(self.wake*sun_is_up + low_light*(1 - sun_is_up))
        
        elif schedule == 2: # Constant darkness 
            
            return 0
        
        elif schedule == 3: # Burke 2013 with light intervention 
            
            if 0 <= t < 24*1: # Enter the lab 4hrs before habitual sleep
                if t < self.wake_time:
                    lux = self.dark
                elif self.wake_time <= t < self.sleep_time-4: 
                    lux = self.wake
                elif self.sleep_time-4 <= t < self.sleep_time:
                    lux = self.dim
                else:
                    lux = self.dark

            elif 24*1 <= t < 24*2: # Start of CR1 (28 h)
                t = np.mod(t,24)
                if t < self.wake_time:
                    lux = self.dark
                else:
                    lux = self.dim
                
            elif 24*2 <= t <  24*3: 
                t = np.mod(t,24)
                if t < self.wake_time+4: # Need 4hrs more of dim to complete the 28h CR
                    lux = self.dim
                elif self.wake_time+4 <= t < self.wake_time+9:
                    lux = self.dark
                elif self.wake_time+9 <= t < self.sleep_time:
                    lux = self.dim
                else:
                    lux = self.dark
                    
            elif 24*3 <= t <  24*4: # Light intervention and start of CR2 (19h)
                t = np.mod(t,24)
                if t < self.wake_time-1:
                    lux = self.dark
                elif self.wake_time-1 <= t < self.wake_time+2:
                    lux = self.bright 
                else: 
                    lux = self.dim
            
            elif 24*4 <= t <= 24*5: # Released from lab
                t = np.mod(t,24)
                if t < self.wake_time-3:
                    lux = self.dim
                elif self.wake_time-3 <= t < self.wake_time+3:
                    lux = self.dark 
                else: 
                    lux = self.wake
            
            return lux
        
        elif schedule == 4: # Burke 2013 without light intervention 
            
            if 0 <= t < 24*1: # Enter the lab 4hrs before habitual sleep
                if t < self.wake_time:
                    lux = self.dark
                elif self.wake_time <= t < self.sleep_time-4: 
                    lux = self.wake
                elif self.sleep_time-4 <= t < self.sleep_time:
                    lux = self.dim
                else:
                    lux = self.dark

            elif 24*1 <= t < 24*2: # Start of CR1 (28 h)
                t = np.mod(t,24)
                if t < self.wake_time:
                    lux = self.dark
                else:
                    lux = self.dim
                
            elif 24*2 <= t <  24*3: 
                t = np.mod(t,24)
                if t < self.wake_time+4: # Need 4hrs more of dim to complete the 28h CR
                    lux = self.dim
                elif self.wake_time+4 <= t < self.wake_time+9:
                    lux = self.dark
                elif self.wake_time+9 <= t < self.sleep_time:
                    lux = self.dim
                else:
                    lux = self.dark
                    
            elif 24*3 <= t <  24*4: # Light intervention and start of CR2 (19h)
                t = np.mod(t,24)
                if t < self.wake_time-1:
                    lux = self.dark
                elif self.wake_time-1 <= t < self.wake_time+2:
                    lux = self.dim 
                else: 
                    lux = self.dim
            
            elif 24*4 <= t <= 24*5: # Released from lab
                t = np.mod(t,24)
                if t < self.wake_time-3:
                    lux = self.dim
                elif self.wake_time-3 <= t < self.wake_time+3:
                    lux = self.dark 
                else: 
                    lux = self.wake
            
            return lux
        
        elif schedule == 5: # Burke 2013 without light intervention extended to 7 days
            wake = 1000
            dim = 2
            dark = 0
            
            if 0 <= t < 24*1: # Enter the lab
                if t < 7:
                    lux = dark
                elif 7 <= t < 19: 
                    lux = wake
                elif 19 <= t < 23:
                    lux = dim
                else:
                    lux = dark

            elif 24*1 <= t < 24*2: # Start of CR1 (28 h)
                t = np.mod(t,24)
                if t < 7:
                    lux = dark
                else:
                    lux = dim
                
            elif 24*2 <= t <  24*3:# ex mel
                t = np.mod(t,24)
                if t < 11: 
                    lux = dim
                elif 11 <= t < 16:
                    lux = dark
                elif 16 <= t < 23:
                    lux = dim
                else:
                    lux = dark
                    
            elif 24*3 <= t <  24*4: # ex mel
                t = np.mod(t,24)
                if t < 11: 
                    lux = dim
                elif 11 <= t < 16:
                    lux = dark
                elif 16 <= t < 23:
                    lux = dim
                else:
                    lux = dark
                    
            elif 24*4 <= t <  24*5: # ex mel
                t = np.mod(t,24)
                if t < 11: 
                    lux = dim
                elif 11 <= t < 16:
                    lux = dark
                elif 16 <= t < 23:
                    lux = dim
                else:
                    lux = dark
                    
            elif 24*5 <= t <  24*6: # Start of CR2 (19h)
                t = np.mod(t,24)
                if t < 6:
                    lux = dark
                elif 6 <= t < 9:
                    lux = dim 
                else: 
                    lux = dim
            
            elif 24*6 <= t <= 24*7: # Released from lab
                t = np.mod(t,24)
                if t < 4:
                    lux = dim
                elif 4 <= t < 10:
                    lux = dark 
                else: 
                    lux = wake
            
            return lux
        

# Define the alpha(L) function 
    def alpha0(self,t,schedule):
        """A helper function for modeling the light input processing"""
        
        return(self.alpha_0*pow(self.light(t,schedule), self.p)/(pow(self.light(t,schedule), self.p)+self.I_0));
    

# Set the exogenous melatonin administration schedule VERSION 5
    def ex_melatonin(self,t,melatonin_timing,melatonin_dosage):
        
        if melatonin_timing == None:
            return 0 # set exogenous melatonin to zero
        else: 
            if 60 <= t <= 66:#64 <= t <= 66:
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
        #y_line = 832.37*pow(x_line,3) - 4840.4*pow(x_line,2) + 64949*x_line + 2189.4 # Cubic fit to 5 points
        y_line = 962.48*pow(x_line,3) - 6316.5*pow(x_line,2) + 66719*x_line + 1975.9 # Cubic fit to 5 points (May 2024)
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
        psi = Psi 
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
    
        #print(LightAmp)
        #print(LightPhase)
    
        # Melatonin interaction with pacemaker
        Mhat = self.M_max/(1 + np.exp((self.H_sat - H2)/(self.sigma_M)))
        
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



 



#--------- Run the model to find initial conditions ---------------

model_IC = HannayBreslowModel() # defining model as a new object built with the HannayBreslowModel class 
model_IC.integrateModel(24*50,schedule=1) # use the integrateModel method with the object model
IC = model_IC.results[-1,:] # get initial conditions from entrained model



#--------- Run the model without exogenous melatonin ---------------

#model = HannayBreslowModel()
#model.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=None, melatonin_dosage=None,schedule=3) 

model_DLP = HannayBreslowModel()
model_DLP.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=None, melatonin_dosage=None,schedule=4) 

model_BLP = HannayBreslowModel()
model_BLP.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=None, melatonin_dosage=None,schedule=3) 


#--------- Run the model with exogenous melatonin ---------------
# Set melatonin_timing to a clock hour 
# Set melatonin dosage to a mg amount

model_BLM = HannayBreslowModel()
model_BLM.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=24-5.75, melatonin_dosage=5.0,schedule=3) 

model_DLM = HannayBreslowModel()
model_DLM.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=24-5.75, melatonin_dosage=5.0,schedule=4) 

#------------------------------------------------------------------------------
#--------- Find DLMO for DLP -----------

## DLMO 
# By threshold definition (10 pg/mL in plasma)
# Updated (May 2024) to perform a linear interpolation 
DLMO_threshold = 10

# Baseline DLMO
plasma_mel_concentrations = model_DLP.results[330:500,4]/4.3 # converting output to pg/mL
times = model_DLP.ts[330:500] 
plasma_mel, = np.where(plasma_mel_concentrations<=DLMO_threshold) # finding all the indices where concentration is below 10pg/mL
DLMO_H2_Baseline_below10 = times[plasma_mel[-1]] # finding the time corresponding to the last index below threshold, DLMO
DLMO_H2_Baseline_above10 = DLMO_H2_Baseline_below10+0.1

DLMO_H2_Baseline = np.interp(DLMO_threshold, [plasma_mel_concentrations[plasma_mel[-1]],plasma_mel_concentrations[plasma_mel[-1]+1]], [DLMO_H2_Baseline_below10, DLMO_H2_Baseline_above10])

# Final DLMO
plasma_mel_concentrations = model_DLP.results[810:980,4]/4.3 # converting output to pg/mL
times = model_DLP.ts[810:980] 
plasma_mel, = np.where(plasma_mel_concentrations<=DLMO_threshold) # finding all the indices where concentration is below 10pg/mL
DLMO_H2_Final_below10 = times[plasma_mel[-1]] # finding the time corresponding to the last index below threshold, DLMO
DLMO_H2_Final_above10 = DLMO_H2_Final_below10+0.1

DLMO_H2_Final = np.interp(DLMO_threshold, [plasma_mel_concentrations[plasma_mel[-1]],plasma_mel_concentrations[plasma_mel[-1]+1]], [DLMO_H2_Final_below10, DLMO_H2_Final_above10])


# Calculate Phase Shift 
phase_shift_DLP = np.mod(DLMO_H2_Baseline,24) - np.mod(DLMO_H2_Final,24)


#--------- Find DLMO for BLP -----------

## DLMO 
# By threshold definition (10 pg/mL in plasma)
# Updated (May 2024) to perform a linear interpolation 
DLMO_threshold = 10

# Baseline DLMO
plasma_mel_concentrations = model_BLP.results[330:500,4]/4.3 # converting output to pg/mL
times = model_BLP.ts[330:500] 
plasma_mel, = np.where(plasma_mel_concentrations<=DLMO_threshold) # finding all the indices where concentration is below 10pg/mL
DLMO_H2_Baseline_below10 = times[plasma_mel[-1]] # finding the time corresponding to the last index below threshold, DLMO
DLMO_H2_Baseline_above10 = DLMO_H2_Baseline_below10+0.1

DLMO_H2_Baseline = np.interp(DLMO_threshold, [plasma_mel_concentrations[plasma_mel[-1]],plasma_mel_concentrations[plasma_mel[-1]+1]], [DLMO_H2_Baseline_below10, DLMO_H2_Baseline_above10])

# Final DLMO
plasma_mel_concentrations = model_BLP.results[810:980,4]/4.3 # converting output to pg/mL
times = model_BLP.ts[810:980] 
plasma_mel, = np.where(plasma_mel_concentrations<=DLMO_threshold) # finding all the indices where concentration is below 10pg/mL
DLMO_H2_Final_below10 = times[plasma_mel[-1]] # finding the time corresponding to the last index below threshold, DLMO
DLMO_H2_Final_above10 = DLMO_H2_Final_below10+0.1

DLMO_H2_Final = np.interp(DLMO_threshold, [plasma_mel_concentrations[plasma_mel[-1]],plasma_mel_concentrations[plasma_mel[-1]+1]], [DLMO_H2_Final_below10, DLMO_H2_Final_above10])


# Calculate Phase Shift 
phase_shift_BLP = np.mod(DLMO_H2_Baseline,24) - np.mod(DLMO_H2_Final,24)


#--------- Find DLMO for BLM -----------

## DLMO 
# By threshold definition (10 pg/mL in plasma)
# Updated (May 2024) to perform a linear interpolation 
DLMO_threshold = 10

# Baseline DLMO
plasma_mel_concentrations = model_BLM.results[330:500,4]/4.3 # converting output to pg/mL
times = model_BLM.ts[330:500] 
plasma_mel, = np.where(plasma_mel_concentrations<=DLMO_threshold) # finding all the indices where concentration is below 10pg/mL
DLMO_H2_Baseline_below10 = times[plasma_mel[-1]] # finding the time corresponding to the last index below threshold, DLMO
DLMO_H2_Baseline_above10 = DLMO_H2_Baseline_below10+0.1

DLMO_H2_Baseline = np.interp(DLMO_threshold, [plasma_mel_concentrations[plasma_mel[-1]],plasma_mel_concentrations[plasma_mel[-1]+1]], [DLMO_H2_Baseline_below10, DLMO_H2_Baseline_above10])

# Final DLMO
plasma_mel_concentrations = model_BLM.results[810:980,4]/4.3 # converting output to pg/mL
times = model_BLM.ts[810:980] 
plasma_mel, = np.where(plasma_mel_concentrations<=DLMO_threshold) # finding all the indices where concentration is below 10pg/mL
DLMO_H2_Final_below10 = times[plasma_mel[-1]] # finding the time corresponding to the last index below threshold, DLMO
DLMO_H2_Final_above10 = DLMO_H2_Final_below10+0.1

DLMO_H2_Final = np.interp(DLMO_threshold, [plasma_mel_concentrations[plasma_mel[-1]],plasma_mel_concentrations[plasma_mel[-1]+1]], [DLMO_H2_Final_below10, DLMO_H2_Final_above10])


# Calculate Phase Shift 
phase_shift_BLM = np.mod(DLMO_H2_Baseline,24) - np.mod(DLMO_H2_Final,24)


#--------- Find DLMO for DLM -----------

## DLMO 
# By threshold definition (10 pg/mL in plasma)
# Updated (May 2024) to perform a linear interpolation 
DLMO_threshold = 10

# Baseline DLMO
plasma_mel_concentrations = model_DLM.results[330:500,4]/4.3 # converting output to pg/mL
times = model_DLM.ts[330:500] 
plasma_mel, = np.where(plasma_mel_concentrations<=DLMO_threshold) # finding all the indices where concentration is below 10pg/mL
DLMO_H2_Baseline_below10 = times[plasma_mel[-1]] # finding the time corresponding to the last index below threshold, DLMO
DLMO_H2_Baseline_above10 = DLMO_H2_Baseline_below10+0.1

DLMO_H2_Baseline = np.interp(DLMO_threshold, [plasma_mel_concentrations[plasma_mel[-1]],plasma_mel_concentrations[plasma_mel[-1]+1]], [DLMO_H2_Baseline_below10, DLMO_H2_Baseline_above10])

# Final DLMO
plasma_mel_concentrations = model_DLM.results[810:980,4]/4.3 # converting output to pg/mL
times = model_DLM.ts[810:980] 
plasma_mel, = np.where(plasma_mel_concentrations<=DLMO_threshold) # finding all the indices where concentration is below 10pg/mL
DLMO_H2_Final_below10 = times[plasma_mel[-1]] # finding the time corresponding to the last index below threshold, DLMO
DLMO_H2_Final_above10 = DLMO_H2_Final_below10+0.1

DLMO_H2_Final = np.interp(DLMO_threshold, [plasma_mel_concentrations[plasma_mel[-1]],plasma_mel_concentrations[plasma_mel[-1]+1]], [DLMO_H2_Final_below10, DLMO_H2_Final_above10])


# Calculate Phase Shift 
phase_shift_DLM = np.mod(DLMO_H2_Baseline,24) - np.mod(DLMO_H2_Final,24)


# --------------------------------------------------------------------------
# ------ Print all four phase shift predictions ------

print([phase_shift_DLP, phase_shift_DLM,phase_shift_BLP,phase_shift_BLM])

#--------- Plot Model Output -------------------

plt.rcParams.update({'font.size': 15})

model = model_DLP

'''
# Plotting H1, H2, and H3 (melatonin concentrations, pmol/L)
plt.plot(model.ts,model.results[:,3],lw=2)
plt.plot(model.ts,model.results[:,4],lw=2)
plt.plot(model.ts,model.results[:,5],lw=2)
#plt.axvline(x=6)
#plt.axvline(x=20.3)
#plt.axvline(x=12)
plt.axhline(200)
plt.xlabel("Time (hours)")
plt.ylabel("Melatonin Concentration (pmol/L)")
plt.title("Melatonin Concentrations (pmol/L)")
plt.legend(["Pineal","Plasma", "Exogenous"])
plt.show()
'''


# Plotting H1, H2, and H3 (melatonin concentrations, pg/mL)
plt.plot(model.ts,model.results[:,3]/4.3,lw=3,color='mediumblue')
plt.plot(model.ts,model.results[:,4]/4.3,lw=3,color='darkorchid')
#plt.plot(model.ts,model.results[:,5]/4.3,lw=3,color='hotpink')
plt.axvline(x=DLMO_H2_Baseline,color='black',linestyle='dashed') 
plt.axvline(x=DLMO_H2_Final,color='black',linestyle='dashdot') 
#plt.axvline(x=20.6) # Checking pineal on 
#plt.axvline(x=5.7) # Checking pineal off
#plt.axvline(5.7+24)
#plt.axvline(CBTmin,color='grey')
#plt.axvline(63,color='hotpink') # 15 h dose
#plt.axvline(65.25,color='hotpink') # 17.25 h dose
#plt.axvline(65.25+24,color='hotpink') # 17.25 h dose
#plt.axvline(65.25+24+24,color='hotpink') # 17.25 h dose
#plt.axvspan(78, 81, facecolor='gold', alpha=0.4)
plt.axhline(10, color='black',lw=1)
plt.ylim([-3,73])
plt.xlabel("Clock Time (hours)")
plt.ylabel("Melatonin Concentration (pg/mL)")
#plt.title("Melatonin Concentrations (pg/mL)")
plt.legend(["Pineal","Plasma","Baseline DLMO","Final DLMO"],loc='upper left')
plt.show()

'''
# Plotting R - DLP
plt.plot(model_DLP.ts,model_DLP.results[:,0],lw=3,color='forestgreen',linestyle='dotted') #black
plt.ylim([0.675,0.899])
plt.xticks(np.arange(0, 122, 24))
plt.xlabel("Time (hours)")
plt.ylabel("R, Collective Amplitude")
plt.title("Dim Light-Placebo (DLP)")
plt.legend(["DLP"],loc='lower left')
plt.show()

# Plotting R - DLM
plt.plot(model_DLM.ts,model_DLM.results[:,0],lw=3,color='forestgreen',linestyle='dashed') #dimgrey
plt.plot(model_DLP.ts,model_DLP.results[:,0],lw=3,color='black',linestyle='dotted') #black
plt.axvline(65.25,color='mediumvioletred',lw=3) # 17.25 h dose
plt.ylim([0.675,0.899])
plt.xticks(np.arange(0, 122, 24))
plt.xlabel("Time (hours)")
plt.ylabel("R, Collective Amplitude")
plt.title("Dim Light-Melatonin (DLM)")
plt.legend(["DLM", "DLP"],loc='lower left')
plt.show()

# Plotting R - BLP
plt.plot(model_BLP.ts,model_BLP.results[:,0],lw=3,color='forestgreen',linestyle='dashdot') #darkgray
plt.plot(model_DLM.ts,model_DLM.results[:,0],lw=3,color='dimgrey',linestyle='dashed') #dimgrey
plt.plot(model_DLP.ts,model_DLP.results[:,0],lw=3,color='black',linestyle='dotted') #black
plt.axvspan(78, 81, facecolor='gold', alpha=0.4)
plt.ylim([0.675,0.899])
plt.xticks(np.arange(0, 122, 24))
plt.xlabel("Time (hours)")
plt.ylabel("R, Collective Amplitude")
plt.title("Bright Light-Placebo (BLP)")
plt.legend(["BLP", "DLM", "DLP"],loc='lower left')
plt.show()

# Plotting R - BLM
plt.plot(model.ts,model.results[:,0],lw=6,color='forestgreen')
plt.plot(model_BLP.ts,model_BLP.results[:,0],lw=3,color='darkgray',linestyle='dashdot') #darkgray
plt.plot(model_DLM.ts,model_DLM.results[:,0],lw=3,color='dimgrey',linestyle='dashed') #dimgrey
plt.plot(model_DLP.ts,model_DLP.results[:,0],lw=3,color='black',linestyle='dotted') #black
#plt.axvline(x=20.6)
#plt.axvline(x=5.7)
#plt.axvline(x=8.1)
#plt.axvline(15)
#plt.axvline(5.7+24)
#plt.axvline(63,color='hotpink') # 15 h dose
plt.axvline(65.25,color='mediumvioletred',lw=3) # 17.25 h dose
plt.axvspan(78, 81, facecolor='gold', alpha=0.4)
plt.ylim([0.675,0.899])
plt.xticks(np.arange(0, 122, 24))
plt.xlabel("Time (hours)")
plt.ylabel("R, Collective Amplitude")
plt.title("Bright Light-Melatonin (BLM)")
plt.legend(["BLM", "BLP", "DLM", "DLP"],loc='lower left')
plt.show()



# Plotting psi
plt.plot(model.ts,model.results[:,1],lw=2)
plt.xlabel("Time (hours)")
plt.ylabel("Psi, Mean Phase (radians)")
plt.title("Time Trace of Psi, Mean Phase")
plt.show()

# Plotting psi mod 2pi
plt.axvline(x=DLMO_psi_Baseline,color='black',linestyle='dashed')
plt.axvline(x=DLMO_psi_Final,color='black',linestyle='dashdot') # Checking DLMO
plt.plot(model.ts,np.mod(model.results[:,1],2*np.pi),'.',markersize=10,color='mediumturquoise')
plt.axhline(5*np.pi/12, color='black',lw=1)
#plt.axhline(np.pi)
#plt.axvline(CBTmin)
#plt.axvline(x=20.6) # Checking pineal on 
#plt.axvline(x=5.7) # Checking pineal off
#plt.axvline(x=8.1)
#plt.axvline(15)
#plt.axvline(x=7)
#plt.axvline(x=23)
#plt.axvline(63,color='hotpink') # 15 h dose
plt.axvline(65.25,color='hotpink') # 17.25 h dose
plt.axvline(65.25+24,color='hotpink') # 17.25 h dose
plt.axvline(65.25+24+24,color='hotpink') # 17.25 h dose
#plt.axvspan(78, 81, facecolor='gold', alpha=0.4)
plt.xlabel("Clock Time (hours)")
plt.ylabel("Psi, Mean Phase (radians)")
plt.legend(["Baseline DLMO","Final DLMO"],loc='upper left')
#plt.title("Time Trace of Psi, Mean Phase")
plt.show()



# Plotting n
plt.plot(model.ts,model.results[:,2],lw=3,color='goldenrod')
#plt.axvline(x=7)
#plt.axvline(x=23)
plt.axvline(63,color='hotpink') # 15 h dose
#plt.axvline(65.25,color='hotpink') # 17.25 h dose
#plt.axvspan(78, 81, facecolor='gold', alpha=0.4)
plt.xlabel("Clock Time (hours)")
plt.ylabel("n, Proportion of Activated Photoreceptors")
#plt.title("Time Trace of Photoreceptor Activation")
plt.show()
'''
