# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:54:04 2024

@author: sstowe

Simulating Burgess 2008 Ultradian Protocol

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

        self.a = (0.0675)*60*60 # pmol/L/sec converted to hours, I determined
        self.delta_M = 600 # sec, Breslow 2013
        self.r = 15.36 # sec, Breslow 2013
        
        self.psi_on = 1.1345 #1.13446401 # radians, I determined 
        self.psi_off = 3.6652 #3.66519143 # radians, I determined
        
        self.M_max = 0.019513 # Breslow 2013
        self.H_sat = 861 # Breslow 2013
        self.sigma_M = 50 # Breslow 2013
        self.m = 4.7147 # I determined by fitting to Zeitzer using differential evolution
        
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
        #x = [-0.74545016, 0.05671999, -0.76024892, 0.05994563, 0.18366069]
        # Switched sign of all five 
        #x = [0.74545016, -0.05671999, 0.76024892, -0.05994563, -0.18366069]
        # Differential Evolution: 
        #x = [0, 0, 0, 0, 0]
        #x = [0.95248293,  0.78097113,  0.82623692,  1.56471182, -0.24311406] # Error = 4.579093796811765
        #x = [0.95248293,  0.78097113,  0.82623692,  1.56471182, -0.24311406] # Error = 4.579093796811765
        #x = [0.95454361,  0.51195699,  0.89647909,  1.1634005,   0.00778411] # Error = 4.479966517732024
        #x = [1.77933371,  0.60638676,  2.8676035,   1.07183046, -0.14709443] # Error = 3.5704481511429305
        x = [1.83748124,  0.61746707,  1.83032292,  1.24868776, -0.06290714]  # Error = 3.481680628661965
        self.B_1 = x[0]
        self.theta_M1 = x[1]
        self.B_2 = x[2]
        self.theta_M2 = x[3]
        self.epsilon = x[4]
        
        
# Set the light schedule (timings and intensities)
    def light(self,t,schedule,melatonin_timing):
        
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
        
        else: # Laboratory days (5 total)
            bright_light = 40 # ~Mean reported in Breslow 2008 & 2010
            dim_light = 5
            dark = 0
            
            DLMO = 21 # Rounded baseline DLMO (21.1) to nearest whole number to make light schedule work correctly
            
            if 24 <= t < 96: # Days 2,3,4
                t = np.mod(t,24)
                if melatonin_timing == None:
                    melatonin_timing = 11 # If no exogenous melatonin is administered, have the ultradian schedule start at 0h on day 2 (but it does not change the phase shift prediction if another hour is chosen)
                    
                dose_time = np.mod(DLMO+melatonin_timing,24) # Find the clock time of the first exogenous melatonin dosage 
                
                # Find the lights on time for the first lights on 
                if dose_time >= 4:
                    lights_on = np.mod(dose_time,4)
                else: 
                    lights_on = dose_time

                # Ultradian light schedule (2.5:1.5 light:dark)
                if 0 <= np.mod(t-lights_on, 4) < 2.5: # Turn the lights on for 2.5h 
                    light = bright_light
                else: # Turn the lights off for 1.5
                    light = dark
                    
            else: # Days 1,5
                light = dim_light # Constant routine during phase assesments
        
        return light


# Define the alpha(L) function 
    def alpha0(self,t,schedule,melatonin_timing):
        """A helper function for modeling the light input processing"""
        
        return(self.alpha_0*pow(self.light(t,schedule,melatonin_timing), self.p)/(pow(self.light(t,schedule,melatonin_timing), self.p)+self.I_0));
    

# Set the exogenous melatonin administration schedule
    def ex_melatonin(self,t,melatonin_timing,melatonin_dosage):
        
        if melatonin_timing == None:
            return 0 # set exogenous melatonin to zero
        else: 
            if 24 < t < 96: 
                t = np.mod(t,24)
                
                DLMO = 21 # Rounded baseline DLMO (21.1) to nearest whole number to make light schedule work correctly
                dose_time = np.mod(DLMO+melatonin_timing,24) # Find the clock time of the first exogenous melatonin dosage 
                
                if dose_time-0.1 <= t <= dose_time+0.3: # Only calculate Guassian close to exogneous melatonin administration
                    sigma = np.sqrt(0.002)
                    mu = dose_time+0.1 # Shift mean of Guassian so exogenous melatonin is not entering the system too early

                    ex_mel = (1/sigma*np.sqrt(2*np.pi))*np.exp((-pow(t-mu,2))/(2*pow(sigma,2))) # Guassian function

                    x = np.arange(0, 1, 0.01)
                    melatonin_values = self.max_value(x, sigma)
                    max_value = max(melatonin_values)
                    #print(max_value)
                
                    #converted_dose = self.mg_conversion(melatonin_dosage)
                    converted_dose = 70000
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
    #def mg_conversion(self, melatonin_dosage):
        #x_line = melatonin_dosage
        #y_line = (56383*x_line) + 3085.1 # 2pts fit (Wyatt 2006)
        #y_line = 70000
        #return y_line

        

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
        Bhat = self.G*(1.0-n)*self.alpha0(t,schedule,melatonin_timing)
    
        # Light forcing equations
        LightAmp = (self.A_1/2.0)*Bhat*(1.0 - pow(R,4.0))*np.cos(Psi + self.beta_L1) + (self.A_2/2.0)*Bhat*R*(1.0 - pow(R,8.0))*np.cos(2.0*Psi + self.beta_L2) # L_R
        LightPhase = self.sigma*Bhat - (self.A_1/2.0)*Bhat*(pow(R,3.0) + 1.0/R)*np.sin(Psi + self.beta_L1) - (self.A_2/2.0)*Bhat*(1.0 + pow(R,8.0))*np.sin(2.0*Psi + self.beta_L2) # L_psi
    
        # Melatonin interaction with pacemaker
        Mhat = self.M_max/(1 + np.exp((self.H_sat - H2)/self.sigma_M))
        MelAmp = (self.B_1/2)*Mhat*(1.0 - pow(R,4.0))*np.cos(Psi + self.theta_M1) + (self.B_2/2.0)*Mhat*R*(1.0 - pow(R,8.0))*np.cos(2.0*Psi + self.theta_M2) # M_R
        MelPhase = self.epsilon*Mhat - (self.B_1/2.0)*Mhat*(pow(R,3.0)+1.0/R)*np.sin(Psi + self.theta_M1) - (self.B_2/2.0)*Mhat*(1.0 + pow(R,8.0))*np.sin(2.0*Psi + self.theta_M2) # M_psi
    
        # Switch pineal on and off in the presence of light 
        tmp = 1 - self.m*Bhat
        S = np.piecewise(tmp, [tmp >= 0, tmp < 0 and H1 < 0.001], [1, 0])
    
    
        dydt=np.zeros(6)
    
        # ODE System  
        dydt[0] = -(self.D + self.gamma)*R + (self.K/2)*np.cos(self.beta)*R*(1 - pow(R,4.0)) + LightAmp + MelAmp # dR/dt
        dydt[1] = self.omega_0 + (self.K/2)*np.sin(self.beta)*(1 + pow(R,4.0)) + LightPhase + MelPhase # dpsi/dt 
        dydt[2] = 60.0*(self.alpha0(t,schedule,melatonin_timing)*(1.0-n)-(self.delta*n)) # dn/dt
        
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




#------------- Running the model under all condtions for the PRC ----------

# Set the DLMO threshold that will be used throughout to determine DLMO time
DLMO_threshold = 10



#--------- Run the model to find initial conditions ---------------

model_IC = HannayBreslowModel() # defining model as a new object built with the HannayBreslowModel class 
model_IC.integrateModel(24*50,schedule=1) # use the integrateModel method with the object model
IC = model_IC.results[-1,:] # get initial conditions from entrained model



#--------- Run the model under the placebo condition ---------------

# Run Burgess 2008 placebo protocol
model_placebo = HannayBreslowModel()
model_placebo.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=None, melatonin_dosage=None,schedule=2) # run the model from entrained ICs

# Calculate shift due to protocol
# Baseline day 
baseline_plasma_mel = model_placebo.results[0:240,4]/4.3 # converting output to pg/mL
baseline_times = model_placebo.ts[0:240] # defining times from first 24hrs 
baseline, = np.where(baseline_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below 4pg/mL
baseline_DLMO = baseline_times[baseline[-1]] # finding the time corresponding to the first index below threshold, DLMO
#print(baseline_DLMO)

# Final day 
final_plasma_mel = model_placebo.results[960:1199,4]/4.3 # converting output to pg/mL
final_times = model_placebo.ts[960:1199] # defining times from last 24hrs
final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below 4pg/mL
final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO
#print(final_DLMO)

# Calculate phase shift (final - baseline; negative = delay, positive = advance) 
phase_shift_placebo = final_DLMO - baseline_DLMO



#--------- Run the model with exogenous melatonin 0h after DLMO (Clock time: 21:00)---------------

# Burgess 2008, exogenous melatonin given at start of wake episode 
model_0 = HannayBreslowModel()
model_0.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=0, melatonin_dosage=3.0,schedule=2) 

# Calculate shift due to protocol
# Baseline day DLMO determined above

# Final day 
final_plasma_mel = model_0.results[960:1199,4]/4.3 # converting output to pg/mL
final_times = model_0.ts[960:1199] # defining times from last 24hrs
final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below threshold
final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO
#print("Final - 0h")
#print(final_DLMO)


# Calculate phase shift (final - baseline; negative = delay, positive = advance) 
phase_shift_0 = final_DLMO - baseline_DLMO 



#--------- Run the model with exogenous melatonin 2h after DLMO (Clock time: 23:00)---------------

# Burgess 2008, exogenous melatonin given at start of wake episode 
model_2 = HannayBreslowModel()
model_2.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=2, melatonin_dosage=3.0,schedule=2) 

# Calculate shift due to protocol
# Baseline day DLMO determined above

# Final day 
final_plasma_mel = model_2.results[960:1199,4]/4.3 # converting output to pg/mL
final_times = model_2.ts[960:1199] # defining times from last 24hrs
final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below threshold
final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO
#print("Final - 2h")
#print(final_DLMO)


# Calculate phase shift (final - baseline; negative = delay, positive = advance) 
phase_shift_2 = final_DLMO - baseline_DLMO 



#--------- Run the model with exogenous melatonin 3h after DLMO (Clock time: 0:00)---------------

# Burgess 2008, exogenous melatonin given at start of wake episode 
model_3 = HannayBreslowModel()
model_3.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=3, melatonin_dosage=3.0,schedule=2) 

# Calculate shift due to protocol
# Baseline day DLMO determined above

# Final day 
final_plasma_mel = model_3.results[960:1199,4]/4.3 # converting output to pg/mL
final_times = model_3.ts[960:1199] # defining times from last 24hrs
final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below threshold
final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO
#print("Final - 3h")
#print(final_DLMO)


# Calculate phase shift (final - baseline; negative = delay, positive = advance) 
phase_shift_3 = final_DLMO - baseline_DLMO 



#--------- Run the model with exogenous melatonin 4h after DLMO (Clock time: 1:00)---------------

# Burgess 2008, exogenous melatonin given at start of wake episode 
model_4 = HannayBreslowModel()
model_4.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=4, melatonin_dosage=3.0,schedule=2) 

# Calculate shift due to protocol
# Baseline day DLMO determined above

# Final day 
final_plasma_mel = model_4.results[960:1199,4]/4.3 # converting output to pg/mL
final_times = model_4.ts[960:1199] # defining times from last 24hrs
final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below threshold
final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO
#print("Final - 4h")
#print(final_DLMO)


# Calculate phase shift (final - baseline; negative = delay, positive = advance) 
phase_shift_4 = final_DLMO - baseline_DLMO 



#--------- Run the model with exogenous melatonin 7h after DLMO (Clock time: 4:00)---------------

# Burgess 2008, exogenous melatonin given at start of wake episode 
model_7 = HannayBreslowModel()
model_7.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=7, melatonin_dosage=3.0,schedule=2) 

# Calculate shift due to protocol
# Baseline day DLMO determined above

# Final day 
final_plasma_mel = model_7.results[960:1199,4]/4.3 # converting output to pg/mL
final_times = model_7.ts[960:1199] # defining times from last 24hrs
final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below threshold
final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO
#print("Final - 7h")
#print(final_DLMO)


# Calculate phase shift (final - baseline; negative = delay, positive = advance) 
phase_shift_7 = final_DLMO - baseline_DLMO 



#--------- Run the model with exogenous melatonin 10h after DLMO (Clock time: 7:00)---------------

# Burgess 2008, exogenous melatonin given at start of wake episode 
model_10 = HannayBreslowModel()
model_10.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=10, melatonin_dosage=3.0,schedule=2)

# Calculate shift due to protocol
# Baseline day DLMO determined above

# Final day 
final_plasma_mel = model_10.results[960:1199,4]/4.3 # converting output to pg/mL
final_times = model_10.ts[960:1199] # defining times from last 24hrs
final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below threshold
final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO
#print("Final - 10h")
#print(final_DLMO)


# Calculate phase shift (final - baseline; negative = delay, positive = advance) 
phase_shift_10 = final_DLMO - baseline_DLMO 



#--------- Run the model with exogenous melatonin 11h after DLMO (Clock time: 8:00)---------------

# Burgess 2008, exogenous melatonin given at start of wake episode 
model_11 = HannayBreslowModel()
model_11.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=11, melatonin_dosage=3.0,schedule=2)

# Calculate shift due to protocol
# Baseline day DLMO determined above

# Final day 
final_plasma_mel = model_11.results[960:1199,4]/4.3 # converting output to pg/mL
final_times = model_11.ts[960:1199] # defining times from last 24hrs
final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below threshold
final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO
#print("Final - 11h")
#print(final_DLMO)


# Calculate phase shift (final - baseline; negative = delay, positive = advance) 
phase_shift_11 = final_DLMO - baseline_DLMO 



#--------- Run the model with exogenous melatonin 12h after DLMO (Clock time: 9:00)---------------

# Burgess 2008, exogenous melatonin given at start of wake episode 
model_12 = HannayBreslowModel()
model_12.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=12, melatonin_dosage=3.0,schedule=2)

# Calculate shift due to protocol
# Baseline day DLMO determined above

# Final day 
final_plasma_mel = model_12.results[960:1199,4]/4.3 # converting output to pg/mL
final_times = model_12.ts[960:1199] # defining times from last 24hrs
final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below threshold
final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO
#print("Final - 12h")
#print(final_DLMO)


# Calculate phase shift (final - baseline; negative = delay, positive = advance) 
phase_shift_12 = final_DLMO - baseline_DLMO 



#--------- Run the model with exogenous melatonin 13h after DLMO (Clock time: 10:00)---------------

# Burgess 2008, exogenous melatonin given at start of wake episode 
model_13 = HannayBreslowModel()
model_13.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=13, melatonin_dosage=3.0,schedule=2)

# Calculate shift due to protocol
# Baseline day DLMO determined above

# Final day 
final_plasma_mel = model_13.results[960:1199,4]/4.3 # converting output to pg/mL
final_times = model_13.ts[960:1199] # defining times from last 24hrs
final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below threshold
final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO
#print("Final - 13h")
#print(final_DLMO)


# Calculate phase shift (final - baseline; negative = delay, positive = advance) 
phase_shift_13 = final_DLMO - baseline_DLMO 



#--------- Run the model with exogenous melatonin 15h after DLMO (Clock time: 12:00)---------------

# Burgess 2008, exogenous melatonin given at start of wake episode 
model_15 = HannayBreslowModel()
model_15.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=15, melatonin_dosage=3.0,schedule=2)

# Calculate shift due to protocol
# Baseline day DLMO determined above

# Final day 
final_plasma_mel = model_15.results[960:1199,4]/4.3 # converting output to pg/mL
final_times = model_15.ts[960:1199] # defining times from last 24hrs
final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below threshold
final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO
#print("Final - 15h")
#print(final_DLMO)


# Calculate phase shift (final - baseline; negative = delay, positive = advance) 
phase_shift_15 = final_DLMO - baseline_DLMO 



#--------- Run the model with exogenous melatonin 16h after DLMO (Clock time: 13:00)---------------

# Burgess 2008, exogenous melatonin given at start of wake episode 
model_16 = HannayBreslowModel()
model_16.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=16, melatonin_dosage=3.0,schedule=2)

# Calculate shift due to protocol
# Baseline day DLMO determined above

# Final day 
final_plasma_mel = model_16.results[960:1199,4]/4.3 # converting output to pg/mL
final_times = model_16.ts[960:1199] # defining times from last 24hrs
final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below threshold
final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO
#print("Final - 16h")
#print(final_DLMO)


# Calculate phase shift (final - baseline; negative = delay, positive = advance) 
phase_shift_16 = final_DLMO - baseline_DLMO 



#--------- Run the model with exogenous melatonin 17h after DLMO (Clock time: 14:00)---------------

# Burgess 2008, exogenous melatonin given at start of wake episode 
model_17 = HannayBreslowModel()
model_17.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=17, melatonin_dosage=3.0,schedule=2)

# Calculate shift due to protocol
# Baseline day DLMO determined above

# Final day 
final_plasma_mel = model_17.results[960:1199,4]/4.3 # converting output to pg/mL
final_times = model_17.ts[960:1199] # defining times from last 24hrs
final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below threshold
final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO
#print("Final - 17h")
#print(final_DLMO)


# Calculate phase shift (final - baseline; negative = delay, positive = advance) 
phase_shift_17 = final_DLMO - baseline_DLMO 



#--------- Run the model with exogenous melatonin 18h after DLMO (Clock time: 15:00)---------------

# Burgess 2008, exogenous melatonin given at start of wake episode 
model_18 = HannayBreslowModel()
model_18.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=18, melatonin_dosage=3.0,schedule=2)

# Calculate shift due to protocol
# Baseline day DLMO determined above

# Final day 
final_plasma_mel = model_18.results[960:1199,4]/4.3 # converting output to pg/mL
final_times = model_18.ts[960:1199] # defining times from last 24hrs
final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below threshold
final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO
#print("Final - 18h")
#print(final_DLMO)


# Calculate phase shift (final - baseline; negative = delay, positive = advance) 
phase_shift_18 = final_DLMO - baseline_DLMO 



#--------- Run the model with exogenous melatonin 19h after DLMO (Clock time: 16:00)---------------

# Burgess 2008, exogenous melatonin given at start of wake episode 
model_19 = HannayBreslowModel()
model_19.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=19, melatonin_dosage=3.0,schedule=2)

# Calculate shift due to protocol
# Baseline day DLMO determined above

# Final day 
final_plasma_mel = model_19.results[960:1199,4]/4.3 # converting output to pg/mL
final_times = model_19.ts[960:1199] # defining times from last 24hrs
final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below threshold
final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO
#print("Final - 19h")
#print(final_DLMO)


# Calculate phase shift (final - baseline; negative = delay, positive = advance) 
phase_shift_19 = final_DLMO - baseline_DLMO



#--------- Run the model with exogenous melatonin 21h after DLMO (Clock time: 18:00)---------------

# Burgess 2008, exogenous melatonin given at start of wake episode 
model_21 = HannayBreslowModel()
model_21.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=21, melatonin_dosage=3.0,schedule=2)

# Calculate shift due to protocol
# Baseline day DLMO determined above

# Final day 
final_plasma_mel = model_21.results[960:1199,4]/4.3 # converting output to pg/mL
final_times = model_21.ts[960:1199] # defining times from last 24hrs
final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below threshold
final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO
#print("Final - 21h")
#print(final_DLMO)


# Calculate phase shift (final - baseline; negative = delay, positive = advance) 
phase_shift_21 = final_DLMO - baseline_DLMO



#--------- Run the model with exogenous melatonin 22h after DLMO (Clock time: 19:00)---------------

# Burgess 2008, exogenous melatonin given at start of wake episode 
model_22 = HannayBreslowModel()
model_22.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=22, melatonin_dosage=3.0,schedule=2)

# Calculate shift due to protocol
# Baseline day DLMO determined above

# Final day 
final_plasma_mel = model_22.results[960:1199,4]/4.3 # converting output to pg/mL
final_times = model_22.ts[960:1199] # defining times from last 24hrs
final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below threshold
final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO
#print("Final - 22h")
#print(final_DLMO)


# Calculate phase shift (final - baseline; negative = delay, positive = advance) 
phase_shift_22 = final_DLMO - baseline_DLMO



#--------- Run the model with exogenous melatonin 23h after DLMO (clock time: 20:00)---------------

# Burgess 2008, exogenous melatonin given at start of wake episode 
model_23 = HannayBreslowModel()
model_23.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=23, melatonin_dosage=3.0,schedule=2)

# Calculate shift due to protocol
# Baseline day DLMO determined above

# Final day 
final_plasma_mel = model_23.results[960:1199,4]/4.3 # converting output to pg/mL
final_times = model_23.ts[960:1199] # defining times from last 24hrs
final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below threshold
final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO
#print("Final - 23h")
#print(final_DLMO)


# Calculate phase shift (final - baseline; negative = delay, positive = advance) 
phase_shift_23 = final_DLMO - baseline_DLMO




#---------- Make an array of all predicted phase shifts --------------

phase_shifts = [phase_shift_0, 
                phase_shift_2, 
                phase_shift_3, 
                phase_shift_4, 
                phase_shift_7, 
                phase_shift_10, 
                phase_shift_11, 
                phase_shift_12, 
                phase_shift_13, 
                phase_shift_15, 
                phase_shift_16, 
                phase_shift_17, 
                phase_shift_18, 
                phase_shift_19,
                phase_shift_21, 
                phase_shift_22,
                phase_shift_23
                ];

# Subtract off the shift due to the protocol
phase_shifts_corrected = (phase_shifts - phase_shift_placebo)

# Make array of administration times for plotting PRC 
ExMel_times = [0,2,3,4,7,10,11,12,13,15,16,17,18,19,21,22,23]

# Plot PRC points
plt.plot(ExMel_times,phase_shifts_corrected,'o')
plt.plot(ExMel_times,phase_shifts_corrected, lw=2)
plt.axhline(0)
plt.axvline(21.1)
plt.xlabel("Time Since DLMO (hours)")
plt.ylabel("Phase Shift (hours)")
plt.title("3 Pulse Melatonin (3mg) Phase Response Curve")
plt.show()


#----------- Load Burgess 2008 PRC data -------------
# From WebPlotDigitizer (n = 27)

Burgess_2008_PRC = [-0.43,
                    -0.08,
                    0.45,
                    0.78,
                    -0.17,
                    0.08,
                    -0.37,
                    -0.55,
                    -1.17,
                    -2.69,
                    -1.02,
                    -1.78,
                    -1.37,
                    -1.05,
                    -0.43,
                    0.35,
                    -0.62,
                    1.00,
                    -0.32,
                    1.58,
                    2.60,
                    1.09,
                    2.18,
                    1.66,
                    0.60,
                    1.68,
                    1.63
                    ]

Burgess_2008_PRC_times = [0.3,
                          0.4,
                          1.9,
                          3.4,
                          3.7,
                          3.8,
                          7.5,
                          10.0,
                          10.1,
                          11.0,
                          11.1,
                          11.5,
                          11.7,
                          12.5,
                          13.3,
                          13.4,
                          14.9,
                          16.3,
                          16.7,
                          16.8,
                          17.7,
                          19.3,
                          19.4,
                          21.0,
                          22.0,
                          22.4,
                          23.4
                          ]
'''
# What I am actually fitting to (rounded to nearest hour): 
Burgess_2008_PRC_times = [0,
                          0,
                          2,
                          3,
                          4,
                          4,
                          7,
                          10,
                          10,
                          11,
                          11,
                          11,
                          12,
                          12,
                          13,
                          13,
                          15,
                          16,
                          17,
                          17,
                          18,
                          19,
                          19,
                          21,
                          22,
                          22,
                          23
                          ]
'''
    
plt.plot(Burgess_2008_PRC_times, Burgess_2008_PRC, 'o')
plt.plot(ExMel_times,phase_shifts_corrected, lw=2)
plt.axvline(x=19.4) # Burgess fitted peak (max phase advance)
plt.axhline(y=1.8) # Burgess fitted max advance 
plt.axhline(y=-1.3) # Burgess fitted max delay
plt.axvline(x=11.1) # Burgess fitted trough (max phase delay)
plt.show()


#--------- Plot Model Output -------------------

# pick one to plot 
#model = model_placebo
#model = model_0 # Clock time: 21:00
#model = model_2 # Clock time: 23:00
#model = model_3 # Clock time: 0:00
#model = model_4 # Clock time: 1:00
#model = model_7 # Clock time: 4:00
#model = model_10 # Clock time: 7:00
model = model_11 # Clock time: 8:00
#model = model_12 # Clock time: 9:00
#model = model_13 # Clock time: 10:00
#model = model_15 # Clock time: 12:00
#model = model_16 # Clock time: 13:00
#model = model_17 # Clock time: 14:00
#model = model_18 # Clock time: 15:00
#model = model_19 # Clock time: 16:00
#model = model_21 # Clock time: 18:00
#model = model_22 # Clock time: 19:00
#model = model_23 # Clock time: 20:00

#check = 20


# Plotting H1, H2, and H3 (melatonin concentrations, pmol/L)
plt.plot(model.ts,model.results[:,3],lw=2)
plt.plot(model.ts,model.results[:,4],lw=2)
plt.plot(model.ts,model.results[:,5],lw=2)
#plt.axvline(x=2.5)
#plt.axvline(x=26.5)
#plt.axvline(x=50.5)
plt.xlabel("Time (hours)")
plt.ylabel("Melatonin Concentration (pmol/L)")
plt.title("Time Trace of Melatonin Concentrations (pmol/L)")
plt.legend(["Pineal","Plasma", "Exogenous"])
plt.show()


# Plotting H1, H2, and H3 (melatonin concentrations, pg/mL)
plt.plot(model.ts,model.results[:,3]/4.3,lw=2)
plt.plot(model.ts,model.results[:,4]/4.3,lw=2)
plt.plot(model.ts,model.results[:,5]/4.3,lw=2)
plt.axhline(DLMO_threshold)
#plt.axvline(8+24)
plt.xlabel("Time (hours)")
plt.ylabel("Melatonin Concentration (pg/mL)")
plt.title("Time Trace of Melatonin Concentrations (pg/mL)")
plt.legend(["Pineal","Plasma", "Exogenous"])
plt.show()


# Plotting H1, H2, and H3 on baseline day (melatonin concentrations, pg/mL)
plt.plot(model.ts[0:240],model.results[0:240,3]/4.3,lw=2)
plt.plot(model.ts[0:240],model.results[0:240,4]/4.3,lw=2)
plt.plot(model.ts[0:240],model.results[0:240,5]/4.3,lw=2)
plt.axhline(DLMO_threshold)
plt.axvline(21.1)
plt.xlabel("Time (hours)")
plt.ylabel("Melatonin Concentration (pg/mL)")
plt.title("Baseline Day Melatonin Concentrations (DLMO = 10pg/mL)")
plt.legend(["Pineal","Plasma", "Exogenous"])
plt.show()

# Plotting H1, H2 on Day 1 of ultradian schedule (melatonin concentrations, pg/mL)
plt.plot(model.ts[241:480],model.results[241:480,3]/4.3,lw=2)
plt.plot(model.ts[241:480],model.results[241:480,4]/4.3,lw=2)
#plt.plot(model.ts[241:480],model.results[241:480,5]/4.3,lw=2)
plt.axhline(DLMO_threshold)
#plt.ylim(0, 500)
#plt.axvline(check+24)
#plt.axvline(22+24)
plt.xlabel("Time (hours)")
plt.ylabel("Melatonin Concentration (pg/mL)")
plt.title("Day 1 Melatonin Concentrations (DLMO = 10pg/mL)")
plt.legend(["Pineal","Plasma"])
plt.show()

# Plotting H1, H2 on Day 2 of ultradian schedule (melatonin concentrations, pg/mL)
plt.plot(model.ts[481:720],model.results[481:720,3]/4.3,lw=2)
plt.plot(model.ts[481:720],model.results[481:720,4]/4.3,lw=2)
#plt.plot(model.ts[481:720],model.results[481:720,5]/4.3,lw=2)
plt.axhline(DLMO_threshold)
#plt.ylim(0, 500)
#plt.axvline(check+24+24)
plt.xlabel("Time (hours)")
plt.ylabel("Melatonin Concentration (pg/mL)")
plt.title("Day 2 Melatonin Concentrations (DLMO = 10pg/mL)")
plt.legend(["Pineal","Plasma"])
plt.show()

# Plotting H1, H2 on Day 3 of ultradian schedule (melatonin concentrations, pg/mL)
plt.plot(model.ts[721:960],model.results[721:960,3]/4.3,lw=2)
plt.plot(model.ts[721:960],model.results[721:960,4]/4.3,lw=2)
#plt.plot(model.ts[721:960],model.results[721:960,5]/4.3,lw=2)
plt.axhline(DLMO_threshold)
#plt.ylim(0, 500)
#plt.axvline(check+24+24+24)
plt.xlabel("Time (hours)")
plt.ylabel("Melatonin Concentration (pg/mL)")
plt.title("Day 3 Melatonin Concentrations (DLMO = 10pg/mL)")
plt.legend(["Pineal","Plasma"])
plt.show()

# Plotting H1, H2, and H3 on final day (melatonin concentrations, pg/mL)
plt.plot(np.mod(model.ts[960:1199],24),model.results[960:1199,3]/4.3,lw=2)
plt.plot(np.mod(model.ts[960:1199],24),model.results[960:1199,4]/4.3,lw=2)
plt.plot(np.mod(model.ts[960:1199],24),model.results[960:1199,5]/4.3,lw=2)
plt.axhline(DLMO_threshold)
#plt.axvline(21.5)
#plt.ylim(0, 50)
#plt.xlim(15, 24)
plt.xlabel("Time (hours)")
plt.ylabel("Melatonin Concentration (pg/mL)")
plt.title("FInal Day Melatonin Concentrations (DLMO = 10pg/mL)")
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
plt.plot(model.ts,np.mod(model.results[:,1],2*np.pi),lw=2)
plt.xlabel("Time (hours)")
plt.ylabel("Psi, Mean Phase (radians)")
plt.title("Time Trace of Psi, Mean Phase")
plt.show()

# Plotting n
plt.plot(model.ts,model.results[:,2],lw=2)
#plt.axvline(check+24)
#plt.axvline(check+24+24)
#plt.axvline(check+24+24+24)
#plt.axvline(72)
#plt.axvline(96)
plt.xlabel("Time (hours)")
plt.ylabel("Proportion of Activated Photoreceptors")
plt.title("Time Trace of Photoreceptor Activation")
plt.show()


