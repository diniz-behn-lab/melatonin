# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:50:14 2024

@author: sstowe

Hannay-Breslow Model code to be imported into the differential evolution code
(DifferentialEvolution.py)

"""

import numpy as np
import scipy as sp

# ---------- Create Model Class -------------------

class HannayBreslowModel(object):
    """Driver of ForwardModel simulation"""

# Initialize the class
    def __init__(self):
        self.set_params() # setting parameters every time an object of the class is created
        #self.set_melatonin_prc_params()
        
        ## Melatonin Forcing Parameters 
        # TO BE OPTIMIZED
    def set_melatonin_prc_params(self, theta):
        self.B_1 = theta[0] #0.74545016
        self.theta_M1 = theta[1] #-0.05671999
        self.B_2 = theta[2] #0.76024892
        self.theta_M2 = theta[3] #-0.05994563
        self.epsilon = theta[4] #-0.18366069    
        
        print("melatonin parameters are set")
        print(theta)

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
        self.m = 4.9278 # I determined by fitting to Zeitzer using differential evolution

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
        
        self.B_1 = 0#0.74545016
        self.theta_M1 = 0#-0.05671999
        self.B_2 = 0#0.76024892
        self.theta_M2 = 0#-0.05994563
        self.epsilon = 0#-0.18366069
        
        print("parameters are set")
    
# Set the exogenous melatonin administration schedule VERSION 5
    def ex_melatonin(self,t,melatonin_timing,melatonin_dosage):
        
        if melatonin_timing == None:
            return 0 # set exogenous melatonin to zero
        else: 
            if 24 < t < 96: 
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
        y_line = (56383*x_line) + 3085.1 # 2pts fit (Wyatt 2006)
        return y_line
    

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
        else: # Laboratory days (5 total)
            if 24 < t < 96: # Days 2-4, ultradian schedule 
                full_light = 40
                if 1 <= np.mod(t,4) <= 2.5 :
                    return 0 
                else:
                    return full_light
            else: # Days 1 and 5, constant routine 
                dim_light = 5
                return dim_light


# Define the alpha(L) function 
    def alpha0(self,t,schedule):
        """A helper function for modeling the light input processing"""
        return(self.alpha_0*pow(self.light(t,schedule), self.p)/(pow(self.light(t,schedule), self.p)+self.I_0));

 


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
        dydt[0] = -(self.D + self.gamma)*R + (self.K/2)*np.cos(self.beta)*R*(1 - pow(R,4.0)) + LightAmp + MelAmp# dR/dt
        dydt[1] = self.omega_0 + (self.K/2)*np.sin(self.beta)*(1 + pow(R,4.0)) + LightPhase + MelPhase # dpsi/dt 
        dydt[2] = 60.0*(self.alpha0(t,schedule)*(1.0-n)-(self.delta*n)) # dn/dt
        
        dydt[3] = -self.beta_IP*H1 + A*(1 - self.m*Bhat)*S # dH1/dt
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


   
def run_HannayBreslow_PRC(params):
    
    #--------- Run the model to find initial conditions --------------- 
    '''
    model_IC = HannayBreslowModel()
    
    model_IC.set_melatonin_prc_params(params)
    model_IC.integrateModel(24*50, schedule=1) # use the integrateModel method with the object model
    IC = model_IC.results[-1,:] # get initial conditions from entrained model
    '''
    IC = ([0.832531, 2.06503, 0.470175, 80.229, 108.099, 0])
    

    #--------- Run the model under the placebo condition ---------------
    model_placebo = HannayBreslowModel()
    model_placebo.set_melatonin_prc_params(params)
    model_placebo.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=None, melatonin_dosage=None,schedule=2) # run the model from entrained ICs

    # Calculate shift due to protocol
    # Baseline day 
    baseline_plasma_mel = model_placebo.results[0:240,4]/4.3 # converting output to pg/mL
    baseline_times = model_placebo.ts[0:240] # defining times from first 24hrs 
    baseline, = np.where(baseline_plasma_mel<=3)#self.DLMO_threshold) # finding all the indices where concentration is below 4pg/mL
    baseline_DLMO = baseline_times[baseline[-1]] # finding the time corresponding to the first index below threshold, DLMO

    # Final day 
    final_plasma_mel = model_placebo.results[960:1199,4]/4.3 # converting output to pg/mL
    final_times = model_placebo.ts[960:1199] # defining times from last 24hrs
    final, = np.where(final_plasma_mel<=3)#self.DLMO_threshold) # finding all the indices where concentration is below 4pg/mL
    final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO

    # Calculate phase shift (final - baseline; negative = delay, positive = advance) 
    phase_shift_placebo = final_DLMO - baseline_DLMO
        
    print("placebo shift found")    

    #--------- Run the model with exogenous melatonin at 2.5h ---------------
    model_2 = HannayBreslowModel()
    model_2.set_melatonin_prc_params(params)
    model_2.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=2.5, melatonin_dosage=3.0,schedule=2) 

    # Calculate shift due to protocol
    # Baseline day 
    baseline_plasma_mel = model_2.results[0:240,4]/4.3 # converting output to pg/mL
    baseline_times = model_2.ts[0:240] # defining times from first 24hrs 
    baseline, = np.where(baseline_plasma_mel<=3)#self.DLMO_threshold) # finding all the indices where concentration is below 4pg/mL
    baseline_DLMO = baseline_times[baseline[-1]] # finding the time corresponding to the first index below threshold, DLMO

    # Final day  
    final_plasma_mel = model_2.results[960:1199,4]/4.3 # converting output to pg/mL
    final_times = model_2.ts[960:1199] # defining times from last 24hrs
    final, = np.where(final_plasma_mel<=3)#self.DLMO_threshold) # finding all the indices where concentration is below 4pg/mL\
    final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO

    # Calculate phase shift (final - baseline; negative = delay, positive = advance) 
    phase_shift_2 = final_DLMO - baseline_DLMO # 2.5h, or 6.4h after DLMO
       
    print("2.5 shift found") 
        
    #--------- Run the model with exogenous melatonin at 6.5h ---------------
    model_6 = HannayBreslowModel()
    model_6.set_melatonin_prc_params(params)
    model_6.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=6.5, melatonin_dosage=3.0,schedule=2) 

    # Calculate shift due to protocol
    # Baseline day 
    baseline_plasma_mel = model_6.results[0:240,4]/4.3 # converting output to pg/mL
    baseline_times = model_6.ts[0:240] # defining times from first 24hrs 
    baseline, = np.where(baseline_plasma_mel<=3)#self.DLMO_threshold) # finding all the indices where concentration is below 4pg/mL
    baseline_DLMO = baseline_times[baseline[-1]] # finding the time corresponding to the first index below threshold, DLMO

    # Final day 
    final_plasma_mel = model_6.results[960:1199,4]/4.3 # converting output to pg/mL
    final_times = model_6.ts[960:1199] # defining times from last 24hrs
    final, = np.where(final_plasma_mel<=3)#self.DLMO_threshold) # finding all the indices where concentration is below 4pg/mL
    final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO

    # Calculate phase shift (final - baseline; negative = delay, positive = advance) 
    phase_shift_6 = final_DLMO - baseline_DLMO # 6.5h, or 10.4h after DLMO

    print("6.5 shift found") 

    #--------- Run the model with exogenous melatonin at 10.5h ---------------
    model_10 = HannayBreslowModel()
    model_10.set_melatonin_prc_params(params)
    model_10.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=10.5, melatonin_dosage=3.0,schedule=2)

    # Calculate shift due to protocol
    # Baseline day 
    baseline_plasma_mel = model_10.results[0:240,4]/4.3 # converting output to pg/mL
    baseline_times = model_10.ts[0:240] # defining times from first 24hrs 
    baseline, = np.where(baseline_plasma_mel<=3)#self.DLMO_threshold) # finding all the indices where concentration is below 4pg/mL
    baseline_DLMO = baseline_times[baseline[-1]] # finding the time corresponding to the first index below threshold, DLMO

    # Final day 
    final_plasma_mel = model_10.results[960:1199,4]/4.3 # converting output to pg/mL
    final_times = model_10.ts[960:1199] # defining times from last 24hrs
    final, = np.where(final_plasma_mel<=3)#self.DLMO_threshold) # finding all the indices where concentration is below 4pg/mL
    final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO
    
    # Calculate phase shift (final - baseline; negative = delay, positive = advance) 
    phase_shift_10 = final_DLMO - baseline_DLMO # 10.5h, or 14.4h after DLMO
    
    print("10.5 shift found") 

    #--------- Run the model with exogenous melatonin at 14.5h ---------------
    model_14 = HannayBreslowModel()
    model_14.set_melatonin_prc_params(params)
    model_14.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=14.5, melatonin_dosage=3.0,schedule=2)

    # Calculate shift due to protocol
    # Baseline day 
    baseline_plasma_mel = model_14.results[0:240,4]/4.3 # converting output to pg/mL
    baseline_times = model_14.ts[0:240] # defining times from first 24hrs 
    baseline, = np.where(baseline_plasma_mel<=3)#self.DLMO_threshold) # finding all the indices where concentration is below 4pg/mL
    baseline_DLMO = baseline_times[baseline[-1]] # finding the time corresponding to the first index below threshold, DLMO

    # Final day 
    final_plasma_mel = model_14.results[960:1199,4]/4.3 # converting output to pg/mL
    final_times = model_14.ts[960:1199] # defining times from last 24hrs
    final, = np.where(final_plasma_mel<=3)#self.DLMO_threshold) # finding all the indices where concentration is below 4pg/mL
    final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO

    # Calculate phase shift (final - baseline; negative = delay, positive = advance) 
    phase_shift_14 = final_DLMO - baseline_DLMO # 14.5h, or 6.4h before DLMO

    print("14.5 shift found") 

    #--------- Run the model with exogenous melatonin at 18.5h ---------------
    model_18 = HannayBreslowModel()
    model_18.set_melatonin_prc_params(params)
    model_18.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=18.5, melatonin_dosage=3.0,schedule=2)

    # Calculate shift due to protocol
    # Baseline day 
    baseline_plasma_mel = model_18.results[0:240,4]/4.3 # converting output to pg/mL
    baseline_times = model_18.ts[0:240] # defining times from first 24hrs 
    baseline, = np.where(baseline_plasma_mel<=3)#self.DLMO_threshold) # finding all the indices where concentration is below 4pg/mL
    baseline_DLMO = baseline_times[baseline[-1]] # finding the time corresponding to the first index below threshold, DLMO

    # Final day 
    final_plasma_mel = model_18.results[960:1199,4]/4.3 # converting output to pg/mL
    final_times = model_18.ts[960:1199] # defining times from last 24hrs
    final, = np.where(final_plasma_mel<=3)#self.DLMO_threshold) # finding all the indices where concentration is below 4pg/mL
    final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO

    # Calculate phase shift (final - baseline; negative = delay, positive = advance) 
    phase_shift_18 = final_DLMO - baseline_DLMO # 18.5h, or 2.4h before DLMO
    
    print("18.5 shift found") 

    #--------- Run the model with exogenous melatonin at 22.5h ---------------
    model_22 = HannayBreslowModel()
    model_22.set_melatonin_prc_params(params)
    model_22.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=22.5, melatonin_dosage=3.0,schedule=2)

    # Calculate shift due to protocol
    # Baseline day 
    baseline_plasma_mel = model_22.results[0:240,4]/4.3 # converting output to pg/mL
    baseline_times = model_22.ts[0:240] # defining times from first 24hrs 
    baseline, = np.where(baseline_plasma_mel<=3)#self.DLMO_threshold) # finding all the indices where concentration is below 4pg/mL
    baseline_DLMO = baseline_times[baseline[-1]] # finding the time corresponding to the first index below threshold, DLMO

    # Final day 
    final_plasma_mel = model_22.results[960:1199,4]/4.3 # converting output to pg/mL
    final_times = model_22.ts[960:1199] # defining times from last 24hrs
    final, = np.where(final_plasma_mel<=3)#self.DLMO_threshold) # finding all the indices where concentration is below 4pg/mL
    final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO

    # Calculate phase shift (final - baseline; negative = delay, positive = advance) 
    phase_shift_22 = final_DLMO - baseline_DLMO # 22.5h, or 1.4h after DLMO
    
    print("22.5 shift found")     

    #---------- Make an array of all predicted phase shifts --------------

    phase_shifts = [phase_shift_2, phase_shift_6, phase_shift_10, phase_shift_14, phase_shift_18, phase_shift_22];

    # Subtract off the shift due to the protocol
    phase_shifts_corrected = (phase_shifts - phase_shift_placebo)
    
    #print(phase_shifts_corrected)
    
    return phase_shifts_corrected
    


def objective_func(params,data_vals):
    try: 
        print(params)
        phase_shifts = run_HannayBreslow_PRC(params)
        print(phase_shifts)
        
        print(phase_shifts - data_vals)
        #print(abs(phase_shifts - data_vals))
    
        error = np.mean(abs(phase_shifts - data_vals))
        print(error)
    
        return error
    except: 
        return 1e6
