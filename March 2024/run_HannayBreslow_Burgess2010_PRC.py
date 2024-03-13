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
        self.B_1 = theta[0] 
        self.theta_M1 = theta[1] 
        self.B_2 = theta[2] 
        self.theta_M2 = theta[3]
        self.epsilon = theta[4]     
        
        print("melatonin parameters are set")
        print(theta)

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
        
        self.M_max = 0.019513 # Breslow 2013
        self.H_sat = 861 # Breslow 2013
        self.sigma_M = 50 # Breslow 2013
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
        
        self.B_1 = 0 
        self.theta_M1 = 0
        self.B_2 = 0
        self.theta_M2 = 0
        self.epsilon = 0
    
    
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
                
                    #converted_dose = 179893 #70000
                    converted_dose = self.mg_conversion(melatonin_dosage)
            
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
        #y_line = 70000
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
    
    #--------- Set initial conditions --------------- 

    IC = ([0.832603, 2.06472, 0.470183, 119.162, 164.444, 0])


    # ------- Set the DLMO threshold that determines DLMO time ----------
    DLMO_threshold = 10


    #--------- Run the model under the placebo condition ---------------
    model_placebo = HannayBreslowModel()
    model_placebo.set_melatonin_prc_params(params)
    model_placebo.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=None, melatonin_dosage=None,schedule=2) # run the model from entrained ICs

    # Calculate shift due to protocol
    # Baseline day 
    baseline_plasma_mel = model_placebo.results[0:240,4]/4.3 # converting output to pg/mL
    baseline_times = model_placebo.ts[0:240] # defining times from first 24hrs 
    baseline, = np.where(baseline_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below 4pg/mL
    baseline_DLMO = baseline_times[baseline[-1]] # finding the time corresponding to the first index below threshold, DLMO
   
    # Final day
    final_plasma_mel = model_placebo.results[960:1199,4]/4.3 # converting output to pg/mL
    final_times = model_placebo.ts[960:1199] # defining times from last 24hrs
    final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below 4pg/mL
    final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO

    # Calculate phase shift (baseline - final; negative = delay, positive = advance) 
    phase_shift_placebo = baseline_DLMO - final_DLMO
        
    print("placebo shift found")    


    #--------- Run the model with exogenous melatonin 0h after DLMO (Clock time: 21:00)---------------

    # Burgess 2010, exogenous melatonin given at start of wake episode 
    model_0 = HannayBreslowModel()
    model_0.set_melatonin_prc_params(params)
    model_0.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=0, melatonin_dosage=0.5,schedule=2) 

    # Calculate shift due to protocol
    # Baseline day DLMO determined above

    # Final day 
    final_plasma_mel = model_0.results[960:1199,4]/4.3 # converting output to pg/mL
    final_times = model_0.ts[960:1199] # defining times from last 24hrs
    final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below threshold
    final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO

    # Calculate phase shift (baseline - final; negative = delay, positive = advance) 
    phase_shift_0 = baseline_DLMO - final_DLMO
    
    print("0h shift found") 
    

    #--------- Run the model with exogenous melatonin 1h after DLMO (Clock time: 22:00)---------------

    # Burgess 2010, exogenous melatonin given at start of wake episode 
    model_1 = HannayBreslowModel()
    model_1.set_melatonin_prc_params(params)
    model_1.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=1, melatonin_dosage=0.5,schedule=2) 

    # Calculate shift due to protocol
    # Baseline day DLMO determined above

    # Final day 
    final_plasma_mel = model_1.results[960:1199,4]/4.3 # converting output to pg/mL
    final_times = model_1.ts[960:1199] # defining times from last 24hrs
    final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below threshold
    final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO

    # Calculate phase shift (baseline - final; negative = delay, positive = advance) 
    phase_shift_1 = baseline_DLMO - final_DLMO 

    print("1h shift found")
    
    
    #--------- Run the model with exogenous melatonin 2h after DLMO (Clock time: 23:00)---------------

    # Burgess 2010, exogenous melatonin given at start of wake episode 
    model_2 = HannayBreslowModel()
    model_2.set_melatonin_prc_params(params)
    model_2.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=2, melatonin_dosage=0.5,schedule=2) 

    # Calculate shift due to protocol
    # Baseline day DLMO determined above

    # Final day 
    final_plasma_mel = model_2.results[960:1199,4]/4.3 # converting output to pg/mL
    final_times = model_2.ts[960:1199] # defining times from last 24hrs
    final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below threshold
    final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO

    # Calculate phase shift (baseline - final; negative = delay, positive = advance) 
    phase_shift_2 = baseline_DLMO - final_DLMO 

    print("2h shift found")
    

    #--------- Run the model with exogenous melatonin 3h after DLMO (Clock time: 0:00)---------------

    # Burgess 2010, exogenous melatonin given at start of wake episode 
    model_3 = HannayBreslowModel()
    model_3.set_melatonin_prc_params(params)
    model_3.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=3, melatonin_dosage=0.5,schedule=2) 

    # Calculate shift due to protocol
    # Baseline day DLMO determined above

    # Final day 
    final_plasma_mel = model_3.results[960:1199,4]/4.3 # converting output to pg/mL
    final_times = model_3.ts[960:1199] # defining times from last 24hrs
    final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below threshold
    final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO

    # Calculate phase shift (baseline - final; negative = delay, positive = advance) 
    phase_shift_3 = baseline_DLMO - final_DLMO 

    print("3h shift found")
    

    #--------- Run the model with exogenous melatonin 5h after DLMO (Clock time: 4:00)---------------

    # Burgess 2010, exogenous melatonin given at start of wake episode 
    model_5 = HannayBreslowModel()
    model_5.set_melatonin_prc_params(params)
    model_5.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=5, melatonin_dosage=0.5,schedule=2) 

    # Calculate shift due to protocol
    # Baseline day DLMO determined above

    # Final day 
    final_plasma_mel = model_5.results[960:1199,4]/4.3 # converting output to pg/mL
    final_times = model_5.ts[960:1199] # defining times from last 24hrs
    final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below threshold
    final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO

    # Calculate phase shift (baseline - final; negative = delay, positive = advance) 
    phase_shift_5 = baseline_DLMO - final_DLMO

    print("5h shift found")
    
    
    #--------- Run the model with exogenous melatonin 6h after DLMO (Clock time: 5:00)---------------

    # Burgess 2010, exogenous melatonin given at start of wake episode 
    model_6 = HannayBreslowModel()
    model_6.set_melatonin_prc_params(params)
    model_6.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=6, melatonin_dosage=0.5,schedule=2) 

    # Calculate shift due to protocol
    # Baseline day DLMO determined above

    # Final day 
    final_plasma_mel = model_6.results[960:1199,4]/4.3 # converting output to pg/mL
    final_times = model_6.ts[960:1199] # defining times from last 24hrs
    final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below threshold
    final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO

    # Calculate phase shift (baseline - final; negative = delay, positive = advance) 
    phase_shift_6 = baseline_DLMO - final_DLMO

    print("6h shift found")
    

    #--------- Run the model with exogenous melatonin 7h after DLMO (Clock time: 4:00)---------------

    # Burgess 2010, exogenous melatonin given at start of wake episode 
    model_7 = HannayBreslowModel()
    model_7.set_melatonin_prc_params(params)
    model_7.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=7, melatonin_dosage=0.5,schedule=2) 

    # Calculate shift due to protocol
    # Baseline day DLMO determined above

    # Final day 
    final_plasma_mel = model_7.results[960:1199,4]/4.3 # converting output to pg/mL
    final_times = model_7.ts[960:1199] # defining times from last 24hrs
    final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below threshold
    final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO

    # Calculate phase shift (baseline - final; negative = delay, positive = advance) 
    phase_shift_7 = baseline_DLMO - final_DLMO 

    print("7h shift found")
    
    
    #--------- Run the model with exogenous melatonin 8h after DLMO (Clock time: 5:00)---------------

    # Burgess 2010, exogenous melatonin given at start of wake episode 
    model_8 = HannayBreslowModel()
    model_8.set_melatonin_prc_params(params)
    model_8.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=8, melatonin_dosage=0.5,schedule=2) 

    # Calculate shift due to protocol
    # Baseline day DLMO determined above

    # Final day 
    final_plasma_mel = model_8.results[960:1199,4]/4.3 # converting output to pg/mL
    final_times = model_8.ts[960:1199] # defining times from last 24hrs
    final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below threshold
    final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO

    # Calculate phase shift (baseline - final; negative = delay, positive = advance) 
    phase_shift_8 = baseline_DLMO - final_DLMO 

    print("8h shift found")
    

    #--------- Run the model with exogenous melatonin 10h after DLMO (Clock time: 7:00)---------------

    # Burgess 2010, exogenous melatonin given at start of wake episode 
    model_10 = HannayBreslowModel()
    model_10.set_melatonin_prc_params(params)
    model_10.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=10, melatonin_dosage=0.5,schedule=2)

    # Calculate shift due to protocol
    # Baseline day DLMO determined above

    # Final day 
    final_plasma_mel = model_10.results[960:1199,4]/4.3 # converting output to pg/mL
    final_times = model_10.ts[960:1199] # defining times from last 24hrs
    final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below threshold
    final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO

    # Calculate phase shift (baseline - final; negative = delay, positive = advance) 
    phase_shift_10 = baseline_DLMO - final_DLMO

    print("10h shift found")
    

    #--------- Run the model with exogenous melatonin 11h after DLMO (Clock time: 8:00)---------------

    # Burgess 2010, exogenous melatonin given at start of wake episode 
    model_11 = HannayBreslowModel()
    model_11.set_melatonin_prc_params(params)
    model_11.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=11, melatonin_dosage=0.5,schedule=2)

    # Calculate shift due to protocol
    # Baseline day DLMO determined above

    # Final day 
    final_plasma_mel = model_11.results[960:1199,4]/4.3 # converting output to pg/mL
    final_times = model_11.ts[960:1199] # defining times from last 24hrs
    final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below threshold
    final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO

    # Calculate phase shift (baseline - final; negative = delay, positive = advance) 
    phase_shift_11 = baseline_DLMO - final_DLMO 

    print("11h shift found")
    

    #--------- Run the model with exogenous melatonin 12h after DLMO (Clock time: 9:00)---------------

    # Burgess 2010, exogenous melatonin given at start of wake episode 
    model_12 = HannayBreslowModel()
    model_12.set_melatonin_prc_params(params)
    model_12.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=12, melatonin_dosage=0.5,schedule=2)

    # Calculate shift due to protocol
    # Baseline day DLMO determined above

    # Final day 
    final_plasma_mel = model_12.results[960:1199,4]/4.3 # converting output to pg/mL
    final_times = model_12.ts[960:1199] # defining times from last 24hrs
    final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below threshold
    final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO

    # Calculate phase shift (baseline - final; negative = delay, positive = advance) 
    phase_shift_12 = baseline_DLMO - final_DLMO 

    print("12h shift found")
    

    #--------- Run the model with exogenous melatonin 13h after DLMO (Clock time: 10:00)---------------

    # Burgess 2010, exogenous melatonin given at start of wake episode 
    model_13 = HannayBreslowModel()
    model_13.set_melatonin_prc_params(params)
    model_13.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=13, melatonin_dosage=0.5,schedule=2)

    # Calculate shift due to protocol
    # Baseline day DLMO determined above

    # Final day 
    final_plasma_mel = model_13.results[960:1199,4]/4.3 # converting output to pg/mL
    final_times = model_13.ts[960:1199] # defining times from last 24hrs
    final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below threshold
    final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO

    # Calculate phase shift (baseline - final; negative = delay, positive = advance) 
    phase_shift_13 = baseline_DLMO - final_DLMO

    print("13h shift found")
    

    #--------- Run the model with exogenous melatonin 16h after DLMO (Clock time: 13:00)---------------

    # Burgess 2010, exogenous melatonin given at start of wake episode 
    model_16 = HannayBreslowModel()
    model_16.set_melatonin_prc_params(params)
    model_16.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=16, melatonin_dosage=0.5,schedule=2)

    # Calculate shift due to protocol
    # Baseline day DLMO determined above

    # Final day 
    final_plasma_mel = model_16.results[960:1199,4]/4.3 # converting output to pg/mL
    final_times = model_16.ts[960:1199] # defining times from last 24hrs
    final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below threshold
    final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO

    # Calculate phase shift (baseline - final; negative = delay, positive = advance) 
    phase_shift_16 = baseline_DLMO - final_DLMO

    print("16h shift found")
    

    #--------- Run the model with exogenous melatonin 17h after DLMO (Clock time: 14:00)---------------

    # Burgess 2010, exogenous melatonin given at start of wake episode 
    model_17 = HannayBreslowModel()
    model_17.set_melatonin_prc_params(params)
    model_17.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=17, melatonin_dosage=0.5,schedule=2)

    # Calculate shift due to protocol
    # Baseline day DLMO determined above

    # Final day 
    final_plasma_mel = model_17.results[960:1199,4]/4.3 # converting output to pg/mL
    final_times = model_17.ts[960:1199] # defining times from last 24hrs
    final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below threshold
    final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO
    
    # Calculate phase shift (baseline - final; negative = delay, positive = advance) 
    phase_shift_17 = baseline_DLMO - final_DLMO

    print("17h shift found")
    

    #--------- Run the model with exogenous melatonin 19h after DLMO (Clock time: 16:00)---------------

    # Burgess 2010, exogenous melatonin given at start of wake episode 
    model_19 = HannayBreslowModel()
    model_19.set_melatonin_prc_params(params)
    model_19.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=19, melatonin_dosage=0.5,schedule=2)

    # Calculate shift due to protocol
    # Baseline day DLMO determined above

    # Final day 
    final_plasma_mel = model_19.results[960:1199,4]/4.3 # converting output to pg/mL
    final_times = model_19.ts[960:1199] # defining times from last 24hrs
    final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below threshold
    final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO

    # Calculate phase shift (baseline - final; negative = delay, positive = advance) 
    phase_shift_19 = baseline_DLMO - final_DLMO

    print("19h shift found")
    
    
    #--------- Run the model with exogenous melatonin 20h after DLMO (Clock time: 17:00)---------------

    # Burgess 2010, exogenous melatonin given at start of wake episode 
    model_20 = HannayBreslowModel()
    model_20.set_melatonin_prc_params(params)
    model_20.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=20, melatonin_dosage=0.5,schedule=2)

    # Calculate shift due to protocol
    # Baseline day DLMO determined above

    # Final day 
    final_plasma_mel = model_20.results[960:1199,4]/4.3 # converting output to pg/mL
    final_times = model_20.ts[960:1199] # defining times from last 24hrs
    final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below threshold
    final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO

    # Calculate phase shift (baseline - final; negative = delay, positive = advance) 
    phase_shift_20 = baseline_DLMO - final_DLMO

    print("20h shift found")
    

    #--------- Run the model with exogenous melatonin 21h after DLMO (Clock time: 18:00)---------------

    # Burgess 2010, exogenous melatonin given at start of wake episode 
    model_21 = HannayBreslowModel()
    model_21.set_melatonin_prc_params(params)
    model_21.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=21, melatonin_dosage=0.5,schedule=2)

    # Calculate shift due to protocol
    # Baseline day DLMO determined above

    # Final day 
    final_plasma_mel = model_21.results[960:1199,4]/4.3 # converting output to pg/mL
    final_times = model_21.ts[960:1199] # defining times from last 24hrs
    final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below threshold
    final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO

    # Calculate phase shift (baseline - final; negative = delay, positive = advance) 
    phase_shift_21 = baseline_DLMO - final_DLMO

    print("21h shift found")
    

    #--------- Run the model with exogenous melatonin 22h after DLMO (Clock time: 19:00)---------------

    # Burgess 2010, exogenous melatonin given at start of wake episode 
    model_22 = HannayBreslowModel()
    model_22.set_melatonin_prc_params(params)
    model_22.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=22, melatonin_dosage=0.5,schedule=2)

    # Calculate shift due to protocol
    # Baseline day DLMO determined above

    # Final day 
    final_plasma_mel = model_22.results[960:1199,4]/4.3 # converting output to pg/mL
    final_times = model_22.ts[960:1199] # defining times from last 24hrs
    final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below threshold
    final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO

    # Calculate phase shift (baseline - final; negative = delay, positive = advance) 
    phase_shift_22 = baseline_DLMO - final_DLMO

    print("22h shift found")
    

    #--------- Run the model with exogenous melatonin 23h after DLMO (clock time: 20:00)---------------

    # Burgess 2010, exogenous melatonin given at start of wake episode 
    model_23 = HannayBreslowModel()
    model_23.set_melatonin_prc_params(params)
    model_23.integrateModel(24*5,tstart=0.0,initial=IC, melatonin_timing=23, melatonin_dosage=0.5,schedule=2)

    # Calculate shift due to protocol
    # Baseline day DLMO determined above

    # Final day 
    final_plasma_mel = model_23.results[960:1199,4]/4.3 # converting output to pg/mL
    final_times = model_23.ts[960:1199] # defining times from last 24hrs
    final, = np.where(final_plasma_mel<=DLMO_threshold) # finding all the indices where concentration is below threshold
    final_DLMO = np.mod(final_times[final[-1]],24) # finding the time corresponding to the first index below threshold, DLMO

    # Calculate phase shift (baseline - final; negative = delay, positive = advance) 
    phase_shift_23 = baseline_DLMO - final_DLMO

    print("23h shift found")
    

    #---------- Make an array of all predicted phase shifts --------------
    # n = 34
    
    phase_shifts = [phase_shift_0,
                    phase_shift_1,
                    phase_shift_2,
                    phase_shift_2,
                    phase_shift_2, 
                    phase_shift_3, 
                    phase_shift_5, 
                    phase_shift_6, 
                    phase_shift_7,
                    phase_shift_7,
                    phase_shift_8,
                    phase_shift_10,
                    phase_shift_10,
                    phase_shift_11,
                    phase_shift_11,
                    phase_shift_12,
                    phase_shift_12,
                    phase_shift_13,
                    phase_shift_13,
                    phase_shift_13,
                    phase_shift_16,
                    phase_shift_17,
                    phase_shift_19,
                    phase_shift_19,
                    phase_shift_20,
                    phase_shift_20,
                    phase_shift_21,
                    phase_shift_21,
                    phase_shift_21,
                    phase_shift_21,
                    phase_shift_22,
                    phase_shift_22,
                    phase_shift_23,
                    phase_shift_23
                    ];

    # Subtract off the shift due to the protocol
    phase_shifts_corrected = (phase_shifts - phase_shift_placebo)
    
    #print(phase_shifts_corrected)
    
    return phase_shifts_corrected
    


def objective_func(params,data_vals):
    try: 
        print(params)
        phase_shifts = run_HannayBreslow_PRC(params)
        print(phase_shifts)
        
        a = np.array(phase_shifts)
        b = np.array(data_vals)
        
        difference = a - b
        print(difference)
    
        #error = np.mean(abs(phase_shifts - data_vals))
        error = np.linalg.norm(difference)
        
        print(error)
    
        return error
    except: 
        return 1e6
