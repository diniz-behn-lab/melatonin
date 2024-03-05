# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 11:49:43 2024

@author: sstowe

Hannay-Breslow Model code to be imported into the differential evolution code
(DifferentialEvolution_Zeitzer2000.py) to optimize the parameter "m"

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

        # TO BE OPTIMIZED
    def set_m_param(self, theta):
        self.m = theta[0] 
        
        print("Parameter m is set")
        print(theta)


# Set parameter values 
    def set_params(self):
        ## Breslow Model
        self.beta_IP = (7.83e-4)*60*60 # converting 1/sec to 1/hr, Breslow 2013
        self.beta_CP = (3.35e-4)*60*60 # converting 1/sec to 1/hr, Breslow 2013
        self.beta_AP = (1.62e-4)*60*60 # converting 1/sec to 1/hr, Breslow 2013

        self.a = (0.1)*60*60 #(0.0675)*60*60 # CHANGED, pmol/L/sec converted to hours, I determined
        self.delta_M = 600 # sec, Breslow 2013
        self.r = 15.36 # sec, Breslow 2013
        
        self.psi_on = 1.0472 # radians, I determined 
        self.psi_off = 3.92699 # radians, I determined
        
        self.M_max = 0.019513 # Breslow 2013
        self.H_sat = 861 # Breslow 2013
        self.sigma_M = 50 # Breslow 2013
        self.m = 5 #7 # 1/sec, I determined by fitting (roughly) to Zeitzer
        
        
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
    def light(self,t,schedule,light_pulse):
        
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
        elif schedule == 2: # Constant conditions 
            full_light = 150
            dim_light = 150
            wake_time = 7
            sleep_time = 23
            sun_up = 8
            sun_down = 19
            
            is_awake = np.mod(t - wake_time,24) <= np.mod(sleep_time - wake_time,24)
            sun_is_up = np.mod(t - sun_up,24) <= np.mod(sun_down - sun_up,24)

            return is_awake*(full_light*sun_is_up + dim_light*(1 - sun_is_up))
        else: 
            bright_light = light_pulse
            dim_light = 10
            CBTmin = 52.2
            if 0 <= t < 24:
                if t > np.mod(CBTmin-6.75,24):
                    return bright_light
                else: 
                    return dim_light
            else:
                if t < np.mod(CBTmin - 0.25,24)+24:
                    return bright_light
                else:
                    return dim_light


# Define the alpha(L) function 
    def alpha0(self,t,schedule,light_pulse):
        """A helper function for modeling the light input processing"""
        
        return(self.alpha_0*pow(self.light(t,schedule,light_pulse), self.p)/(pow(self.light(t,schedule,light_pulse), self.p)+self.I_0));

        

# Defining the system of ODEs (2-dimensional system)
    def ODESystem(self,t,y,schedule,light_pulse):
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
        Bhat = self.G*(1.0-n)*self.alpha0(t,schedule,light_pulse)
    
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
        dydt[2] = 60.0*(self.alpha0(t,schedule,light_pulse)*(1.0-n)-(self.delta*n)) # dn/dt
        
        dydt[3] = -self.beta_IP*H1 + A*(1 - self.m*Bhat)*S # dH1/dt
        dydt[4] = self.beta_IP*H1 - self.beta_CP*H2 + self.beta_AP*H3 # dH2/dt
        dydt[5] = -self.beta_AP*H3 # dH3/dt

        return(dydt)



    def integrateModel(self, tend, tstart=0.0, initial=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], schedule = 1, light_pulse = 1000):
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
        
        r_variable = sp.integrate.solve_ivp(self.ODESystem,(tstart,tend), initial, t_eval=self.ts, method='Radau', args=(schedule,light_pulse,))
        self.results = np.transpose(r_variable.y)

        return
#-------- end of HannayBreslowModel class ---------




def run_HannayBreslow_MelSupp(params):

    IC_2 = [0.810081, 2.02662, 0.348864, 121.033, 192.803, 0]    

    #-------------- Run the model with a 6.5 h light exposure (3 lux) -----------

    # Run Zeitzer 2000 light pulse  
    model_light_3 = HannayBreslowModel()
    model_light_3.integrateModel(24*2,tstart=0.0,initial=IC_2,schedule=3,light_pulse=3) # run the model from baseline ICs

    # Calculate % suppression 
    before = model_light_3.results[30,4]
    during = model_light_3.results[270,4]

    percent_supp_3 = (before - during)/before 

    print("3 lux suppression found") 


    #-------------- Run the model with a 6.5 h light exposure (10 lux) -----------
    
    # Run Zeitzer 2000 light pulse  
    model_light_10 = HannayBreslowModel()
    model_light_10.integrateModel(24*2,tstart=0.0,initial=IC_2,schedule=3,light_pulse=10) # run the model from baseline ICs
    
    # Calculate % suppression 
    before = model_light_10.results[30,4]
    during = model_light_10.results[270,4]
    
    percent_supp_10 = (before - during)/before 
    
    print("10 lux suppression found") 
    

    #-------------- Run the model with a 6.5 h light exposure (50 lux) -----------
    
    # Run Zeitzer 2000 light pulse  
    model_light_50 = HannayBreslowModel()
    model_light_50.integrateModel(24*2,tstart=0.0,initial=IC_2,schedule=3,light_pulse=50) # run the model from baseline ICs
    
    # Calculate % suppression 
    before = model_light_50.results[30,4]
    during = model_light_50.results[270,4]
    
    percent_supp_50 = (before - during)/before 
    
    print("50 lux suppression found") 
    
    
    #-------------- Run the model with a 6.5 h light exposure (60 lux) -----------
    
    # Run Zeitzer 2000 light pulse  
    model_light_60 = HannayBreslowModel()
    model_light_60.integrateModel(24*2,tstart=0.0,initial=IC_2,schedule=3,light_pulse=60) # run the model from baseline ICs
    
    # Calculate % suppression 
    before = model_light_60.results[30,4]
    during = model_light_60.results[270,4]
    
    percent_supp_60 = (before - during)/before 
    
    print("60 lux suppression found") 
    
    
    #-------------- Run the model with a 6.5 h light exposure (80 lux) -----------
    
    # Run Zeitzer 2000 light pulse  
    model_light_80 = HannayBreslowModel()
    model_light_80.integrateModel(24*2,tstart=0.0,initial=IC_2,schedule=3,light_pulse=80) # run the model from baseline ICs
    
    # Calculate % suppression 
    before = model_light_80.results[30,4]
    during = model_light_80.results[270,4]
    
    percent_supp_80 = (before - during)/before 
    
    print("80 lux suppression found") 
    
    
    #-------------- Run the model with a 6.5 h light exposure (100 lux) -----------
    
    # Run Zeitzer 2000 light pulse  
    model_light_100 = HannayBreslowModel()
    model_light_100.integrateModel(24*2,tstart=0.0,initial=IC_2,schedule=3,light_pulse=100) # run the model from baseline ICs
    
    # Calculate % suppression 
    before = model_light_100.results[30,4]
    during = model_light_100.results[270,4]
    
    percent_supp_100 = (before - during)/before 
    
    print("100 lux suppression found") 
    
    
    #-------------- Run the model with a 6.5 h light exposure (200 lux) -----------
    
    # Run Zeitzer 2000 light pulse  
    model_light_200 = HannayBreslowModel()
    model_light_200.integrateModel(24*2,tstart=0.0,initial=IC_2,schedule=3,light_pulse=200) # run the model from baseline ICs
    
    # Calculate % suppression 
    before = model_light_200.results[30,4]
    during = model_light_200.results[270,4]
    
    percent_supp_200 = (before - during)/before 
    
    print("200 lux suppression found") 
    
    
    #-------------- Run the model with a 6.5 h light exposure (500 lux) -----------
    
    # Run Zeitzer 2000 light pulse  
    model_light_500 = HannayBreslowModel()
    model_light_500.integrateModel(24*2,tstart=0.0,initial=IC_2,schedule=3,light_pulse=500) # run the model from baseline ICs
    
    # Calculate % suppression 
    before = model_light_500.results[30,4]
    during = model_light_500.results[270,4]
    
    percent_supp_500 = (before - during)/before 
    
    print("500 lux suppression found") 
    
    
    #-------------- Run the model with a 6.5 h light exposure (1000 lux) -----------
    
    # Run Zeitzer 2000 light pulse  
    model_light_1000 = HannayBreslowModel()
    model_light_1000.integrateModel(24*2,tstart=0.0,initial=IC_2,schedule=3,light_pulse=1000) # run the model from baseline ICs
    
    # Calculate % suppression 
    before = model_light_1000.results[30,4]
    during = model_light_1000.results[270,4]
    
    percent_supp_1000 = (before - during)/before 
    
    print("1000 lux suppression found") 
    
    
    #-------------- Run the model with a 6.5 h light exposure (5000 lux) -----------
    
    # Run Zeitzer 2000 light pulse  
    model_light_5000 = HannayBreslowModel()
    model_light_5000.integrateModel(24*2,tstart=0.0,initial=IC_2,schedule=3,light_pulse=5000) # run the model from baseline ICs
    
    # Calculate % suppression 
    before = model_light_5000.results[30,4]
    during = model_light_5000.results[270,4]
    
    percent_supp_5000 = (before - during)/before 
    
    print("5000 lux suppression found") 
    
    
    #---------- Make an array of all predicted phase shifts --------------
    
    melatonin_suppression = [percent_supp_3, percent_supp_10, percent_supp_50, percent_supp_60, percent_supp_80, percent_supp_100, percent_supp_200, percent_supp_500, percent_supp_1000, percent_supp_5000];
    
    return melatonin_suppression


def objective_func(params,data_vals):
    try: 
        print(params)
        melatonin_suppression = run_HannayBreslow_MelSupp(params)
        print(melatonin_suppression)
    
        error = np.mean(abs(melatonin_suppression - data_vals))
        print(error)
    
        return error
    except: 
        return 1e6