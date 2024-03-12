# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 17:50:09 2024

@author: sstowe

Same as HannayBreslow_Zeitzer2000 but suppression is calculated as an AUC 

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
        
        self.M_max = 0.019513 # Breslow 2013
        self.H_sat = 861 # Breslow 2013
        self.sigma_M = 50 # Breslow 2013
        # Differential evolution: 
            # Switched to L2 norm
            # Fitting to data (determined from WebPlotDigitizer) with AUC
                # 4.92083843, Error = 0.8007471773658505
                
                # 4.71470534, Error = 0.7979262129459022
        self.m = 4.9208 # I determined by fitting to Zeitzer using differential evolution
        
        
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
        elif schedule == 2: # Constant conditions with sleep opportunities  
            full_light = 150
            dim_light = 150
            wake_time = 7
            sleep_time = 23
            sun_up = 8
            sun_down = 19
            
            is_awake = np.mod(t - wake_time,24) <= np.mod(sleep_time - wake_time,24)
            sun_is_up = np.mod(t - sun_up,24) <= np.mod(sun_down - sun_up,24)

            return is_awake*(full_light*sun_is_up + dim_light*(1 - sun_is_up))
        else: # Light pulse timed based on CBTmin
            bright_light = light_pulse
            dim_light = 10
            if 0 <= t < 24:
                if t > np.mod(CBTmin - 6.75,24):
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





#--------- Run the model to find initial conditions ---------------

model_IC = HannayBreslowModel() # defining model as a new object built with the HannayBreslowModel class 
model_IC.integrateModel(24*50,schedule=1) # use the integrateModel method with the object model
IC = model_IC.results[-1,:] # get initial conditions from entrained model





#--------- Run the model under constant conditions ---------------
# Find baseline CBTmin

# Run Zeitzer 2000 constant routine 
model_baseline = HannayBreslowModel()
model_baseline.integrateModel(24*3,tstart=0.0,initial=IC,schedule=2) # run the model from entrained ICs
IC_2 = model_baseline.results[-1,:] 


last_day = model_baseline.results[480:720,1]
last_day = np.mod(last_day,2*np.pi)
last_day = np.around(last_day, 2)
last_day_list = last_day.tolist()

try: 
    CBTmin_index = last_day_list.index(3.14)
except: 
    print("3.14 not found")
    
try: 
    CBTmin_index = last_day_list.index(3.13)
except: 
    print("3.13 not found")
    
try: 
    CBTmin_index = last_day_list.index(3.15)
except: 
    print("3.15 not found")

CBTmin_index = CBTmin_index + 480

CBTmin = model_baseline.ts[CBTmin_index]



#-------------- Run the model with a 6.5 h light exposure (3 lux) -----------

# Run Zeitzer 2000 light pulse  
model_light_3 = HannayBreslowModel()
model_light_3.integrateModel(24*2,tstart=0.0,initial=IC_2,schedule=3,light_pulse=3) # run the model from baseline ICs

# Calculate % suppression as the AUC 
before = np.trapz(model_light_3.results[0:40,4])
during = np.trapz(model_light_3.results[240:280,4])

percent_supp_3 = (before - during)/before 


#-------------- Run the model with a 6.5 h light exposure (15 lux) -----------

# Run Zeitzer 2000 light pulse  
model_light_15 = HannayBreslowModel()
model_light_15.integrateModel(24*2,tstart=0.0,initial=IC_2,schedule=3,light_pulse=15) # run the model from baseline ICs

# Calculate % suppression as the AUC 
before = np.trapz(model_light_15.results[0:40,4])
during = np.trapz(model_light_15.results[240:280,4])

percent_supp_15 = (before - during)/before


#-------------- Run the model with a 6.5 h light exposure (20 lux) -----------

# Run Zeitzer 2000 light pulse  
model_light_20 = HannayBreslowModel()
model_light_20.integrateModel(24*2,tstart=0.0,initial=IC_2,schedule=3,light_pulse=20) # run the model from baseline ICs

# Calculate % suppression as the AUC 
before = np.trapz(model_light_20.results[0:40,4])
during = np.trapz(model_light_20.results[240:280,4])

percent_supp_20 = (before - during)/before


#-------------- Run the model with a 6.5 h light exposure (50 lux) -----------

# Run Zeitzer 2000 light pulse  
model_light_50 = HannayBreslowModel()
model_light_50.integrateModel(24*2,tstart=0.0,initial=IC_2,schedule=3,light_pulse=50) # run the model from baseline ICs

# Calculate % suppression as the AUC 
before = np.trapz(model_light_50.results[0:40,4])
during = np.trapz(model_light_50.results[240:280,4])

percent_supp_50 = (before - during)/before 


#-------------- Run the model with a 6.5 h light exposure (60 lux) -----------

# Run Zeitzer 2000 light pulse  
model_light_60 = HannayBreslowModel()
model_light_60.integrateModel(24*2,tstart=0.0,initial=IC_2,schedule=3,light_pulse=60) # run the model from baseline ICs

# Calculate % suppression as the AUC 
before = np.trapz(model_light_60.results[0:40,4])
during = np.trapz(model_light_60.results[240:280,4])

percent_supp_60 = (before - during)/before 


#-------------- Run the model with a 6.5 h light exposure (90 lux) -----------

# Run Zeitzer 2000 light pulse  
model_light_90 = HannayBreslowModel()
model_light_90.integrateModel(24*2,tstart=0.0,initial=IC_2,schedule=3,light_pulse=90) # run the model from baseline ICs

# Calculate % suppression as the AUC 
before = np.trapz(model_light_90.results[0:40,4])
during = np.trapz(model_light_90.results[240:280,4])

percent_supp_90 = (before - during)/before 


#-------------- Run the model with a 6.5 h light exposure (106 lux) -----------

# Run Zeitzer 2000 light pulse  
model_light_106 = HannayBreslowModel()
model_light_106.integrateModel(24*2,tstart=0.0,initial=IC_2,schedule=3,light_pulse=106) # run the model from baseline ICs

# Calculate % suppression as the AUC 
before = np.trapz(model_light_106.results[0:40,4])
during = np.trapz(model_light_106.results[240:280,4])

percent_supp_106 = (before - during)/before 


#-------------- Run the model with a 6.5 h light exposure (130 lux) -----------

# Run Zeitzer 2000 light pulse  
model_light_130 = HannayBreslowModel()
model_light_130.integrateModel(24*2,tstart=0.0,initial=IC_2,schedule=3,light_pulse=130) # run the model from baseline ICs

# Calculate % suppression as the AUC 
before = np.trapz(model_light_130.results[0:40,4])
during = np.trapz(model_light_130.results[240:280,4])

percent_supp_130 = (before - during)/before


#-------------- Run the model with a 6.5 h light exposure (160 lux) -----------

# Run Zeitzer 2000 light pulse  
model_light_160 = HannayBreslowModel()
model_light_160.integrateModel(24*2,tstart=0.0,initial=IC_2,schedule=3,light_pulse=160) # run the model from baseline ICs

# Calculate % suppression as the AUC 
before = np.trapz(model_light_160.results[0:40,4])
during = np.trapz(model_light_160.results[240:280,4])

percent_supp_160 = (before - during)/before 

#-------------- Run the model with a 6.5 h light exposure (175 lux) -----------

# Run Zeitzer 2000 light pulse  
model_light_175 = HannayBreslowModel()
model_light_175.integrateModel(24*2,tstart=0.0,initial=IC_2,schedule=3,light_pulse=175) # run the model from baseline ICs

# Calculate % suppression as the AUC 
before = np.trapz(model_light_175.results[0:40,4])
during = np.trapz(model_light_175.results[240:280,4])

percent_supp_175 = (before - during)/before 


#-------------- Run the model with a 6.5 h light exposure (300 lux) -----------

# Run Zeitzer 2000 light pulse  
model_light_300 = HannayBreslowModel()
model_light_300.integrateModel(24*2,tstart=0.0,initial=IC_2,schedule=3,light_pulse=300) # run the model from baseline ICs

# Calculate % suppression as the AUC 
before = np.trapz(model_light_300.results[0:40,4])
during = np.trapz(model_light_300.results[240:280,4])

percent_supp_300 = (before - during)/before 


#-------------- Run the model with a 6.5 h light exposure (375 lux) -----------

# Run Zeitzer 2000 light pulse  
model_light_375 = HannayBreslowModel()
model_light_375.integrateModel(24*2,tstart=0.0,initial=IC_2,schedule=3,light_pulse=375) # run the model from baseline ICs

# Calculate % suppression as the AUC 
before = np.trapz(model_light_375.results[0:40,4])
during = np.trapz(model_light_375.results[240:280,4])

percent_supp_375 = (before - during)/before 


#-------------- Run the model with a 6.5 h light exposure (400 lux) -----------

# Run Zeitzer 2000 light pulse  
model_light_400 = HannayBreslowModel()
model_light_400.integrateModel(24*2,tstart=0.0,initial=IC_2,schedule=3,light_pulse=400) # run the model from baseline ICs

# Calculate % suppression as the AUC 
before = np.trapz(model_light_400.results[0:40,4])
during = np.trapz(model_light_400.results[240:280,4])

percent_supp_400 = (before - during)/before 


#-------------- Run the model with a 6.5 h light exposure (550 lux) -----------

# Run Zeitzer 2000 light pulse  
model_light_550 = HannayBreslowModel()
model_light_550.integrateModel(24*2,tstart=0.0,initial=IC_2,schedule=3,light_pulse=550) # run the model from baseline ICs

# Calculate % suppression as the AUC 
before = np.trapz(model_light_550.results[0:40,4])
during = np.trapz(model_light_550.results[240:280,4])

percent_supp_550 = (before - during)/before 


#-------------- Run the model with a 6.5 h light exposure (650 lux) -----------

# Run Zeitzer 2000 light pulse  
model_light_650 = HannayBreslowModel()
model_light_650.integrateModel(24*2,tstart=0.0,initial=IC_2,schedule=3,light_pulse=650) # run the model from baseline ICs

# Calculate % suppression as the AUC 
before = np.trapz(model_light_650.results[0:40,4])
during = np.trapz(model_light_650.results[240:280,4])

percent_supp_650 = (before - during)/before 


#-------------- Run the model with a 6.5 h light exposure (1500 lux) -----------

# Run Zeitzer 2000 light pulse  
model_light_1500 = HannayBreslowModel()
model_light_1500.integrateModel(24*2,tstart=0.0,initial=IC_2,schedule=3,light_pulse=1500) # run the model from baseline ICs

# Calculate % suppression as the AUC 
before = np.trapz(model_light_1500.results[0:40,4])
during = np.trapz(model_light_1500.results[240:280,4])

percent_supp_1500 = (before - during)/before 


#-------------- Run the model with a 6.5 h light exposure (3000 lux) -----------

# Run Zeitzer 2000 light pulse  
model_light_3000 = HannayBreslowModel()
model_light_3000.integrateModel(24*2,tstart=0.0,initial=IC_2,schedule=3,light_pulse=3000) # run the model from baseline ICs

# Calculate % suppression as the AUC 
before = np.trapz(model_light_3000.results[0:40,4])
during = np.trapz(model_light_3000.results[240:280,4])

percent_supp_3000 = (before - during)/before 


#-------------- Run the model with a 6.5 h light exposure (3700 lux) -----------

# Run Zeitzer 2000 light pulse  
model_light_3700 = HannayBreslowModel()
model_light_3700.integrateModel(24*2,tstart=0.0,initial=IC_2,schedule=3,light_pulse=3700) # run the model from baseline ICs

# Calculate % suppression as the AUC 
before = np.trapz(model_light_3700.results[0:40,4])
during = np.trapz(model_light_3700.results[240:280,4])

percent_supp_3700 = (before - during)/before 


#-------------- Run the model with a 6.5 h light exposure (7000 lux) -----------

# Run Zeitzer 2000 light pulse  
model_light_7000 = HannayBreslowModel()
model_light_7000.integrateModel(24*2,tstart=0.0,initial=IC_2,schedule=3,light_pulse=7000) # run the model from baseline ICs

# Calculate % suppression as the AUC 
before = np.trapz(model_light_7000.results[0:40,4])
during = np.trapz(model_light_7000.results[240:280,4])

percent_supp_7000 = (before - during)/before 


#-------------- Run the model with a 6.5 h light exposure (8100 lux) -----------

# Run Zeitzer 2000 light pulse  
model_light_8100 = HannayBreslowModel()
model_light_8100.integrateModel(24*2,tstart=0.0,initial=IC_2,schedule=3,light_pulse=8100) # run the model from baseline ICs

# Calculate % suppression as the AUC 
before = np.trapz(model_light_8100.results[0:40,4])
during = np.trapz(model_light_8100.results[240:280,4])

percent_supp_8100 = (before - during)/before 


#-------------- Run the model with a 6.5 h light exposure (9100 lux) -----------

# Run Zeitzer 2000 light pulse  
model_light_9100 = HannayBreslowModel()
model_light_9100.integrateModel(24*2,tstart=0.0,initial=IC_2,schedule=3,light_pulse=9100) # run the model from baseline ICs

# Calculate % suppression as the AUC 
before = np.trapz(model_light_9100.results[0:40,4])
during = np.trapz(model_light_9100.results[240:280,4])

percent_supp_9100 = (before - during)/before 


#---------- Make an array of all predicted % suppressions --------------

melatonin_suppression = [percent_supp_3, percent_supp_15, percent_supp_20, percent_supp_50, percent_supp_60, percent_supp_90, percent_supp_106, percent_supp_130, percent_supp_160, percent_supp_175, percent_supp_300, percent_supp_375, percent_supp_400, percent_supp_550, percent_supp_650, percent_supp_1500, percent_supp_3000, percent_supp_3700, percent_supp_7000, percent_supp_8100, percent_supp_9100];


# Make array of administration times for plotting PRC 
Lux = [3,15,25,50,60,106,120,130,170,175,300,360,400,550,650,2000,3000,4000,7000,9000,9100]

# Plot PRC points
#plt.plot(Lux,melatonin_suppression,'o')
plt.plot(Lux,melatonin_suppression, lw=2)
#plt.axhline(0)
#plt.axvline(20.9)
plt.xlabel("Light Exposure (Lux)")
plt.ylabel("Melatonin Suppression (%)")
plt.title("Simumalted Illuminance Response Curve")
plt.show()


#----------- Load Burgess 2008 PRC data -------------
# Fitting to the curve 
'''
Zeitzer_2000 = [
0, # 3
0, # 10
0, # 20
0, # 30
0, # 50
0.1, # 60
0.2, # 70
0.3, # 80
0.5, # 100
0.7, # 150
0.8, # 180
0.9, # 300
0.9, # 500
0.95, # 1000
0.95 # 5000
]

# Eyeballed it
Lux_2 = [3,10,20,30,50,60,70,80,100,150,180,300,500,1000,5000]
'''

# Fitting to the data (determined from WebPlotDigitizer)
# n = 21
Zeitzer_2000 = [0.11,
0.08,
-0.08,
-0.01,
-0.12,
0.22,
0.88,
0.26,
0.80,
0.75,
0.69,
0.85,
0.96,
0.99,
0.92,
0.96,
0.94,
0.97,
0.98,
0.95,
0.98
]

Lux_2 = [3,
15,
20,
50,
60,
90,
106,
130,
160,
175,
300,
375,
400,
550,
650,
1500,
3000,
3700,
7000,
8100,
9100
]

plt.plot(Lux_2, Zeitzer_2000, 'o')
plt.plot(Lux,melatonin_suppression, lw=2)
plt.xscale('log')
plt.title("Illuminance Response Curve")
plt.xlabel("Illuminance (lux)")
plt.ylabel("Melatonin Suppression (%)")
plt.legend(["Zeitzer 2000","Model"])
plt.show()


#--------- Plot Model Output -------------------

# pick one to plot 
#model = model_baseline
model = model_light_3
#model = model_light_15
#model = model_light_25
#model = model_light_50
#model = model_light_60
#model = model_light_106
#model = model_light_120
#model = model_light_130
#model = model_light_170
#model = model_light_175
#model = model_light_300
#model = model_light_360
#model = model_light_400
#model = model_light_550
#model = model_light_650
#model = model_light_2000
#model = model_light_3000
#model = model_light_4000
#model = model_light_7000
#model = model_light_9000
#model = model_light_9100


# Plotting H1, H2, and H3 (melatonin concentrations, pmol/L)
plt.plot(model.ts,model.results[:,3],lw=2)
plt.plot(model.ts,model.results[:,4],lw=2)
plt.plot(model.ts,model.results[:,5],lw=2)
plt.axvline(x=0)
plt.axvline(x=4)
#plt.axvline(x=21.45)
#plt.axvline(x=27.95)
#plt.axvline(x=28.2)
plt.axvline(x=24)
plt.axvline(x=28)
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
#plt.axhline(DLMO_threshold)
plt.xlabel("Time (hours)")
plt.ylabel("Melatonin Concentration (pg/mL)")
plt.title("Zeitzer 2000 Protocol - 3 lux")
plt.axvspan(21.45, 3.95+(24*1), facecolor='y', alpha=0.4)
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
#plt.axvline(x=21.75)
#plt.axvline(x=28.25)
plt.xlabel("Time (hours)")
plt.ylabel("Proportion of Activated Photoreceptors")
plt.title("Time Trace of Photoreceptor Activation")
plt.show()



