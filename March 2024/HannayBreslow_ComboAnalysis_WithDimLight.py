# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:05:46 2024

@author: sstowe
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
        
        self.psi_on = 1.2217 # radians, I determined 
        self.psi_off = 3.5779  # radians, I determined
        
        self.M_max = 0.019513 # Breslow 2013
        self.H_sat = 301 # Scaled for our peak concentration #861 # Breslow 2013
        self.sigma_M = 17.5 # Scaled for our peak concentration #50 # Breslow 2013
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
        
        # Standard 16:8 schedule
        if schedule == 1:  
            full_light = 1000
            dim_light = 300
            wake_time = 7
            sleep_time = 23
            sun_up = 8
            sun_down = 19
            
            is_awake = np.mod(t - wake_time,24) <= np.mod(sleep_time - wake_time,24)
            sun_is_up = np.mod(t - sun_up,24) <= np.mod(sun_down - sun_up,24)

            return is_awake*(full_light*sun_is_up + dim_light*(1 - sun_is_up))
        
        # Constant Dark schedule
        elif schedule == 2: 
            
            return 0
        
        # Constant dim schedule
        elif schedule == 3:  
            
            return 5
        
        # Advance protocol intervention
        elif schedule == 4:
            dim_light = 5
            bright_light = 1000
            
            if 5+24 <= t <= 7+24: # 2 Hours before habitual wake 
                return bright_light 
            else:
                return dim_light
            
        # Delay protocol intervention     
        elif schedule == 5:
            dim_light = 5
            bright_light = 1000
            
            if 21+24 <= t <= 23+24: # 2 Hours before habitual bedtime 
                return bright_light
            else:
                return dim_light
        

# Define the alpha(L) function 
    def alpha0(self,t,schedule):
        """A helper function for modeling the light input processing"""
        
        return(self.alpha_0*pow(self.light(t,schedule), self.p)/(pow(self.light(t,schedule), self.p)+self.I_0));
    

# Set the exogenous melatonin administration schedule VERSION 5
    def ex_melatonin(self,t,melatonin_timing,melatonin_dosage):
        
        if melatonin_timing == None:
            return 0 # set exogenous melatonin to zero
        else: 
            if 0+24 < t < 24+24:
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
#model.integrateModel(24*3,tstart=0.0,initial=IC, melatonin_timing=None, melatonin_dosage=None,schedule=4) 




#--------- Run the model with exogenous melatonin ---------------
# Set melatonin_timing to a clock hour 
# Set melatonin dosage to a mg amount

model = HannayBreslowModel()
model.integrateModel(24*3,tstart=0.0,initial=IC, melatonin_timing=15, melatonin_dosage=3.0,schedule=3) 



#--------- Find Baseline and Final DLMO -----------

# By Hannay model definition 

## Baseline 
psi_mod2pi_baseline = np.mod(model.results[120:300,1],2*np.pi)

DLMO_index_baseline = min(range(len(psi_mod2pi_baseline)), key=lambda i: abs(psi_mod2pi_baseline[i]-1.30899))

DLMO_psi_baseline = model.ts[DLMO_index_baseline+120] # closest to psi = 1.30899

## Final 
psi_mod2pi_final = np.mod(model.results[600:720,1],2*np.pi)

DLMO_index_final = min(range(len(psi_mod2pi_final)), key=lambda i: abs(psi_mod2pi_final[i]-1.30899))
#CBTmin_index = min(range(len(psi_mod2pi)), key=lambda i: abs(psi_mod2pi[i]-3.14159))

DLMO_psi_final = model.ts[DLMO_index_final+600] # closest to psi = 1.30899
#CBTmin = model.ts[CBTmin_index+400] # closest to psi = 3.14159



# By threshold definition (10 pg/mL in plasma)
DLMO_threshold = 10

## Baseline 
plasma_mel_concentrations_baseline = model.results[120:300,4]/4.3 # converting output to pg/mL
times_baseline = model.ts[120:300] # defining times from first 24hrs 
plasma_mel_baseline, = np.where(plasma_mel_concentrations_baseline<=DLMO_threshold) # finding all the indices where concentration is below 10pg/mL
DLMO_H2_baseline = times_baseline[plasma_mel_baseline[-1]] # finding the time corresponding to the last index below threshold, DLMO


## Final
plasma_mel_concentrations_final = model.results[600:720,4]/4.3 # converting output to pg/mL
times_final = model.ts[600:720] # defining times from first 24hrs 
plasma_mel_final, = np.where(plasma_mel_concentrations_final<=DLMO_threshold) # finding all the indices where concentration is below 10pg/mL
DLMO_H2_final = times_final[plasma_mel_final[-1]] # finding the time corresponding to the last index below threshold, DLMO
#DLMOff = times[plasma_mel[0]] # finding the time corresponding to the first index below threshold, DLMOff


#------- Calculate Phase Shift --------------

# By Hannay model defintion 
DLMO_psi_shift = np.mod(DLMO_psi_baseline,24) - np.mod(DLMO_psi_final,24)


# By threshold definition 
DLMO_H2_shift = np.mod(DLMO_H2_baseline,24) - np.mod(DLMO_H2_final,24)




#--------- Plot Model Output -------------------

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
plt.plot(model.ts,model.results[:,5]/4.3,lw=3,color='hotpink')
#plt.axvline(x=DLMOff,color='black',linestyle='dotted') # Checking DLMOff
plt.axvline(x=DLMO_H2_baseline,color='black',linestyle='dashed') # Checking DLMO
plt.axvline(x=DLMO_H2_final,color='grey',linestyle='dashed')
#plt.axvline(x=20.6) # Checking pineal on 
#plt.axvline(x=5.7) # Checking pineal off
#plt.axvline(5.7+24)
#plt.axvline(CBTmin,color='grey')
plt.axhline(10, color='black',lw=1)
#plt.axvspan(5+24, 7+24, facecolor='gold', alpha=0.4)
plt.xlabel("Clock Time (hours)")
plt.ylabel("Melatonin Concentration (pg/mL)")
#plt.title("Melatonin Concentrations (pg/mL)")
plt.legend(["Pineal","Plasma","Exogenous","Baseline DLMO","Final DLMO"])
plt.ylim([0,100])
plt.show()



# Plotting R
plt.plot(model.ts,model.results[:,0],lw=3,color='forestgreen')
#plt.axvline(x=20.6)
#plt.axvline(x=5.7)
#plt.axvline(x=8.1)
#plt.axvline(5.7+24)
plt.axvline(15+24,lw=3,color='hotpink')
#plt.axvspan(5+24, 7+24, facecolor='gold', alpha=0.4)
plt.xlabel("Clock Time (hours)")
plt.ylabel("R, Collective Amplitude")
#plt.title("Time Trace of R, Collective Amplitude")
plt.show()



# Plotting psi
plt.plot(model.ts,model.results[:,1],lw=2)
plt.xlabel("Time (hours)")
plt.ylabel("Psi, Mean Phase (radians)")
plt.title("Time Trace of Psi, Mean Phase")
plt.show()

# Plotting psi mod 2pi
plt.axvline(x=DLMO_psi_baseline,color='black',linestyle='dashed') # Checking DLMO
plt.axvline(x=DLMO_psi_final,color='grey',linestyle='dashed')
plt.plot(model.ts,np.mod(model.results[:,1],2*np.pi),'.',markersize=10,color='mediumturquoise')
plt.axhline(5*np.pi/12, color='black',lw=1)
#plt.axhline(np.pi)
#plt.axvline(CBTmin)
#plt.axvline(x=20.6) # Checking pineal on 
#plt.axvline(x=5.7) # Checking pineal off
#plt.axvline(x=8.1)
plt.axvline(15+24,lw=3,color='hotpink')
#plt.axvspan(5+24, 7+24, facecolor='gold', alpha=0.4)
plt.xlabel("Clock Time (hours)")
plt.ylabel("Psi, Mean Phase (radians)")
plt.legend(["Baseline DLMO", "Final DlMO"])
#plt.title("Time Trace of Psi, Mean Phase")
plt.show()



# Plotting n
plt.plot(model.ts,model.results[:,2],lw=3,color='goldenrod')
#plt.axvline(x=7)
#plt.axvline(x=23)
plt.axvline(15+24,lw=3,color='hotpink')
plt.xlabel("Clock Time (hours)")
plt.ylabel("n, Proportion of Activated Photoreceptors")
#plt.title("Time Trace of Photoreceptor Activation")
plt.show()

