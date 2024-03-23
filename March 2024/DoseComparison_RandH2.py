# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 12:29:16 2024

@author: sstowe

Plotting R and H2 with different exogenous doses

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
            if 0 <= t < 24:
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



 
timing = 15
days = 1


#--------- Run the model to find initial conditions ---------------

model_IC = HannayBreslowModel() # defining model as a new object built with the HannayBreslowModel class 
model_IC.integrateModel(24*50,schedule=1) # use the integrateModel method with the object model
IC = model_IC.results[-1,:] # get initial conditions from entrained model



#--------- Run the model without exogenous melatonin ---------------

model_none = HannayBreslowModel()
model_none.integrateModel(24*days,tstart=0.0,initial=IC, melatonin_timing=None, melatonin_dosage=None,schedule=2) 



#--------- Run the model with 0.1 mg exogenous melatonin ---------------

model_01 = HannayBreslowModel()
model_01.integrateModel(24*days,tstart=0.0,initial=IC, melatonin_timing=timing, melatonin_dosage=0.1,schedule=2) 



#--------- Run the model with 0.3 mg exogenous melatonin ---------------

model_03 = HannayBreslowModel()
model_03.integrateModel(24*days,tstart=0.0,initial=IC, melatonin_timing=timing, melatonin_dosage=0.3,schedule=2) 



#--------- Run the model with 0.5 mg exogenous melatonin ---------------

model_05 = HannayBreslowModel()
model_05.integrateModel(24*days,tstart=0.0,initial=IC, melatonin_timing=timing, melatonin_dosage=0.5,schedule=2) 



#--------- Run the model with 1.0 mg exogenous melatonin ---------------

model_1 = HannayBreslowModel()
model_1.integrateModel(24*days,tstart=0.0,initial=IC, melatonin_timing=timing, melatonin_dosage=1.0,schedule=2) 



#--------- Run the model with 2.0 mg exogenous melatonin ---------------

model_2 = HannayBreslowModel()
model_2.integrateModel(24*days,tstart=0.0,initial=IC, melatonin_timing=timing, melatonin_dosage=2.0,schedule=2) 



#--------- Run the model with 3.0 mg exogenous melatonin ---------------

model_3 = HannayBreslowModel()
model_3.integrateModel(24*days,tstart=0.0,initial=IC, melatonin_timing=timing, melatonin_dosage=3.0,schedule=2) 



#--------- Run the model with 4.0 mg exogenous melatonin ---------------

model_4 = HannayBreslowModel()
model_4.integrateModel(24*days,tstart=0.0,initial=IC, melatonin_timing=timing, melatonin_dosage=4.0,schedule=2) 



#--------- Run the model with 5.0 mg exogenous melatonin ---------------

model_5 = HannayBreslowModel()
model_5.integrateModel(24*days,tstart=0.0,initial=IC, melatonin_timing=timing, melatonin_dosage=5.0,schedule=2) 



#--------- Run the model with 6.0 mg exogenous melatonin ---------------

model_6 = HannayBreslowModel()
model_6.integrateModel(24*days,tstart=0.0,initial=IC, melatonin_timing=timing, melatonin_dosage=6.0,schedule=2) 



#--------- Run the model with 10.0 mg exogenous melatonin ---------------

model_10 = HannayBreslowModel()
model_10.integrateModel(24*days,tstart=0.0,initial=IC, melatonin_timing=timing, melatonin_dosage=10.0,schedule=2) 



#--------- Plot Model Output -------------------




# Plotting H2 (melatonin concentration, pg/mL)
plt.plot(model_none.ts,model_none.results[:,4]/4.3,lw=2)
plt.plot(model_01.ts,model_01.results[:,4]/4.3,lw=2)
plt.plot(model_03.ts,model_03.results[:,4]/4.3,lw=2)
plt.plot(model_05.ts,model_05.results[:,4]/4.3,lw=2)
plt.plot(model_1.ts,model_1.results[:,4]/4.3,lw=2)
#plt.plot(model_2.ts,model_2.results[:,4]/4.3,lw=2)
plt.plot(model_3.ts,model_3.results[:,4]/4.3,lw=2)
#plt.plot(model_4.ts,model_4.results[:,4]/4.3,lw=2)
plt.plot(model_5.ts,model_5.results[:,4]/4.3,lw=2)
#plt.plot(model_6.ts,model_6.results[:,4]/4.3,lw=2)
#plt.plot(model_10.ts,model_10.results[:,4]/4.3,lw=2)
plt.axhline(70, linestyle='dashed',color = 'black')
plt.axvline(timing, color = 'black',lw=1)
plt.axvline(timing+1.5, color = 'black',lw=1)
plt.xlabel("Time (hours)")
plt.ylabel("Melatonin Concentration (pg/mL)")
plt.title("Plasma Melatonin Concentrations (pg/mL)")
#plt.legend(["None","0.1 mg", "0.3 mg", "0.5 mg", "1.0 mg", "2.0 mg", "3.0 mg"])
plt.legend(["None","0.1 mg", "0.3 mg", "0.5 mg", "1.0 mg", "3.0 mg","5.0 mg"])
plt.show()


# Plotting H2 (melatonin concentration, pg/mL), zoomed in
plt.plot(model_none.ts[timing*10:timing*10+100],model_none.results[timing*10:timing*10+100,4]/4.3,lw=2)
plt.plot(model_01.ts[timing*10:timing*10+100],model_01.results[timing*10:timing*10+100,4]/4.3,lw=2)
plt.plot(model_03.ts[timing*10:timing*10+100],model_03.results[timing*10:timing*10+100,4]/4.3,lw=2)
plt.plot(model_05.ts[timing*10:timing*10+100],model_05.results[timing*10:timing*10+100,4]/4.3,lw=2)
plt.plot(model_1.ts[timing*10:timing*10+100],model_1.results[timing*10:timing*10+100,4]/4.3,lw=2)
#plt.plot(model_2.ts[timing*10:timing*10+100],model_2.results[timing*10:timing*10+100,4]/4.3,lw=2)
plt.plot(model_3.ts[timing*10:timing*10+100],model_3.results[timing*10:timing*10+100,4]/4.3,lw=2)
#plt.plot(model_4.ts[timing*10:timing*10+100],model_4.results[timing*10:timing*10+100,4]/4.3,lw=2)
plt.plot(model_5.ts[timing*10:timing*10+100],model_5.results[timing*10:timing*10+100,4]/4.3,lw=2)
#plt.plot(model_6.ts[timing*10:timing*10+100],model_6.results[timing*10:timing*10+100,4]/4.3,lw=2)
#plt.plot(model_10.ts[timing*10:timing*10+100],model_10.results[timing*10:timing*10+100,4]/4.3,lw=2)
plt.axhline(70, linestyle='dashed',color = 'black')
plt.axvline(timing, color = 'black',lw=1)
plt.axvline(timing+1.5, color = 'black',lw=1)
plt.xlabel("Time (hours)")
plt.ylabel("Melatonin Concentration (pg/mL)")
plt.title("Plasma Melatonin Concentrations (pg/mL)")
#plt.legend(["None","0.1 mg", "0.3 mg", "0.5 mg", "1.0 mg", "2.0 mg", "3.0 mg"])
plt.legend(["None","0.1 mg", "0.3 mg", "0.5 mg", "1.0 mg", "3.0 mg","5.0 mg"])
plt.show()


# Plotting R
plt.plot(model_none.ts,model_none.results[:,0],lw=2)
plt.plot(model_01.ts,model_01.results[:,0],lw=2)
plt.plot(model_03.ts,model_03.results[:,0],lw=2)
plt.plot(model_05.ts,model_05.results[:,0],lw=2)
plt.plot(model_1.ts,model_1.results[:,0],lw=2)
#plt.plot(model_2.ts,model_2.results[:,0],lw=2)
plt.plot(model_3.ts,model_3.results[:,0],lw=2)
#plt.plot(model_4.ts,model_4.results[:,0],lw=2)
plt.plot(model_5.ts,model_5.results[:,0],lw=2)
#plt.plot(model_6.ts,model_6.results[:,0],lw=2)
plt.axvline(timing, color = 'black',lw=1)
plt.xlabel("Time (hours)")
plt.ylabel("R, Collective Amplitude")
plt.title("Time Trace of R, Collective Amplitude")
#plt.legend(["None","0.1 mg", "0.3 mg", "0.5 mg", "1.0 mg", "2.0 mg", "3.0 mg", "4.0 mg", "5.0 mg", "6.0 mg"])
plt.legend(["None","0.1 mg", "0.3 mg", "0.5 mg", "1.0 mg", "3.0 mg","5.0 mg"])
plt.show()



# Plotting R, zoomed in
plt.plot(model_none.ts[timing*10:timing*10+100],model_none.results[timing*10:timing*10+100,0],lw=2)
plt.plot(model_01.ts[timing*10:timing*10+100],model_01.results[timing*10:timing*10+100,0],lw=2)
plt.plot(model_03.ts[timing*10:timing*10+100],model_03.results[timing*10:timing*10+100,0],lw=2)
plt.plot(model_05.ts[timing*10:timing*10+100],model_05.results[timing*10:timing*10+100,0],lw=2)
plt.plot(model_1.ts[timing*10:timing*10+100],model_1.results[timing*10:timing*10+100,0],lw=2)
#plt.plot(model_2.ts[timing*10:timing*10+100],model_2.results[timing*10:timing*10+100,0],lw=2)
plt.plot(model_3.ts[timing*10:timing*10+100],model_3.results[timing*10:timing*10+100,0],lw=2)
#plt.plot(model_4.ts[timing*10:timing*10+100],model_4.results[timing*10:timing*10+100,0],lw=2)
plt.plot(model_5.ts[timing*10:timing*10+100],model_5.results[timing*10:timing*10+100,0],lw=2)
#plt.plot(model_6.ts[timing*10:timing*10+100],model_6.results[timing*10:timing*10+100,0],lw=2)
plt.xlabel("Time (hours)")
plt.ylabel("R, Collective Amplitude")
plt.title("Time Trace of R, Collective Amplitude")
#plt.legend(["None","0.1 mg", "0.3 mg", "0.5 mg", "1.0 mg", "2.0 mg", "3.0 mg", "4.0 mg", "5.0 mg", "6.0 mg"])
plt.legend(["None","0.1 mg", "0.3 mg", "0.5 mg", "1.0 mg", "3.0 mg","5.0 mg"])
plt.show()



# Plotting Psi
plt.plot(model_none.ts,np.mod(model_none.results[:,1],2*np.pi),lw=2)
plt.plot(model_01.ts,np.mod(model_01.results[:,1],2*np.pi),lw=2)
plt.plot(model_03.ts,np.mod(model_03.results[:,1],2*np.pi),lw=2)
plt.plot(model_05.ts,np.mod(model_05.results[:,1],2*np.pi),lw=2)
plt.plot(model_1.ts,np.mod(model_1.results[:,1],2*np.pi),lw=2)
plt.plot(model_2.ts,np.mod(model_2.results[:,1],2*np.pi),lw=2)
plt.plot(model_3.ts,np.mod(model_3.results[:,1],2*np.pi),lw=2)
plt.plot(model_4.ts,np.mod(model_4.results[:,1],2*np.pi),lw=2)
plt.plot(model_5.ts,np.mod(model_5.results[:,1],2*np.pi),lw=2)
plt.plot(model_6.ts,np.mod(model_6.results[:,1],2*np.pi),lw=2)
plt.axvline(timing, color = 'black',lw=1)
plt.xlabel("Time (hours)")
plt.ylabel("Psi, Mean Phase")
plt.title("Time Trace of Psi, Mean Phase")
plt.legend(["None","0.1 mg", "0.3 mg", "0.5 mg", "1.0 mg", "2.0 mg", "3.0 mg", "4.0 mg", "5.0 mg", "6.0 mg"],loc = 'upper left')
plt.show()



# Plotting Pineal Time Trace (pg/mL)
plt.plot(model_none.ts,model_none.results[:,3]/4.3,lw=2)
plt.plot(model_01.ts,model_01.results[:,3]/4.3,lw=2)
plt.plot(model_03.ts,model_03.results[:,3]/4.3,lw=2)
plt.plot(model_05.ts,model_05.results[:,3]/4.3,lw=2)
plt.plot(model_1.ts,model_1.results[:,3]/4.3,lw=2)
plt.plot(model_2.ts,model_2.results[:,3]/4.3,lw=2)
plt.plot(model_3.ts,model_3.results[:,3]/4.3,lw=2)
plt.plot(model_4.ts,model_4.results[:,3]/4.3,lw=2)
plt.plot(model_5.ts,model_5.results[:,3]/4.3,lw=2)
plt.plot(model_6.ts,model_6.results[:,3]/4.3,lw=2)
#plt.axhline(70, linestyle='dashed',color = 'black')
plt.axvline(timing, color = 'black',lw=1)
plt.xlabel("Time (hours)")
plt.ylabel("Melatonin Concentration (pg/mL)")
plt.title("Pineal Melatonin Concentrations (pg/mL)")
plt.legend(["None","0.1 mg", "0.3 mg", "0.5 mg", "1.0 mg", "2.0 mg", "3.0 mg"])
plt.show()
