from HCRSimPY.models import SinglePopModel
import numpy as np
import scipy as sp

from scipy import integrate

class IntegratedModel(SinglePopModel):
    """Driver of ForwardModel simulation"""

    def __init__(self, LightFun):
        super().__init__(LightFun)
        self.set_melatonin_params()

    def set_melatonin_prc_params(self, theta):
        self.epsilon = theta[0] # Bias term (equivalent to sigma in Hannay Model for Light)
        self.B1 = theta[1] # First harmonic amplitude
        self.B2 = theta[2] # Second harmonic amplitude
        self.theta1 = theta[3] # First harmonic phase
        self.theta2 = theta[4] # Second harmonic phase

    def set_melatonin_params(self):
        self.m = 7*60 # maybe m = 7*3600?
        self.beta_cp = 3.35e-4*3600
        self.beta_ap = 1.62e-4*3600
        self.beta_ip = 7.83e-4*3600

        self.H_sat = 861
        self.M_max = 0.019513
        self.sigma_m = 50
        self.delta_m = 600/3600
        r2 = self.beta_cp
        r3 = self.beta_ap
        max_ratio = (r3/(r2-r3))*(np.power(r2/r3,-r3/(r2-r3))-np.power(r2/r3,-r2/(r2-r3)))
        mid_dose = (0.1+0.3)/2
        dose_ratio = 200/mid_dose
        self.aborption_conversion = dose_ratio/max_ratio

        self.epsilon = 0 # Bias term (equivalent to sigma in Hannay Model for Light)
        self.B1 = 0 # First harmonic amplitude
        self.B2 = 0 # Second harmonic amplitude
        self.theta1 = 0 # First harmonic phase
        self.theta2 = 0 # Second harmonic phase

    def m_process(self,u):
        H2 = max(u[4],0)
        H2_conc = H2*861/200 # Convert from pg/L to pmol/L
        try:
            if H2_conc > 861*10:
                output = self.M_max
            else:
                output = self.M_max/(1 + np.exp((self.H_sat - H2_conc)/self.sigma_m))
        except:
            output = self.M_max/(1 + np.exp((self.H_sat - H2_conc)/self.sigma_m))
        return output

    def circ_response(self,phi):
        dlmo_phase = 5*np.pi/12
        phi = np.mod(phi - dlmo_phase,2*np.pi)
        a = 4*60
        phi_off = 4.3 
        phi_on = 6.1
        delta = 600/3600
        r = 15.36/3600

        if phi > phi_off and phi <= phi_on:
            return a * (1 - np.exp(-delta*np.mod(phi_on - phi,2*np.pi))) / (1 - np.exp(-delta*np.mod(phi_on - phi_off,2*np.pi)))
        else:
            return a*np.exp(-r*np.mod(phi_on - phi_off,2*np.pi))

    def mel_derv(self,t,u,bhat):
        H1 = max(u[3],0)
        H2 = max(u[4],0)
        H3 = max(u[5],0)

        m = 7*60 # maybe m = 7*3600?
        beta_ip = 7.83e-4*3600
        beta_cp = 3.35e-4*3600
        beta_ap = 1.62e-4*3600
        
        tmp = 1 - m*bhat # This m might need to be altered
        S = not(H1 < 0.001 and tmp < 0)

        dH1dt = -beta_ip*H1 + self.circ_response(u[1])*tmp*S
        dH2dt = beta_ip*H1 - beta_cp*H2 + beta_ap*H3
        dH3dt = -beta_ap*H3

        dHdt = np.array([dH1dt,dH2dt,dH3dt])
        dHdt = np.sign(dHdt)*np.minimum(60*200*np.ones(3),np.abs(dHdt))
        return dHdt

    def derv(self,t,y):
        """
        This defines the ode system for the single population model.
        derv(self,t,y)
        returns dydt numpy array.
        """
        R=y[0]
        Psi=y[1]
        n=y[2]
        H1 = y[3]
        H2 = y[4]
        H3 = y[5]

        # Light interaction with pacemaker
        Bhat=self.G*(1.0-n)*self.alpha0(t)
        LightAmp=self.A1*0.5*Bhat*(1.0-pow(R,4.0))*np.cos(Psi+self.BetaL1)+self.A2*0.5*Bhat*R*(1.0-pow(R,8.0))*np.cos(2.0*Psi+self.BetaL2)
        LightPhase=self.sigma*Bhat-self.A1*Bhat*0.5*(pow(R,3.0)+1.0/R)*np.sin(Psi+self.BetaL1)-self.A2*Bhat*0.5*(1.0+pow(R,8.0))*np.sin(2.0*Psi+self.BetaL2)

        dydt=np.zeros(6)

        # Melatonin variables
        dHdt = self.mel_derv(t,y,Bhat)
        dydt[3:] = dHdt
        
        # Melatonin interaction with pacemaker
        Mhat = self.m_process(y)
        MelAmp = self.B1*0.5*Mhat*(1.0-pow(R,4.0))*np.cos(Psi+self.theta1)+self.B2*0.5*Mhat*R*(1.0-pow(R,8.0))*np.cos(2.0*Psi+self.theta2)
        MelPhase = self.epsilon*Mhat-self.B1*Mhat*0.5*(pow(R,3.0)+1.0/R)*np.sin(Psi+self.theta1)-self.B2*Mhat*0.5*(1.0+pow(R,8.0))*np.sin(2.0*Psi+self.theta2)

        dydt[0]=-1.0*self.gamma*R+self.K*np.cos(self.Beta1)/2.0*R*(1.0-pow(R,4.0))+LightAmp+MelAmp
        dydt[1]=self.w0+self.K/2.0*np.sin(self.Beta1)*(1+pow(R,4.0))+LightPhase+MelPhase
        dydt[2]=60.0*(self.alpha0(t)*(1.0-n)-self.delta*n)


        return(dydt)



    def integrateModel(self, tend, tstart=0.0, initial=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],melatonin_timing = None,melatonin_dosage=None):
        """ Integrate the model forward in time.
        integrateModel(tend, initial=[1.0,0.0, 0.0])
        tend: float giving the final time to integrate to.
        initial: initial dynamical state
        The parameters are tend= the end time to stop the simulation and initial=[R, Psi, n]
        Writes the integration results into the scipy array self.results.
        Returns the circadian phase (in hours) at the ending time for the system.
        """
        dt = 0.1
        self.ts = np.arange(tstart,tend,dt)
        initial[1] = np.mod(initial[1], 2*sp.pi) #start the initial phase between 0 and 2pi

        # Sanitize input: sometimes np.arange witll output values outside of the range of tstart -> tend
        self.ts = self.ts[self.ts <= tend]
        self.ts = self.ts[self.ts >= tstart]
        
        if melatonin_timing is None:
            r = sp.integrate.solve_ivp(self.derv,(tstart,tend), initial, t_eval=self.ts, method='Radau')
            self.results = np.transpose(r.y)
        else:
            t_start = self.ts[0]
            all_arrays = []
            for t_end in melatonin_timing:
                local_ts = self.ts[np.logical_and(self.ts >= t_start, self.ts <= t_end)]
                r = sp.integrate.solve_ivp(self.derv,(t_start,t_end+0.1), initial, t_eval = local_ts, method="Radau")
                all_arrays.append(r.y[:,:-1])
                initial = r.y[:,-1]
                initial[5] = initial[5] + melatonin_dosage*self.aborption_conversion


                t_start = t_end
            t_end = tend
            local_ts = self.ts[np.logical_and(self.ts >= t_start, self.ts <= t_end)]
            r = sp.integrate.solve_ivp(self.derv,(t_start,t_end+0.1), initial, t_eval = local_ts, method="Radau")
            all_arrays.append(r.y)
            results = all_arrays[0]

            for i in np.arange(1,len(all_arrays)):
                results = np.hstack((results,all_arrays[i]))

            self.results = np.transpose(results)

        
        
        return

# Initial conditions function
def get_baseline(model):
    model.Light = light
    model.integrateModel(24*7+15.1)
    initial = model.results[-1,:]
    model.Light = dim_light
    model.integrateModel(15+24.1,tstart=15.0,initial=initial)
    baseline_day = model.results[:-1,:]
    phi = np.mod(baseline_day[:,1],2*np.pi)
    dlmo_time = np.arange(15-24,15,0.1)[np.argmin(np.abs(phi - 5*np.pi/12))]
    dlmo_time = np.round(np.mod(dlmo_time,24.0),1)
    initial = baseline_day[-1,:]
    return dlmo_time, initial


# Code to simulate PRC methodology
def prc_test(model, first_dlmo_time, initial, timing, dose):
    mel_timing = np.round(np.mod(first_dlmo_time + timing,24.0),1)
    mel_timing = mel_timing + 24*np.arange(5)
    mel_timing = mel_timing[mel_timing >= 15.0]
    mel_timing = mel_timing[mel_timing <= 15+24*3]
    experiment_dlmo_time = simulate_prc_test(model, initial, melatonin_timing=mel_timing, melatonin_dose=dose)

    return  experiment_dlmo_time

def simulate_prc_test(model, initial, melatonin_timing=None, melatonin_dose=0):
    tstart = 15
    tend = 15.1+24*3
    model.Light = ultradian_light
    # Control ultradian timing
    if melatonin_timing is None:
        model.integrateModel(tend,tstart=tstart,initial=initial)
    else:
        model.integrateModel(tend,tstart=tstart,initial=initial,melatonin_timing=melatonin_timing, melatonin_dosage=melatonin_dose)

    new_initial = model.results[-1,:]
    new_start = 15+24*3
    model.light = dim_light
    model.integrateModel(new_start+24,tstart=new_start,initial=new_initial)
    dlmo_arg = np.argmax(np.cos(model.results[:,1] - 5*np.pi/12))
    dlmo_time = np.round(np.mod(model.ts[dlmo_arg],24.0),1)

    return dlmo_time


def objective_func(params, data):
    try:
        
        model = IntegratedModel(light)
        model.set_melatonin_prc_params(params)
        timings = np.round(data[:,0],1)
        observed = data[:,1]
        doses = data[:,2]
        first_dlmo, initial = get_baseline(model)
        control_dlmo = simulate_prc_test(model,initial)
        results = np.array([prc_test(model, first_dlmo, initial, t, d) for t,d in zip(timings,doses)])

        shifts = np.mod(control_dlmo - results,24)
        shifts[shifts > 12] = shifts[shifts > 12] - 24.0

        delta = observed - shifts
        delta[np.abs(delta) > 12] = delta[np.abs(delta) > 12] - 24*np.sign(delta[np.abs(delta) > 12])
        result = np.mean(np.abs(delta))
        print(result)
        return result

    except:
        
        return 1e6



def light(t):
    full_light = 1000
    dim_light = 300
    wake_time = 7
    sleep_time = 23
    sun_up = 8
    sun_down = 19

    is_awake = np.mod(t - wake_time,24) <= np.mod(sleep_time - wake_time,24)
    sun_is_up = np.mod(t - sun_up,24) <= np.mod(sun_down - sun_up,24)

    return is_awake*(full_light*sun_is_up + dim_light*(1 - sun_is_up))

def dim_light(t):
    return 5

def ultradian_light(t):
    lights_on = 150
    return lights_on*(np.mod(t-14.5,4) < 2.5)

