import numpy as np
from numpy.core.numeric import full
from scipy import integrate
import matplotlib.pyplot as plt

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



def alpha(I):
    I0 = 9500
    I1 = 100
    alpha0 = 0.1
    p = 0.5

    return alpha0*np.power(I/I0,p)*(I/(I+I1))


def b_process(t,u):
    G = 37.0
    b = 0.4
    x = u[0]
    xc = u[1]
    n = u[2]
    I = light(t)

    return G*(1-b*x)*(1-b*xc)*alpha(I)*(1-n)


def derv(t, u):
    x = u[0]
    xc = u[1]
    n = u[2]
    
    kappa = 12/np.pi
    beta = 0.007
    K = 1
    gamma = 0.13
    tau_c = 24.1
    f = 0.99729

    B = b_process(t,u)

    dxdt = (xc + gamma*(x/3 + 4*(x**3)/3 - 256*(x**7)/105) + B)/kappa
    dxcdt = (B*xc/3 - x*((24/(f*tau_c))**2) + K*B)/kappa
    dndt = alpha(light(t))*(1-n) - beta*n

    dudt = np.array([dxdt,dxcdt,dndt])
    return dudt
    

def integrate_model(tend, tstart=0.0, initial=[1.0,0.0, 0.0]):
    dt = 0.1
    t = np.arange(tstart,tend,dt)

    # Sanitize input: sometimes np.arange witll output values outside of the range of tstart -> tend
    t = t[t <= tend]
    t = t[t >= tstart]

    r = integrate.solve_ivp(derv,(tstart,tend), initial, t_eval=t, method='Radau')
    return t, np.transpose(r.y)

t, results = integrate_model(21*24.0)
phi = np.mod(np.arctan2(results[:,0],results[:,1]),2*np.pi)
#plt.plot(t[-480:],phi[-480:],lw=2)
#plt.show()

r = np.sqrt(results[:,0]**2 + results[:,1]**2)
r = r[-7*240:]
t = t[-7*240:]
print(min(r))
print(max(r) - min(r))
plt.plot(t,r,lw=2)
plt.show()
plt.plot(results[7*240:,0],results[7*240:,1],lw=2)
plt.show()