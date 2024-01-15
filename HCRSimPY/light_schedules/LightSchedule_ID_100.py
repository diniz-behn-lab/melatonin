"""
This code is modified to run data from LeBourgeois protocol, participant ID#100



This python module contains functions for creating light schedules giving the light levels in lux as a function of time in hours.

Used as input to simulate the circadian models.

Contains some common schedules as well as some custom data driven schedules


"""


from __future__ import division
from __future__ import print_function

from builtins import str
from builtins import zip
from builtins import map
from builtins import range
from builtins import object
from past.utils import old_div
import numpy as np
import scipy as sp
import pandas as pd
from math import fmod
from scipy import interpolate
import pylab as plt
import sys
import datetime


from numba import jit



pd.options.mode.chained_assignment = None  # default='warn'


def cal_days_diff(a,b):
    """Get the calander days between two time dates"""
    A = a.replace(hour = 0, minute = 0, second = 0, microsecond = 0)
    B = b.replace(hour = 0, minute = 0, second = 0, microsecond = 0)
    return (A - B).days

@jit(nopython=True)
def interpolateLinear(t, xvals, yvals):
    """Implement a faster method to get linear interprolations of the light functions"""

    if (t>= xvals[-1]):
        return(0.0)
    if (t<=xvals[0]):
        t+=24.0

    i=np.searchsorted(xvals,t)-1
    ans=(yvals[i+1]-yvals[i])/((xvals[i+1]-xvals[i])*(t-xvals[i]))+yvals[i]
    return(ans)


@jit(nopython=True)
def interpolateLinearExt(t, xvals, yvals):
    """Implement a faster method to get linear interprolations of the light functions, exclude non-full days"""
    i=np.searchsorted(xvals,t)-1
    ans=(yvals[i+1]-yvals[i])/((xvals[i+1]-xvals[i])*(t-xvals[i]))+yvals[i]
    return(ans)




def parse_dt(date, time):
    strDate=date+' '+ time
    return pd.to_datetime(strDate, format='%m/%d/%Y %I:%M %p')


@jit(nopython=True)
def LightLog(lightlevel, threshold=1.0):
    """Take the log10 of the light levels, but map 0 to zero and not negative numbers"""
    if (lightlevel<threshold):
        return(0.0)
    if (lightlevel>=threshold):
        return(np.log10(lightlevel))




def RegularLight(t, Intensity, PP, period):
    """Define a basic light schedule"""
    val=0.5*sp.tanh(100*(fmod(t, period))) - 0.5*sp.tanh(100*(fmod(t, period) - PP))
    return(Intensity*val)


def RegularLightSimple(t, Intensity=150.0, wakeUp=8.0, workday=16.0):
    """Define a basic light schedule with a given intensity of the light, wakeup time and length of the active period (non-sleeping)
    This schedule will automatically repeat on a daily basis, so each day will be the same.....
    """

    s=fmod(t,24.0)-wakeUp

    if (s<0):
        s+=24.0

    val=0.5*sp.tanh(100*(s)) - 0.5*sp.tanh(100*(s - workday))
    return(Intensity*val)


def ShiftWorkLight(t, dayson=5, daysoff=2):
    """Simulate a night shift worker. Assume they are working a night shift for dayson number of days followed by daysoff normal days where they revert to a normal schedule
    ShiftWorkLight(t, dayson=5, daysoff=2)
    """

    t=fmod(t, (dayson+daysoff)*24) #make it repeat
    if (t<=24*dayson):

        return(RegularLightSimple(t, Intensity=150.0, wakeUp=16.0, workday=16.0))

    else:
        return(RegularLightSimple(t, Intensity=250.0, wakeUp=9.0, workday=16.0))


def SocialJetLag(t, weekdayWake=7.0, weekdayBed=24.0, weekendWake=11.0, weekendBed=2.0):
    """Simulate a social jetlag schedule. """

    t=fmod(t, (7)*24) #make it repeat each week


    if (t<=24*5):
        #Monday through thursday
        duration=fmod(weekdayBed-weekdayWake, 24.0)
        if duration<0.0:
            duration+=24.0
        return(RegularLightSimple(t, Intensity=150.0, wakeUp=weekdayWake, workday=duration))
    if (t> 24*5 and t<= 24*7):
        #Friday, stay up late
        duration=fmod(weekendBed-weekendWake, 24.0)
        if duration<0.0:
            duration+=24.0
        return(RegularLightSimple(t, Intensity=250.0, wakeUp=weekendWake, workday=duration))


def SlamShift(t, shift=8.0, intensity=150.0, beforeDays=10):
    """Simulate a sudden shift in the light schedule"""

    t=fmod(t, 40*24) #make it repeat every 40 days

    if (t<= 24*beforeDays):
        return(RegularLightSimple(t, intensity, wakeUp=8.0, workday=16.0))
    else:
        newVal=fmod(8.0+shift, 24.0)
        if (newVal<0):
            newVal+=24.0
        return(RegularLightSimple(t, intensity, wakeUp=newVal, workday=16.0))


def OneDayShift(t, pulse=23.0, wakeUp=8.0, workday=16.0, shift=8.0):

    t=fmod(t, 40*24)

    beforeDays=10.0

    if (t< 24*beforeDays-3*24.0):
        return(RegularLightSimple(t, 150.0, wakeUp=wakeUp, workday=workday))

    if ((t>= 24*beforeDays-3*24.0) and (t<=24*beforeDays)):
        #adjustment day get to add one hour of bright light
        pulse+=24*beforeDays
        Light=RegularLightSimple(t, 150.0, wakeUp=wakeUp, workday=workday)
        Light+=10000.0*(0.5*sp.tanh(100*(t-pulse+72.0)) - 0.5*sp.tanh(100*(t- 1.0-pulse+72.0)))
        Light+=10000.0*(0.5*sp.tanh(100*(t-pulse+48.0)) - 0.5*sp.tanh(100*(t- 1.0-pulse+48.0)))
        Light+=10000.0*(0.5*sp.tanh(100*(t-pulse+24.0)) - 0.5*sp.tanh(100*(t- 1.0-pulse+24.0)))
        return(Light)

    if (t>24*(beforeDays)):
        newVal=fmod(8.0+shift, 24.0)
        if (newVal<0):
            newVal+=24.0
        return(RegularLightSimple(t, 150.0, wakeUp=newVal, workday=16.0))




class ML_light(object):

    def __init__(self, filename, smoothWindow=20):

        fileData=pd.read_excel(filename, delimiter=',', skiprows=0)
        #fileData=fileData.set_index('Date_Time')
        fileData['Lux']=fileData.whitelight+fileData.bluelight+fileData.redlight+fileData.greenlight
        fileData.time = pd.to_datetime(fileData.time, format='%H:%M:%S')
        self.data=fileData

        startingDay=self.data.dayofweek.iloc[0]
        dow=list(self.data.dayofweek)
        count=0
        day_list=[0]
        for i in range(1,len(dow)):
            if dow[i-1]==dow[i]:
                day_list.append(count)
            else:
                count+=1
                day_list.append(count)

        self.data.day=day_list


        #Create an index to count the hours of the data (UNITS of hours)

        self.data['TimeTotal']=(self.data.day)*24.0+pd.DatetimeIndex(self.data.time).hour+pd.DatetimeIndex(self.data.time).minute/60.0+pd.DatetimeIndex(self.data.time).second/3600.0
        self.data['TimeCount']=pd.DatetimeIndex(self.data.time).hour+pd.DatetimeIndex(self.data.time).minute/60.0+pd.DatetimeIndex(self.data.time).second/3600.0
        self.data['Lux']=self.data.Lux.rolling(smoothWindow, center=True).max()
        #Smooth the wake scores as well
        self.data['wake']=self.data.wake.rolling(smoothWindow, center=True).mean()

        #Drop the missing values introduced
        self.data=self.data.dropna()

        self.startTime=self.data.TimeTotal.iloc[0]
        self.endTime=self.data.TimeTotal.iloc[-1]

        #Make a function which gives the light level
        self.LightFunction=lambda t: interpolateLinear(fmod(t, self.endTime), np.array(self.data['TimeTotal']), np.array(self.data['Lux']))


        #Make a extended function for lomg simulations
        xvals=np.array(self.data['TimeTotal'])
        yvals=np.array(self.data.Lux)
        yvals=yvals[xvals>=24.0]
        xvals=xvals[xvals>=24.0]

        num_days=int(xvals[-1]/24.0)
        final_time=num_days*24.0
        #Exclude the final day
        yvals=yvals[xvals<=final_time]
        xvals=xvals[xvals<=final_time]

        #start at zero
        xvals=xvals-24.0

        self.LightFunctionInitial=lambda t: interpolateLinearExt(fmod(t, xvals[-1]),xvals,yvals)
        #Trim off the non-full days within the data set and make a light function using that data


    def findMidSleep(self, cutoff=0.5, show=False):
        """Estimate the midpoint of the sleep. Actually the angular mean of the sleep times"""

        av_data=self.data.groupby(by=['TimeCount']).mean()

        #Now round the sleep scores:
        av_data.loc[av_data['wake']>=cutoff, 'wake']=1.0
        av_data.loc[av_data['wake']<cutoff, 'wake']=0.0

        sleepTimes=pd.Series(av_data.loc[av_data['wake']==0.0,:].index)
        toangle=lambda x: np.exp(complex(0,1)*(2*sp.pi*x/24.0))
        all_angles=np.angle(list(map(toangle, sleepTimes)))
        Z=np.mean(list(map(toangle, sleepTimes)))
        angle=np.fmod(np.angle(Z)+2.0*sp.pi, 2.0*sp.pi)
        amp=abs(Z)

        timeEquivalent=old_div(angle*24.0,(2.0*sp.pi))

        if (show==True):
            plt.scatter(sp.cos(all_angles), sp.sin(all_angles))
            plt.scatter(sp.cos(angle), sp.sin(angle), color='green')
            plt.xlim(-1,1)
            plt.ylim(-1,1)
            plt.show()

            plt.plot(av_data.index, av_data.wake, color='green')
            plt.plot(av_data.index, list(map(LightLog, list(av_data.Lux))), color='blue')
            plt.show()


        return(timeEquivalent)


    def plot_light(self):
        """Make a plot of the light schedule with each day on top of the last"""
        ts=np.arange(self.data.TimeTotal.iloc[0], self.data.TimeTotal.iloc[-1], 0.01)
        y=list(map(self.LightFunction, ts))
        plt.plot(ts,list(map(LightLog, y)))
        #plt.scatter(np.array(self.data.TimeTotal), map(LightLog, self.data.Lux), color='red', s=0.1)
        plt.plot(np.array(self.data.TimeTotal), self.data.wake)
        plt.show()


if __name__=='__main__':

    ML=ML_light(r'./../light_schedules/ID_100_light_history.xlsx')
    ML.plot_light()
    print(ML.findMidSleep(show=True))
