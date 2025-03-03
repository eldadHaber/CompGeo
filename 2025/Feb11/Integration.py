import torch
import numpy as np
from matplotlib import pyplot as plt

def forwardEuler(func, y0, h, nsteps):
    y = torch.zeros(nsteps+1)
    t = torch.zeros(nsteps+1)
    y[0] = y0
    
    for i in range(1, nsteps+1):
        y[i] = y[i-1] + h * func(y[i-1])
        t[i] = t[i-1] + h
    
    return y, t

def RK4(func, y0, h, nsteps):
    y = torch.zeros(nsteps+1)
    t = torch.zeros(nsteps+1)
    y[0] = y0
    
    for i in range(1, nsteps+1):
        k1 = func(y[i-1])
        k2 = func(y[i-1] + h/2 * k1)
        k3 = func(y[i-1] + h/2 * k2)
        k4 = func(y[i-1] + h * k3)
        
        y[i] = y[i-1] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t[i] = t[i-1] + h
    
    return y, t



def func(y):
    return -y

nsteps = 3
h = 1/nsteps
y0 = 1.0
for j in range(7):
    #yout, t = forwardEuler(func, y0, h, nsteps)
    yout, t = RK4(func, y0, h, nsteps)
    
    ytrue = torch.exp(-t)
    
    #plt.plot(t, yout)
    
    err = torch.abs(yout - ytrue)
    plt.plot(t, err)    
    
    h = h/2
    nsteps = 2*nsteps
    print('h = %3.2e  Error = %3.2e' % (h, torch.max(err)))
print(' ')