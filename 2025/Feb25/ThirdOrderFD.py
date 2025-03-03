import torch
import numpy as np
from matplotlib import pyplot as plt

# y[i+1] - y[i]
# y[1:] - y[:-1]


def get_third_order_der(y, h):
# (y[i+2] - (2*y[i+1] - 2*y[i-1]) - y[i-2])/(2*h**3) = y'''(x_i)
#
    dy1 =  2*(y[2:] - y[:-2]) # 2*y[i+1] - 2*y[i-1]
    dy2 =  y[4:] - y[:-4] # y[i+2] - y[i-2]
    
    d3y = (dy2 - dy1[1:-1])/(2*h**3)
    
    return d3y

# test it
for n in range(5):
    m = 2**(n+3)
    t = torch.linspace(0, 2*torch.pi, m)
    y = torch.sin(4*t)
    h = t[1] - t[0]

    
    d3ydt3 =  get_third_order_der(y, h)  
    d3ydt3True = -torch.cos(4*t)*(4**3)

    r = (d3ydt3True[2:-2] - d3ydt3).abs().mean()

    print('h = %3.2e  Max error = %3.2e' %(h, r))

plt.plot(t[2:-2], d3ydt3)
plt.plot(t, d3ydt3True)
 
print(' ')