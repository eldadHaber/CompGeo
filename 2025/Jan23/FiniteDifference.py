import torch
import numpy as np
from matplotlib import pyplot as plt

def finite_diff(f, h):
    df = f[1:] - f[:-1]
    return df/h

k = 8
res = []
for i in range(8):
    x = torch.linspace(0, 2*np.pi, k)
    n = len(x)
    h=2*np.pi/(n-1)
    x_mid = x[:-1] + h/2
    f = torch.sin(10*x)
    fprime = finite_diff(f, h)
    fprime_true = 10*torch.cos(10*x_mid)
    r = torch.max((fprime - fprime_true).abs())
    res.append(r)
    plt.plot(x[1:], fprime)
    plt.plot(x_mid, fprime_true,'-.r')

    print('k =', k, 'Error =', r)
    k = 2*k

res = torch.tensor(res)
plt.semilogy(res)
#plt.plot(x[1:], fprime)
#plt.plot(x[1:], fprime_true,'-.r')

print(' ')