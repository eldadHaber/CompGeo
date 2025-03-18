import torch
import numpy as np
from matplotlib import pyplot as plt

# points where we measure the function
# Xobs = torch.randn(100, 2)
# plt.scatter(Xobs[:, 0], Xobs[:, 1])
# f = (Xobs**2).sum(dim=1)

# # find pairwise distance
# D = torch.cdist(Xobs, Xobs, p=2)

# # setup the linear system
# eps = 1
# A = torch.exp(-D*eps)
# c = torch.linalg.solve(A, f)

# N = 100
# M = 120
# xI = torch.linspace(-4, 4, N)
# yI = torch.linspace(-3, 3, M)
# X, Y = torch.meshgrid(xI, yI, indexing='xy')

# XI = torch.stack([X.flatten(), Y.flatten()], dim=1)

# DI = torch.cdist(XI, Xobs, p=2)
# AI = torch.exp(-DI*eps)

# fI = AI @ c
# fI = fI.reshape(M, N)

# plt.contourf(xI, yI, fI)
# plt.scatter(Xobs[:, 0], Xobs[:, 1], c=f)

# print(' ')


import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.spatial.distance import cdist

def RBFinterp(Xobs, f, XI, eps=0.01):
  # Given data f(x,y)
  # interpolate to new points s(xI, yI)

  # find pairwise distance
  D = cdist(Xobs, Xobs, metric='euclidean')

  # setup the linear system
  A = np.exp(-D * eps)
  c = np.linalg.solve(A, f)

  DI = cdist(XI, Xobs, metric='euclidean')
  AI = np.exp(-DI * eps)

  fI = AI @ c
  return fI

def peaks(x, y):
    """
    Generate a peaks function, similar to MATLAB's peaks.
    
    Parameters:
    n : int, optional
        Number of points to generate in each dimension (default is 49)
    
    Returns:
    x, y : 2D arrays
        Meshgrid of x and y coordinates
    z : 2D array
        The peaks function evaluated at (x, y)
    """
    # Calculate z
    z = (3 * (1-x)**2 * np.exp(-(x**2) - (y+1)**2) 
         - 10 * (x/5 - x**3 - y**5) * np.exp(-x**2 - y**2) 
         - 1/3 * np.exp(-(x+1)**2 - y**2))
    
    return z

xObs = np.random.randn(1000)
yObs = np.random.randn(1000)
fObs = peaks(xObs, yObs) + np.random.randn(1000)

Xobs = np.column_stack((xObs, yObs))
x, y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))    
XI = np.column_stack((x.flatten(), y.flatten()))

fI = RBFinterp(Xobs, fObs, XI, eps=0.1)
fI = fI.reshape(100, 100)

# To visualize (requires matplotlib):
plt.contourf(x, y, fI)
plt.scatter(xObs, yObs, c=fObs)
plt.colorbar()
plt.show()



print(' ')