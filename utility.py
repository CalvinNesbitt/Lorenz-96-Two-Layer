# Utilities
# Some functions we will need for Lorenz 63 Lyapunov Analysis

import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
import scipy.integrate
from scipy.linalg import expm
from mpl_toolkits.mplot3d import Axes3D
import time as tm
import pickle

# Loading parameters from dictionary
infile = open('parameters','rb')
param_dict = pickle.load(infile)
infile.close()

# Assigning parameters for use

tau = param_dict['tau'] # How many steps TLE pushes you forward
kA = param_dict['kA']
kB = param_dict['kB']
kC = param_dict['kC']
tA = kA * tau # BLV convergence steps
tB = kB * tau # Sampling steps
tC = kC * tau # CLV convergence steps
steps = tA + tB + tC # total number of steps

# Strength of initial perturbation in Ginelli algorithm
eps = param_dict['eps']

# L63 Parameters

a = param_dict['a']
b = param_dict['b']
c = param_dict['c']
p = [a, b, c]

# Function to ensure QR decomposition has positive diagonals

def posQR(M):
    """ Returns QR decomposition of a matrix with positive diagonals on R.
    Parameter, M: Array that is being decomposed
    """
    Q, R = np.linalg.qr(M) # Performing QR decomposition
    signs = np.diag(np.sign(np.diagonal(R))) # Matrix with signs of R diagonal on the diagonal
    Q, R = np.dot(Q, signs), np.dot(signs, R) # Ensuring R Diagonal is positive
    return Q, R

# Vector field of the coupled Nonlinear dynamics and TLE

def TL63(t, state):
    """ ODEs defining coupled Lorenz 63 system and tangent dynamcis
    Parameter, state: current state.
    Parameter, t: time.
    Parameter, p: array, parameters for L63.
    """
    a = 10.0
    b = 8/3
    c = 28.0
    x, y, z, dx, dy, dz = state # x, y, z (nonlinear) dx, dy, dz (pertubation)

    # Nonlinear Lorenz Dynamics
    dxdt = a * (y - x)
    dydt = (c * x) - y - (x * z)
    dzdt = (x * y) - (b * z)

    # TLE
    ddxdt = a * (dy - dx)
    ddydt = (c - z) * dx - dy - (x * dz)
    ddzdt = (y * dx) + (x * dy) - (b * dz)

    return [dxdt, dydt, dzdt, ddxdt, ddydt, ddzdt]

# Function for solving ODEs

def solve(where, oldQ, t, tau):
    """ Simultaneuosly solve TLE and nonlinear dynamics. Returns stretched P = QR matrix.
    Parameter, where: where we are on attractor. [x, y, z]
    Parameter, oldQ: IC for tangent linear dynamics. Matrix, normally comes from Q, R decomposition.
    Parameter, t: time of where we are in the integration
    Parameter, tau: how long we solve TLE for.
    Returns [P_k, trajectory, time]
    """
    # Evolving first column of Q
    IC = [*where, *oldQ[:, 0]] # * unpacks list
    everything = scipy.integrate.solve_ivp(TL63, (t, t + tau), IC, dense_output = True) # Solving coupled system
    col1 = everything.y[3:, -1] # everythin.y is solution of coupled system. Last 3 index are TLE evolution

    # Evolving second column of Q
    IC = [*where, *oldQ[:, 1]] # * unpacks list
    everything = scipy.integrate.solve_ivp(TL63, (t, t + tau), IC, dense_output = True) # Solving coupled system
    col2 = everything.y[3:, -1] # everythin.y is solution of coupled system. Last 3 index are TLE evolution

    # Evolving third column of Q
    IC = [*where, *oldQ[:, 2]] # * unpacks list
    everything = scipy.integrate.solve_ivp(TL63, (t, t + tau), IC, dense_output = True, rtol = 1e-9) # Solving coupled system
    col3 = everything.y[3:, -1] # everythin.y is solution of coupled system. Last 3 index are TLE evolution

    # Getting current trajectory
    trajectory = everything.y[:3] # This is the trajectory we computed the above over. Note it's from last solution

    # Getting time trajectory was calculated over
    time = everything.t
    Pk = np.column_stack((col1, col2, col3))

    return [Pk, trajectory, time]
