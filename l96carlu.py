""" Lorenz 96 tangent Integrator class. 
Based on equations 1a, 1b, 18a and 18b of Carlu et al. 2019 paper (https://doi.org/10.5194/npg-26-73-2019).
----------
Contents
----------
- Integrator, class for integrating L96 two layer and corresponding tangent
dynamics simultaneously.

- TrajectoryObserver, class for observing the trajectory of the L96
tangent integration.

- make_observations, function that makes many observations L96 tangent
integration.
"""
# ----------------------------------------
# Imports
# ----------------------------------------
import numpy as np
import xarray as xr
import sys
from tqdm.notebook import tqdm 

# ------------------------------------------
# Integrator
# ------------------------------------------

class Integrator:

    """Integrates the L96 ODEs and it's tangent dynamics simultaneously."""
    def __init__(self, K=36, J=10, h=1, Ff=6, Fs=10, b=10, c=10, dt=0.001,
                 X_init=None, Y_init=None, dx_init=None, dy_init=None, tangent_dynamics=False):

        # Model parameters
        self.K, self.J, self.h, self.Ff, self.Fs, self.b, self.c, self.dt = K, J, h, Ff, Fs, b, c, dt
        self.size = self.K + (self.J * self.K) # Number of variables
        
        self.tangent_dynamics = tangent_dynamics # Do you integrate TLE Simultaneously

        # Step counts
        self.step_count = 0 # Number of integration steps

        # Non-linear Variables
        self.X = np.random.rand(self.K) if X_init is None else X_init.copy() # Random IC if none given
        self.Y = np.random.rand(self.K * self.J) if Y_init is None else Y_init.copy()  # ALL the y's

        # TLE Variables
        self.dx = np.array([i/10000 for i in np.random.rand(self.K)]) if dx_init is None else dx_init.copy()
        self.dy = np.array([i/10000 for i in np.random.rand(self.K * self.J)]) if dy_init is None else dy_init.copy()

    def _rhs_X_dt(self, X, Y):
        """Compute the right hand side of the X-ODE."""

        dXdt = (
                np.roll(X, 1) * (np.roll(X, -1) - np.roll(X, 2)) -
                X + self.Fs - ((self.h * self.c)/self.b) * Y.reshape(self.K, self.J).sum(1)
        )
        return self.dt * dXdt

    def _rhs_Y_dt(self, X, Y):
        """Compute the right hand side of the Y-ODE."""
        dYdt = self.c * (
                          self.b * np.roll(Y, -1) * ( np.roll(Y, 1) - np.roll(Y, -2) )
                       - Y + self.Ff/self.b + (self.h/self.b) * np.repeat(X, self.J) # repeat so x's match y's
               )
        return self.dt * dYdt

    def _rhs_dx_dt(self, X, dx, dy):
        """Computhe rhs of the dx-ODE"""
        ddxdt = (
                    np.roll(dx, 1) * ( np.roll(X, -1) - np.roll(X, 2) )
                    + np.roll(X, 1) * ( np.roll(dx, -1) - np.roll(dx, 2) ) - dx
                   - ((self.h * self.c)/self.b) * dy.reshape(self.K, self.J).sum(1)
        )
        return self.dt * ddxdt

    def _rhs_dy_dt(self, Y, dx, dy):
        """Computhe rhs of the dy-ODE"""
        ddydt = self.c * self.b  * (
                            np.roll(dy, -1) * (np.roll(Y, 1) - np.roll(Y, -2) )
                          + np.roll(Y, -1) * (np.roll(dy, 1) - np.roll(dy, -2) ))
        - self.c * dy + ((self.h * self.c)/self.b) * np.repeat(dx, self.J)
        return self.dt * ddydt

    def _rhs_dt(self, X, Y, dx, dy):
        return self._rhs_X_dt(X, Y), self._rhs_Y_dt(X, Y), self._rhs_dx_dt(X, dx, dy), self._rhs_dy_dt(Y, dx, dy)

    def _step(self):
        """Integrate one time step"""

        # RK Coefficients
        k1_X, k1_Y, k1_dx, k1_dy = self._rhs_dt(self.X, self.Y,
                                                self.dx, self.dy)
        k2_X, k2_Y, k2_dx, k2_dy = self._rhs_dt(self.X + k1_X / 2, self.Y + k1_Y / 2,
                                                self.X + k1_X / 2, self.dy + k1_dy / 2)
        k3_X, k3_Y, k3_dx, k3_dy = self._rhs_dt(self.X + k2_X / 2, self.Y + k2_Y / 2,
                                               self.dx + k2_dx / 2, self.dy + k2_dy / 2)
        k4_X, k4_Y, k4_dx, k4_dy = self._rhs_dt(self.X + k3_X, self.Y + k3_Y,
                                               self.dx + k3_dx / 2, self.dy + k3_dy / 2)

        # Update State
        self.X += 1 / 6 * (k1_X + 2 * k2_X + 2 * k3_X + k4_X)
        self.Y += 1 / 6 * (k1_Y + 2 * k2_Y + 2 * k3_Y + k4_Y)
        
        if (self.tangent_dynamics):
            self.dx += 1 / 6 * (k1_dx + 2 * k2_dx + 2 * k3_dx + k4_dx)
            self.dy += 1 / 6 * (k1_dy + 2 * k2_dy + 2 * k3_dy + k4_dy)
        self.step_count += 1

    def integrate(self, time, noprog=True):
        """time: how long we integrate for in adimensional time."""
        steps = int(time / self.dt)
        for n in tqdm(range(steps), disable=noprog):
            self._step()

    def set_state(self, x, tangent_x):
        """x is [X, Y]. tangent_x is [dx, dy]"""
        self.X = x[:self.K]
        self.Y = x[self.K:] 
        self.dx = tangent_x[: self.K]
        self.dy = tangent_x[self.K: ] 

    @property
    def state(self):
        """Where we are in phase space."""
        return np.concatenate([self.X, self.Y])

    @property
    def tangent_state(self):
        """Where we are in tangent space"""
        return np.concatenate([self.dx, self.dy])

    @property
    def time(self):
        """a-dimensional time"""
        return self.dt * self.step_count

    @property
    def parameter_dict(self):
        param = {
        'h': self.h, # L96
        'Fs': self.Fs,
        'Ff': self.Ff,
        'c': self.c,
        'J': self.J,
        'K': self.K,
        'Number of variables': self.size,
        'b': self.b,
        'dt': self.dt
        }
        return param

    def reset_count(self):
        """Reset Step count"""
        self.step_count = 0

# ------------------------------------------
# TrajectoryObserver
# ------------------------------------------

class TrajectoryObserver():
    """Observes the trajectory of L96 ODE integrator. Dumps to netcdf."""

    def __init__(self, integrator, name='L96 Trajectory'):
        """param, integrator: integrator being observed."""

        # Needed knowledge of the integrator
        self._K = integrator.K
        self._J = integrator.J
        self._parameters = integrator.parameter_dict

        # Trajectory Observation logs
        self.time_obs = [] # Times we've made observations
        self.x_obs = []
        self.y_obs = []
        self.dx_obs = []
        self.dy_obs = []

    def look(self, integrator):
        """Observes trajectory of L96 trajectory"""

        # Note the time
        self.time_obs.append(integrator.time)

        # Making Observations
        self.x_obs.append(integrator.X.copy())
        self.y_obs.append(integrator.Y.copy()) # Integrator solves transformed equations
        self.dx_obs.append(integrator.dx.copy())
        self.dy_obs.append(integrator.dy.copy())

def make_observations(runner, looker, obs_num, obs_freq, noprog=False):
    """Makes observations given runner and looker.
    runner, integrator object.
    looker, observer object.
    obs_num, how many observations you want.
    obs_freq, adimensional time between observations"""
    for step in tqdm(np.repeat(obs_freq, obs_num), disable=noprog):
        runner.integrate(obs_freq)
        looker.look(runner)