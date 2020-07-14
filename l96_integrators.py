""" Lorenz 96  Integrator classes.
Based on equations 1a, 1b, 18a and 18b of Carlu et al. 2019 paper (https://doi.org/10.5194/npg-26-73-2019).
Uses adaptive RK54 method.
----------
Contents
----------
- Integrator, class for integrating L96 two layer dynamics.

- TangentIntegrator, class for integrating L96 two layer and corresponding tangent
dynamics simultaneously.

- TrajectoryObserver, class for observing the trajectory of the L96 integration.

- TangentTrajectoryObserver, class for observing the trajectory of the L96 tangent integration.

- make_observations, function that makes many observations given integrator and observer.
"""
# ----------------------------------------
# Imports
# ----------------------------------------
import numpy as np
import scipy.integrate
import xarray as xr
import sys
from tqdm import tqdm 

# ------------------------------------------
# Integrator
# ------------------------------------------

class Integrator:

    """Integrates the 2 layer L96 ODEs."""
    def __init__(self, K=36, J=10, h=1, Ff=6, Fs=10, b=10, c=10,
                 X_init=None, Y_init=None):

        # Model parameters
        self.K, self.J, self.h, self.Ff, self.Fs, self.b, self.c= K, J, h, Ff, Fs, b, c
        self.size = self.K + (self.J * self.K) # Number of variables

        self.time = 0

        # Non-linear Variables
        self.X = np.random.rand(self.K) if X_init is None else X_init.copy() # Random IC if none given
        self.Y = np.random.rand(self.K * self.J) if Y_init is None else Y_init.copy()  # ALL the y's


    def _rhs_X_dt(self, X, Y):
        """Compute the right hand side of the X-ODE."""

        dXdt = (
                np.roll(X, 1) * (np.roll(X, -1) - np.roll(X, 2)) -
                X + self.Fs - ((self.h * self.c)/self.b) * Y.reshape(self.K, self.J).sum(1)
        )
        return dXdt

    def _rhs_Y_dt(self, X, Y):
        """Compute the right hand side of the Y-ODE."""
        dYdt = self.c * (
                          self.b * np.roll(Y, -1) * ( np.roll(Y, 1) - np.roll(Y, -2) )
                       - Y + self.Ff/self.b + (self.h/self.b) * np.repeat(X, self.J) # repeat so x's match y's
               )
        return dYdt


    def _rhs_dt(self, t, state):
        X, Y = state[:self.K], state[self.K:]
        return [*self._rhs_X_dt(X, Y), *self._rhs_Y_dt(X, Y)]

    def integrate(self, how_long):
        """time: how long we integrate for in adimensional time."""

        # Where We are
        t = self.time
        IC = self.state

        # Integration, uses RK45 with adaptive stepping. THIS IS THE HEART.
        solver_return = scipy.integrate.solve_ivp(self._rhs_dt, (t, t + how_long), IC, dense_output = True)

        # Updating variables
        new_state = solver_return.y[:,-1]
        self.X = new_state[:self.K]
        self.Y = new_state[self.K: self.size]

        self.time = t + how_long

    def set_state(self, x):
        """x is [X, Y]. tangent_x is [dx, dy]"""
        self.X = x[:self.K]
        self.Y = x[self.K:]

    @property
    def state(self):
        """Where we are in phase space."""
        return np.concatenate([self.X, self.Y])

    @property
    def time(self):
        """a-dimensional time"""
        return self.__time

    @time.setter
    def time(self, when):
        self.__time = when

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
        }
        return param

# ------------------------------------------
# TangentIntegrator
# ------------------------------------------

class TangentIntegrator:

    """Integrates the L96 ODEs and it's tangent dynamics simultaneously."""
    def __init__(self, K=36, J=10, h=1, Ff=6, Fs=10, b=10, c=10,
                 X_init=None, Y_init=None, dx_init=None, dy_init=None):

        # Model parameters
        self.K, self.J, self.h, self.Ff, self.Fs, self.b, self.c= K, J, h, Ff, Fs, b, c
        self.size = self.K + (self.J * self.K) # Number of variables

        self.time = 0

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
        return dXdt

    def _rhs_Y_dt(self, X, Y):
        """Compute the right hand side of the Y-ODE."""
        dYdt = self.c * (
                          self.b * np.roll(Y, -1) * ( np.roll(Y, 1) - np.roll(Y, -2) )
                       - Y + self.Ff/self.b + (self.h/self.b) * np.repeat(X, self.J) # repeat so x's match y's
               )
        return dYdt

    def _rhs_dx_dt(self, X, dx, dy):
        """Compute rhs of the dx-ODE"""
        ddxdt = (
                    np.roll(dx, 1) * ( np.roll(X, -1) - np.roll(X, 2) )
                    + np.roll(X, 1) * ( np.roll(dx, -1) - np.roll(dx, 2) ) - dx
                   - ((self.h * self.c)/self.b) * dy.reshape(self.K, self.J).sum(1)
        )
        return ddxdt

    def _rhs_dy_dt(self, Y, dx, dy):
        """Compute rhs of the dy-ODE"""
        ddydt = self.c * self.b  * (
                            np.roll(dy, -1) * (np.roll(Y, 1) - np.roll(Y, -2) )
                          + np.roll(Y, -1) * (np.roll(dy, 1) - np.roll(dy, -2) ))
        - self.c * dy + ((self.h * self.c)/self.b) * np.repeat(dx, self.J)
        return ddydt

    def _rhs_dt(self, t, state):
        X, Y = state[:self.K], state[self.K: self.size]
        dx, dy = state[self.size: self.size + self.K], state[self.size + self.K: ]
        return [*self._rhs_X_dt(X, Y), *self._rhs_Y_dt(X, Y), *self._rhs_dx_dt(X, dx, dy), *self._rhs_dy_dt(Y, dx, dy)]

    def integrate(self, how_long):
        """time: how long we integrate for in adimensional time."""

        # Where We are
        t = self.time
        IC = np.hstack((self.state, self.tangent_state))

        # Integration, uses RK45 with adaptive stepping. THIS IS THE HEART.
        solver_return = scipy.integrate.solve_ivp(self._rhs_dt, (t, t + how_long), IC)

        # Updating variables
        new_state = solver_return.y[:,-1]
        self.X = new_state[:self.K]
        self.Y = new_state[self.K: self.size]
        self.dx = new_state[self.size: self.size + self.K]
        self.dy = new_state[self.size + self.K: ]

        self.time = t + how_long

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
        return self.__time

    @time.setter
    def time(self, when):
        self.__time = when

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
        }
        return param

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

    def look(self, integrator):
        """Observes trajectory of L96 trajectory"""

        # Note the time
        self.time_obs.append(integrator.time)

        # Making Observations
        self.x_obs.append(integrator.X.copy())
        self.y_obs.append(integrator.Y.copy())

    @property
    def observations(self):
        """cupboard: Directory where to write netcdf."""
        if (len(self.x_obs) == 0):
            print('I have no observations! :(')
            return

        dic = {}
        _time = self.time_obs
        dic['X'] = xr.DataArray(self.x_obs, dims=['time', 'K'], name='X',
                                coords = {'time': _time,'K': np.arange(1, 1 + self._K)})
        dic['Y'] = xr.DataArray(self.y_obs, dims=['time', 'KJ'], name='Y',
                                coords = {'time': _time, 'KJ': np.arange(1, 1 + self._K * self._J)})
        return xr.Dataset(dic, attrs= self._parameters)

    def wipe(self):
        """Erases observations"""
        self.time_obs = []
        self.x_obs = []
        self.y_obs = []

    def dump(self, cupboard, name=None):
        """ Saves observations to netcdf and wipes.
        cupboard: Directory where to write netcdf.
        name: file name"""

        if (len(self.x_obs) == 0):
            print('I have no observations! :(')
            return

        if name == None:
            name=self.name

        save = cupboard + f'/{name}' + f'{self.dump_count + 1}.nc'
        self.observations.to_netcdf(save)
        print(f'Observations written to {save}. Erasing personal log.\n')
        self.wipe()
        self.dump_count +=1

# ------------------------------------------
# TangentTrajectoryObserver
# ------------------------------------------

class TangentTrajectoryObserver():
    """Observes the trajectory of L96 tangent integrator. Dumps to netcdf."""

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
        self.y_obs.append(integrator.Y.copy())
        self.dx_obs.append(integrator.dx.copy())
        self.dy_obs.append(integrator.dy.copy())

    @property
    def observations(self):
        """cupboard: Directory where to write netcdf."""
        if (len(self.x_obs) == 0):
            print('I have no observations! :(')
            return

        dic = {}
        _time = self.time_obs
        dic['X'] = xr.DataArray(self.x_obs, dims=['time', 'K'], name='X',
                                coords = {'time': _time,'K': np.arange(1, 1 + self._K)})
        dic['Y'] = xr.DataArray(self.y_obs, dims=['time', 'KJ'], name='Y',
                                coords = {'time': _time, 'KJ': np.arange(1, 1 + self._K * self._J)})
        dic['dx'] = xr.DataArray(self.dx_obs, dims=['time', 'K'], name='dx',
                                coords = {'time': _time,'K': np.arange(1, self._K + 1)})
        dic['dy'] = xr.DataArray(self.dy_obs, dims=['time', 'KJ'], name='dy',
                                coords = {'time': _time, 'KJ': np.arange(1, 1 + self._K * self._J)})

        return xr.Dataset(dic, attrs= self._parameters)

    def wipe(self):
        """Erases observations"""
        self.time_obs = []
        self.x_obs = []
        self.y_obs = []
        self.dx_obs = []
        self.dy_obs = []

    def dump(self, cupboard, name=None):
        """ Saves observations to netcdf and wipes.
        cupboard: Directory where to write netcdf.
        name: file name"""

        if (len(self.x_obs) == 0):
            print('I have no observations! :(')
            return

        if name == None:
            name=self.name

        save = cupboard + f'/{name}' + f'{self.dump_count + 1}.nc'
        self.observations.to_netcdf(save)
        print(f'Observations written to {save}. Erasing personal log.\n')
        self.wipe()
        self.dump_count +=1

# ------------------------------------------
# make_observations
# ------------------------------------------

def make_observations(runner, looker, obs_num, obs_freq, noprog=False):
    """Makes observations given runner and looker.
    runner, integrator object.
    looker, observer object.
    obs_num, how many observations you want.
    obs_freq, adimensional time between observations"""
    for step in tqdm(np.repeat(obs_freq, obs_num), disable=noprog):
        runner.integrate(obs_freq)
        looker.look(runner)
