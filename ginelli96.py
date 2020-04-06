import numpy as np
import time as tm
import pickle
from tqdm import tqdm
import xarray as xr
import sys

"""
Ginelli algorithm applied to the two layer L96 system.

Created on 2019-04-16-12-28
Author: Stephan Rasp, raspstephan@gmail.com

Edited on 2020-03-20
by Calvin Nesbitt
"""


class Ginelli96(object):
    def __init__(self, K=36, J=10, h=1, Ff=6, Fs=10, c=10, dt=0.001, tau=0.01, transient=500, kA=500,
                 kB=1000, kC=100, X_init=None, Y_init=None, dx_init=None, dy_init=None,
                 oldQ_init=None, noprog=False): # X's and Y's will be np arrays
        # Model parameters
        self.K, self.J, self.h, self.Ff, self.Fs, self.c, self.dt = K, J, h, Ff, Fs, c, dt
        self.b = np.sqrt(J * c) # b restriction
        self.size = self.K + (self.J * self.K) # Number of variables

        # Step counts
        self.step_count = 0 # Number of integration steps
        self.g_step_count = 0 # Number of Ginelli Steps

        # Ginelli Parameters
        self.tau = tau # How long in adimensional time before doing a QR decomposition
        self.transient = int(transient) # Number of transient steps
        self.kA = int(kA) # BLV convergence steps
        self.kB = int(kB) # Sampling steps
        self.kC = int(kC) # CLV convergence steps

        # Progress Bars
        self.noprog = noprog

        # Non-linear Variables
        self.X = np.random.rand(self.K) if X_init is None else X_init.copy() # Random IC if none given
        self.Y = np.zeros(self.K * self.J) if Y_init is None else Y_init.copy() * self.b # ALL the y's
        self._history_X = [self.X.copy()]
        self._history_Y = [self.Y.copy()] # Y's in a list, remember they are in natural K lists

        # TLE Variables
        self.dx = np.random.rand(self.K) if dx_init is None else dx_init.copy()
        self.dy = np.random.rand(self.K * self.J) if dy_init is None else dy_init.copy()

        # Ginelli Variables
        self.P = np.random.rand(self.size, self.size) # Stretched Matrix
        eps = 0.001
        self.oldQ = eps * np.identity(self.size) if oldQ_init is None else oldQ_init.copy()
        self.oldQ[0, 1] = eps * 1
        self.R = np.random.rand(self.size, self.size)  # Stretching rates
        self._history_R = []
        self.oldA = np.identity(self.size) # Initial A
        self.oldA[0, 1] = 1

        # Lyapunov Spectra
        self.FTBLE = np.random.rand(int(kB), self.size)
        self.FTCLE = np.random.rand(int(kB), self.size)
        self._history_FTBLE = [] # For storing time series
        self._history_FTCLE = []

        # Lyapunov Vectors
        self.BLV = np.random.rand(self.size, self.size)
        self.CLV = np.random.rand(self.size, self.size)
        self._history_BLV = []
        self._history_CLV = []

        # Ginelli Timings
        self.timet, self.time1, self.time2, self.time3, self.time4, self.time5 = np.zeros(6)

    def _rhs_X_dt(self, X, Y):
        """Compute the right hand side of the X-ODE. Note this has been scaled."""

        dXdt = (
                np.roll(X, 1) * (np.roll(X, -1) - np.roll(X, 2)) -
                X + self.Fs - self.h * Y.reshape(self.K, self.J).mean(1) # Using Y mean
        )
        return self.dt * dXdt

    def _rhs_Y_dt(self, X, Y):
        """Compute the right hand side of the Y-ODE."""
        dYdt = self.c * (
                          np.roll(Y, -1) * ( np.roll(Y, 1) - np.roll(Y, -2) )
                       - Y + self.Ff + self.h * np.repeat(X, self.J) # repeat so x's match y's
               )
        return self.dt * dYdt

    def _rhs_dx_dt(self, X, dx, dy):
        """Computhe rhs of the dx-ODE"""
        ddxdt = (
                    np.roll(dx, 1) * ( np.roll(X, -1) - np.roll(X, 2) )
                    + np.roll(X, 1) * ( np.roll(dx, -1) - np.roll(dx, 2) ) - dx
                   - self.h * dy.reshape(self.K, self.J).mean(1)
        )
        return self.dt * ddxdt

    def _rhs_dy_dt(self, Y, dx, dy):
        """Computhe rhs of the dy-ODE"""
        ddydt = self.c * (
                            np.roll(dy, -1) * (np.roll(Y, 1) - np.roll(Y, -2) )
                          + np.roll(Y, -1) * (np.roll(dy, 1) - np.roll(dy, -2) )
                        - dy + self.h * np.repeat(dx, self.J)
        )
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

        # Update
        self.X += 1 / 6 * (k1_X + 2 * k2_X + 2 * k3_X + k4_X)
        self.Y += 1 / 6 * (k1_Y + 2 * k2_Y + 2 * k3_Y + k4_Y)
        self.dx += 1 / 6 * (k1_dx + 2 * k2_dx + 2 * k3_dx + k4_dx)
        self.dy += 1 / 6 * (k1_dy + 2 * k2_dy + 2 * k3_dy + k4_dy)

    def _integrate(self, time):
        """parameter, time: how longer we integrate for in adimensional time."""
        steps = int(time / self.dt)
        for n in range(steps):
            self._step()

    # Ginelli Algorithm

    def posQR(self, M):
        """ Returns QR decomposition of a matrix with positive diagonals on R.
        Parameter, M: Array that is being decomposed
        """
        Q, R = np.linalg.qr(M) # Performing QR decomposition
        signs = np.diag(np.sign(np.diagonal(R))) # Matrix with signs of R diagonal on the diagonal
        Q, R = np.dot(Q, signs), np.dot(signs, R) # Ensuring R Diagonal is positive
        return Q, R

    def _ginelli_step(self):
        """One QR step. Take old Q, stretch it, do a QR decomposition."""

        # Where we are in phase space before ginelli step
        phase_state = self.state[:self.size]

        # Stretching first column
        self.set_state(phase_state, self.oldQ.T[0]) # First column of Q is ic for TLE
        self._integrate(self.tau)

        # Saving Output
        # We only save phase space evolution in this step
        #self.step_count += int(self.tau/self.dt)
        self.P[:, 0] = np.append(self.dx, self.dy)
        self._history_X.append(self.X.copy()) # These are only saved at intervals of size tau
        self._history_Y.append(self.Y.copy())

        # Stretching the rest of the columns
        for i, column in enumerate(self.oldQ.T[1:]): # Evolve each CLV
            self.set_state(phase_state, column)
            self._integrate(self.tau)
            self.P[:, i] = np.append(self.dx, self.dy) # Building P

        # QR decomposition
        self.oldQ, self.R = self.posQR(self.P)
        self.g_step_count += 1

    def _run_ginelli(self, no_pbars=True):
        """Ginelli Algorithm"""

        self.noprog = no_pbars
        start = tm.time()
        # Transient
        print('Transient Beginning\n')
        for n in tqdm(range(self.transient), disable=self.noprog):
            self._ginelli_step()
        self.timet = tm.time() - start

        start = tm.time()
        # BLV Convergence Steps
        print(f'Transient took {self.timet} seconds. Starting Ginelli Step 1.\n')
        for n in tqdm(range(self.kA), disable=self.noprog):
            self._ginelli_step()
        self.time1 = tm.time() - start

        start = tm.time()
        # BLV Sampling Steps
        print(f'Step 1 took {self.time1} seconds. Starting Ginelli Step 2.\n')
        for n in tqdm(range(self.kB), disable=self.noprog):
            self._ginelli_step() # Updates Q and R
            self.FTBLE = np.log(np.diag(self.R))/(self.tau) # Note division by tau is done here
            self._history_FTBLE.append(self.FTBLE.copy())
            self._history_BLV.append(self.oldQ.copy())
            self._history_R.append(self.R.copy())

        self.time2 = tm.time() - start

        start = tm.time()
        # CLV Convergence Steps
        r2_index = len(self._history_R) # These are the R's we will invert
        print(f'Step 2 took {self.time2} seconds. Starting Ginelli Step 3.\n')
        for n in tqdm(range(self.kC), disable=self.noprog):
            self._ginelli_step()
            self._history_R.append(self.R.copy()) # Only store R's at this stage

        self.time3 = tm.time() - start

        start = tm.time()
        # Pushing A's back with inverse R's
        print(f'Step 3 took {self.time3} seconds. Starting Ginelli Step 4.\n')
        for i in tqdm(range(self.kC), disable=self.noprog):
            # Pushing A- backwards with R's
            self.R = self._history_R[self.kC - i - 1]
            newA = np.linalg.solve(self.R, self.oldA)

            # Normalises A's to prevent overflow
            norms = np.linalg.norm(newA, axis=0, ord=2) # L2 of column norms.
            self.oldA = newA/norms
        self.time4 = tm.time() - start

        start = tm.time()

        # CLV Sampling
        print(f'Step 4 took {self.time4} seconds. Starting Ginelli Step 5.\n')
        for i in tqdm(range(self.kB), disable=self.noprog):

            # Pushing A- backwards with R's
            self.R = self._history_R[self.kB - i - 1]
            newA = np.linalg.solve(self.R, self.oldA)

            # Sampling FTCLE
            #ftcle = - np.log(np.diag(newA))/(tao * dt) # Norm for matching FTBLE
            norms = np.linalg.norm(newA, axis=0, ord=2) # L2 of column norms. Ensures CLVs are unit length
            self.FTCLE = - np.log(norms)/(self.tau)# Notice minus sign for contraction

            # Storage
            self._history_FTCLE.append(self.FTCLE.copy()) # These are stored backwards

            # Normalises A's to prevent overflow
            self.oldA = newA/norms

            # Calculate CLV, using A- and BLV
            self.BLV = self._history_BLV[self.kB - i - 1]
            self.CLV = np.matmul(self.BLV, self.oldA)
            self._history_CLV.append(self.CLV.copy())

        self._history_CLV.reverse() # Reordering lists
        self._history_FTCLE.reverse()
        self.time5 = tm.time() - start
        print(f'Step 5 took {self.time5} seconds. Finishing up.\n')

    @property
    def state(self):
        """Where we are"""
        return np.concatenate([self.X, self.Y, self.dx, self.dy])

    def set_state(self, x, tangent_x):
        """x is [X, Y]. tangent_x is [dx, dy]"""
        self.X = x[:self.K]
        self.Y = x[self.K:]
        self.dx = tangent_x[: self.K]
        self.dy = tangent_x[self.K: ]

    def erase_history(self):
        self._history_X = []
        self._history_Y_mean = []
        self._history_Y = []

    @property
    def metadata(self):
        param = {
        'h': self.h, # L96
        'Fs': self.Fs,
        'Ff': self.Ff,
        'c': self.c,
        'J': self.J,
        'K': self.K,
        'b': self.b,
        'dt': self.dt,
        'tau': self.tau, # Ginelli
        'kA': self.kA,
        'kB': self.kB,
        'kC': self.kC,
        'transient': self.transient, # Timings
        'Transient (s)': self.timet,
        'Step 1': self.time1,
        'Step 2': self.time2,
        'Step 3': self.time3,
        'Step 4': self.time4,
        'Step 5': self.time5,
        'Total (seconds)': (self.timet + self.time1 + self.time2 + self.time3 + self.time4 + self.time5)
        }
        return param

    @property
    def run_data(self):
        delT = self.tau # How oftern we save
        dic = {}
        dic['X'] = xr.DataArray(self._history_X, dims=['time', 'x'], name='X',
                                coords = {'time': np.arange(len(self._history_X)) * delT,'x': np.arange(self.K)})
        dic['Y'] = xr.DataArray(self._history_Y, dims=['time', 'y'], name='Y',
                                coords = {'time': np.arange(len(self._history_X)) * delT,'y': np.arange(self.K * self.J)})
        dic['FTBLE'] = xr.DataArray(self._history_FTBLE, dims=['time', 'le_index'], name='FTBLE',
                                coords = {'time': (self.transient + self.kA + np.arange(len(self._history_FTBLE))) * delT,
                                    'le_index':np.arange(1, 1 + self.size)})
        dic['FTCLE'] = xr.DataArray(self._history_FTCLE, dims=['time', 'le_index'], name='FTCLE',
                                coords = {'time': (self.transient + self.kA + np.arange(len(self._history_FTBLE))) * delT,
                                    'le_index':np.arange(1, 1 + self.size)})
        dic['BLV'] = xr.DataArray(self._history_BLV, dims=['time', 'component', 'le_index'], name='BLV',
                                coords = {'time': (self.transient + self.kA + np.arange(len(self._history_FTBLE))) * delT,
                                  'component': np.arange(self.size), 'le_index':np.arange(1, 1 + self.size)})
        dic['CLV'] = xr.DataArray(self._history_CLV, dims=['time', 'component', 'le_index'], name='CLV',
                                coords = {'time': (self.transient + self.kA + np.arange(len(self._history_FTBLE))) * delT,
                                  'component': np.arange(self.size), 'le_index':np.arange(1, 1 + self.size)})

        # Slow Variables above fast ones
        dic['X_repeat'] = xr.DataArray(np.repeat(self._history_X, self.J, axis=1),
                                   dims=['time', 'y'], name='X_repeat',
                                    coords = {'time': np.arange(len(self._history_X)) * delT,
                                              'y': np.arange(self.K * self.J)})# X's above the y's

        return xr.Dataset(dic, attrs= self.metadata)


      #WHEN LOADING DATA MIGHT BE NICE TO UNPACK META DATA LIKE THIS
#     def meta_data(self):
#         L96_parameters = {
#             'h': self.h,
#             'Fs': self.Fs,
#             'Ff': self.Ff,
#             'c': self.c,
#             'J': self.J,
#             'K': self.K,
#             'b': self.b,
#             'dt': self.dt
#         }

#         Ginelli_parameters = {
#             'tau': self.tau,
#             'kA': self.kA,
#             'kB': self.kB,
#             'kC': self.kC,
#             'transient': self.transient
#         }

#         timings = {
#             'Transient': self.timet,
#             'Step 1': self.time1,
#             'Step 2': self.time2,
#             'Step 3': self.time3,
#             'Step 4': self.time4,
#             'Step 5': self.time5,
#         }
#         total_time = sum(timings.values())
#         timings.update({'Total (seconds)': total_time})

#         return {
#             'L96_p': L96_parameters,
#             'Ginelli_p': Ginelli_parameters,
#             'Timings': timings
#         }

    def save_data(self):
        print('Saving Output')
        self.run_data.to_netcdf(f'L96_Ginelli_kc{self.kC:.2f}'.replace('.','_') + '.nc')
