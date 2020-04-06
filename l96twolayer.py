"""
Definition of the Lorenz96 model.

Created on 2019-04-16-12-28
Author: Stephan Rasp, raspstephan@gmail.com

Edited on 2020-03-20
by Calvin Nesbitt 
"""


class TL96TwoLevel(object):
    def __init__(self, K=36, J=10, h=1, Ff=6, Fs=10, c=10, dt=0.001,
                 X_init=None, Y_init=None, dx_init=None, dy_init=None, noprog=False, save_dt=0.1,
                 integration_type='uncoupled'): # X's and Y's will be np arrays
        # Model parameters
        self.K, self.J, self.h, self.Ff, self.Fs, self.c, self.dt = K, J, h, Ff, Fs, c, dt
        self.b = np.sqrt(J * c) # b restriction
        self.size = self.K + (self.J * self.K) # Number of variables
        
        # When we save
        self.step_count = 0
        self.save_dt = save_dt
        self.save_steps = int(save_dt / dt)
        
        # Progress Bar
        self.noprog = noprog

        # Non-linear Variables
        self.X = np.random.rand(self.K) if X_init is None else X_init.copy() # Random IC if none given
        self.Y = np.zeros(self.K * self.J) if Y_init is None else Y_init.copy() * self.b # ALL the y's
        self._history_X = [self.X.copy()]
        self._history_Y = [self.Y.copy()] # Y's in a list, remember they are in natural K lists
        self._history_Y_mean = [self.Y.reshape(self.K, self.J).mean(1).copy()] # This is reshaping all the Y's in to groups K, then storing K means.
        
        # TLE Variables
        self.dx = np.random.rand(self.K) if dx_init is None else dx_init.copy()
        self.dy = np.random.rand(self.K * self.J) if dy_init is None else dy_init.copy()
        self._history_dx = [self.dx.copy()]
        self._history_dy = [self.dy.copy()]
        self._history_dy_mean = [self.dy.reshape(self.K, self.J).mean(1).copy()] # This is reshaping all the Y's in to groups K, then storing K means.

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

    def step(self):
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
        
        # Saving output
        self.step_count += 1
        if self.step_count % self.save_steps == 0:
            Y_mean = self.Y.reshape(self.K, self.J).mean(1)
            self._history_X.append(self.X.copy())
            self._history_Y_mean.append(Y_mean.copy())
            self._history_Y.append(self.Y.copy())
            self._history_dx.append(self.dx.copy())
            self._history_dy.append(self.dy.copy())

    def iterate(self, time):
        """parameter, time: how longer we iterate for in adimensional time."""
        steps = int(time / self.dt)
        for n in tqdm(range(steps), disable=self.noprog):
            self.step()

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

    @property
    def parameters(self):
        return np.array([self.K, self.J, self.h, self.Ff, self.Fs, self.c, self.dt, self.b])

    def erase_history(self):
        self._history_X = []
        self._history_Y_mean = []
        self._history_Y = []

    @property
    def history(self):
        dic = {}
        dic['X'] = xr.DataArray(self._history_X, dims=['time', 'x'], name='X') 
        dic['Y'] = xr.DataArray(self._history_Y, dims=['time', 'y'], name='Y')
        dic['dX'] = xr.DataArray(self._history_dx, dims=['time', 'x'], name='dX')
        dic['dY'] = xr.DataArray(self._history_dy, dims=['time', 'y'], name='dY')
        dic['Y_mean'] = xr.DataArray(self._history_Y_mean, dims=['time', 'x'], name='Y_mean')
        
        # Slow Variables above fast ones
        dic['X_repeat'] = xr.DataArray(np.repeat(self._history_X, self.J, axis=1),
                                   dims=['time', 'y'], name='X_repeat') # X's above the y's
        dic['dx_repeat'] = xr.DataArray(np.repeat(self._history_dx, self.J, 1),
                                   dims=['time', 'y'], name='dX_repeat')
        return xr.Dataset(
            dic,
            coords={'time': np.arange(len(self._history_X)) * self.save_dt, 'x': np.arange(self.K),
                    'y': np.arange(self.K * self.J)}
        )

    def mean_stats(self, ax=None, fn=np.mean):
        h = self.history
        return np.concatenate([
            np.atleast_1d(fn(h.X, ax)),
            np.atleast_1d(fn(h.Y_mean, ax)),
            np.atleast_1d(fn((h.X ** 2), ax)),
            np.atleast_1d(fn((h.X * h.Y_mean), ax)),
        ])