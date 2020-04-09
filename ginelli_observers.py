"""Observer classes for the Ginelli Algorithm.
----------
Contents
----------
- RMatrixObserver, class for observing the R matrices. Note the FTBLEs can be calculated from the diagonals of these Rs.

- BLVMatrixObserver, class for observing the R matrices."""

import numpy as np
import xarray as xr

class RMatrixObserver:
    """Observes the R Matrix in the forward Ginelli Steps."""

    def __init__(self, ginelli):
        """param, ginelli: Ginelli Forward Stepper being obseved."""
        
        # Knowledge associated with this observer
        self.ginelli = ginelli # stepper we're associated with
        self.parameters = ginelli.parameter_dict

        # R Observation log
        self.R_obs = []
        self.time_obs = []

    def look(self, ginelli):
        """Observes trajectory of L96 trajectory"""

        # Note the time
        self.time_obs.append(ginelli.time)

        # Making Observations
        self.R_obs.append(ginelli.R.copy())
    
    @property
    def observations(self):
        """cupboard: Directory where to write netcdf."""
        if (len(self.R_obs) == 0):
            print('I have no observations! :(')
            return

        dic = {}
        _time = self.time_obs
        dic['R'] = xr.DataArray(self.R_obs, dims=['time', 'row', 'column'], name='R',
                                coords = {'time': _time})

        return xr.Dataset(dic, attrs=self.parameters)

    def wipe(self):
        """Erases observations"""
        self.time_obs = []
        self.R_obs = []
    
    def dump(self, cupboard):
        """ Saves observations to netcdf and wipes.
        cupboard: Directory where to write netcdf."""
        self.observations.to_netcdf(cupboard)
        print(f'Observations written to {cupboard}. Erasing personal log.\n')
        self.wipe()
        

class BLVMatrixObserver:
    """Observes the Q Matrix in the forward Ginelli Steps."""

    def __init__(self, ginelli):
        """param, ginelli: Ginelli Forward Stepper being obseved."""
        
        # Knowledge from Ginelli object
        self.ginelli = ginelli # stepper we're associated with
        self.parameters = ginelli.parameter_dict
        self.le_index = np.arange(1, 1 + ginelli.size)

        # BLV Observation log
        self.BLV_obs = []
        self.time_obs = []

    def look(self, ginelli):
        """Observes trajectory of L96 trajectory"""

        # Note the time
        self.time_obs.append(ginelli.time)

        # Making Observations
        self.BLV_obs.append(ginelli.oldQ.copy())
    
    @property
    def observations(self):
        """cupboard: Directory where to write netcdf."""
        if (len(self.BLV_obs) == 0):
            print('I have no observations! :(')
            return

        dic = {}
        _time = self.time_obs
        dic['BLV'] = xr.DataArray(self.BLV_obs, dims=['time', 'row', 'le_index'], name='BLV',
                                coords = {'time': _time, 'le_index': self.le_index})

        return xr.Dataset(dic, attrs=self.parameters)

    def wipe(self):
        """Erases observations"""
        self.time_obs = []
        self.BLV_obs = []
    
    def dump(self, cupboard):
        """ Saves observations to netcdf and wipes.
        cupboard: Directory where to write netcdf."""
        self.observations.to_netcdf(cupboard)
        print(f'Observations written to {cupboard}. Erasing personal log.\n')
        self.wipe()