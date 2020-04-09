"""Functions and Classes needed for implementation of Ginelli Algorithm.
----------
Contents
----------
- posQR, function that performs QR decomposition with postivie entries on the R diagonal

- Forward, class for performing the forward integration steps of the Ginelli algorithm"""

import numpy as np
import xarray as xr

def posQR(M):
    """ Returns QR decomposition of a matrix with positive diagonals on R.
    Parameter, M: Array that is being decomposed
    """
    Q, R = np.linalg.qr(M) # Performing QR decomposition
    signs = np.diag(np.sign(np.diagonal(R))) # Matrix with signs of R diagonal on the diagonal
    Q, R = np.dot(Q, signs), np.dot(signs, R) # Ensuring R Diagonal is positive
    return Q, R

class Forward:
    """Performs forward steps in Ginelli algorithm. Relies on a tangent integrator object"""
    
    def __init__(self, integrator, tau, oldQ = None):
        """param, integrator: object to integrate both TLE and system itself.
        param, tau: adimensional time between orthonormalisations."""
        
        self.integrator, self.tau = integrator, tau
        self.step_count = 0
        
         # Info we need from the integrator
        self.size = integrator.size # size of original + linearised system.
       
        # Stretched matrix.
        self.P = np.random.rand(self.size, self.size) # Stretched Matrix
        
        # Initialising orthogonal matrix
        
        if (oldQ == None):
            eps = 0.0001
            self.oldQ = eps * np.identity(self.size)
            self.oldQ[0, 1] = eps * 1    
        
        else:
            self.oldQ = oldQ
        
        # Stretching rates after QR decomposition
        self.R = np.random.rand(self.size, self.size)
        
    @property
    def time(self):
        return self.integrator.time
    @property
    def parameter_dict(self):
        ginelli_params = {'tau':self.tau}
        combined_dict = {**self.integrator.parameter_dict , **ginelli_params}
        return combined_dict
             
    def _step(self):
            """Perform one QR step. Take old Q, stretch it, do a QR decomposition.
            param, location: where we are in phase space.
            param, """

            # Where we are in phase space before ginelli step
            phase_state = self.integrator.state
            step = self.integrator.step_count

            # Stretching first column
            self.integrator.set_state(phase_state, self.oldQ.T[0]) # First column of Q is ic for TLE
            self.integrator.integrate(self.tau)

            # First column of Stretched matirx
            self.P[:, 0] = self.integrator.tangent_state

            # Stretching the rest of the columns
            for i, column in enumerate(self.oldQ.T[1:]):
               
                # Reseting to where we were in phase space
                self.integrator.set_state(phase_state, column)
                self.integrator.step_count = step
                
                self.integrator.integrate(self.tau)
                self.P[:, i] = self.integrator.tangent_state
                
            # QR decomposition
            self.oldQ, self.R = posQR(self.P)
            self.step_count += 1
            
    def run(self, steps):
        """Performs specified number of Ginelli Steps"""
        for i in range(steps):
            self._step()