"""
Script that runs the Ginelli Algorithm.
Relies on the ginelli_observers and ginelli_utilities files.

- User specifies integrator, memory and time parameters.
- Script will output directories containing xarrays and a dictionary of timings.
- The 'Step 5' directory will contain the BLVs, CLVs and corresponding spectra.
"""
# -----------------------------------------
# Imports
# -----------------------------------------
import numpy as np
import pickle
import time as tm
import xarray as xr
import os
import sys

# Lorenz 96 integrator
import l96_integrators as l96

# Dependencies
import ginelli_utilities as utilities
from ginelli_observers import *

# -----------------------------------------
# Setup & Parameter Choices
# -----------------------------------------

# h experiment values
values = np.linspace(0, 15, 5)
c = values[int(sys.argv[1]) - 1]

dump_size = 500 # How many observations before output

# Time Parameter Choices
tau = 0.01 # time between QR decompositions
transient = 50
ka = 2500 # BLV convergence
kb = 10000 # Number of observations
kc = 2500 # CLV convergence

# Integrator
runner = l96.Integrator(c=c)
tangent_runner = l96.TangentIntegrator(c=c)
ginelli_runner = utilities.Forward(tangent_runner, tau)

# Observables
Rlooker = RMatrixObserver(ginelli_runner)
BLVlooker = BLVMatrixObserver(ginelli_runner)
TrajectoryLooker = TrajectoryObserver(ginelli_runner)

# Observables
Rlooker = RMatrixObserver(ginelli_runner)
BLVlooker = BLVMatrixObserver(ginelli_runner)
TrajectoryLooker = TrajectoryObserver(ginelli_runner)

# Timing the algorithm
timings = {}
start = tm.time()

# Making Output Storage
utilities.make_cupboard()

# -----------------------------------------
# Forward Part of Ginelli Algorithm
# -----------------------------------------

# -----------------------------------------
# Transient and Step 1, BLV Convergence
# -----------------------------------------

print('\nTransient beginning.\n')
runner.integrate(transient)
tangent_runner.set_state(runner.state, tangent_runner.tangent_state)
print('\nTransient finished. Beginning BLV convergence steps.\n')

timings.update({'transient': tm.time() - start})

# BLV Convergence steps

ginelli_runner.run(ka, noprog=False)

timings.update({'Step1': tm.time() - timings['transient'] - start})
pickle.dump(timings, open("ginelli/timings.p", "wb" ))

# -----------------------------------------
# Step 2, BLV Observation.
# -----------------------------------------

print('\nBLV convergence finished. Beginning to observe BLVs.\n')
blocks = int(kb/dump_size) # How many times we dump
remainder = kb%dump_size # Rest of observations

for i in tqdm(range(blocks)):

    utilities.make_observations(ginelli_runner, [Rlooker, BLVlooker, TrajectoryLooker], dump_size, 1, noprog=False)
    # Observation frequency has to be 1 if we're reversing CLVs
    Rlooker.dump('ginelli/step2/R')
    BLVlooker.dump('ginelli/step2/BLV')
    TrajectoryLooker.dump('ginelli/trajectory')

if (remainder !=0):
    utilities.make_observations(ginelli_runner, [Rlooker, BLVlooker, TrajectoryLooker], remainder, 1, noprog=False)
    Rlooker.dump('ginelli/step2/R')
    BLVlooker.dump('ginelli/step2/BLV')
    TrajectoryLooker.dump('ginelli/trajectory')

timings.update({'Step2': tm.time() - timings['Step1'] - start})
pickle.dump(timings, open("ginelli/timings.p", "wb" ))

# -----------------------------------------
# Step 3, CLV Convergence Step, Forward
# -----------------------------------------

print('\nBLV observations finished. CLV convergence beginning. Just observing Rs.\n')
blocks = int(kc/dump_size)
remainder = kc%dump_size

for i in range(blocks):

    utilities.make_observations(ginelli_runner, [Rlooker], dump_size, 1, noprog=False)
    Rlooker.dump('ginelli/step3')

if (remainder !=0):
    utilities.make_observations(ginelli_runner, [Rlooker], remainder, 1, noprog=False)
    Rlooker.dump('ginelli/step3')

print('\n\n****************************************************************')
print('Forward part all done :)')
print('****************************************************************\n\n')

timings.update({'Step3': tm.time() - timings['Step2'] - start})
pickle.dump(timings, open("ginelli/timings.p", "wb" ))

# -----------------------------------------
# Backward Part of Ginelli Algorithm
# -----------------------------------------

# -----------------------------------------
# Step 4, Reversing CLV Convergence Steps
# -----------------------------------------

R_files = os.listdir('ginelli/step3')
R_files.sort(reverse=True) # Ensuring files are in the right order

A = np.identity(ginelli_runner.size)

for file in R_files:
    R_history = xr.open_dataset('ginelli/step3/' + file)
    A = utilities.block_squish_norm(R_history, A) # This A is one timestep ahead of the R that pushed it
    R_history.close()
    print(f'Pushed A through {file}. Overwriting A.npy.\n')
    np.save('ginelli/step4/A.npy', A)

timings.update({'Step4': tm.time() - timings['Step3'] - start})
pickle.dump(timings, open("ginelli/timings.p", "wb" ))

# -----------------------------------------
# Step 5, Observing LVs and LEs
# -----------------------------------------

# Sorting files we will be working with

R_files = os.listdir('ginelli/step2/R')
R_files.sort(reverse=True)

BLV_files = os.listdir('ginelli/step2/BLV')
BLV_files.sort(reverse=True)

# Setting up observable storage

parameters = ginelli_runner.parameter_dict.copy()
parameters.update({'transient':transient,'ka':ka, 'kb':kb, 'kc':kc})
LyapunovLooker = LyapunovObserver(parameters, len(BLV_files))

for [rfile, bfile] in zip(R_files, BLV_files): # Loop over files that were dumped
    R_history = xr.open_dataset('ginelli/step2/R/' + rfile)
    BLV_history = xr.open_dataset('ginelli/step2/BLV/' + bfile)

    Rs, BLVs = np.flip(R_history.R, axis = 0), np.flip(BLV_history.BLV, axis = 0) # Times series reversed

    for R, BLV in zip(Rs, BLVs):

        # CLV Calculation
        CLV = np.matmul(BLV.values, A)

        # FTCLE Calculation
        squishedA = np.linalg.solve(R, A)
        norms = np.linalg.norm(squishedA, axis=0, ord=2)
        ftcle = - np.log(norms)/(tau)
        A = squishedA/norms

        # FTBLE Calculation
        ftble = np.log(np.diag(R))/(tau)

        # Making observation
        time = R.time.item()
        LyapunovLooker.look(time, CLV, BLV.values, ftcle, ftble)


    LyapunovLooker.dump('ginelli/step5')
    R_history.close()
    BLV_history.close()

timings.update({'Step5': tm.time() - timings['Step4'] - start})
pickle.dump(timings, open("ginelli/timings.p", "wb" ))

print('Ginelli Algorithm finished.')
