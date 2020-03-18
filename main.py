# Implementation of the Ginelli algorithm for L63

from utility import *
from scipy.integrate import odeint
import numpy as np
import time as tm
import pickle

# Loading parameters from dictionary
infile = open('parameters','rb')
param_dict = pickle.load(infile)
infile.close()

## Initial conditions

# Assigning parameters for use
t0 = param_dict['t0']
tau = param_dict['tau'] # How many steps TLE pushes you forward
kA = param_dict['kA']
kB = param_dict['kB']
kC = param_dict['kC']
steps = kA + kB + kC # total number of steps
where = param_dict['where']

# Strength of initial perturbation in Ginelli algorithm
eps = param_dict['eps']
oldQ = eps * np.identity(3)

# L63 Parameters

a = param_dict['a']
b = param_dict['b']
c = param_dict['c']
p = [a, b, c]

timing = open("Timing.txt", mode='a') # Making a file to store timings
timing.write(f'Timings from Ginelli algorithm. tau: {tau}, Steps: {steps}\n')


# Lyapunov Vectors are indexed via step, row, column
BLVs = np.zeros((int(kB), 3, 3))
Rs = np.zeros((int(kB), 3, 3))  # Stretching rates
Rs2 = np.zeros((int(kC), 3, 3))  # Stretching rates
CLVs = np.zeros((int(kB), 3, 3))

# Lyapunov exponents are indexed via step, then index of LE, beggining with largest LE
FTBLE = np.zeros((int(kB), 3)) # To store time series of FTBLES
BLE = np.zeros((kB, 3)) # Storing running mean of FTBLEs
FTCLE = np.zeros((int(kB), 3)) # To store time series of FTBLES
CLE = np.zeros((kB, 3)) # Storing running mean of FTBLEs

solution = [[where[0]], [where[1]], [where[2]]] # For storing trajectory


# Step 1: Converging to the BLVs via Benettin Stepping

start = tm.time() #Timing
for i in np.arange(kA):
    # i will be storage index. Remember here storage(i) corresponds to time i * tau

    # Where we are
    step = i * tau
    #print([step, step + tau])

    # Solving L63 + TLE
    Pk, trajectory, time = solve(where, oldQ, step, tau) # Pk is matrix solution of TLE, trajectory is L63 solution
    where = trajectory[:, -1]

    # Storing L63 Trajectory
    solution[0].extend(trajectory[0, 1:]) # 1 is so we don't repeat points
    solution[1].extend(trajectory[1, 1:])
    solution[2].extend(trajectory[2, 1:])

    # QR Decomposition
    oldQ = Pk
    oldQ, R = posQR(Pk)

end = tm.time()
timing.write(f'It took {end - start} for step 1.\n')

print(f'It took {end - start} for step 1.\n')

# Step 2: More Benettin stepping. Here we sample BLVs and FTBLEs

start = tm.time()
tA = time[-1]
print(f'tA = {tA}')

for i in np.arange(kB):
    # i will be storage index. Remember here storage(i) corresponds to time: (tA + i * tau) * dt

    # Where we are
    step = tA + (i * tau)
    where = trajectory[:, -1]

    # Solving L63 + TLE
    Pk, trajectory, time = solve(where, oldQ, step, tau) # Pk is matrix solution of TLE, trajectory is L63 solution
    where = trajectory[:, -1]

    # QR
    oldQ = Pk
    oldQ, R = posQR(Pk) # Performing Q, R decomposition with positive diagonal

    # Storage of FTBLEs
    ftble = np.log(np.diag(R))/(tau) # Note division by tau is done here
    FTBLE[i] = ftble
    BLE[i] = np.mean(FTBLE[0:i + 1], axis = 0)
    BLVs[i]= oldQ
    Rs[i] = R

    # Storing L63 Trajectory
    solution[0].extend(trajectory[0, 1:]) # 1 is so we don't repeat points
    solution[1].extend(trajectory[1, 1:])
    solution[2].extend(trajectory[2, 1:])
end = tm.time()
timing.write(f'It took {end - start} for step 2.\n')

print(f'It took {end - start} for step 2.\n')


# Step 3: More Bennetin Stepping. Now we only store the Rs.

start = tm.time()

tB = time[-1]
print(tB)

for i in np.arange(kC):

    # Where we are
    step = tB + (i * tau)
    where = trajectory[:, -1]

    # Solving L63 + TLE
    Pk, trajectory, time = solve(where, oldQ, step, tau) # Pk is matrix solution of TLE, trajectory is L63 solution
    where = trajectory[:, -1]

    #QR
    oldQ = Pk
    oldQ, R = posQR(Pk)

    # Storage
    Rs2[i] =  R

    # Storing L63 Trajectory
    solution[0].extend(trajectory[0, 1:]) # 1 is so we don't repeat points
    solution[1].extend(trajectory[1, 1:])
    solution[2].extend(trajectory[2, 1:])

end = tm.time()
timing.write(f'It took {end - start} for step 3.\n')

print(f'It took {end - start} for step 3.\n')


# Step 4: Time to go back, converging to A- matrix (Coefficient matrix of CLVs in BLV basis)

# Initialise an upper triangular matrix
A = np.identity(3)
A[0,1] = 1
oldA = A

start = tm.time()

tC = time[-1]
print(tC)

for i in np.arange(kC):

    # Where we are
    step = tC - (i * tau)
    #print([step, step - tau])

    # Pushing A- backwards with R's
    R = Rs2[kC - i - 1]
    newA = np.linalg.solve(R, oldA)

    # Normalises A's to prevent overflow
    norms = np.linalg.norm(newA, axis=0, ord=2) # L2 of column norms.
    oldA = newA/norms

    # Storing L63 Trajectory
    solution[0].extend(trajectory[0, 1:]) # 1 is so we don't repeat points
    solution[1].extend(trajectory[1, 1:])
    solution[2].extend(trajectory[2, 1:])

end = tm.time()
timing.write(f'It took {end - start} for step 4.\n')
print(f'It took {end - start} for step 4.\n')

# Step 5: Keep going back, finding CLVs. Sample FTCLEs here.

start = tm.time()
for i in np.arange(kB):

    # Where we are
    step = tB - (i * tau) # Time is step * tau

    # Pushing A- backwards with R's
    R = Rs[kB - i - 1]
    newA = np.linalg.solve(R, oldA)

    # Sampling FTCLE
    #ftcle = - np.log(np.diag(newA))/(tao * dt) # Norm for matching FTBLE
    norms = np.linalg.norm(newA, axis=0, ord=2) # L2 of column norms. Ensures CLVs are unit length
    ftcle = - np.log(norms)/(tau)# Notice minus sign for contraction

    # Storage
    FTCLE[kB - i - 1] = ftcle
    flipped = np.flip(FTCLE, axis = 0) #Flipped so we can calculate running mean
    CLE[kB - i - 1] = np.mean(flipped[0:i + 1], axis = 0) # Time series of estimated LE spectrum from FTCLEs

    # Storing L63 Trajectory
    solution[0].extend(trajectory[0, 1:]) # 1 is so we don't repeat points
    solution[1].extend(trajectory[1, 1:])
    solution[2].extend(trajectory[2, 1:])

    # Normalises A's to prevent overflow
    oldA = newA/norms

    # Calculate CLV, using A- and BLV
    BLV = BLVs[kB - i - 1]
    CLVs[kB - i - 1] = np.matmul(BLV, oldA)
end = tm.time()
timing.write(f'It took {end - start} for step 5.\n')

# Saving Results of run

print('Ginelli Algorithm Ran. Saving Data.')
path ='Data/'# Need /
np.save(f'{path}solution',solution) #Trajectory
np.save(f'{path}BLVs',BLVs)
np.save(f'{path}Rs',Rs)
np.save(f'{path}Rs2',Rs2)
np.save(f'{path}CLVs',CLVs)
np.save(f'{path}FTBLE',FTBLE)
np.save(f'{path}BLE',BLE)
np.save(f'{path}FTCLE',FTCLE)
np.save(f'{path}CLE',CLE)
